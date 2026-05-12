import "./style.css";
import { WORKGROUP_SIZE, defaultSettings, type SimulationSettings } from "./config";
import {
  createGridBuffers,
  createMaterialBuffer,
  createParticleBuffers,
  createUniformBuffer,
  getGridDimensions,
  type MaterialPreset,
  writeMaterialParams,
  writeSimParams
} from "./gpu/buffers";
import { initWebGPU } from "./gpu/initWebGPU";
import { createPipelines } from "./gpu/pipelines";
import { PointerInput } from "./input";

const canvasElement = document.querySelector<HTMLCanvasElement>("#sim-canvas");
const statusElement = document.querySelector<HTMLElement>("#gpu-status");

if (!canvasElement || !statusElement) {
  throw new Error("Missing required DOM nodes.");
}

const canvas: HTMLCanvasElement = canvasElement;
const statusLabel: HTMLElement = statusElement;

void start();

async function start() {
  const settings = defaultSettings();
  const ui = bindControls(settings);

  try {
    const gpu = await initWebGPU(canvas);
    const pointer = new PointerInput(canvas);
    const particles = createParticleBuffers(gpu.device);
    const grid = createGridBuffers(gpu.device);
    const uniforms = createUniformBuffer(gpu.device);
    const materials = createMaterialBuffer(gpu.device);
    const pipelines = createPipelines(gpu.device, gpu.format, particles.buffers, uniforms, grid, materials);

    gpu.device.lost.then((info) => {
      statusLabel.textContent = `WebGPU device lost: ${info.message || info.reason}`;
    });

    gpu.resize();
    const world = gpu.getWorldSize();
    particles.reset(gpu.device, settings.particleCount, world.width, world.height);
    statusLabel.textContent = `${gpu.adapter.info?.vendor || "GPU"} adapter ready`;

    ui.onReset = () => {
      const size = gpu.getWorldSize();
      particles.reset(gpu.device, settings.particleCount, size.width, size.height);
    };

    ui.onMaterialPresetChange = (preset) => {
      writeMaterialParams(gpu.device, materials, preset);
    };

    let paused = false;
    let lastTime = performance.now();

    ui.onPauseToggle = () => {
      paused = !paused;
      ui.setPaused(paused);
      lastTime = performance.now();
    };

    const frame = (now: number) => {
      const elapsed = Math.min((now - lastTime) / 1000, 1 / 30);
      lastTime = now;
      gpu.resize();
      const size = gpu.getWorldSize();
      const deltaTime = paused ? 0 : elapsed;

      writeSimParams(gpu.device, uniforms, settings, pointer.state, size.width, size.height, deltaTime);

      const encoder = gpu.device.createCommandEncoder({ label: "Frame command encoder" });

      if (!paused) {
        const activeGrid = getGridDimensions(size.width, size.height);
        const activeCellCount = activeGrid.columns * activeGrid.rows;

        const clearGridPass = encoder.beginComputePass({ label: "Clear spatial grid pass" });
        clearGridPass.setPipeline(pipelines.clearGridPipeline);
        clearGridPass.setBindGroup(0, pipelines.clearGridBindGroup);
        clearGridPass.dispatchWorkgroups(Math.ceil(activeCellCount / WORKGROUP_SIZE));
        clearGridPass.end();

        const countGridPass = encoder.beginComputePass({ label: "Count spatial grid pass" });
        countGridPass.setPipeline(pipelines.countGridPipeline);
        countGridPass.setBindGroup(0, pipelines.countGridBindGroups[particles.activeIndex]);
        countGridPass.dispatchWorkgroups(Math.ceil(settings.particleCount / WORKGROUP_SIZE));
        countGridPass.end();

        const cellScanGroups = Math.ceil(activeCellCount / grid.scanBlockSize);

        const scanCellStartsPass = encoder.beginComputePass({ label: "Scan cell starts pass" });
        scanCellStartsPass.setPipeline(pipelines.scanCellStartsPipeline);
        scanCellStartsPass.setBindGroup(0, pipelines.scanCellStartsBindGroup);
        scanCellStartsPass.dispatchWorkgroups(cellScanGroups);
        scanCellStartsPass.end();

        const scanGroupOffsetsPass = encoder.beginComputePass({ label: "Scan grid group offsets pass" });
        scanGroupOffsetsPass.setPipeline(pipelines.scanGroupOffsetsPipeline);
        scanGroupOffsetsPass.setBindGroup(0, pipelines.scanGroupOffsetsBindGroup);
        scanGroupOffsetsPass.dispatchWorkgroups(1);
        scanGroupOffsetsPass.end();

        const addCellOffsetsPass = encoder.beginComputePass({ label: "Add cell offsets pass" });
        addCellOffsetsPass.setPipeline(pipelines.addCellOffsetsPipeline);
        addCellOffsetsPass.setBindGroup(0, pipelines.addCellOffsetsBindGroup);
        addCellOffsetsPass.dispatchWorkgroups(Math.ceil(activeCellCount / WORKGROUP_SIZE));
        addCellOffsetsPass.end();

        const scatterGridPass = encoder.beginComputePass({ label: "Scatter spatial grid pass" });
        scatterGridPass.setPipeline(pipelines.scatterGridPipeline);
        scatterGridPass.setBindGroup(0, pipelines.scatterGridBindGroups[particles.activeIndex]);
        scatterGridPass.dispatchWorkgroups(Math.ceil(settings.particleCount / WORKGROUP_SIZE));
        scatterGridPass.end();

        const simulatePass = encoder.beginComputePass({ label: "Particle simulation pass" });
        simulatePass.setPipeline(pipelines.simulatePipeline);
        simulatePass.setBindGroup(0, pipelines.simulateBindGroups[particles.activeIndex]);
        simulatePass.dispatchWorkgroups(Math.ceil(settings.particleCount / WORKGROUP_SIZE));
        simulatePass.end();

        particles.swap();
      }

      const renderPass = encoder.beginRenderPass({
        label: "Particle render pass",
        colorAttachments: [
          {
            view: gpu.context.getCurrentTexture().createView(),
            clearValue: { r: 0.025, g: 0.03, b: 0.035, a: 1 },
            loadOp: "clear",
            storeOp: "store"
          }
        ]
      });

      renderPass.setPipeline(pipelines.renderPipeline);
      renderPass.setBindGroup(0, pipelines.renderBindGroups[particles.activeIndex]);
      renderPass.draw(6, settings.particleCount);
      renderPass.end();

      gpu.device.queue.submit([encoder.finish()]);
      requestAnimationFrame(frame);
    };

    requestAnimationFrame(frame);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    statusLabel.textContent = message;
    console.error(error);
  }
}

type ControlBindings = {
  onReset: () => void;
  onPauseToggle: () => void;
  onMaterialPresetChange: (preset: MaterialPreset) => void;
  setPaused: (paused: boolean) => void;
};

function bindControls(settings: SimulationSettings): ControlBindings {
  const particleCount = getInput("particle-count");
  const particleCountOutput = getOutput("particle-count-output");
  const gravity = getInput("gravity");
  const gravityOutput = getOutput("gravity-output");
  const damping = getInput("damping");
  const dampingOutput = getOutput("damping-output");
  const mouseForce = getInput("mouse-force");
  const mouseForceOutput = getOutput("mouse-force-output");
  const repulsion = getInput("repulsion");
  const repulsionOutput = getOutput("repulsion-output");
  const cohesion = getInput("cohesion");
  const cohesionOutput = getOutput("cohesion-output");
  const materialPreset = getSelect("material-preset");
  const pauseButton = getButton("pause-button");
  const resetButton = getButton("reset-button");

  const bindings: ControlBindings = {
    onReset: () => undefined,
    onPauseToggle: () => undefined,
    onMaterialPresetChange: () => undefined,
    setPaused(paused: boolean) {
      pauseButton.textContent = paused ? "Resume" : "Pause";
    }
  };

  const refresh = () => {
    particleCountOutput.value = settings.particleCount.toLocaleString();
    gravityOutput.value = settings.gravityY.toFixed(0);
    dampingOutput.value = settings.damping.toFixed(2);
    mouseForceOutput.value = settings.mouseForce.toLocaleString();
    repulsionOutput.value = settings.particleRepulsion.toFixed(0);
    cohesionOutput.value = settings.cohesion.toFixed(2);
  };

  particleCount.addEventListener("input", () => {
    settings.particleCount = particleCount.valueAsNumber;
    refresh();
  });
  particleCount.addEventListener("change", () => bindings.onReset());

  gravity.addEventListener("input", () => {
    settings.gravityY = gravity.valueAsNumber;
    refresh();
  });

  damping.addEventListener("input", () => {
    settings.damping = damping.valueAsNumber;
    refresh();
  });

  mouseForce.addEventListener("input", () => {
    settings.mouseForce = mouseForce.valueAsNumber;
    refresh();
  });

  repulsion.addEventListener("input", () => {
    settings.particleRepulsion = repulsion.valueAsNumber;
    refresh();
  });

  cohesion.addEventListener("input", () => {
    settings.cohesion = cohesion.valueAsNumber;
    refresh();
  });

  materialPreset.addEventListener("change", () => {
    bindings.onMaterialPresetChange(materialPreset.value as MaterialPreset);
  });

  pauseButton.addEventListener("click", () => bindings.onPauseToggle());
  resetButton.addEventListener("click", () => bindings.onReset());
  refresh();
  return bindings;
}

function getInput(id: string): HTMLInputElement {
  const input = document.getElementById(id);
  if (!(input instanceof HTMLInputElement)) {
    throw new Error(`Missing input #${id}`);
  }
  return input;
}

function getOutput(id: string): HTMLOutputElement {
  const output = document.getElementById(id);
  if (!(output instanceof HTMLOutputElement)) {
    throw new Error(`Missing output #${id}`);
  }
  return output;
}

function getButton(id: string): HTMLButtonElement {
  const button = document.getElementById(id);
  if (!(button instanceof HTMLButtonElement)) {
    throw new Error(`Missing button #${id}`);
  }
  return button;
}

function getSelect(id: string): HTMLSelectElement {
  const select = document.getElementById(id);
  if (!(select instanceof HTMLSelectElement)) {
    throw new Error(`Missing select #${id}`);
  }
  return select;
}
