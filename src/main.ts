import "./style.css";
import { WORKGROUP_SIZE, defaultSettings, type SimulationSettings } from "./config";
import {
  createBodyBuffer,
  createGridBuffers,
  createMaterialBuffer,
  createParticleBuffers,
  createUniformBuffer,
  getGridDimensions,
  type MaterialPreset,
  type SoftBodyShape,
  writeMaterialParams,
  writeSimParams,
  type GridBuffers,
  type ParticleBuffers
} from "./gpu/buffers";
import { initWebGPU } from "./gpu/initWebGPU";
import { createPipelines, type Pipelines } from "./gpu/pipelines";
import { PointerInput } from "./input";

const canvasElement = document.querySelector<HTMLCanvasElement>("#sim-canvas");
const statusElement = document.querySelector<HTMLElement>("#gpu-status");
const debugElement = document.querySelector<HTMLElement>("#debug-stats");

if (!canvasElement || !statusElement || !debugElement) {
  throw new Error("Missing required DOM nodes.");
}

const canvas: HTMLCanvasElement = canvasElement;
const statusLabel: HTMLElement = statusElement;
const debugLabel: HTMLElement = debugElement;

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
    const bodies = createBodyBuffer(gpu.device);
    const pipelines = createPipelines(gpu.device, gpu.format, particles.buffers, uniforms, grid, materials, bodies);

    gpu.device.lost.then((info) => {
      statusLabel.textContent = `WebGPU device lost: ${info.message || info.reason}`;
    });

    gpu.resize();
    particles.clear(gpu.device, bodies);
    settings.particleCount = particles.particleCount;
    ui.setBodyStats(particles.bodyCount, particles.particleCount);
    statusLabel.textContent = `${gpu.adapter.info?.vendor || "GPU"} adapter ready`;

    ui.onReset = () => {
      particles.clear(gpu.device, bodies);
      settings.particleCount = particles.particleCount;
      ui.setBodyStats(particles.bodyCount, particles.particleCount);
    };

    ui.onAddBody = (shape, bodySize, particleRadius) => {
      const size = gpu.getWorldSize();
      const result = particles.addSoftBody(gpu.device, bodies, shape, bodySize, particleRadius, size.width, size.height);
      if (result.added) {
        settings.particleCount = result.particleCount;
        ui.setBodyStats(result.bodyCount, result.particleCount);
        statusLabel.textContent = `Added ${shape} body (${result.addedParticles.toLocaleString()} particles)`;
      } else {
        statusLabel.textContent = result.reason;
      }
    };

    ui.onMaterialPresetChange = (preset) => {
      writeMaterialParams(gpu.device, materials, preset);
    };

    let paused = false;
    let lastTime = performance.now();
    let latestMaxCellOccupancy = 0;
    let latestGridOverflow = 0;
    let debugReadbackPending = false;
    let lastDebugReadback = 0;
    let lastDebugUiRefresh = 0;
    let smoothedFrameMs = 0;
    let smoothedSimulationEncodeMs = 0;

    ui.onPauseToggle = () => {
      paused = !paused;
      ui.setPaused(paused);
      lastTime = performance.now();
    };

    const frame = (now: number) => {
      const frameStart = performance.now();
      const elapsed = Math.min((now - lastTime) / 1000, 1 / 30);
      lastTime = now;
      gpu.resize();
      const size = gpu.getWorldSize();
      const deltaTime = paused ? 0 : elapsed;
      let simulationEncodeMs = 0;

      if (!paused && settings.particleCount > 0) {
        const substeps = clampSubsteps(settings.substeps);
        const subDeltaTime = deltaTime / substeps;
        const activeGrid = getGridDimensions(size.width, size.height);
        const activeCellCount = activeGrid.columns * activeGrid.rows;
        const cellScanGroups = Math.ceil(activeCellCount / grid.scanBlockSize);
        writeSimParams(gpu.device, uniforms, settings, pointer.state, size.width, size.height, subDeltaTime);

        const simulationStart = performance.now();
        const simulationEncoder = gpu.device.createCommandEncoder({ label: "Simulation command encoder" });
        for (let step = 0; step < substeps; step += 1) {
          encodeSimulationStep(
            simulationEncoder,
            pipelines,
            particles,
            grid,
            activeCellCount,
            cellScanGroups,
            settings.particleCount
          );
          particles.swap();
        }
        gpu.device.queue.submit([simulationEncoder.finish()]);
        simulationEncodeMs = performance.now() - simulationStart;
      } else {
        writeSimParams(gpu.device, uniforms, settings, pointer.state, size.width, size.height, 0);
      }

      const renderEncoder = gpu.device.createCommandEncoder({ label: "Render command encoder" });
      const shouldReadDebug = !debugReadbackPending && !paused && now - lastDebugReadback > 250;
      if (shouldReadDebug) {
        renderEncoder.copyBufferToBuffer(grid.debugCounters, 0, grid.debugReadback, 0, grid.debugCounterBytes);
      }

      const renderPass = renderEncoder.beginRenderPass({
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

      gpu.device.queue.submit([renderEncoder.finish()]);
      if (shouldReadDebug) {
        debugReadbackPending = true;
        lastDebugReadback = now;
        grid.debugReadback
          .mapAsync(GPUMapMode.READ)
          .then(() => {
            try {
              const counters = new Uint32Array(grid.debugReadback.getMappedRange());
              latestMaxCellOccupancy = counters[0] ?? 0;
              latestGridOverflow = counters[1] ?? 0;
            } finally {
              grid.debugReadback.unmap();
            }
          })
          .catch(() => {
            latestMaxCellOccupancy = 0;
            latestGridOverflow = 0;
          })
          .finally(() => {
            debugReadbackPending = false;
          });
      }

      const frameMs = performance.now() - frameStart;
      smoothedFrameMs = smoothTiming(smoothedFrameMs, frameMs);
      smoothedSimulationEncodeMs = smoothTiming(smoothedSimulationEncodeMs, simulationEncodeMs);
      if (now - lastDebugUiRefresh > 100) {
        ui.setDebugStats({
          substeps: clampSubsteps(settings.substeps),
          maxCellOccupancy: latestMaxCellOccupancy,
          gridOverflow: latestGridOverflow,
          frameMs: smoothedFrameMs,
          simulationEncodeMs: smoothedSimulationEncodeMs
        });
        lastDebugUiRefresh = now;
      }

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
  onAddBody: (shape: SoftBodyShape, size: number, particleRadius: number) => void;
  onPauseToggle: () => void;
  onMaterialPresetChange: (preset: MaterialPreset) => void;
  setPaused: (paused: boolean) => void;
  setBodyStats: (bodyCount: number, particleCount: number) => void;
  setDebugStats: (stats: DebugStats) => void;
};

type DebugStats = {
  substeps: number;
  maxCellOccupancy: number;
  gridOverflow: number;
  frameMs: number;
  simulationEncodeMs: number;
};

function bindControls(settings: SimulationSettings): ControlBindings {
  const bodyShape = getSelect("body-shape");
  const bodySize = getInput("body-size");
  const bodySizeOutput = getOutput("body-size-output");
  const particleRadius = getInput("particle-radius");
  const particleRadiusOutput = getOutput("particle-radius-output");
  const bodyStats = getOutput("body-stats-output");
  const gravity = getInput("gravity");
  const gravityOutput = getOutput("gravity-output");
  const damping = getInput("damping");
  const dampingOutput = getOutput("damping-output");
  const substeps = getInput("substeps");
  const substepsOutput = getOutput("substeps-output");
  const bondIterations = getInput("bond-iterations");
  const bondIterationsOutput = getOutput("bond-iterations-output");
  const softBodyStrength = getInput("soft-body-strength");
  const softBodyStrengthOutput = getOutput("soft-body-strength-output");
  const viscosity = getInput("viscosity");
  const viscosityOutput = getOutput("viscosity-output");
  const mouseForce = getInput("mouse-force");
  const mouseForceOutput = getOutput("mouse-force-output");
  const repulsion = getInput("repulsion");
  const repulsionOutput = getOutput("repulsion-output");
  const cohesion = getInput("cohesion");
  const cohesionOutput = getOutput("cohesion-output");
  const materialPreset = getSelect("material-preset");
  const pauseButton = getButton("pause-button");
  const resetButton = getButton("reset-button");
  const addBodyButton = getButton("add-body-button");

  const bindings: ControlBindings = {
    onReset: () => undefined,
    onAddBody: () => undefined,
    onPauseToggle: () => undefined,
    onMaterialPresetChange: () => undefined,
    setPaused(paused: boolean) {
      pauseButton.textContent = paused ? "Resume" : "Pause";
    },
    setBodyStats(bodyCount: number, particleCount: number) {
      bodyStats.value = `${bodyCount.toLocaleString()} / ${particleCount.toLocaleString()}`;
    },
    setDebugStats(stats) {
      const overflow = stats.gridOverflow > 0 ? ` | overflow ${stats.gridOverflow.toLocaleString()}` : "";
      debugLabel.textContent = `substeps ${stats.substeps}x | cell max ${stats.maxCellOccupancy.toLocaleString()}${overflow} | encode ${stats.simulationEncodeMs.toFixed(1)}ms | frame ${stats.frameMs.toFixed(1)}ms`;
    }
  };

  gravity.value = String(settings.gravityY);
  damping.value = String(settings.damping);
  substeps.value = String(settings.substeps);
  bondIterations.value = String(settings.bondIterations);
  softBodyStrength.value = String(settings.softBodyStrength);
  viscosity.value = String(settings.viscosity);
  mouseForce.value = String(settings.mouseForce);
  repulsion.value = String(settings.particleRepulsion);
  cohesion.value = String(settings.cohesion);

  const refresh = () => {
    bodySizeOutput.value = bodySize.valueAsNumber.toFixed(0);
    particleRadiusOutput.value = particleRadius.valueAsNumber.toFixed(1);
    gravityOutput.value = settings.gravityY.toFixed(0);
    dampingOutput.value = settings.damping.toFixed(2);
    substepsOutput.value = clampSubsteps(settings.substeps).toFixed(0);
    bondIterationsOutput.value = clampBondIterations(settings.bondIterations).toFixed(0);
    softBodyStrengthOutput.value = settings.softBodyStrength.toFixed(0);
    viscosityOutput.value = settings.viscosity.toFixed(1);
    mouseForceOutput.value = settings.mouseForce.toLocaleString();
    repulsionOutput.value = settings.particleRepulsion.toFixed(0);
    cohesionOutput.value = settings.cohesion.toFixed(2);
  };

  bodySize.addEventListener("input", () => {
    refresh();
  });

  particleRadius.addEventListener("input", () => {
    refresh();
  });

  gravity.addEventListener("input", () => {
    settings.gravityY = gravity.valueAsNumber;
    refresh();
  });

  damping.addEventListener("input", () => {
    settings.damping = damping.valueAsNumber;
    refresh();
  });

  substeps.addEventListener("input", () => {
    settings.substeps = clampSubsteps(substeps.valueAsNumber);
    refresh();
  });

  bondIterations.addEventListener("input", () => {
    settings.bondIterations = clampBondIterations(bondIterations.valueAsNumber);
    refresh();
  });

  softBodyStrength.addEventListener("input", () => {
    settings.softBodyStrength = softBodyStrength.valueAsNumber;
    refresh();
  });

  viscosity.addEventListener("input", () => {
    settings.viscosity = viscosity.valueAsNumber;
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
  addBodyButton.addEventListener("click", () =>
    bindings.onAddBody(bodyShape.value as SoftBodyShape, bodySize.valueAsNumber, particleRadius.valueAsNumber)
  );
  bindings.setBodyStats(0, settings.particleCount);
  refresh();
  return bindings;
}

function encodeSimulationStep(
  encoder: GPUCommandEncoder,
  pipelines: Pipelines,
  particles: ParticleBuffers,
  grid: GridBuffers,
  activeCellCount: number,
  cellScanGroups: number,
  particleCount: number
) {
  const clearGridPass = encoder.beginComputePass({ label: "Clear spatial grid pass" });
  clearGridPass.setPipeline(pipelines.clearGridPipeline);
  clearGridPass.setBindGroup(0, pipelines.clearGridBindGroup);
  clearGridPass.dispatchWorkgroups(Math.ceil(activeCellCount / WORKGROUP_SIZE));
  clearGridPass.end();

  const countGridPass = encoder.beginComputePass({ label: "Count spatial grid pass" });
  countGridPass.setPipeline(pipelines.countGridPipeline);
  countGridPass.setBindGroup(0, pipelines.countGridBindGroups[particles.activeIndex]);
  countGridPass.dispatchWorkgroups(Math.ceil(particleCount / WORKGROUP_SIZE));
  countGridPass.end();

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
  scatterGridPass.dispatchWorkgroups(Math.ceil(particleCount / WORKGROUP_SIZE));
  scatterGridPass.end();

  const simulatePass = encoder.beginComputePass({ label: "Particle simulation pass" });
  simulatePass.setPipeline(pipelines.simulatePipeline);
  simulatePass.setBindGroup(0, pipelines.simulateBindGroups[particles.activeIndex]);
  simulatePass.dispatchWorkgroups(Math.ceil(particleCount / WORKGROUP_SIZE));
  simulatePass.end();
}

function clampSubsteps(value: number) {
  return Math.min(8, Math.max(1, Math.floor(value || 1)));
}

function clampBondIterations(value: number) {
  return Math.min(8, Math.max(1, Math.floor(value || 1)));
}

function smoothTiming(previous: number, next: number) {
  if (previous === 0) {
    return next;
  }
  return previous * 0.88 + next * 0.12;
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
