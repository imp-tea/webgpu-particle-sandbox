import "./style.css";
import { PARTICLE_STRIDE_BYTES, WORKGROUP_SIZE, defaultSettings, type SimulationSettings } from "./config";
import {
  createBondBuffer,
  createBodyBuffer,
  createGridBuffers,
  createMaterialBuffer,
  createParticleBuffers,
  createRestShapeBuffer,
  createUniformBuffer,
  getGridDimensions,
  type BodyProperties,
  type SoftBodyShape,
  writeSimParams,
  type GridBuffers,
  type ParticleBuffers
} from "./gpu/buffers";
import { initWebGPU } from "./gpu/initWebGPU";
import { createPipelines, type Pipelines } from "./gpu/pipelines";
import { PointerInput, type PointerState } from "./input";

const canvasElement = document.querySelector<HTMLCanvasElement>("#sim-canvas");
const vectorCanvasElement = document.querySelector<HTMLCanvasElement>("#vector-canvas");
const statusElement = document.querySelector<HTMLElement>("#gpu-status");
const debugElement = document.querySelector<HTMLElement>("#debug-stats");
const dragBandElement = document.querySelector<HTMLElement>("#drag-band");

if (!canvasElement || !vectorCanvasElement || !statusElement || !debugElement || !dragBandElement) {
  throw new Error("Missing required DOM nodes.");
}

const canvas: HTMLCanvasElement = canvasElement;
const vectorCanvas: HTMLCanvasElement = vectorCanvasElement;
const statusLabel: HTMLElement = statusElement;
const debugLabel: HTMLElement = debugElement;
const dragBand: HTMLElement = dragBandElement;

void start();

async function start() {
  const settings = defaultSettings();
  const ui = bindControls(settings);
  const vectorContext = vectorCanvas.getContext("2d");
  if (!vectorContext) {
    throw new Error("Could not create vector render context.");
  }

  try {
    const gpu = await initWebGPU(canvas);
    const pointer = new PointerInput(canvas);
    const particles = createParticleBuffers(gpu.device);
    const grid = createGridBuffers(gpu.device);
    const uniforms = createUniformBuffer(gpu.device);
    const materials = createMaterialBuffer(gpu.device);
    const bodies = createBodyBuffer(gpu.device);
    const bonds = createBondBuffer(gpu.device);
    const restShapes = createRestShapeBuffer(gpu.device);
    const particlePickReadback = gpu.device.createBuffer({
      label: "Particle picking readback",
      size: particles.maxParticles * PARTICLE_STRIDE_BYTES,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });
    const particleSnapshotReadback = gpu.device.createBuffer({
      label: "Particle snapshot readback",
      size: particles.maxParticles * PARTICLE_STRIDE_BYTES,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });
    const bodyProperties: BodyProperties[] = [];
    const bodyRenderInfos: BodyRenderInfo[] = [];
    const pipelines = createPipelines(
      gpu.device,
      gpu.format,
      particles.buffers,
      uniforms,
      grid,
      materials,
      bodies,
      bonds,
      restShapes
    );

    gpu.device.lost.then((info) => {
      statusLabel.textContent = `WebGPU device lost: ${info.message || info.reason}`;
    });

    gpu.resize();
    particles.clear(gpu.device, bodies, bonds, restShapes);
    settings.particleCount = particles.particleCount;
    ui.setBodyStats(particles.bodyCount, particles.particleCount);
    statusLabel.textContent = `${gpu.adapter.info?.vendor || "GPU"} adapter ready`;

    ui.onReset = () => {
      particles.clear(gpu.device, bodies, bonds, restShapes);
      bodyProperties.length = 0;
      bodyRenderInfos.length = 0;
      pointer.state.selectedParticleIndex = 0xffffffff;
      pointer.state.selectedBodyId = 0xffffffff;
      particleSnapshot = undefined;
      particleSnapshotCount = 0;
      ui.setSelectedBody(undefined);
      settings.particleCount = particles.particleCount;
      ui.setBodyStats(particles.bodyCount, particles.particleCount);
    };

    ui.onAddBody = (shape, bodySize, particleRadius) => {
      const size = gpu.getWorldSize();
      const properties = ui.getSpawnBodyProperties();
      const result = particles.addSoftBody(
        gpu.device,
        bodies,
        bonds,
        restShapes,
        shape,
        bodySize,
        particleRadius,
        properties,
        size.width,
        size.height
      );
      if (result.added) {
        bodyProperties.push({ ...properties });
        bodyRenderInfos.push({
          bodyId: result.bodyId,
          startIndex: result.startIndex,
          perimeterParticleCount: result.perimeterParticleCount,
          particleRadius: result.particleRadius,
          materialId: result.materialId
        });
        settings.particleCount = result.particleCount;
        ui.setBodyStats(result.bodyCount, result.particleCount);
        statusLabel.textContent = `Added ${shape} body (${result.addedParticles.toLocaleString()} particles)`;
      } else {
        statusLabel.textContent = result.reason;
      }
    };

    ui.onBodyPropertiesChange = (bodyId, properties) => {
      if (bodyId === undefined) {
        return;
      }

      bodyProperties[bodyId] = { ...properties };
      particles.updateBodyProperties(gpu.device, bodies, bodyId, properties);
    };

    let pickRequestId = 0;
    let pickingParticle = false;
    let dragAnchor: { x: number; y: number } | undefined;
    let particleSnapshot: ArrayBuffer | undefined;
    let particleSnapshotCount = 0;
    let particleSnapshotPending = false;
    let lastParticleSnapshot = 0;
    pointer.onPointerDown = (position) => {
      const snapshotPick = pickNearestParticleFromSnapshot(
        particleSnapshot,
        particleSnapshotCount,
        position.x,
        position.y
      );
      if (snapshotPick) {
        pointer.state.selectedParticleIndex = snapshotPick.particleIndex;
        pointer.state.selectedBodyId = snapshotPick.bodyId;
        dragAnchor = { x: snapshotPick.x, y: snapshotPick.y };
        ui.setSelectedBody(snapshotPick.bodyId, bodyProperties[snapshotPick.bodyId]);
        return;
      }

      if (pickingParticle) {
        return;
      }

      const requestId = ++pickRequestId;
      pickingParticle = true;
      void pickNearestParticle(gpu.device, particles, particlePickReadback, position.x, position.y).then((pick) => {
        if (requestId !== pickRequestId) {
          return;
        }

        if (!pick) {
          pointer.state.selectedParticleIndex = 0xffffffff;
          pointer.state.selectedBodyId = 0xffffffff;
          dragAnchor = undefined;
          ui.setSelectedBody(undefined);
          return;
        }

        pointer.state.selectedParticleIndex = pointer.state.active ? pick.particleIndex : 0xffffffff;
        pointer.state.selectedBodyId = pick.bodyId;
        dragAnchor = { x: pick.x, y: pick.y };
        ui.setSelectedBody(pick.bodyId, bodyProperties[pick.bodyId]);
      }).finally(() => {
        pickingParticle = false;
      });
    };

    pointer.onPointerUp = () => {
      dragAnchor = undefined;
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
      resizeVectorCanvas(vectorCanvas, vectorContext);
      const size = gpu.getWorldSize();
      const deltaTime = paused ? 0 : elapsed;
      let simulationEncodeMs = 0;
      const liveDragAnchor =
        getParticlePositionFromSnapshot(particleSnapshot, particleSnapshotCount, pointer.state.selectedParticleIndex) ??
        dragAnchor;
      updateDragBand(dragBand, liveDragAnchor, pointer.state);
      const renderingVectors = ui.getRenderMode() === "vectors";
      const snapshotInterval =
        pointer.state.active && pointer.state.selectedParticleIndex !== 0xffffffff ? 16 : renderingVectors ? 16 : 80;
      if (!particleSnapshotPending && settings.particleCount > 0 && now - lastParticleSnapshot > snapshotInterval) {
        particleSnapshotPending = true;
        lastParticleSnapshot = now;
        void readParticleSnapshot(
          gpu.device,
          particles,
          particleSnapshotReadback
        ).then((snapshot) => {
          particleSnapshot = snapshot.data;
          particleSnapshotCount = snapshot.particleCount;
        }).finally(() => {
          particleSnapshotPending = false;
        });
      }

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
          encodeSimulationIntegrate(simulationEncoder, pipelines, particles, settings.particleCount);
          particles.swap();
        }

        if (particles.bodyCount > 1) {
          for (let iteration = 0; iteration < clampContactIterations(settings.contactIterations); iteration += 1) {
            encodeGridBuild(
              simulationEncoder,
              pipelines,
              particles,
              grid,
              activeCellCount,
              cellScanGroups,
              settings.particleCount
            );
            encodeContactSolve(simulationEncoder, pipelines, particles, settings.particleCount);
            particles.swap();
          }
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

      if (!renderingVectors) {
        renderPass.setPipeline(pipelines.renderPipeline);
        renderPass.setBindGroup(0, pipelines.renderBindGroups[particles.activeIndex]);
        renderPass.draw(6, settings.particleCount);
      }
      renderPass.end();

      gpu.device.queue.submit([renderEncoder.finish()]);
      renderVectorLayer(vectorCanvas, vectorContext, renderingVectors, particleSnapshot, particleSnapshotCount, bodyRenderInfos);
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
  onBodyPropertiesChange: (bodyId: number | undefined, properties: BodyProperties) => void;
  onPauseToggle: () => void;
  getRenderMode: () => RenderMode;
  getSpawnBodyProperties: () => BodyProperties;
  setSelectedBody: (bodyId: number | undefined, properties?: BodyProperties) => void;
  setPaused: (paused: boolean) => void;
  setBodyStats: (bodyCount: number, particleCount: number) => void;
  setDebugStats: (stats: DebugStats) => void;
};

type RenderMode = "particles" | "vectors";

type BodyRenderInfo = {
  bodyId: number;
  startIndex: number;
  perimeterParticleCount: number;
  particleRadius: number;
  materialId: number;
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
  const renderMode = getSelect("render-mode");
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
  const contactIterations = getInput("contact-iterations");
  const contactIterationsOutput = getOutput("contact-iterations-output");
  const bondIterations = getInput("bond-iterations");
  const bondIterationsOutput = getOutput("bond-iterations-output");
  const softBodyStrength = getInput("soft-body-strength");
  const softBodyStrengthOutput = getOutput("soft-body-strength-output");
  const viscosity = getInput("viscosity");
  const viscosityOutput = getOutput("viscosity-output");
  const friction = getInput("friction");
  const frictionOutput = getOutput("friction-output");
  const wallBounce = getInput("wall-bounce");
  const wallBounceOutput = getOutput("wall-bounce-output");
  const mouseForce = getInput("mouse-force");
  const mouseForceOutput = getOutput("mouse-force-output");
  const propertyMode = getElement("property-mode");
  const pauseButton = getButton("pause-button");
  const resetButton = getButton("reset-button");
  const addBodyButton = getButton("add-body-button");
  let selectedBodyId: number | undefined;
  let selectedBodyProperties: BodyProperties | undefined;
  let spawnBodyProperties: BodyProperties = {
    softBodyStrength: settings.softBodyStrength,
    viscosity: settings.viscosity,
    friction: settings.friction
  };
  let syncingBodyControls = false;

  const bindings: ControlBindings = {
    onReset: () => undefined,
    onAddBody: () => undefined,
    onBodyPropertiesChange: () => undefined,
    onPauseToggle: () => undefined,
    getRenderMode() {
      return renderMode.value as RenderMode;
    },
    getSpawnBodyProperties() {
      return { ...spawnBodyProperties };
    },
    setSelectedBody(bodyId, properties) {
      selectedBodyId = properties ? bodyId : undefined;
      selectedBodyProperties = properties ? { ...properties } : undefined;
      syncingBodyControls = true;
      if (selectedBodyId === undefined || !selectedBodyProperties) {
        propertyMode.textContent = "Editing spawn properties";
        softBodyStrength.value = String(spawnBodyProperties.softBodyStrength);
        viscosity.value = String(spawnBodyProperties.viscosity);
        friction.value = String(spawnBodyProperties.friction);
      } else {
        propertyMode.textContent = `Editing body ${selectedBodyId + 1}`;
        softBodyStrength.value = String(selectedBodyProperties.softBodyStrength);
        viscosity.value = String(selectedBodyProperties.viscosity);
        friction.value = String(selectedBodyProperties.friction);
      }
      syncingBodyControls = false;
      refresh();
    },
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
  contactIterations.value = String(settings.contactIterations);
  bondIterations.value = String(settings.bondIterations);
  softBodyStrength.value = String(settings.softBodyStrength);
  viscosity.value = String(settings.viscosity);
  friction.value = String(settings.friction);
  wallBounce.value = String(settings.wallBounce);
  mouseForce.value = String(settings.mouseForce);

  const refresh = () => {
    bodySizeOutput.value = bodySize.valueAsNumber.toFixed(0);
    particleRadiusOutput.value = particleRadius.valueAsNumber.toFixed(1);
    gravityOutput.value = settings.gravityY.toFixed(0);
    dampingOutput.value = settings.damping.toFixed(2);
    substepsOutput.value = clampSubsteps(settings.substeps).toFixed(0);
    contactIterationsOutput.value = clampContactIterations(settings.contactIterations).toFixed(0);
    bondIterationsOutput.value = clampBondIterations(settings.bondIterations).toFixed(0);
    softBodyStrengthOutput.value = getCurrentBodyProperties().softBodyStrength.toFixed(0);
    viscosityOutput.value = getCurrentBodyProperties().viscosity.toFixed(1);
    frictionOutput.value = getCurrentBodyProperties().friction.toFixed(2);
    wallBounceOutput.value = settings.wallBounce.toFixed(2);
    mouseForceOutput.value = settings.mouseForce.toLocaleString();
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

  contactIterations.addEventListener("input", () => {
    settings.contactIterations = clampContactIterations(contactIterations.valueAsNumber);
    refresh();
  });

  bondIterations.addEventListener("input", () => {
    settings.bondIterations = clampBondIterations(bondIterations.valueAsNumber);
    refresh();
  });

  softBodyStrength.addEventListener("input", () => {
    updateCurrentBodyProperties({
      ...getCurrentBodyProperties(),
      softBodyStrength: softBodyStrength.valueAsNumber
    });
    refresh();
  });

  viscosity.addEventListener("input", () => {
    updateCurrentBodyProperties({
      ...getCurrentBodyProperties(),
      viscosity: viscosity.valueAsNumber
    });
    refresh();
  });

  friction.addEventListener("input", () => {
    updateCurrentBodyProperties({
      ...getCurrentBodyProperties(),
      friction: friction.valueAsNumber
    });
    refresh();
  });

  wallBounce.addEventListener("input", () => {
    settings.wallBounce = wallBounce.valueAsNumber;
    refresh();
  });

  mouseForce.addEventListener("input", () => {
    settings.mouseForce = mouseForce.valueAsNumber;
    refresh();
  });

  pauseButton.addEventListener("click", () => bindings.onPauseToggle());
  resetButton.addEventListener("click", () => bindings.onReset());
  addBodyButton.addEventListener("click", () =>
    bindings.onAddBody(bodyShape.value as SoftBodyShape, bodySize.valueAsNumber, particleRadius.valueAsNumber)
  );
  bindings.setBodyStats(0, settings.particleCount);
  refresh();
  return bindings;

  function getCurrentBodyProperties(): BodyProperties {
    return selectedBodyId === undefined ? spawnBodyProperties : selectedBodyProperties!;
  }

  function updateCurrentBodyProperties(properties: BodyProperties) {
    if (selectedBodyId === undefined) {
      spawnBodyProperties = properties;
      settings.softBodyStrength = properties.softBodyStrength;
      settings.viscosity = properties.viscosity;
      settings.friction = properties.friction;
      return;
    }

    selectedBodyProperties = properties;
    if (!syncingBodyControls) {
      bindings.onBodyPropertiesChange(selectedBodyId, properties);
    }
  }
}

function encodeGridBuild(
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
}

function encodeSimulationIntegrate(
  encoder: GPUCommandEncoder,
  pipelines: Pipelines,
  particles: ParticleBuffers,
  particleCount: number
) {
  const simulatePass = encoder.beginComputePass({ label: "Particle simulation pass" });
  simulatePass.setPipeline(pipelines.simulatePipeline);
  simulatePass.setBindGroup(0, pipelines.simulateBindGroups[particles.activeIndex]);
  simulatePass.dispatchWorkgroups(Math.ceil(particleCount / WORKGROUP_SIZE));
  simulatePass.end();
}

function encodeContactSolve(
  encoder: GPUCommandEncoder,
  pipelines: Pipelines,
  particles: ParticleBuffers,
  particleCount: number
) {
  const contactPass = encoder.beginComputePass({ label: "Contact projection pass" });
  contactPass.setPipeline(pipelines.solveContactsPipeline);
  contactPass.setBindGroup(0, pipelines.solveContactBindGroups[particles.activeIndex]);
  contactPass.dispatchWorkgroups(Math.ceil(particleCount / WORKGROUP_SIZE));
  contactPass.end();
}

function clampSubsteps(value: number) {
  return Math.min(8, Math.max(1, Math.floor(value || 1)));
}

function clampContactIterations(value: number) {
  return Math.min(8, Math.max(0, Math.floor(value || 0)));
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

const BODY_FILL_COLORS = ["#47c7f2", "#fab340", "#8aeb7a", "#eb75ba"];

function resizeVectorCanvas(canvas: HTMLCanvasElement, context: CanvasRenderingContext2D) {
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.floor(rect.width * dpr));
  const height = Math.max(1, Math.floor(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function renderVectorLayer(
  canvas: HTMLCanvasElement,
  context: CanvasRenderingContext2D,
  active: boolean,
  snapshot: ArrayBuffer | undefined,
  particleCount: number,
  bodies: BodyRenderInfo[]
) {
  canvas.style.opacity = active ? "1" : "0";
  const width = canvas.width / Math.max(1, Math.min(window.devicePixelRatio || 1, 2));
  const height = canvas.height / Math.max(1, Math.min(window.devicePixelRatio || 1, 2));
  context.clearRect(0, 0, width, height);
  if (!active || !snapshot || particleCount === 0) {
    return;
  }

  const view = new DataView(snapshot);
  context.lineJoin = "round";
  context.lineCap = "round";

  for (const body of bodies) {
    if (body.perimeterParticleCount < 3 || body.startIndex + body.perimeterParticleCount > particleCount) {
      continue;
    }

    context.beginPath();
    for (let i = 0; i < body.perimeterParticleCount; i += 1) {
      const offset = (body.startIndex + i) * PARTICLE_STRIDE_BYTES;
      const x = view.getFloat32(offset, true);
      const y = view.getFloat32(offset + 4, true);
      if (i === 0) {
        context.moveTo(x, y);
      } else {
        context.lineTo(x, y);
      }
    }

    context.closePath();
    const bodyColor = BODY_FILL_COLORS[body.materialId % BODY_FILL_COLORS.length];
    context.fillStyle = bodyColor;
    context.globalAlpha = 0.82;
    context.fill();
    context.globalAlpha = 1;
    context.strokeStyle = bodyColor;
    context.lineWidth = Math.max(1, body.particleRadius * 2);
    context.stroke();
  }
}

function updateDragBand(element: HTMLElement, anchor: { x: number; y: number } | undefined, pointer: PointerState) {
  if (!anchor || !pointer.active || pointer.selectedParticleIndex === 0xffffffff) {
    element.style.opacity = "0";
    return;
  }

  const dx = pointer.x - anchor.x;
  const dy = pointer.y - anchor.y;
  const length = Math.hypot(dx, dy);
  const angle = Math.atan2(dy, dx);
  element.style.opacity = "1";
  element.style.width = `${length}px`;
  element.style.transform = `translate(${anchor.x}px, ${anchor.y - 1}px) rotate(${angle}rad)`;
}

async function pickNearestParticle(
  device: GPUDevice,
  particles: ParticleBuffers,
  readback: GPUBuffer,
  x: number,
  y: number
) {
  if (particles.particleCount === 0) {
    return undefined;
  }

  const byteLength = particles.particleCount * PARTICLE_STRIDE_BYTES;
  const encoder = device.createCommandEncoder({ label: "Particle picking readback encoder" });
  encoder.copyBufferToBuffer(particles.buffers[particles.activeIndex], 0, readback, 0, byteLength);
  device.queue.submit([encoder.finish()]);

  await readback.mapAsync(GPUMapMode.READ, 0, byteLength);
  try {
    return pickNearestParticleFromView(new DataView(readback.getMappedRange(0, byteLength)), particles.particleCount, x, y);
  } finally {
    readback.unmap();
  }
}

async function readParticleSnapshot(device: GPUDevice, particles: ParticleBuffers, readback: GPUBuffer) {
  const particleCount = particles.particleCount;
  const byteLength = particleCount * PARTICLE_STRIDE_BYTES;
  const encoder = device.createCommandEncoder({ label: "Particle snapshot readback encoder" });
  encoder.copyBufferToBuffer(particles.buffers[particles.activeIndex], 0, readback, 0, byteLength);
  device.queue.submit([encoder.finish()]);

  await readback.mapAsync(GPUMapMode.READ, 0, byteLength);
  try {
    const source = readback.getMappedRange(0, byteLength);
    return {
      data: source.slice(0),
      particleCount
    };
  } finally {
    readback.unmap();
  }
}

function pickNearestParticleFromSnapshot(
  snapshot: ArrayBuffer | undefined,
  particleCount: number,
  x: number,
  y: number
) {
  if (!snapshot || particleCount === 0) {
    return undefined;
  }

  return pickNearestParticleFromView(new DataView(snapshot), particleCount, x, y);
}

function getParticlePositionFromSnapshot(
  snapshot: ArrayBuffer | undefined,
  particleCount: number,
  particleIndex: number
) {
  if (!snapshot || particleIndex >= particleCount || particleIndex === 0xffffffff) {
    return undefined;
  }

  const view = new DataView(snapshot);
  const offset = particleIndex * PARTICLE_STRIDE_BYTES;
  const radius = view.getFloat32(offset + 24, true);
  if (radius <= 0) {
    return undefined;
  }

  return {
    x: view.getFloat32(offset, true),
    y: view.getFloat32(offset + 4, true)
  };
}

function pickNearestParticleFromView(view: DataView, particleCount: number, x: number, y: number) {
  let bestParticleIndex = -1;
  let bestBodyId = -1;
  let bestDistanceSq = Number.POSITIVE_INFINITY;
  let bestRadius = 0;
  let bestX = 0;
  let bestY = 0;

  for (let index = 0; index < particleCount; index += 1) {
    const offset = index * PARTICLE_STRIDE_BYTES;
    const px = view.getFloat32(offset, true);
    const py = view.getFloat32(offset + 4, true);
    const flags = view.getUint32(offset + 20, true);
    const radius = view.getFloat32(offset + 24, true);
    if (radius <= 0) {
      continue;
    }

    const dx = px - x;
    const dy = py - y;
    const distanceSq = dx * dx + dy * dy;
    if (distanceSq < bestDistanceSq) {
      bestParticleIndex = index;
      bestBodyId = flags & 0x0000ffff;
      bestDistanceSq = distanceSq;
      bestRadius = radius;
      bestX = px;
      bestY = py;
    }
  }

  const pickRadius = Math.max(36, bestRadius * 3);
  if (bestParticleIndex < 0 || bestDistanceSq > pickRadius * pickRadius) {
    return undefined;
  }

  return {
    particleIndex: bestParticleIndex,
    bodyId: bestBodyId,
    x: bestX,
    y: bestY
  };
}

function getInput(id: string): HTMLInputElement {
  const input = document.getElementById(id);
  if (!(input instanceof HTMLInputElement)) {
    throw new Error(`Missing input #${id}`);
  }
  return input;
}

function getElement(id: string): HTMLElement {
  const element = document.getElementById(id);
  if (!(element instanceof HTMLElement)) {
    throw new Error(`Missing element #${id}`);
  }
  return element;
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
