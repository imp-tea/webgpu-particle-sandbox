import "./style.css";
import { PARTICLE_STRIDE_BYTES, WORKGROUP_SIZE, defaultSettings, type SimulationSettings } from "./config";
import {
  createBondBuffer,
  createBodyBuffer,
  createGridBuffers,
  createJointBuffer,
  createMaterialBuffer,
  createParticleBuffers,
  createRestShapeBuffer,
  createUniformBuffer,
  getGridDimensions,
  type BodyKind,
  type BodyProperties,
  type JointDefinition,
  type SampledBodyPoint,
  type SoftBodyShape,
  writeJointBuffer,
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
const placementMarkerElement = document.querySelector<HTMLElement>("#placement-marker");
const placementBandElement = document.querySelector<HTMLElement>("#placement-band");
const placementHintElement = document.querySelector<HTMLElement>("#placement-hint");
const jointMarkerElement = document.querySelector<HTMLElement>("#joint-marker");
const jointBandElement = document.querySelector<HTMLElement>("#joint-band");

if (
  !canvasElement ||
  !vectorCanvasElement ||
  !statusElement ||
  !debugElement ||
  !dragBandElement ||
  !placementMarkerElement ||
  !placementBandElement ||
  !placementHintElement ||
  !jointMarkerElement ||
  !jointBandElement
) {
  throw new Error("Missing required DOM nodes.");
}

const canvas: HTMLCanvasElement = canvasElement;
const vectorCanvas: HTMLCanvasElement = vectorCanvasElement;
const statusLabel: HTMLElement = statusElement;
const debugLabel: HTMLElement = debugElement;
const dragBand: HTMLElement = dragBandElement;
const placementMarker: HTMLElement = placementMarkerElement;
const placementBand: HTMLElement = placementBandElement;
const placementHint: HTMLElement = placementHintElement;
const jointMarker: HTMLElement = jointMarkerElement;
const jointBand: HTMLElement = jointBandElement;
const placementHintIdle = placementHint.textContent ?? "";
const placementHintPendingRope = "Click again to set the rope end. Esc to cancel.";
const placementHintPendingJoint = "Shift+Click a second body to complete the joint. Esc to cancel.";
const defaultSvgUrl = new URL("../test.svg", import.meta.url).href;

void start();

async function start() {
  const settings = defaultSettings();
  const vehicle = createVehicleController();
  const ui = bindControls(settings, vehicle);
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
    const joints = createJointBuffer(gpu.device);
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
    let jointDefinitions: JointDefinition[] = [];
    let jointCount = 0;
    const pipelines = createPipelines(
      gpu.device,
      gpu.format,
      particles.buffers,
      uniforms,
      grid,
      materials,
      bodies,
      bonds,
      restShapes,
      joints
    );

    gpu.device.lost.then((info) => {
      statusLabel.textContent = `WebGPU device lost: ${info.message || info.reason}`;
    });

    gpu.resize();
    particles.clear(gpu.device, bodies, bonds, restShapes);
    jointCount = writeJointBuffer(gpu.device, joints, jointDefinitions);
    const initialVehicle = addVehicleScene(
      gpu.device,
      particles,
      bodies,
      bonds,
      restShapes,
      joints,
      gpu.getWorldSize(),
      settings,
      bodyProperties,
      bodyRenderInfos,
      vehicle
    );
    jointDefinitions = initialVehicle.joints;
    jointCount = initialVehicle.jointCount;
    settings.particleCount = particles.particleCount;
    ui.setBodyStats(particles.bodyCount, particles.particleCount);
    statusLabel.textContent = `${gpu.adapter.info?.vendor || "GPU"} ready | A/D drive wheels`;

    ui.onReset = () => {
      particles.clear(gpu.device, bodies, bonds, restShapes);
      jointDefinitions = [];
      jointCount = writeJointBuffer(gpu.device, joints, jointDefinitions);
      bodyProperties.length = 0;
      bodyRenderInfos.length = 0;
      vehicle.clear();
      pointer.state.selectedParticleIndex = 0xffffffff;
      pointer.state.selectedBodyId = 0xffffffff;
      particleSnapshot = undefined;
      particleSnapshotCount = 0;
      ui.setSelectedBody(undefined);
      cancelPendingPlacement();
      settings.particleCount = particles.particleCount;
      refreshLiveStats();
    };

    const refreshLiveStats = () => {
      const liveParticles = bodyRenderInfos.reduce((sum, info) => sum + info.particleCount, 0);
      ui.setBodyStats(bodyRenderInfos.length, liveParticles);
    };

    const spawnBody = async (
      shape: BodyShapeChoice,
      bodySize: number,
      particleRadius: number,
      svgFile: File | undefined,
      spawnPoint?: { x: number; y: number },
      endPoint?: { x: number; y: number }
    ) => {
      const size = gpu.getWorldSize();
      const properties = ui.getSpawnBodyProperties();
      if (shape === "svg") {
        statusLabel.textContent = "Sampling SVG body...";
        try {
          const source = await createSvgBodySource(svgFile, bodySize, particleRadius);
          const result = particles.addSampledBody(
            gpu.device,
            bodies,
            bonds,
            restShapes,
            source.points,
            source.size,
            source.spacing,
            particleRadius,
            properties,
            size.width,
            size.height,
            spawnPoint
          );
          if (result.added) {
            bodyProperties.push({ ...properties });
            bodyRenderInfos.push({
              kind: properties.kind,
              bodyId: result.bodyId,
              startIndex: result.startIndex,
              particleCount: result.addedParticles,
              perimeterParticleCount: result.perimeterParticleCount,
              particleRadius: result.particleRadius,
              materialId: result.materialId,
              restPositions: result.restPositions,
              svgRender: source.render
            });
            settings.particleCount = result.particleCount;
            refreshLiveStats();
            statusLabel.textContent = `Added SVG body (${result.addedParticles.toLocaleString()} particles)`;
          } else {
            statusLabel.textContent = result.reason;
          }
        } catch (error) {
          statusLabel.textContent = error instanceof Error ? error.message : String(error);
        }
        return;
      }

      if (shape === "rope") {
        const ropeOptions = ui.getRopeOptions();
        const result = particles.addRope(
          gpu.device,
          bodies,
          bonds,
          restShapes,
          bodySize,
          particleRadius,
          properties,
          size.width,
          size.height,
          ropeOptions.pinnedStart,
          ropeOptions.pinnedEnd,
          spawnPoint,
          endPoint,
          ropeOptions.lengthMultiplier,
          ropeOptions.density
        );
        if (result.added) {
          const ropeProperties = { ...properties, kind: "rope" as const };
          bodyProperties.push(ropeProperties);
          bodyRenderInfos.push({
            kind: "rope",
            bodyId: result.bodyId,
            startIndex: result.startIndex,
            particleCount: result.addedParticles,
            perimeterParticleCount: result.perimeterParticleCount,
            particleRadius: result.particleRadius,
            materialId: result.materialId,
            restPositions: result.restPositions
          });
          settings.particleCount = result.particleCount;
          refreshLiveStats();
          statusLabel.textContent = `Added rope (${result.addedParticles.toLocaleString()} particles)`;
        } else {
          statusLabel.textContent = result.reason;
        }
        return;
      }

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
        size.height,
        spawnPoint
      );
      if (result.added) {
        bodyProperties.push({ ...properties });
        bodyRenderInfos.push({
          kind: properties.kind,
          bodyId: result.bodyId,
          startIndex: result.startIndex,
          particleCount: result.addedParticles,
          perimeterParticleCount: result.perimeterParticleCount,
          particleRadius: result.particleRadius,
          materialId: result.materialId,
          restPositions: result.restPositions
        });
        settings.particleCount = result.particleCount;
        refreshLiveStats();
        statusLabel.textContent = `Added ${shape} body (${result.addedParticles.toLocaleString()} particles)`;
      } else {
        statusLabel.textContent = result.reason;
      }
    };

    ui.onAddBody = spawnBody;

    let pendingRopeStart: { x: number; y: number } | undefined;
    let pendingJointAnchor:
      | { bodyId: number; particleIndex: number; world: { x: number; y: number }; localAnchor: { x: number; y: number } }
      | undefined;
    const updatePlacementHint = () => {
      if (pendingRopeStart) {
        placementHint.textContent = placementHintPendingRope;
        placementHint.classList.add("active");
      } else if (pendingJointAnchor) {
        placementHint.textContent = placementHintPendingJoint;
        placementHint.classList.add("active");
      } else {
        placementHint.textContent = placementHintIdle;
        placementHint.classList.remove("active");
      }
    };
    const cancelPendingPlacement = () => {
      if (!pendingRopeStart && !pendingJointAnchor) {
        return;
      }
      pendingRopeStart = undefined;
      pendingJointAnchor = undefined;
      updatePlacementHint();
    };
    ui.onShapeChange = () => {
      if (pendingRopeStart) {
        pendingRopeStart = undefined;
        updatePlacementHint();
      }
    };
    pointer.onPlace = (position) => {
      const args = ui.getSpawnArgs();
      if (args.shape === "rope") {
        if (!pendingRopeStart) {
          pendingRopeStart = { x: position.x, y: position.y };
          updatePlacementHint();
          statusLabel.textContent = "Rope start placed. Click to set the end.";
          return;
        }
        const start = pendingRopeStart;
        pendingRopeStart = undefined;
        updatePlacementHint();
        void spawnBody(args.shape, args.bodySize, args.particleRadius, args.svgFile, start, position);
        return;
      }
      cancelPendingPlacement();
      void spawnBody(args.shape, args.bodySize, args.particleRadius, args.svgFile, position);
    };

    pointer.onAddJoint = (position) => {
      const pick = pickNearestParticleFromSnapshot(particleSnapshot, particleSnapshotCount, position.x, position.y);
      if (!pick) {
        statusLabel.textContent = "No body at cursor.";
        return;
      }
      const info = bodyRenderInfos.find((entry) => entry.bodyId === pick.bodyId);
      if (!info) {
        return;
      }
      const localIndex = pick.particleIndex - info.startIndex;
      if (localIndex < 0 || localIndex >= info.particleCount) {
        return;
      }
      const localAnchor = {
        x: info.restPositions[localIndex * 2],
        y: info.restPositions[localIndex * 2 + 1]
      };

      if (!pendingJointAnchor) {
        pendingJointAnchor = {
          bodyId: pick.bodyId,
          particleIndex: pick.particleIndex,
          world: { x: pick.x, y: pick.y },
          localAnchor
        };
        updatePlacementHint();
        statusLabel.textContent = `Joint anchor set on body ${pick.bodyId + 1}.`;
        return;
      }

      if (pendingJointAnchor.bodyId === pick.bodyId) {
        statusLabel.textContent = "Joint endpoints must be on different bodies.";
        return;
      }

      if (jointDefinitions.length >= 64) {
        statusLabel.textContent = "Joint limit reached.";
        pendingJointAnchor = undefined;
        updatePlacementHint();
        return;
      }

      const anchorWorldDistance = Math.hypot(
        pendingJointAnchor.world.x - pick.x,
        pendingJointAnchor.world.y - pick.y
      );
      const radiusGuess = Math.max(info.particleRadius * 12, 32);
      const newJoint = {
        bodyA: pendingJointAnchor.bodyId,
        bodyB: pick.bodyId,
        localAnchorA: pendingJointAnchor.localAnchor,
        localAnchorB: localAnchor,
        restLength: 0,
        stiffness: 0.78,
        influenceRadius: Math.max(radiusGuess, anchorWorldDistance * 0.5)
      };
      jointDefinitions = [...jointDefinitions, newJoint];
      jointCount = writeJointBuffer(gpu.device, joints, jointDefinitions);
      const completedBodyId = pendingJointAnchor.bodyId;
      pendingJointAnchor = undefined;
      updatePlacementHint();
      statusLabel.textContent = `Joined body ${completedBodyId + 1} to body ${pick.bodyId + 1}.`;
    };

    pointer.onDelete = (position) => {
      const pick = pickNearestParticleFromSnapshot(particleSnapshot, particleSnapshotCount, position.x, position.y);
      if (!pick) {
        return;
      }
      const renderIndex = bodyRenderInfos.findIndex((info) => info.bodyId === pick.bodyId);
      if (renderIndex < 0) {
        return;
      }
      const info = bodyRenderInfos[renderIndex];
      particles.deleteBody(
        gpu.device,
        bodies,
        bonds,
        restShapes,
        pick.bodyId,
        info.startIndex,
        info.particleCount
      );
      bodyRenderInfos.splice(renderIndex, 1);
      delete bodyProperties[pick.bodyId];
      const filteredJoints = jointDefinitions.filter(
        (joint) => joint.bodyA !== pick.bodyId && joint.bodyB !== pick.bodyId
      );
      if (filteredJoints.length !== jointDefinitions.length) {
        jointDefinitions = filteredJoints;
        jointCount = writeJointBuffer(gpu.device, joints, jointDefinitions);
      }
      vehicle.forgetBody(pick.bodyId);
      if (pointer.state.selectedBodyId === pick.bodyId) {
        pointer.state.selectedBodyId = 0xffffffff;
        pointer.state.selectedParticleIndex = 0xffffffff;
        dragAnchor = undefined;
        ui.setSelectedBody(undefined);
      }
      if (pendingJointAnchor?.bodyId === pick.bodyId) {
        pendingJointAnchor = undefined;
        updatePlacementHint();
      }
      refreshLiveStats();
      statusLabel.textContent = `Deleted body ${pick.bodyId + 1}`;
    };
    window.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        cancelPendingPlacement();
      }
    });

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
    let solverFramePhase = 0;

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
      updatePlacementPreview(placementMarker, placementBand, pendingRopeStart, pointer.state);
      const livePendingJointAnchor = pendingJointAnchor
        ? getParticlePositionFromSnapshot(particleSnapshot, particleSnapshotCount, pendingJointAnchor.particleIndex) ??
          pendingJointAnchor.world
        : undefined;
      updatePlacementPreview(jointMarker, jointBand, livePendingJointAnchor, pointer.state);
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
        vehicle.updateMotors(gpu.device, particles, bodies);
        writeSimParams(
          gpu.device,
          uniforms,
          settings,
          pointer.state,
          size.width,
          size.height,
          subDeltaTime,
          jointCount,
          solverFramePhase
        );

        const simulationStart = performance.now();
        const simulationEncoder = gpu.device.createCommandEncoder({ label: "Simulation command encoder" });
        for (let step = 0; step < substeps; step += 1) {
          encodeSimulationIntegrate(simulationEncoder, pipelines, particles, settings.particleCount);
          particles.swap();
          for (let iteration = 0; iteration < clampJointIterations(settings.jointIterations); iteration += 1) {
            encodeJointSolve(simulationEncoder, pipelines, particles, settings.particleCount);
            particles.swap();
          }
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
        solverFramePhase = (solverFramePhase + 1) & 0xffff;
        simulationEncodeMs = performance.now() - simulationStart;
      } else {
        vehicle.updateMotors(gpu.device, particles, bodies);
        writeSimParams(gpu.device, uniforms, settings, pointer.state, size.width, size.height, 0, jointCount, solverFramePhase);
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
            clearValue: { r: 0.051, g: 0.09, b: 0.137, a: 1 },
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
  onAddBody: (
    shape: BodyShapeChoice,
    size: number,
    particleRadius: number,
    svgFile: File | undefined,
    spawnPoint?: { x: number; y: number },
    endPoint?: { x: number; y: number }
  ) => void | Promise<void>;
  onBodyPropertiesChange: (bodyId: number | undefined, properties: BodyProperties) => void;
  onPauseToggle: () => void;
  onShapeChange: (shape: BodyShapeChoice) => void;
  getRenderMode: () => RenderMode;
  getSpawnBodyProperties: () => BodyProperties;
  getRopeOptions: () => {
    pinnedStart: boolean;
    pinnedEnd: boolean;
    lengthMultiplier: number;
    density: number;
  };
  getSpawnArgs: () => {
    shape: BodyShapeChoice;
    bodySize: number;
    particleRadius: number;
    svgFile: File | undefined;
  };
  setSelectedBody: (bodyId: number | undefined, properties?: BodyProperties) => void;
  setPaused: (paused: boolean) => void;
  setBodyStats: (bodyCount: number, particleCount: number) => void;
  setDebugStats: (stats: DebugStats) => void;
};

type RenderMode = "particles" | "vectors";
type BodyShapeChoice = SoftBodyShape | "svg" | "rope";

type BodyRenderInfo = {
  kind: BodyKind;
  bodyId: number;
  startIndex: number;
  particleCount: number;
  perimeterParticleCount: number;
  particleRadius: number;
  materialId: number;
  restPositions: Float32Array;
  svgRender?: SvgRenderInfo;
};

type SvgRenderInfo = {
  image: HTMLCanvasElement;
  points: SvgRenderPoint[];
  triangles: SvgTriangle[];
};

type SvgRenderPoint = {
  x: number;
  y: number;
  sourceX: number;
  sourceY: number;
};

type SvgTriangle = [number, number, number];
type SvgSamplePoint = SampledBodyPoint & SvgRenderPoint;

type DebugStats = {
  substeps: number;
  maxCellOccupancy: number;
  gridOverflow: number;
  frameMs: number;
  simulationEncodeMs: number;
};

type VehicleController = {
  setWheels: (leftWheelBodyId: number, rightWheelBodyId: number) => void;
  clear: () => void;
  forgetBody: (bodyId: number) => void;
  getMotorStrength: (bodyId: number | undefined) => number | undefined;
  setMotorStrength: (bodyId: number | undefined, motorStrength: number) => void;
  updateMotors: (device: GPUDevice, particles: ParticleBuffers, bodyBuffer: GPUBuffer) => void;
};

function createVehicleController(): VehicleController {
  const maxMotorStrength = 500;
  const defaultMotorStrength = maxMotorStrength;
  const keys = new Set<string>();
  const motorStrengths = new Map<number, number>();
  let leftWheelBodyId: number | undefined;
  let rightWheelBodyId: number | undefined;

  window.addEventListener("keydown", (event) => {
    if (event.code === "KeyA" || event.code === "KeyD") {
      keys.add(event.code);
    }
  });

  window.addEventListener("keyup", (event) => {
    if (event.code === "KeyA" || event.code === "KeyD") {
      keys.delete(event.code);
    }
  });

  return {
    setWheels(left, right) {
      leftWheelBodyId = left;
      rightWheelBodyId = right;
      motorStrengths.set(left, defaultMotorStrength);
      motorStrengths.set(right, defaultMotorStrength);
    },
    clear() {
      leftWheelBodyId = undefined;
      rightWheelBodyId = undefined;
      keys.clear();
      motorStrengths.clear();
    },
    forgetBody(bodyId) {
      motorStrengths.delete(bodyId);
      if (leftWheelBodyId === bodyId) {
        leftWheelBodyId = undefined;
      }
      if (rightWheelBodyId === bodyId) {
        rightWheelBodyId = undefined;
      }
    },
    getMotorStrength(bodyId) {
      return bodyId === undefined ? undefined : motorStrengths.get(bodyId);
    },
    setMotorStrength(bodyId, motorStrength) {
      if (bodyId === undefined || !motorStrengths.has(bodyId)) {
        return;
      }

      motorStrengths.set(bodyId, Math.min(maxMotorStrength, Math.max(0, motorStrength)));
    },
    updateMotors(device, particles, bodyBuffer) {
      const driveAxis = (keys.has("KeyD") ? 1 : 0) - (keys.has("KeyA") ? 1 : 0);
      const targetAngularVelocity = driveAxis * 18;

      if (leftWheelBodyId !== undefined) {
        const motorStrength = driveAxis === 0 ? 0 : (motorStrengths.get(leftWheelBodyId) ?? defaultMotorStrength);
        particles.updateBodyMotor(device, bodyBuffer, leftWheelBodyId, targetAngularVelocity, motorStrength);
      }
      if (rightWheelBodyId !== undefined) {
        const motorStrength = driveAxis === 0 ? 0 : (motorStrengths.get(rightWheelBodyId) ?? defaultMotorStrength);
        particles.updateBodyMotor(device, bodyBuffer, rightWheelBodyId, targetAngularVelocity, motorStrength);
      }
    }
  };
}

function addVehicleScene(
  device: GPUDevice,
  particles: ParticleBuffers,
  bodyBuffer: GPUBuffer,
  bondBuffer: GPUBuffer,
  restShapeBuffer: GPUBuffer,
  jointBuffer: GPUBuffer,
  worldSize: { width: number; height: number },
  settings: SimulationSettings,
  bodyProperties: BodyProperties[],
  bodyRenderInfos: BodyRenderInfo[],
  vehicle: VehicleController
) {
  const centerX = worldSize.width > 620 ? Math.max(370, worldSize.width * 0.5) : worldSize.width * 0.5;
  const wheelOffset = worldSize.width > 520 ? 74 : 54;
  const wheelSize = worldSize.width > 520 ? 72 : 58;
  const chassisSize = worldSize.width > 520 ? 132 : 104;
  const wheelCenterY = Math.max(120, worldSize.height - wheelSize * 0.82);
  const axleY = wheelCenterY;
  const chassisCenterY = axleY - wheelSize * 0.78;
  const chassisAnchorY = axleY - chassisCenterY;
  const vehicleParticleRadius = 8;
  const defaultBodyProperties = {
    kind: "soft" as const,
    softBodyStrength: settings.softBodyStrength,
    viscosity: settings.viscosity,
    friction: settings.friction
  };
  const wheelBodyProperties = {
    ...defaultBodyProperties,
    friction: 1
  };

  const chassis = addSceneBody(
    device,
    particles,
    bodyBuffer,
    bondBuffer,
    restShapeBuffer,
    "rectangle",
    chassisSize,
    vehicleParticleRadius,
    defaultBodyProperties,
    worldSize,
    { x: centerX, y: chassisCenterY },
    bodyProperties,
    bodyRenderInfos
  );
  const leftWheel = addSceneBody(
    device,
    particles,
    bodyBuffer,
    bondBuffer,
    restShapeBuffer,
    "circle",
    wheelSize,
    vehicleParticleRadius,
    wheelBodyProperties,
    worldSize,
    { x: centerX - wheelOffset, y: wheelCenterY },
    bodyProperties,
    bodyRenderInfos
  );
  const rightWheel = addSceneBody(
    device,
    particles,
    bodyBuffer,
    bondBuffer,
    restShapeBuffer,
    "circle",
    wheelSize,
    vehicleParticleRadius,
    wheelBodyProperties,
    worldSize,
    { x: centerX + wheelOffset, y: wheelCenterY },
    bodyProperties,
    bodyRenderInfos
  );

  if (!chassis || !leftWheel || !rightWheel) {
    const jointCount = writeJointBuffer(device, jointBuffer, []);
    return { joints: [], jointCount };
  }

  vehicle.setWheels(leftWheel.bodyId, rightWheel.bodyId);
  const joints: JointDefinition[] = [
    {
      bodyA: chassis.bodyId,
      bodyB: leftWheel.bodyId,
      localAnchorA: { x: -wheelOffset, y: chassisAnchorY },
      localAnchorB: { x: 0, y: 0 },
      restLength: 0,
      stiffness: 0.78,
      influenceRadius: wheelOffset * 0.92
    },
    {
      bodyA: chassis.bodyId,
      bodyB: rightWheel.bodyId,
      localAnchorA: { x: wheelOffset, y: chassisAnchorY },
      localAnchorB: { x: 0, y: 0 },
      restLength: 0,
      stiffness: 0.78,
      influenceRadius: wheelOffset * 0.92
    }
  ];

  const jointCount = writeJointBuffer(device, jointBuffer, joints);
  return { joints, jointCount };
}

function addSceneBody(
  device: GPUDevice,
  particles: ParticleBuffers,
  bodyBuffer: GPUBuffer,
  bondBuffer: GPUBuffer,
  restShapeBuffer: GPUBuffer,
  shape: SoftBodyShape,
  size: number,
  particleRadius: number,
  properties: BodyProperties,
  worldSize: { width: number; height: number },
  center: { x: number; y: number },
  bodyProperties: BodyProperties[],
  bodyRenderInfos: BodyRenderInfo[]
) {
  const result = particles.addSoftBody(
    device,
    bodyBuffer,
    bondBuffer,
    restShapeBuffer,
    shape,
    size,
    particleRadius,
    properties,
    worldSize.width,
    worldSize.height,
    center
  );

  if (!result.added) {
    return undefined;
  }

  bodyProperties[result.bodyId] = { ...properties };
  bodyRenderInfos.push({
    kind: properties.kind,
    bodyId: result.bodyId,
    startIndex: result.startIndex,
    particleCount: result.addedParticles,
    perimeterParticleCount: result.perimeterParticleCount,
    particleRadius: result.particleRadius,
    materialId: result.materialId,
    restPositions: result.restPositions
  });

  return result;
}

type SvgBodySource = {
  points: SampledBodyPoint[];
  size: number;
  spacing: number;
  render: SvgRenderInfo;
};

type SvgRaster = {
  data: Uint8ClampedArray;
  width: number;
  height: number;
  canvas: HTMLCanvasElement;
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
};

async function createSvgBodySource(file: File | undefined, requestedSize: number, particleRadius: number): Promise<SvgBodySource> {
  const svgText = file ? await file.text() : await fetchDefaultSvg();
  const image = await loadSvgImage(svgText);
  const raster = rasterizeSvg(image);
  const size = Math.max(40, Math.min(480, requestedSize));
  const targetAcross = Math.max(6, Math.min(34, Math.round(size / 8)));
  const spacing = size / Math.max(1, targetAcross - 1);
  const maskWidth = raster.maxX - raster.minX + 1;
  const maskHeight = raster.maxY - raster.minY + 1;
  const scale = size / Math.max(maskWidth, maskHeight);
  const localWidth = maskWidth * scale;
  const localHeight = maskHeight * scale;
  const rowSpacing = spacing * Math.sqrt(3) * 0.5;
  const columns = Math.ceil(localWidth / spacing) + 3;
  const rows = Math.ceil(localHeight / rowSpacing) + 3;
  const boundaryPoints: SvgSamplePoint[] = [];
  const interiorCandidates: SvgSamplePoint[] = [];
  const sampleStep = Math.max(1, (particleRadius * 0.75) / scale);

  for (let row = 0; row < rows; row += 1) {
    const y = (row - (rows - 1) * 0.5) * rowSpacing;
    const rowOffset = row % 2 === 0 ? 0 : spacing * 0.5;

    for (let column = 0; column < columns; column += 1) {
      const x = (column - (columns - 1) * 0.5) * spacing + rowOffset;
      const sample = sampleSvgRaster(raster, x, y, scale);
      if (!sample.inside) {
        continue;
      }

      const point = {
        x,
        y,
        boundary: isSvgBoundarySample(raster, x, y, scale, sampleStep),
        color: sample.color,
        sourceX: sample.sourceX,
        sourceY: sample.sourceY
      };
      if (point.boundary) {
        boundaryPoints.push(point);
      } else {
        interiorCandidates.push(point);
      }
    }
  }

  const minInteriorDistanceSq = spacing * spacing * 0.62 * 0.62;
  const interiorPoints = interiorCandidates.filter((point) => !isTooCloseToPoints(point, boundaryPoints, minInteriorDistanceSq));
  const points = [...boundaryPoints, ...interiorPoints];
  if (points.length === 0) {
    throw new Error("SVG did not produce any filled particles.");
  }

  const renderPoints = points.map((point) => ({
    x: point.x,
    y: point.y,
    sourceX: point.sourceX,
    sourceY: point.sourceY
  }));

  return {
    points,
    size,
    spacing,
    render: {
      image: raster.canvas,
      points: renderPoints,
      triangles: createSvgRenderTriangles(renderPoints, spacing)
    }
  };
}

async function fetchDefaultSvg() {
  const response = await fetch(defaultSvgUrl);
  if (!response.ok) {
    throw new Error("Could not load default test.svg.");
  }
  return response.text();
}

async function loadSvgImage(svgText: string) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const url = URL.createObjectURL(new Blob([svgText], { type: "image/svg+xml" }));
    const image = new Image();
    image.onload = () => {
      URL.revokeObjectURL(url);
      resolve(image);
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Could not decode SVG image."));
    };
    image.src = url;
  });
}

function rasterizeSvg(image: HTMLImageElement): SvgRaster {
  const maxRasterSize = 640;
  const sourceWidth = Math.max(1, image.naturalWidth || maxRasterSize);
  const sourceHeight = Math.max(1, image.naturalHeight || maxRasterSize);
  const aspect = sourceWidth / sourceHeight;
  const canvas = document.createElement("canvas");
  canvas.width = aspect >= 1 ? maxRasterSize : Math.max(1, Math.round(maxRasterSize * aspect));
  canvas.height = aspect >= 1 ? Math.max(1, Math.round(maxRasterSize / aspect)) : maxRasterSize;

  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) {
    throw new Error("Could not create SVG sampling canvas.");
  }

  context.clearRect(0, 0, canvas.width, canvas.height);
  context.drawImage(image, 0, 0, canvas.width, canvas.height);
  const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
  const bounds = findAlphaBounds(imageData.data, canvas.width, canvas.height);
  if (!bounds) {
    throw new Error("SVG has no visible pixels to sample.");
  }

  return {
    data: imageData.data,
    width: canvas.width,
    height: canvas.height,
    canvas,
    ...bounds
  };
}

function findAlphaBounds(data: Uint8ClampedArray, width: number, height: number) {
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (alphaAt(data, width, height, x, y) <= 12) {
        continue;
      }

      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
    }
  }

  return maxX < minX || maxY < minY ? undefined : { minX, minY, maxX, maxY };
}

function sampleSvgRaster(raster: SvgRaster, localX: number, localY: number, scale: number) {
  const pixel = localToSvgPixel(raster, localX, localY, scale);
  const x = Math.round(pixel.x);
  const y = Math.round(pixel.y);
  const alpha = alphaAt(raster.data, raster.width, raster.height, x, y);
  return {
    inside: alpha > 24,
    color: colorAt(raster.data, raster.width, raster.height, x, y),
    sourceX: pixel.x,
    sourceY: pixel.y
  };
}

function isSvgBoundarySample(raster: SvgRaster, localX: number, localY: number, scale: number, sampleStep: number) {
  const pixel = localToSvgPixel(raster, localX, localY, scale);
  const offsets = [
    { x: sampleStep, y: 0 },
    { x: -sampleStep, y: 0 },
    { x: 0, y: sampleStep },
    { x: 0, y: -sampleStep },
    { x: sampleStep, y: sampleStep },
    { x: -sampleStep, y: sampleStep },
    { x: sampleStep, y: -sampleStep },
    { x: -sampleStep, y: -sampleStep }
  ];

  for (const offset of offsets) {
    const x = Math.round(pixel.x + offset.x);
    const y = Math.round(pixel.y + offset.y);
    if (alphaAt(raster.data, raster.width, raster.height, x, y) <= 24) {
      return true;
    }
  }

  return false;
}

function localToSvgPixel(raster: SvgRaster, localX: number, localY: number, scale: number) {
  return {
    x: (raster.minX + raster.maxX) * 0.5 + localX / scale,
    y: (raster.minY + raster.maxY) * 0.5 + localY / scale
  };
}

function alphaAt(data: Uint8ClampedArray, width: number, height: number, x: number, y: number) {
  if (x < 0 || y < 0 || x >= width || y >= height) {
    return 0;
  }

  return data[(y * width + x) * 4 + 3];
}

function colorAt(data: Uint8ClampedArray, width: number, height: number, x: number, y: number) {
  if (x < 0 || y < 0 || x >= width || y >= height) {
    return 0xffffff;
  }

  const offset = (y * width + x) * 4;
  return (data[offset] << 16) | (data[offset + 1] << 8) | data[offset + 2];
}

function isTooCloseToPoints(point: SampledBodyPoint, neighbors: SampledBodyPoint[], minDistanceSq: number) {
  for (const neighbor of neighbors) {
    const dx = point.x - neighbor.x;
    const dy = point.y - neighbor.y;
    if (dx * dx + dy * dy < minDistanceSq) {
      return true;
    }
  }

  return false;
}

function createSvgRenderTriangles(points: SvgRenderPoint[], spacing: number): SvgTriangle[] {
  const maxEdge = spacing * 1.35;
  const maxEdgeSq = maxEdge * maxEdge;
  const neighbors = points.map(() => [] as number[]);
  const triangleKeys = new Set<string>();
  const triangles: SvgTriangle[] = [];

  for (let i = 0; i < points.length; i += 1) {
    for (let j = i + 1; j < points.length; j += 1) {
      if (distanceSq(points[i], points[j]) <= maxEdgeSq) {
        neighbors[i].push(j);
        neighbors[j].push(i);
      }
    }
  }

  for (let i = 0; i < points.length; i += 1) {
    const localNeighbors = neighbors[i];
    for (let a = 0; a < localNeighbors.length; a += 1) {
      for (let b = a + 1; b < localNeighbors.length; b += 1) {
        const j = localNeighbors[a];
        const k = localNeighbors[b];
        if (distanceSq(points[j], points[k]) > maxEdgeSq) {
          continue;
        }

        const area = Math.abs(triangleArea(points[i], points[j], points[k]));
        if (area < spacing * spacing * 0.08) {
          continue;
        }

        const ordered = [i, j, k].sort((left, right) => left - right) as SvgTriangle;
        const key = ordered.join(":");
        if (triangleKeys.has(key)) {
          continue;
        }

        triangleKeys.add(key);
        triangles.push(ordered);
      }
    }
  }

  return triangles;
}

function distanceSq(a: { x: number; y: number }, b: { x: number; y: number }) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return dx * dx + dy * dy;
}

function triangleArea(a: { x: number; y: number }, b: { x: number; y: number }, c: { x: number; y: number }) {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

function bindControls(settings: SimulationSettings, vehicle: VehicleController): ControlBindings {
  const bodyShape = getSelect("body-shape");
  const svgFileControl = getElement("svg-file-control");
  const svgFile = getFileInput("svg-file");
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
  const staticBodyControl = getElement("static-body-control");
  const staticBody = getInput("static-body");
  const ropePinsControl = getElement("rope-pins-control");
  const ropePinStart = getInput("rope-pin-start");
  const ropePinEnd = getInput("rope-pin-end");
  const ropeLengthControl = getElement("rope-length-control");
  const ropeLength = getInput("rope-length");
  const ropeLengthOutput = getOutput("rope-length-output");
  const ropeDensityControl = getElement("rope-density-control");
  const ropeDensity = getInput("rope-density");
  const ropeDensityOutput = getOutput("rope-density-output");
  const motorStrengthControl = getElement("motor-strength-control");
  const motorStrength = getInput("motor-strength");
  const motorStrengthOutput = getOutput("motor-strength-output");
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
    kind: "soft",
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
    onShapeChange: () => undefined,
    getRenderMode() {
      return renderMode.value as RenderMode;
    },
    getSpawnBodyProperties() {
      return { ...spawnBodyProperties };
    },
    getRopeOptions() {
      return {
        pinnedStart: ropePinStart.checked,
        pinnedEnd: ropePinEnd.checked,
        lengthMultiplier: ropeLength.valueAsNumber,
        density: ropeDensity.valueAsNumber
      };
    },
    getSpawnArgs() {
      return {
        shape: bodyShape.value as BodyShapeChoice,
        bodySize: bodySize.valueAsNumber,
        particleRadius: particleRadius.valueAsNumber,
        svgFile: svgFile.files?.[0]
      };
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
        staticBody.checked = spawnBodyProperties.kind === "static";
        staticBody.disabled = false;
        motorStrengthControl.hidden = true;
      } else {
        propertyMode.textContent = `Editing body ${selectedBodyId + 1}`;
        softBodyStrength.value = String(selectedBodyProperties.softBodyStrength);
        viscosity.value = String(selectedBodyProperties.viscosity);
        friction.value = String(selectedBodyProperties.friction);
        staticBody.checked = selectedBodyProperties.kind === "static";
        staticBody.disabled = true;
        const selectedMotorStrength = vehicle.getMotorStrength(selectedBodyId);
        motorStrengthControl.hidden = selectedMotorStrength === undefined;
        if (selectedMotorStrength !== undefined) {
          motorStrength.value = String(selectedMotorStrength);
        }
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
    const spawningRope = bodyShape.value === "rope";
    svgFileControl.hidden = bodyShape.value !== "svg";
    ropePinsControl.hidden = !spawningRope;
    ropeLengthControl.hidden = !spawningRope;
    ropeDensityControl.hidden = !spawningRope;
    staticBodyControl.hidden = spawningRope;
    bodySizeOutput.value = bodySize.valueAsNumber.toFixed(0);
    particleRadiusOutput.value = particleRadius.valueAsNumber.toFixed(1);
    ropeLengthOutput.value = `${ropeLength.valueAsNumber.toFixed(2)}x`;
    ropeDensityOutput.value = `${ropeDensity.valueAsNumber.toFixed(2)}x`;
    gravityOutput.value = settings.gravityY.toFixed(0);
    dampingOutput.value = settings.damping.toFixed(2);
    substepsOutput.value = clampSubsteps(settings.substeps).toFixed(0);
    contactIterationsOutput.value = clampContactIterations(settings.contactIterations).toFixed(0);
    bondIterationsOutput.value = clampBondIterations(settings.bondIterations).toFixed(0);
    softBodyStrengthOutput.value = getCurrentBodyProperties().softBodyStrength.toFixed(0);
    viscosityOutput.value = getCurrentBodyProperties().viscosity.toFixed(1);
    frictionOutput.value = getCurrentBodyProperties().friction.toFixed(2);
    motorStrengthOutput.value = motorStrength.valueAsNumber.toFixed(0);
    wallBounceOutput.value = settings.wallBounce.toFixed(2);
    mouseForceOutput.value = settings.mouseForce.toLocaleString();
  };

  bodySize.addEventListener("input", () => {
    refresh();
  });

  bodyShape.addEventListener("change", () => {
    bindings.onShapeChange(bodyShape.value as BodyShapeChoice);
    refresh();
  });

  particleRadius.addEventListener("input", () => {
    refresh();
  });

  ropeLength.addEventListener("input", () => {
    refresh();
  });

  ropeDensity.addEventListener("input", () => {
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

  staticBody.addEventListener("change", () => {
    if (selectedBodyId !== undefined) {
      refresh();
      return;
    }

    updateCurrentBodyProperties({
      ...getCurrentBodyProperties(),
      kind: staticBody.checked ? "static" : "soft"
    });
    refresh();
  });

  motorStrength.addEventListener("input", () => {
    vehicle.setMotorStrength(selectedBodyId, motorStrength.valueAsNumber);
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
  addBodyButton.addEventListener("click", () => {
    void bindings.onAddBody(
      bodyShape.value as BodyShapeChoice,
      bodySize.valueAsNumber,
      particleRadius.valueAsNumber,
      svgFile.files?.[0]
    );
  });
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

function encodeJointSolve(
  encoder: GPUCommandEncoder,
  pipelines: Pipelines,
  particles: ParticleBuffers,
  particleCount: number
) {
  const jointPass = encoder.beginComputePass({ label: "Joint projection pass" });
  jointPass.setPipeline(pipelines.solveJointsPipeline);
  jointPass.setBindGroup(0, pipelines.solveJointBindGroups[particles.activeIndex]);
  jointPass.dispatchWorkgroups(Math.ceil(particleCount / WORKGROUP_SIZE));
  jointPass.end();
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

function clampJointIterations(value: number) {
  return Math.min(8, Math.max(0, Math.floor(value || 0)));
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
    if (body.svgRender && body.startIndex + body.particleCount <= particleCount) {
      renderWarpedSvgBody(context, view, body);
      continue;
    }

    if (body.kind === "rope") {
      renderRopeBody(context, view, body, particleCount);
      continue;
    }

    renderPolygonBody(context, view, body, particleCount);
  }
}

function renderRopeBody(
  context: CanvasRenderingContext2D,
  view: DataView,
  body: BodyRenderInfo,
  particleCount: number
) {
  if (body.particleCount < 2 || body.startIndex + body.particleCount > particleCount) {
    return;
  }

  context.beginPath();
  for (let i = 0; i < body.particleCount; i += 1) {
    const offset = (body.startIndex + i) * PARTICLE_STRIDE_BYTES;
    const x = view.getFloat32(offset, true);
    const y = view.getFloat32(offset + 4, true);
    if (i === 0) {
      context.moveTo(x, y);
    } else {
      context.lineTo(x, y);
    }
  }

  const bodyColor = BODY_FILL_COLORS[body.materialId % BODY_FILL_COLORS.length];
  context.strokeStyle = bodyColor;
  context.lineWidth = Math.max(2, body.particleRadius * 2.2);
  context.globalAlpha = 0.9;
  context.stroke();
  context.globalAlpha = 1;
}

function renderPolygonBody(
  context: CanvasRenderingContext2D,
  view: DataView,
  body: BodyRenderInfo,
  particleCount: number
) {
  if (body.perimeterParticleCount < 3 || body.startIndex + body.perimeterParticleCount > particleCount) {
    return;
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

function renderWarpedSvgBody(context: CanvasRenderingContext2D, view: DataView, body: BodyRenderInfo) {
  const svg = body.svgRender;
  if (!svg || svg.triangles.length === 0) {
    return;
  }

  const dpr = Math.max(1, Math.min(window.devicePixelRatio || 1, 2));
  context.save();
  context.imageSmoothingEnabled = true;
  context.globalAlpha = 0.96;

  for (const triangle of svg.triangles) {
    const sourceA = svg.points[triangle[0]];
    const sourceB = svg.points[triangle[1]];
    const sourceC = svg.points[triangle[2]];
    const destA = readBodyParticlePosition(view, body, triangle[0]);
    const destB = readBodyParticlePosition(view, body, triangle[1]);
    const destC = readBodyParticlePosition(view, body, triangle[2]);

    if (Math.abs(triangleArea(destA, destB, destC)) < 0.001) {
      continue;
    }

    const transform = sourceToDestinationTransform(sourceA, sourceB, sourceC, destA, destB, destC);
    if (!transform) {
      continue;
    }

    context.save();
    context.beginPath();
    context.moveTo(destA.x, destA.y);
    context.lineTo(destB.x, destB.y);
    context.lineTo(destC.x, destC.y);
    context.closePath();
    context.clip();
    context.setTransform(
      transform.a * dpr,
      transform.b * dpr,
      transform.c * dpr,
      transform.d * dpr,
      transform.e * dpr,
      transform.f * dpr
    );
    context.drawImage(svg.image, 0, 0);
    context.restore();
  }

  context.restore();
}

function readBodyParticlePosition(view: DataView, body: BodyRenderInfo, localIndex: number) {
  const offset = (body.startIndex + localIndex) * PARTICLE_STRIDE_BYTES;
  return {
    x: view.getFloat32(offset, true),
    y: view.getFloat32(offset + 4, true)
  };
}

function sourceToDestinationTransform(
  sourceA: SvgRenderPoint,
  sourceB: SvgRenderPoint,
  sourceC: SvgRenderPoint,
  destA: { x: number; y: number },
  destB: { x: number; y: number },
  destC: { x: number; y: number }
) {
  const x0 = sourceA.sourceX;
  const y0 = sourceA.sourceY;
  const x1 = sourceB.sourceX;
  const y1 = sourceB.sourceY;
  const x2 = sourceC.sourceX;
  const y2 = sourceC.sourceY;
  const denominator = x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1);
  if (Math.abs(denominator) < 0.000001) {
    return undefined;
  }

  return {
    a: (destA.x * (y1 - y2) + destB.x * (y2 - y0) + destC.x * (y0 - y1)) / denominator,
    b: (destA.y * (y1 - y2) + destB.y * (y2 - y0) + destC.y * (y0 - y1)) / denominator,
    c: (destA.x * (x2 - x1) + destB.x * (x0 - x2) + destC.x * (x1 - x0)) / denominator,
    d: (destA.y * (x2 - x1) + destB.y * (x0 - x2) + destC.y * (x1 - x0)) / denominator,
    e:
      (destA.x * (x1 * y2 - x2 * y1) +
        destB.x * (x2 * y0 - x0 * y2) +
        destC.x * (x0 * y1 - x1 * y0)) /
      denominator,
    f:
      (destA.y * (x1 * y2 - x2 * y1) +
        destB.y * (x2 * y0 - x0 * y2) +
        destC.y * (x0 * y1 - x1 * y0)) /
      denominator
  };
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

function updatePlacementPreview(
  marker: HTMLElement,
  band: HTMLElement,
  start: { x: number; y: number } | undefined,
  pointer: PointerState
) {
  if (!start) {
    marker.style.opacity = "0";
    band.style.opacity = "0";
    return;
  }

  marker.style.opacity = "1";
  marker.style.transform = `translate(${start.x}px, ${start.y}px)`;

  const dx = pointer.x - start.x;
  const dy = pointer.y - start.y;
  const length = Math.hypot(dx, dy);
  const angle = Math.atan2(dy, dx);
  band.style.opacity = "1";
  band.style.width = `${length}px`;
  band.style.transform = `translate(${start.x}px, ${start.y - 1}px) rotate(${angle}rad)`;
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

function getFileInput(id: string): HTMLInputElement {
  const input = getInput(id);
  if (input.type !== "file") {
    throw new Error(`Expected file input #${id}`);
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
