import {
  BOND_SLOT_COUNT,
  BOND_STRIDE_BYTES,
  BODY_STRIDE_BYTES,
  DEBUG_COUNTER_BYTES,
  DEFAULT_CONFIG,
  GRID_CELL_SIZE,
  GRID_PARTICLE_CAPACITY,
  JOINT_STRIDE_BYTES,
  MATERIAL_COUNT,
  MATERIAL_STRIDE_BYTES,
  MAX_BODIES,
  MAX_GRID_CELLS,
  MAX_GRID_COLUMNS,
  MAX_JOINTS,
  MAX_GRID_ROWS,
  MAX_SCAN_GROUPS,
  PARTICLE_STRIDE_BYTES,
  REST_SHAPE_STRIDE_BYTES,
  SCAN_BLOCK_SIZE,
  SIM_PARAMS_BYTES,
  type SimulationSettings
} from "../config";
import type { PointerState } from "../input";

export type SoftBodyShape = "square" | "rectangle" | "circle" | "triangle";
export type BodyKind = "soft" | "rope" | "static";
export type BodySpawnPoint = { x: number; y: number };
export type SampledBodyPoint = {
  x: number;
  y: number;
  boundary: boolean;
  color?: number;
};

const BODY_ID_MASK = 0x0000ffff;
const PARTICLE_KIND_SHIFT = 16;
const PARTICLE_KIND_SOFT = 0;
const PARTICLE_KIND_STATIC = 1;
const PARTICLE_KIND_ROPE = 2;
const INVALID_BOND_INDEX = 0xffffffff;
const PACKED_COLOR_FLAG = 0x80000000;

export type ParticleBuffers = {
  buffers: [GPUBuffer, GPUBuffer];
  activeIndex: number;
  maxParticles: number;
  particleCount: number;
  bodyCount: number;
  clear: (device: GPUDevice, bodyBuffer: GPUBuffer, bondBuffer: GPUBuffer, restShapeBuffer: GPUBuffer) => void;
  updateBodyProperties: (
    device: GPUDevice,
    bodyBuffer: GPUBuffer,
    bodyId: number,
    properties: BodyProperties
  ) => void;
  updateBodyMotor: (
    device: GPUDevice,
    bodyBuffer: GPUBuffer,
    bodyId: number,
    targetAngularVelocity: number,
    motorStrength: number
  ) => void;
  addSoftBody: (
    device: GPUDevice,
    bodyBuffer: GPUBuffer,
    bondBuffer: GPUBuffer,
    restShapeBuffer: GPUBuffer,
    shape: SoftBodyShape,
    size: number,
    particleRadius: number,
    properties: BodyProperties,
    worldWidth: number,
    worldHeight: number,
    spawnPoint?: BodySpawnPoint
  ) => SoftBodyAddResult;
  addSampledBody: (
    device: GPUDevice,
    bodyBuffer: GPUBuffer,
    bondBuffer: GPUBuffer,
    restShapeBuffer: GPUBuffer,
    points: SampledBodyPoint[],
    size: number,
    spacing: number,
    particleRadius: number,
    properties: BodyProperties,
    worldWidth: number,
    worldHeight: number,
    spawnPoint?: BodySpawnPoint
  ) => SoftBodyAddResult;
  addRope: (
    device: GPUDevice,
    bodyBuffer: GPUBuffer,
    bondBuffer: GPUBuffer,
    restShapeBuffer: GPUBuffer,
    size: number,
    particleRadius: number,
    properties: BodyProperties,
    worldWidth: number,
    worldHeight: number,
    pinnedStart: boolean,
    pinnedEnd: boolean,
    spawnPoint?: BodySpawnPoint,
    endPoint?: BodySpawnPoint,
    lengthMultiplier?: number,
    density?: number
  ) => SoftBodyAddResult;
  deleteBody: (
    device: GPUDevice,
    bodyBuffer: GPUBuffer,
    bondBuffer: GPUBuffer,
    restShapeBuffer: GPUBuffer,
    bodyId: number,
    startIndex: number,
    particleCount: number
  ) => boolean;
  swap: () => void;
};

export type BodyProperties = {
  kind: BodyKind;
  softBodyStrength: number;
  viscosity: number;
  friction: number;
};

export type JointDefinition = {
  bodyA: number;
  bodyB: number;
  localAnchorA: BodySpawnPoint;
  localAnchorB: BodySpawnPoint;
  restLength: number;
  stiffness: number;
  influenceRadius: number;
};

export type SoftBodyAddResult =
  | {
      added: true;
      particleCount: number;
      bodyCount: number;
      addedParticles: number;
      bodyId: number;
      startIndex: number;
      perimeterParticleCount: number;
      particleRadius: number;
      materialId: number;
      restPositions: Float32Array;
    }
  | {
      added: false;
      reason: string;
    };

export type GridBuffers = {
  pairValues: GPUBuffer;
  cellStarts: GPUBuffer;
  cellCounts: GPUBuffer;
  cellWriteOffsets: GPUBuffer;
  cellGroupSums: GPUBuffer;
  cellGroupOffsets: GPUBuffer;
  debugCounters: GPUBuffer;
  debugReadback: GPUBuffer;
  debugCounterBytes: number;
  maxCells: number;
  particleCapacity: number;
  scanBlockSize: number;
  scanGroupCount: number;
};

export function createParticleBuffers(device: GPUDevice, maxParticles = DEFAULT_CONFIG.maxParticles): ParticleBuffers {
  const bufferSize = maxParticles * PARTICLE_STRIDE_BYTES;
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const buffers: [GPUBuffer, GPUBuffer] = [
    device.createBuffer({ label: "Particles A", size: bufferSize, usage }),
    device.createBuffer({ label: "Particles B", size: bufferSize, usage })
  ];

  return {
    buffers,
    activeIndex: 0,
    maxParticles,
    particleCount: 0,
    bodyCount: 0,
    clear(device, bodyBuffer, bondBuffer, restShapeBuffer) {
      this.particleCount = 0;
      this.bodyCount = 0;
      this.activeIndex = 0;
      device.queue.writeBuffer(bodyBuffer, 0, new ArrayBuffer(MAX_BODIES * BODY_STRIDE_BYTES));
      device.queue.writeBuffer(bondBuffer, 0, createEmptyBondData(this.maxParticles));
      device.queue.writeBuffer(restShapeBuffer, 0, new ArrayBuffer(this.maxParticles * REST_SHAPE_STRIDE_BYTES));
    },
    updateBodyProperties(device, bodyBuffer, bodyId, properties) {
      if (bodyId < 0 || bodyId >= this.bodyCount) {
        return;
      }

      device.queue.writeBuffer(bodyBuffer, bodyId * BODY_STRIDE_BYTES + 16, createBodyPropertiesData(properties));
    },
    updateBodyMotor(device, bodyBuffer, bodyId, targetAngularVelocity, motorStrength) {
      if (bodyId < 0 || bodyId >= this.bodyCount) {
        return;
      }

      device.queue.writeBuffer(
        bodyBuffer,
        bodyId * BODY_STRIDE_BYTES + 8,
        createBodyMotorData(targetAngularVelocity, motorStrength)
      );
    },
    addSoftBody(
      device,
      bodyBuffer,
      bondBuffer,
      restShapeBuffer,
      shape,
      size,
      particleRadius,
      properties,
      worldWidth,
      worldHeight,
      spawnPoint
    ) {
      if (this.bodyCount >= MAX_BODIES) {
        return { added: false, reason: `Body limit reached (${MAX_BODIES}).` };
      }

      const body = createSoftBodyData({
        shape,
        size,
        particleRadius,
        bodyId: this.bodyCount,
        startIndex: this.particleCount,
        properties,
        worldWidth,
        worldHeight,
        spawnPoint
      });

      if (this.particleCount + body.particleCount > this.maxParticles) {
        return { added: false, reason: `Particle limit reached (${this.maxParticles.toLocaleString()}).` };
      }

      const particleOffset = this.particleCount * PARTICLE_STRIDE_BYTES;
      device.queue.writeBuffer(buffers[0], particleOffset, body.particleData);
      device.queue.writeBuffer(buffers[1], particleOffset, body.particleData);
      device.queue.writeBuffer(bodyBuffer, this.bodyCount * BODY_STRIDE_BYTES, body.bodyData);
      device.queue.writeBuffer(
        bondBuffer,
        this.particleCount * BOND_SLOT_COUNT * BOND_STRIDE_BYTES,
        body.bondData
      );
      device.queue.writeBuffer(restShapeBuffer, this.particleCount * REST_SHAPE_STRIDE_BYTES, body.restShapeData);

      this.particleCount += body.particleCount;
      this.bodyCount += 1;
      return {
        added: true,
        particleCount: this.particleCount,
        bodyCount: this.bodyCount,
        addedParticles: body.particleCount,
        bodyId: body.bodyId,
        startIndex: body.startIndex,
        perimeterParticleCount: body.perimeterParticleCount,
        particleRadius: body.particleRadius,
        materialId: body.materialId,
        restPositions: body.restPositions
      };
    },
    addSampledBody(
      device,
      bodyBuffer,
      bondBuffer,
      restShapeBuffer,
      points,
      size,
      spacing,
      particleRadius,
      properties,
      worldWidth,
      worldHeight,
      spawnPoint
    ) {
      if (this.bodyCount >= MAX_BODIES) {
        return { added: false, reason: `Body limit reached (${MAX_BODIES}).` };
      }

      const body = createSampledBodyData({
        points,
        size,
        spacing,
        particleRadius,
        bodyId: this.bodyCount,
        startIndex: this.particleCount,
        properties,
        worldWidth,
        worldHeight,
        spawnPoint
      });

      if (!body) {
        return { added: false, reason: "SVG did not produce any filled particles." };
      }

      if (this.particleCount + body.particleCount > this.maxParticles) {
        return { added: false, reason: `Particle limit reached (${this.maxParticles.toLocaleString()}).` };
      }

      const particleOffset = this.particleCount * PARTICLE_STRIDE_BYTES;
      device.queue.writeBuffer(buffers[0], particleOffset, body.particleData);
      device.queue.writeBuffer(buffers[1], particleOffset, body.particleData);
      device.queue.writeBuffer(bodyBuffer, this.bodyCount * BODY_STRIDE_BYTES, body.bodyData);
      device.queue.writeBuffer(
        bondBuffer,
        this.particleCount * BOND_SLOT_COUNT * BOND_STRIDE_BYTES,
        body.bondData
      );
      device.queue.writeBuffer(restShapeBuffer, this.particleCount * REST_SHAPE_STRIDE_BYTES, body.restShapeData);

      this.particleCount += body.particleCount;
      this.bodyCount += 1;
      return {
        added: true,
        particleCount: this.particleCount,
        bodyCount: this.bodyCount,
        addedParticles: body.particleCount,
        bodyId: body.bodyId,
        startIndex: body.startIndex,
        perimeterParticleCount: body.perimeterParticleCount,
        particleRadius: body.particleRadius,
        materialId: body.materialId,
        restPositions: body.restPositions
      };
    },
    addRope(
      device,
      bodyBuffer,
      bondBuffer,
      restShapeBuffer,
      size,
      particleRadius,
      properties,
      worldWidth,
      worldHeight,
      pinnedStart,
      pinnedEnd,
      spawnPoint,
      endPoint,
      lengthMultiplier,
      density
    ) {
      if (this.bodyCount >= MAX_BODIES) {
        return { added: false, reason: `Body limit reached (${MAX_BODIES}).` };
      }

      const body = createRopeBodyData({
        size,
        particleRadius,
        bodyId: this.bodyCount,
        startIndex: this.particleCount,
        properties: {
          ...properties,
          kind: "rope"
        },
        worldWidth,
        worldHeight,
        pinnedStart,
        pinnedEnd,
        spawnPoint,
        endPoint,
        lengthMultiplier,
        density
      });

      if (this.particleCount + body.particleCount > this.maxParticles) {
        return { added: false, reason: `Particle limit reached (${this.maxParticles.toLocaleString()}).` };
      }

      const particleOffset = this.particleCount * PARTICLE_STRIDE_BYTES;
      device.queue.writeBuffer(buffers[0], particleOffset, body.particleData);
      device.queue.writeBuffer(buffers[1], particleOffset, body.particleData);
      device.queue.writeBuffer(bodyBuffer, this.bodyCount * BODY_STRIDE_BYTES, body.bodyData);
      device.queue.writeBuffer(
        bondBuffer,
        this.particleCount * BOND_SLOT_COUNT * BOND_STRIDE_BYTES,
        body.bondData
      );
      device.queue.writeBuffer(restShapeBuffer, this.particleCount * REST_SHAPE_STRIDE_BYTES, body.restShapeData);

      this.particleCount += body.particleCount;
      this.bodyCount += 1;
      return {
        added: true,
        particleCount: this.particleCount,
        bodyCount: this.bodyCount,
        addedParticles: body.particleCount,
        bodyId: body.bodyId,
        startIndex: body.startIndex,
        perimeterParticleCount: body.perimeterParticleCount,
        particleRadius: body.particleRadius,
        materialId: body.materialId,
        restPositions: body.restPositions
      };
    },
    deleteBody(device, bodyBuffer, bondBuffer, restShapeBuffer, bodyId, startIndex, particleCount) {
      if (bodyId < 0 || bodyId >= this.bodyCount || particleCount <= 0) {
        return false;
      }
      if (startIndex < 0 || startIndex + particleCount > this.maxParticles) {
        return false;
      }

      const deadParticles = new ArrayBuffer(particleCount * PARTICLE_STRIDE_BYTES);
      device.queue.writeBuffer(buffers[0], startIndex * PARTICLE_STRIDE_BYTES, deadParticles);
      device.queue.writeBuffer(buffers[1], startIndex * PARTICLE_STRIDE_BYTES, deadParticles);
      device.queue.writeBuffer(
        bondBuffer,
        startIndex * BOND_SLOT_COUNT * BOND_STRIDE_BYTES,
        createEmptyBondData(particleCount)
      );
      device.queue.writeBuffer(
        restShapeBuffer,
        startIndex * REST_SHAPE_STRIDE_BYTES,
        new ArrayBuffer(particleCount * REST_SHAPE_STRIDE_BYTES)
      );
      device.queue.writeBuffer(bodyBuffer, bodyId * BODY_STRIDE_BYTES, new ArrayBuffer(BODY_STRIDE_BYTES));
      return true;
    },
    swap() {
      this.activeIndex = 1 - this.activeIndex;
    }
  };
}

export function createBodyBuffer(device: GPUDevice): GPUBuffer {
  return device.createBuffer({
    label: "Soft body metadata",
    size: MAX_BODIES * BODY_STRIDE_BYTES,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  });
}

export function createBondBuffer(device: GPUDevice, maxParticles = DEFAULT_CONFIG.maxParticles): GPUBuffer {
  const bondBuffer = device.createBuffer({
    label: "Soft body explicit bonds",
    size: maxParticles * BOND_SLOT_COUNT * BOND_STRIDE_BYTES,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  });

  device.queue.writeBuffer(bondBuffer, 0, createEmptyBondData(maxParticles));
  return bondBuffer;
}

export function createRestShapeBuffer(device: GPUDevice, maxParticles = DEFAULT_CONFIG.maxParticles): GPUBuffer {
  return device.createBuffer({
    label: "Soft body rest shape memory",
    size: maxParticles * REST_SHAPE_STRIDE_BYTES,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  });
}

export function createJointBuffer(device: GPUDevice): GPUBuffer {
  return device.createBuffer({
    label: "Body joint constraints",
    size: MAX_JOINTS * JOINT_STRIDE_BYTES,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  });
}

export function createUniformBuffer(device: GPUDevice): GPUBuffer {
  return device.createBuffer({
    label: "Simulation parameters",
    size: SIM_PARAMS_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  });
}

export function createMaterialBuffer(device: GPUDevice): GPUBuffer {
  const materialData = createMaterialData();
  const materialBuffer = device.createBuffer({
    label: "Material parameters",
    size: materialData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  });

  device.queue.writeBuffer(materialBuffer, 0, materialData);
  return materialBuffer;
}

export function createGridBuffers(device: GPUDevice): GridBuffers {
  return {
    pairValues: device.createBuffer({
      label: "Spatial grid particle ids",
      size: GRID_PARTICLE_CAPACITY * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE
    }),
    cellStarts: device.createBuffer({
      label: "Spatial grid cell starts",
      size: MAX_GRID_CELLS * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE
    }),
    cellCounts: device.createBuffer({
      label: "Spatial grid cell counts",
      size: MAX_GRID_CELLS * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE
    }),
    cellWriteOffsets: device.createBuffer({
      label: "Spatial grid scatter offsets",
      size: MAX_GRID_CELLS * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE
    }),
    cellGroupSums: device.createBuffer({
      label: "Spatial grid scan group sums",
      size: MAX_SCAN_GROUPS * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE
    }),
    cellGroupOffsets: device.createBuffer({
      label: "Spatial grid scan group offsets",
      size: MAX_SCAN_GROUPS * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE
    }),
    debugCounters: device.createBuffer({
      label: "Simulation debug counters",
      size: DEBUG_COUNTER_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    }),
    debugReadback: device.createBuffer({
      label: "Simulation debug readback",
      size: DEBUG_COUNTER_BYTES,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    }),
    debugCounterBytes: DEBUG_COUNTER_BYTES,
    maxCells: MAX_GRID_CELLS,
    particleCapacity: GRID_PARTICLE_CAPACITY,
    scanBlockSize: SCAN_BLOCK_SIZE,
    scanGroupCount: MAX_SCAN_GROUPS
  };
}

function createMaterialData(): ArrayBuffer {
  const data = new ArrayBuffer(MATERIAL_COUNT * MATERIAL_STRIDE_BYTES);
  const view = new DataView(data);

  for (let i = 0; i < MATERIAL_COUNT; i += 1) {
    const offset = i * MATERIAL_STRIDE_BYTES;
    const color = MATERIAL_COLORS[i];

    for (let channel = 0; channel < 4; channel += 1) {
      view.setFloat32(offset + channel * Float32Array.BYTES_PER_ELEMENT, color[channel], true);
    }
  }

  return data;
}

const MATERIAL_COLORS: Array<[number, number, number, number]> = [
  [0.28, 0.78, 0.95, 1.0],
  [0.98, 0.70, 0.25, 1.0],
  [0.54, 0.92, 0.48, 1.0],
  [0.92, 0.46, 0.73, 1.0]
];

export function getGridDimensions(worldWidth: number, worldHeight: number) {
  return {
    columns: Math.min(MAX_GRID_COLUMNS, Math.max(1, Math.ceil(worldWidth / GRID_CELL_SIZE))),
    rows: Math.min(MAX_GRID_ROWS, Math.max(1, Math.ceil(worldHeight / GRID_CELL_SIZE))),
    cellSize: GRID_CELL_SIZE
  };
}

export function writeSimParams(
  device: GPUDevice,
  uniformBuffer: GPUBuffer,
  settings: SimulationSettings,
  pointer: PointerState,
  worldWidth: number,
  worldHeight: number,
  deltaTime: number,
  jointCount = 0,
  solverPhase = 0
) {
  const data = new ArrayBuffer(SIM_PARAMS_BYTES);
  const view = new DataView(data);

  const dragForce = pointer.active && pointer.selectedParticleIndex !== 0xffffffff ? settings.mouseForce : 0;
  const grid = getGridDimensions(worldWidth, worldHeight);

  view.setFloat32(0, 0, true);
  view.setFloat32(4, settings.gravityY, true);
  view.setFloat32(8, worldWidth, true);
  view.setFloat32(12, worldHeight, true);
  view.setFloat32(16, pointer.x, true);
  view.setFloat32(20, pointer.y, true);
  view.setFloat32(24, dragForce, true);
  view.setFloat32(28, deltaTime, true);
  view.setFloat32(32, settings.damping, true);
  view.setUint32(36, settings.particleCount, true);
  view.setFloat32(40, settings.wallBounce, true);
  view.setFloat32(44, settings.maxSpeed, true);
  view.setFloat32(48, grid.cellSize, true);
  view.setUint32(52, grid.columns, true);
  view.setUint32(56, grid.rows, true);
  view.setUint32(60, GRID_PARTICLE_CAPACITY, true);
  view.setFloat32(64, 0, true);
  view.setFloat32(68, settings.softBodyStrength, true);
  view.setFloat32(72, settings.viscosity, true);
  view.setUint32(76, settings.contactIterations, true);
  view.setUint32(80, pointer.selectedParticleIndex, true);
  view.setUint32(84, settings.bondIterations, true);
  view.setUint32(88, jointCount, true);
  view.setUint32(92, settings.jointIterations, true);
  view.setUint32(96, solverPhase, true);

  device.queue.writeBuffer(uniformBuffer, 0, data);
}

export function writeJointBuffer(device: GPUDevice, jointBuffer: GPUBuffer, joints: JointDefinition[]) {
  const jointCount = Math.min(MAX_JOINTS, joints.length);
  const data = new ArrayBuffer(MAX_JOINTS * JOINT_STRIDE_BYTES);
  const view = new DataView(data);

  for (let index = 0; index < jointCount; index += 1) {
    const joint = joints[index];
    const offset = index * JOINT_STRIDE_BYTES;
    view.setUint32(offset, joint.bodyA, true);
    view.setUint32(offset + 4, joint.bodyB, true);
    view.setUint32(offset + 8, 1, true);
    view.setUint32(offset + 12, 1, true);
    view.setFloat32(offset + 16, joint.localAnchorA.x, true);
    view.setFloat32(offset + 20, joint.localAnchorA.y, true);
    view.setFloat32(offset + 24, joint.localAnchorB.x, true);
    view.setFloat32(offset + 28, joint.localAnchorB.y, true);
    view.setFloat32(offset + 32, joint.restLength, true);
    view.setFloat32(offset + 36, joint.stiffness, true);
    view.setFloat32(offset + 40, joint.influenceRadius, true);
    view.setFloat32(offset + 44, 0, true);
  }

  device.queue.writeBuffer(jointBuffer, 0, data);
  return jointCount;
}

type SoftBodyDataOptions = {
  shape: SoftBodyShape;
  size: number;
  particleRadius: number;
  bodyId: number;
  startIndex: number;
  properties: BodyProperties;
  worldWidth: number;
  worldHeight: number;
  spawnPoint?: BodySpawnPoint;
};

type SampledBodyDataOptions = Omit<SoftBodyDataOptions, "shape"> & {
  points: SampledBodyPoint[];
  spacing: number;
};

type RopeBodyDataOptions = Omit<SoftBodyDataOptions, "shape"> & {
  pinnedStart: boolean;
  pinnedEnd: boolean;
  endPoint?: BodySpawnPoint;
  lengthMultiplier?: number;
  density?: number;
};

function createSoftBodyData(options: SoftBodyDataOptions) {
  const size = Math.max(40, Math.min(480, options.size));
  const targetAcross = Math.max(6, Math.min(34, Math.round(size / 8)));
  const spacing = size / Math.max(1, targetAcross - 1);
  const rng = mulberry32(0x5eed1234 + options.bodyId * 0x9e3779b9);
  const points = createShapePoints(options.shape, size, spacing, rng);
  return createBodyData({
    ...options,
    points,
    size,
    spacing,
    rng
  });
}

function createSampledBodyData(options: SampledBodyDataOptions) {
  const points = options.points.filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
  if (points.length === 0) {
    return undefined;
  }

  const rng = mulberry32(0x5eed1234 + options.bodyId * 0x9e3779b9);
  return createBodyData({
    ...options,
    points,
    size: Math.max(40, Math.min(480, options.size)),
    spacing: Math.max(1, options.spacing),
    rng
  });
}

function createRopeBodyData(options: RopeBodyDataOptions) {
  const radius = Math.max(1, Math.min(8, options.particleRadius));
  const density = Math.max(0.4, Math.min(2.5, options.density ?? 1));
  const spacing = Math.max(radius * 2.2, 7) / density;
  const hasEndPoints = options.spawnPoint !== undefined && options.endPoint !== undefined;
  const lengthMultiplier = Math.max(0.25, Math.min(2.5, options.lengthMultiplier ?? 1));
  let length: number;
  let originX: number;
  let originY: number;
  let axisX: number;
  let axisY: number;
  let normalX: number;
  let normalY: number;
  let particleSpan: number;
  let initialSag: number;

  if (hasEndPoints) {
    const start = options.spawnPoint!;
    const end = options.endPoint!;
    const dx = end.x - start.x;
    const dy = end.y - start.y;
    const measured = Math.max(Math.hypot(dx, dy), spacing * 3);
    length = Math.max(spacing * 3, Math.min(1500, measured * lengthMultiplier));
    particleSpan = measured;
    const inv = 1 / measured;
    axisX = dx === 0 && dy === 0 ? 1 : dx * inv;
    axisY = dx === 0 && dy === 0 ? 0 : dy * inv;
    normalX = -axisY;
    normalY = axisX;
    originX = (start.x + end.x) * 0.5;
    originY = (start.y + end.y) * 0.5;
    initialSag = 0;
  } else {
    length = Math.max(48, Math.min(480, options.size));
    particleSpan = length;
    const center =
      options.spawnPoint ?? pickBodyCenter(options.bodyId, length, options.worldWidth, options.worldHeight);
    originX = center.x;
    originY = center.y;
    axisX = 1;
    axisY = 0;
    normalX = 0;
    normalY = 1;
    initialSag = Math.min(36, length * 0.12);
  }

  const segmentCount = Math.max(3, Math.min(96, Math.round(length / spacing)));
  const particleCount = segmentCount + 1;
  const bodyId = options.bodyId & BODY_ID_MASK;
  const materialId = options.bodyId % MATERIAL_COUNT;
  const restSag = hasEndPoints ? 0 : Math.min(36, length * 0.12);
  const particleData = new ArrayBuffer(particleCount * PARTICLE_STRIDE_BYTES);
  const bondData = createRopeBondData(particleCount, options.startIndex, length / segmentCount);
  const restShapeData = createRopeRestShapeData(particleCount, length, restSag);
  const restPositions = new Float32Array(particleCount * 2);
  const bodyData = new ArrayBuffer(BODY_STRIDE_BYTES);
  const particleView = new DataView(particleData);
  const bodyView = new DataView(bodyData);

  for (let localIndex = 0; localIndex < particleCount; localIndex += 1) {
    const offset = localIndex * PARTICLE_STRIDE_BYTES;
    const t = localIndex / segmentCount;
    const localX = (t - 0.5) * particleSpan;
    const localY = Math.sin(t * Math.PI) * initialSag;
    const worldX = originX + axisX * localX + normalX * localY;
    const worldY = originY + axisY * localX + normalY * localY;
    const pinned = (localIndex === 0 && options.pinnedStart) || (localIndex === segmentCount && options.pinnedEnd);
    const particleKind = pinned ? "static" : "rope";
    const restLocalX = (t - 0.5) * length;
    const restLocalY = Math.sin(t * Math.PI) * restSag;
    restPositions[localIndex * 2] = restLocalX;
    restPositions[localIndex * 2 + 1] = restLocalY;

    particleView.setFloat32(offset, worldX, true);
    particleView.setFloat32(offset + 4, worldY, true);
    particleView.setFloat32(offset + 8, 0, true);
    particleView.setFloat32(offset + 12, 0, true);
    particleView.setUint32(offset + 16, materialId, true);
    particleView.setUint32(offset + 20, createParticleFlags(bodyId, particleKind), true);
    particleView.setFloat32(offset + 24, radius, true);
    particleView.setFloat32(offset + 28, pinned ? 0 : radius * radius, true);
  }

  bodyView.setUint32(0, options.startIndex, true);
  bodyView.setUint32(4, particleCount, true);
  bodyView.setFloat32(8, 0, true);
  bodyView.setFloat32(12, 0, true);
  bodyView.setFloat32(16, options.properties.softBodyStrength, true);
  bodyView.setFloat32(20, options.properties.viscosity, true);
  bodyView.setFloat32(24, options.properties.friction, true);
  bodyView.setUint32(28, bodyKindFlag("rope"), true);

  return {
    particleData,
    bodyData,
    bondData,
    restShapeData,
    restPositions,
    particleCount,
    perimeterParticleCount: particleCount,
    particleRadius: radius,
    materialId,
    bodyId: options.bodyId,
    startIndex: options.startIndex
  };
}

type BodyDataOptions = Omit<SampledBodyDataOptions, "points"> & {
  points: SampledBodyPoint[];
  rng: () => number;
};

function createBodyData(options: BodyDataOptions) {
  const points = options.points;
  const particleCount = points.length;
  const perimeterParticleCount = points.findIndex((point) => !point.boundary);
  const particleData = new ArrayBuffer(particleCount * PARTICLE_STRIDE_BYTES);
  const bondData = createBondData(points, options.startIndex, options.spacing);
  const restShapeData = createRestShapeData(points);
  const bodyData = new ArrayBuffer(BODY_STRIDE_BYTES);
  const particleView = new DataView(particleData);
  const bodyView = new DataView(bodyData);
  const center = options.spawnPoint ?? pickBodyCenter(options.bodyId, options.size, options.worldWidth, options.worldHeight);
  const radius = Math.max(1, Math.min(8, options.particleRadius));
  const materialId = options.bodyId % MATERIAL_COUNT;
  const bodyId = options.bodyId & BODY_ID_MASK;
  const particleFlags = createParticleFlags(bodyId, options.properties.kind);
  const particleMass = options.properties.kind === "static" ? 0 : radius * radius;
  const bodyFlags = bodyKindFlag(options.properties.kind);

  for (let localIndex = 0; localIndex < points.length; localIndex += 1) {
    const point = points[localIndex];
    const offset = localIndex * PARTICLE_STRIDE_BYTES;
    const angle = options.rng() * Math.PI * 2;
    const speed = options.rng() * 8;
    const pointMaterialId = point.color === undefined ? materialId : packParticleColor(point.color);

    particleView.setFloat32(offset, center.x + point.x, true);
    particleView.setFloat32(offset + 4, center.y + point.y, true);
    particleView.setFloat32(offset + 8, Math.cos(angle) * speed, true);
    particleView.setFloat32(offset + 12, Math.sin(angle) * speed, true);
    particleView.setUint32(offset + 16, pointMaterialId, true);
    particleView.setUint32(offset + 20, particleFlags, true);
    particleView.setFloat32(offset + 24, radius, true);
    particleView.setFloat32(offset + 28, particleMass, true);
  }

  bodyView.setUint32(0, options.startIndex, true);
  bodyView.setUint32(4, particleCount, true);
  bodyView.setFloat32(8, 0, true);
  bodyView.setFloat32(12, 0, true);
  bodyView.setFloat32(16, options.properties.softBodyStrength, true);
  bodyView.setFloat32(20, options.properties.viscosity, true);
  bodyView.setFloat32(24, options.properties.friction, true);
  bodyView.setUint32(28, bodyFlags, true);

  const restPositions = new Float32Array(particleCount * 2);
  for (let i = 0; i < particleCount; i += 1) {
    restPositions[i * 2] = points[i].x;
    restPositions[i * 2 + 1] = points[i].y;
  }

  return {
    particleData,
    bodyData,
    bondData,
    restShapeData,
    restPositions,
    particleCount,
    perimeterParticleCount: perimeterParticleCount === -1 ? particleCount : perimeterParticleCount,
    particleRadius: radius,
    materialId,
    bodyId: options.bodyId,
    startIndex: options.startIndex
  };
}

function packParticleColor(color: number) {
  return (PACKED_COLOR_FLAG | (color & 0x00ffffff)) >>> 0;
}

function createParticleFlags(bodyId: number, kind: BodyKind) {
  return ((bodyKindFlag(kind) << PARTICLE_KIND_SHIFT) | (bodyId & BODY_ID_MASK)) >>> 0;
}

function bodyKindFlag(kind: BodyKind) {
  if (kind === "static") {
    return PARTICLE_KIND_STATIC;
  }

  if (kind === "rope") {
    return PARTICLE_KIND_ROPE;
  }

  return PARTICLE_KIND_SOFT;
}

function createBodyPropertiesData(properties: BodyProperties) {
  const data = new ArrayBuffer(12);
  const view = new DataView(data);
  view.setFloat32(0, properties.softBodyStrength, true);
  view.setFloat32(4, properties.viscosity, true);
  view.setFloat32(8, properties.friction, true);
  return data;
}

function createBodyMotorData(targetAngularVelocity: number, motorStrength: number) {
  const data = new ArrayBuffer(8);
  const view = new DataView(data);
  view.setFloat32(0, targetAngularVelocity, true);
  view.setFloat32(4, motorStrength, true);
  return data;
}

type BodyPoint = SampledBodyPoint;

type BondNeighbor = {
  index: number;
  restLength: number;
  weight: number;
};

function createRestShapeData(points: BodyPoint[]) {
  const data = new ArrayBuffer(points.length * REST_SHAPE_STRIDE_BYTES);
  const view = new DataView(data);

  for (let i = 0; i < points.length; i += 1) {
    const point = points[i];
    const offset = i * REST_SHAPE_STRIDE_BYTES;
    view.setFloat32(offset, point.x, true);
    view.setFloat32(offset + 4, point.y, true);
    view.setFloat32(offset + 8, point.boundary ? 1 : 0.18, true);
  }

  return data;
}

function createRopeRestShapeData(particleCount: number, length: number, ySag: number) {
  const data = new ArrayBuffer(particleCount * REST_SHAPE_STRIDE_BYTES);
  const view = new DataView(data);
  const segmentCount = Math.max(1, particleCount - 1);

  for (let i = 0; i < particleCount; i += 1) {
    const offset = i * REST_SHAPE_STRIDE_BYTES;
    const t = i / segmentCount;
    view.setFloat32(offset, (t - 0.5) * length, true);
    view.setFloat32(offset + 4, Math.sin(t * Math.PI) * ySag, true);
    view.setFloat32(offset + 8, 0, true);
  }

  return data;
}

function createShapePoints(shape: SoftBodyShape, size: number, spacing: number, rng: () => number): BodyPoint[] {
  if (shape === "rectangle") {
    return createRectanglePoints(size * 1.78, size * 0.48, spacing, rng);
  }

  const halfSize = size * 0.5;
  const rowSpacing = spacing * Math.sqrt(3) * 0.5;
  const columns = Math.ceil(size / spacing) + 2;
  const rows = Math.ceil(size / rowSpacing) + 2;
  const outlinePoints = createOutlinePoints(shape, halfSize, spacing);
  const interiorPoints: BodyPoint[] = [];
  const minInteriorDistanceSq = spacing * spacing * 0.62 * 0.62;

  for (let row = 0; row < rows; row += 1) {
    const y = (row - (rows - 1) * 0.5) * rowSpacing;
    const rowOffset = row % 2 === 0 ? 0 : spacing * 0.5;

    for (let column = 0; column < columns; column += 1) {
      const x = (column - (columns - 1) * 0.5) * spacing + rowOffset;
      if (!shapeContainsPoint(shape, x, y, halfSize)) {
        continue;
      }

      if (isTooCloseToOutline(x, y, outlinePoints, minInteriorDistanceSq)) {
        continue;
      }

      interiorPoints.push({
        x,
        y,
        boundary: false
      });
    }
  }

  relaxInteriorPoints(shape, halfSize, spacing, outlinePoints, interiorPoints, rng);

  const points = [...outlinePoints, ...interiorPoints];
  if (points.length === 0) {
    points.push({ x: 0, y: 0, boundary: true });
  }

  return points;
}

function createRectanglePoints(width: number, height: number, spacing: number, rng: () => number): BodyPoint[] {
  const halfWidth = width * 0.5;
  const halfHeight = height * 0.5;
  const rowSpacing = spacing * Math.sqrt(3) * 0.5;
  const columns = Math.ceil(width / spacing) + 2;
  const rows = Math.ceil(height / rowSpacing) + 2;
  const outlinePoints = createRectangleOutlinePoints(halfWidth, halfHeight, spacing);
  const interiorPoints: BodyPoint[] = [];
  const minInteriorDistanceSq = spacing * spacing * 0.62 * 0.62;

  for (let row = 0; row < rows; row += 1) {
    const y = (row - (rows - 1) * 0.5) * rowSpacing;
    const rowOffset = row % 2 === 0 ? 0 : spacing * 0.5;

    for (let column = 0; column < columns; column += 1) {
      const x = (column - (columns - 1) * 0.5) * spacing + rowOffset;
      if (Math.abs(x) > halfWidth || Math.abs(y) > halfHeight) {
        continue;
      }

      if (isTooCloseToOutline(x, y, outlinePoints, minInteriorDistanceSq)) {
        continue;
      }

      interiorPoints.push({
        x,
        y,
        boundary: false
      });
    }
  }

  relaxRectangleInteriorPoints(halfWidth, halfHeight, spacing, outlinePoints, interiorPoints, rng);
  const points = [...outlinePoints, ...interiorPoints];
  if (points.length === 0) {
    points.push({ x: 0, y: 0, boundary: true });
  }

  return points;
}

function createOutlinePoints(shape: SoftBodyShape, halfSize: number, spacing: number): BodyPoint[] {
  if (shape === "circle") {
    return createCircleOutlinePoints(halfSize, spacing);
  }

  const vertices =
    shape === "triangle"
      ? [
          { x: 0, y: -halfSize },
          { x: halfSize, y: halfSize },
          { x: -halfSize, y: halfSize }
        ]
      : [
          { x: -halfSize, y: -halfSize },
          { x: halfSize, y: -halfSize },
          { x: halfSize, y: halfSize },
          { x: -halfSize, y: halfSize }
        ];

  return createPolylineOutlinePoints(vertices, spacing);
}

function createRectangleOutlinePoints(halfWidth: number, halfHeight: number, spacing: number): BodyPoint[] {
  return createPolylineOutlinePoints(
    [
      { x: -halfWidth, y: -halfHeight },
      { x: halfWidth, y: -halfHeight },
      { x: halfWidth, y: halfHeight },
      { x: -halfWidth, y: halfHeight }
    ],
    spacing
  );
}

function createCircleOutlinePoints(radius: number, spacing: number): BodyPoint[] {
  const circumference = Math.PI * 2 * radius;
  const count = Math.max(8, Math.round(circumference / spacing));
  const points: BodyPoint[] = [];

  for (let i = 0; i < count; i += 1) {
    const angle = (i / count) * Math.PI * 2;
    points.push({
      x: Math.cos(angle) * radius,
      y: Math.sin(angle) * radius,
      boundary: true
    });
  }

  return points;
}

function createPolylineOutlinePoints(vertices: Array<{ x: number; y: number }>, spacing: number): BodyPoint[] {
  const points: BodyPoint[] = [];

  for (let vertexIndex = 0; vertexIndex < vertices.length; vertexIndex += 1) {
    const start = vertices[vertexIndex];
    const end = vertices[(vertexIndex + 1) % vertices.length];
    const dx = end.x - start.x;
    const dy = end.y - start.y;
    const edgeLength = Math.hypot(dx, dy);
    const segments = Math.max(1, Math.round(edgeLength / spacing));

    for (let segment = 0; segment < segments; segment += 1) {
      const t = segment / segments;
      points.push({
        x: start.x + dx * t,
        y: start.y + dy * t,
        boundary: true
      });
    }
  }

  return points;
}

function isTooCloseToOutline(
  x: number,
  y: number,
  outlinePoints: BodyPoint[],
  minInteriorDistanceSq: number
) {
  for (const outline of outlinePoints) {
    const dx = x - outline.x;
    const dy = y - outline.y;
    if (dx * dx + dy * dy < minInteriorDistanceSq) {
      return true;
    }
  }

  return false;
}

function relaxInteriorPoints(
  shape: SoftBodyShape,
  halfSize: number,
  spacing: number,
  outlinePoints: BodyPoint[],
  interiorPoints: BodyPoint[],
  rng: () => number
) {
  const iterations = 8;
  const minDistance = spacing * 0.82;
  const minDistanceSq = minDistance * minDistance;
  const maxStep = spacing * 0.18;

  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const deltas = interiorPoints.map(() => ({ x: 0, y: 0 }));

    for (let i = 0; i < interiorPoints.length; i += 1) {
      accumulateSpacingDelta(interiorPoints[i], outlinePoints, minDistanceSq, deltas[i]);

      for (let j = i + 1; j < interiorPoints.length; j += 1) {
        const delta = getRepulsionDelta(interiorPoints[i], interiorPoints[j], minDistanceSq, rng);
        deltas[i].x += delta.x;
        deltas[i].y += delta.y;
        deltas[j].x -= delta.x;
        deltas[j].y -= delta.y;
      }
    }

    for (let i = 0; i < interiorPoints.length; i += 1) {
      const point = interiorPoints[i];
      const delta = deltas[i];
      const length = Math.hypot(delta.x, delta.y);
      if (length <= 0.000001) {
        continue;
      }

      const step = Math.min(maxStep, length);
      const proposed = {
        x: point.x + (delta.x / length) * step,
        y: point.y + (delta.y / length) * step
      };
      const constrained = constrainPointToShape(shape, proposed, point, halfSize);

      point.x = constrained.x;
      point.y = constrained.y;
    }
  }
}

function relaxRectangleInteriorPoints(
  halfWidth: number,
  halfHeight: number,
  spacing: number,
  outlinePoints: BodyPoint[],
  interiorPoints: BodyPoint[],
  rng: () => number
) {
  const iterations = 8;
  const minDistance = spacing * 0.82;
  const minDistanceSq = minDistance * minDistance;
  const maxStep = spacing * 0.18;

  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const deltas = interiorPoints.map(() => ({ x: 0, y: 0 }));

    for (let i = 0; i < interiorPoints.length; i += 1) {
      accumulateSpacingDelta(interiorPoints[i], outlinePoints, minDistanceSq, deltas[i]);

      for (let j = i + 1; j < interiorPoints.length; j += 1) {
        const delta = getRepulsionDelta(interiorPoints[i], interiorPoints[j], minDistanceSq, rng);
        deltas[i].x += delta.x;
        deltas[i].y += delta.y;
        deltas[j].x -= delta.x;
        deltas[j].y -= delta.y;
      }
    }

    for (let i = 0; i < interiorPoints.length; i += 1) {
      const point = interiorPoints[i];
      const delta = deltas[i];
      const length = Math.hypot(delta.x, delta.y);
      if (length <= 0.000001) {
        continue;
      }

      const step = Math.min(maxStep, length);
      point.x = Math.max(-halfWidth, Math.min(halfWidth, point.x + (delta.x / length) * step));
      point.y = Math.max(-halfHeight, Math.min(halfHeight, point.y + (delta.y / length) * step));
    }
  }
}

function accumulateSpacingDelta(
  point: BodyPoint,
  neighbors: BodyPoint[],
  minDistanceSq: number,
  delta: { x: number; y: number }
) {
  for (const neighbor of neighbors) {
    const repulsion = getRepulsionDelta(point, neighbor, minDistanceSq, undefined);
    delta.x += repulsion.x;
    delta.y += repulsion.y;
  }
}

function getRepulsionDelta(
  point: BodyPoint,
  neighbor: BodyPoint,
  minDistanceSq: number,
  rng: (() => number) | undefined
) {
  let dx = point.x - neighbor.x;
  let dy = point.y - neighbor.y;
  let distanceSq = dx * dx + dy * dy;

  if (distanceSq < 0.000001 && rng) {
    const angle = rng() * Math.PI * 2;
    dx = Math.cos(angle) * 0.001;
    dy = Math.sin(angle) * 0.001;
    distanceSq = dx * dx + dy * dy;
  }

  if (distanceSq <= 0.000001 || distanceSq >= minDistanceSq) {
    return { x: 0, y: 0 };
  }

  const distance = Math.sqrt(distanceSq);
  const strength = (Math.sqrt(minDistanceSq) - distance) / Math.sqrt(minDistanceSq);
  return {
    x: (dx / distance) * strength,
    y: (dy / distance) * strength
  };
}

function constrainPointToShape(
  shape: SoftBodyShape,
  proposed: { x: number; y: number },
  fallback: BodyPoint,
  halfSize: number
) {
  if (shapeContainsPoint(shape, proposed.x, proposed.y, halfSize)) {
    return proposed;
  }

  let low = { x: fallback.x, y: fallback.y };
  let high = proposed;

  for (let i = 0; i < 12; i += 1) {
    const mid = {
      x: (low.x + high.x) * 0.5,
      y: (low.y + high.y) * 0.5
    };

    if (shapeContainsPoint(shape, mid.x, mid.y, halfSize)) {
      low = mid;
    } else {
      high = mid;
    }
  }

  return low;
}

function shapeContainsPoint(shape: SoftBodyShape, x: number, y: number, halfSize: number) {
  if (shape === "circle") {
    return x * x + y * y <= halfSize * halfSize;
  }

  if (shape === "triangle") {
    const vertical = (y + halfSize) / (halfSize * 2);
    const halfWidth = halfSize * Math.max(0, Math.min(1, vertical));
    return y >= -halfSize && y <= halfSize && Math.abs(x) <= halfWidth;
  }

  return Math.abs(x) <= halfSize && Math.abs(y) <= halfSize;
}

function createBondData(points: BodyPoint[], startIndex: number, spacing: number) {
  const neighbors = buildBondNeighbors(points, startIndex, spacing);
  const data = createEmptyBondData(points.length);
  const view = new DataView(data);

  for (let particleIndex = 0; particleIndex < neighbors.length; particleIndex += 1) {
    for (let slot = 0; slot < neighbors[particleIndex].length; slot += 1) {
      const neighbor = neighbors[particleIndex][slot];
      const offset = (particleIndex * BOND_SLOT_COUNT + slot) * BOND_STRIDE_BYTES;
      view.setUint32(offset, neighbor.index, true);
      view.setFloat32(offset + 4, neighbor.restLength, true);
      view.setFloat32(offset + 8, neighbor.weight, true);
    }
  }

  return data;
}

function createRopeBondData(particleCount: number, startIndex: number, restLength: number) {
  const data = createEmptyBondData(particleCount);
  const view = new DataView(data);

  for (let localIndex = 0; localIndex < particleCount; localIndex += 1) {
    let slot = 0;
    if (localIndex > 0) {
      writeBond(view, localIndex, slot, startIndex + localIndex - 1, restLength, 1.25);
      slot += 1;
    }

    if (localIndex + 1 < particleCount) {
      writeBond(view, localIndex, slot, startIndex + localIndex + 1, restLength, 1.25);
      slot += 1;
    }

    if (localIndex > 1) {
      writeBond(view, localIndex, slot, startIndex + localIndex - 2, restLength * 2, 0.36);
      slot += 1;
    }

    if (localIndex + 2 < particleCount) {
      writeBond(view, localIndex, slot, startIndex + localIndex + 2, restLength * 2, 0.36);
    }
  }

  return data;
}

function writeBond(
  view: DataView,
  particleIndex: number,
  slot: number,
  neighborIndex: number,
  restLength: number,
  weight: number
) {
  const offset = (particleIndex * BOND_SLOT_COUNT + slot) * BOND_STRIDE_BYTES;
  view.setUint32(offset, neighborIndex, true);
  view.setFloat32(offset + 4, restLength, true);
  view.setFloat32(offset + 8, weight, true);
}

function createEmptyBondData(particleCount: number) {
  const data = new ArrayBuffer(particleCount * BOND_SLOT_COUNT * BOND_STRIDE_BYTES);
  const view = new DataView(data);

  for (let slot = 0; slot < particleCount * BOND_SLOT_COUNT; slot += 1) {
    view.setUint32(slot * BOND_STRIDE_BYTES, INVALID_BOND_INDEX, true);
  }

  return data;
}

function buildBondNeighbors(points: BodyPoint[], startIndex: number, spacing: number) {
  const maxDistance = spacing * 2.15;
  const maxDistanceSq = maxDistance * maxDistance;
  const neighbors: BondNeighbor[][] = points.map(() => []);
  const candidatesByPoint: Array<Array<{ other: number; distance: number; weight: number }>> = points.map(() => []);
  const candidatePairs: Array<{ a: number; b: number; distance: number; weight: number }> = [];

  for (let a = 0; a < points.length; a += 1) {
    for (let b = a + 1; b < points.length; b += 1) {
      const dx = points[a].x - points[b].x;
      const dy = points[a].y - points[b].y;
      const distanceSq = dx * dx + dy * dy;
      if (distanceSq <= maxDistanceSq) {
        const distance = Math.sqrt(distanceSq);
        const weight = getBondWeight(points[a], points[b], distance, spacing);
        candidatePairs.push({
          a,
          b,
          distance,
          weight
        });
        candidatesByPoint[a].push({ other: b, distance, weight });
        candidatesByPoint[b].push({ other: a, distance, weight });
      }
    }
  }

  candidatePairs.sort((left, right) => left.distance - right.distance);
  for (const candidates of candidatesByPoint) {
    candidates.sort((left, right) => left.distance - right.distance);
  }

  for (const pair of candidatePairs) {
    if (neighbors[pair.a].length >= BOND_SLOT_COUNT || neighbors[pair.b].length >= BOND_SLOT_COUNT) {
      continue;
    }

    addBondNeighbor(neighbors[pair.a], startIndex + pair.b, pair.distance, pair.weight);
    addBondNeighbor(neighbors[pair.b], startIndex + pair.a, pair.distance, pair.weight);
  }

  for (let pointIndex = 0; pointIndex < points.length; pointIndex += 1) {
    if (neighbors[pointIndex].length >= BOND_SLOT_COUNT) {
      continue;
    }

    for (const candidate of candidatesByPoint[pointIndex]) {
      if (neighbors[pointIndex].length >= BOND_SLOT_COUNT) {
        break;
      }
      addBondNeighbor(neighbors[pointIndex], startIndex + candidate.other, candidate.distance, candidate.weight);
    }
  }

  return neighbors;
}

function addBondNeighbor(neighbors: BondNeighbor[], index: number, restLength: number, weight: number) {
  if (neighbors.some((neighbor) => neighbor.index === index)) {
    return;
  }

  neighbors.push({ index, restLength, weight });
}

function getBondWeight(a: BodyPoint, b: BodyPoint, distance: number, spacing: number) {
  const lengthWeight = distance > spacing * 1.45 ? 0.55 : 1;
  const boundaryWeight = a.boundary && b.boundary ? 1.25 : 1;
  return lengthWeight * boundaryWeight;
}

function pickBodyCenter(bodyId: number, size: number, worldWidth: number, worldHeight: number) {
  const margin = Math.max(32, size * 0.58);
  const controlClearance = worldWidth > 700 ? 340 : worldWidth > 560 ? 280 : 0;
  const availableWidth = Math.max(1, worldWidth - controlClearance - margin * 2);
  const availableHeight = Math.max(1, worldHeight * 0.42 - margin);

  return {
    x: controlClearance + margin + ((bodyId * 173) % availableWidth),
    y: margin + ((bodyId * 67) % availableHeight)
  };
}

function mulberry32(seed: number) {
  let state = seed;
  return () => {
    state |= 0;
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
