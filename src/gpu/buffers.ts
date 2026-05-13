import {
  BOND_SLOT_COUNT,
  BOND_STRIDE_BYTES,
  BODY_STRIDE_BYTES,
  DEBUG_COUNTER_BYTES,
  DEFAULT_CONFIG,
  GRID_CELL_SIZE,
  GRID_PARTICLE_CAPACITY,
  MATERIAL_COUNT,
  MATERIAL_STRIDE_BYTES,
  MAX_BODIES,
  MAX_GRID_CELLS,
  MAX_GRID_COLUMNS,
  MAX_GRID_ROWS,
  MAX_SCAN_GROUPS,
  PARTICLE_STRIDE_BYTES,
  REST_SHAPE_STRIDE_BYTES,
  SCAN_BLOCK_SIZE,
  SIM_PARAMS_BYTES,
  type SimulationSettings
} from "../config";
import type { PointerState } from "../input";

export type SoftBodyShape = "square" | "circle" | "triangle";

const BODY_ID_MASK = 0x0000ffff;
const INVALID_BOND_INDEX = 0xffffffff;

export type ParticleBuffers = {
  buffers: [GPUBuffer, GPUBuffer];
  activeIndex: number;
  maxParticles: number;
  particleCount: number;
  bodyCount: number;
  clear: (device: GPUDevice, bodyBuffer: GPUBuffer, bondBuffer: GPUBuffer, restShapeBuffer: GPUBuffer) => void;
  addSoftBody: (
    device: GPUDevice,
    bodyBuffer: GPUBuffer,
    bondBuffer: GPUBuffer,
    restShapeBuffer: GPUBuffer,
    shape: SoftBodyShape,
    size: number,
    particleRadius: number,
    worldWidth: number,
    worldHeight: number
  ) => SoftBodyAddResult;
  swap: () => void;
};

export type SoftBodyAddResult =
  | {
      added: true;
      particleCount: number;
      bodyCount: number;
      addedParticles: number;
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
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
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
    addSoftBody(device, bodyBuffer, bondBuffer, restShapeBuffer, shape, size, particleRadius, worldWidth, worldHeight) {
      if (this.bodyCount >= MAX_BODIES) {
        return { added: false, reason: `Body limit reached (${MAX_BODIES}).` };
      }

      const body = createSoftBodyData({
        shape,
        size,
        particleRadius,
        bodyId: this.bodyCount,
        startIndex: this.particleCount,
        worldWidth,
        worldHeight
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
        addedParticles: body.particleCount
      };
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
  deltaTime: number
) {
  const data = new ArrayBuffer(SIM_PARAMS_BYTES);
  const view = new DataView(data);

  const signedMouseForce = pointer.active ? pointer.forceSign * settings.mouseForce : 0;
  const grid = getGridDimensions(worldWidth, worldHeight);

  view.setFloat32(0, 0, true);
  view.setFloat32(4, settings.gravityY, true);
  view.setFloat32(8, worldWidth, true);
  view.setFloat32(12, worldHeight, true);
  view.setFloat32(16, pointer.x, true);
  view.setFloat32(20, pointer.y, true);
  view.setFloat32(24, signedMouseForce, true);
  view.setFloat32(28, deltaTime, true);
  view.setFloat32(32, settings.damping, true);
  view.setUint32(36, settings.particleCount, true);
  view.setFloat32(40, 0, true);
  view.setFloat32(44, settings.maxSpeed, true);
  view.setFloat32(48, grid.cellSize, true);
  view.setUint32(52, grid.columns, true);
  view.setUint32(56, grid.rows, true);
  view.setUint32(60, GRID_PARTICLE_CAPACITY, true);
  view.setFloat32(64, 0, true);
  view.setFloat32(68, settings.softBodyStrength, true);
  view.setFloat32(72, settings.viscosity, true);
  view.setUint32(76, settings.contactIterations, true);
  view.setUint32(80, 0, true);
  view.setUint32(84, settings.bondIterations, true);
  view.setFloat32(88, 1, true);
  view.setFloat32(92, 1, true);

  device.queue.writeBuffer(uniformBuffer, 0, data);
}

type SoftBodyDataOptions = {
  shape: SoftBodyShape;
  size: number;
  particleRadius: number;
  bodyId: number;
  startIndex: number;
  worldWidth: number;
  worldHeight: number;
};

function createSoftBodyData(options: SoftBodyDataOptions) {
  const size = Math.max(40, Math.min(260, options.size));
  const targetAcross = Math.max(6, Math.min(34, Math.round(size / 8)));
  const spacing = size / Math.max(1, targetAcross - 1);
  const rng = mulberry32(0x5eed1234 + options.bodyId * 0x9e3779b9);
  const points = createShapePoints(options.shape, size, spacing, rng);
  const particleCount = points.length;
  const particleData = new ArrayBuffer(particleCount * PARTICLE_STRIDE_BYTES);
  const bondData = createBondData(points, options.startIndex, spacing);
  const restShapeData = createRestShapeData(points);
  const bodyData = new ArrayBuffer(BODY_STRIDE_BYTES);
  const particleView = new DataView(particleData);
  const bodyView = new DataView(bodyData);
  const center = pickBodyCenter(options.bodyId, size, options.worldWidth, options.worldHeight);
  const radius = Math.max(1, Math.min(8, options.particleRadius));
  const materialId = options.bodyId % MATERIAL_COUNT;
  const bodyId = options.bodyId & BODY_ID_MASK;

  for (let localIndex = 0; localIndex < points.length; localIndex += 1) {
    const point = points[localIndex];
    const offset = localIndex * PARTICLE_STRIDE_BYTES;
    const angle = rng() * Math.PI * 2;
    const speed = rng() * 8;

    particleView.setFloat32(offset, center.x + point.x, true);
    particleView.setFloat32(offset + 4, center.y + point.y, true);
    particleView.setFloat32(offset + 8, Math.cos(angle) * speed, true);
    particleView.setFloat32(offset + 12, Math.sin(angle) * speed, true);
    particleView.setUint32(offset + 16, materialId, true);
    particleView.setUint32(offset + 20, bodyId, true);
    particleView.setFloat32(offset + 24, radius, true);
    particleView.setFloat32(offset + 28, radius * radius, true);
  }

  bodyView.setUint32(0, options.startIndex, true);
  bodyView.setUint32(4, particleCount, true);

  return { particleData, bodyData, bondData, restShapeData, particleCount };
}

type BodyPoint = {
  x: number;
  y: number;
  boundary: boolean;
};

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

function createShapePoints(shape: SoftBodyShape, size: number, spacing: number, rng: () => number): BodyPoint[] {
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
