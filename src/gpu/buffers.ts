import {
  DEFAULT_CONFIG,
  GRID_CELL_SIZE,
  GRID_PARTICLE_CAPACITY,
  MATERIAL_COUNT,
  MATERIAL_STRIDE_BYTES,
  MAX_GRID_CELLS,
  MAX_GRID_COLUMNS,
  MAX_GRID_ROWS,
  MAX_SCAN_GROUPS,
  PARTICLE_STRIDE_BYTES,
  SCAN_BLOCK_SIZE,
  SIM_PARAMS_BYTES,
  type SimulationSettings
} from "../config";
import type { PointerState } from "../input";

export type ParticleBuffers = {
  buffers: [GPUBuffer, GPUBuffer];
  activeIndex: number;
  maxParticles: number;
  reset: (device: GPUDevice, count: number, worldWidth: number, worldHeight: number) => void;
  swap: () => void;
};

export type GridBuffers = {
  pairValues: GPUBuffer;
  cellStarts: GPUBuffer;
  cellCounts: GPUBuffer;
  cellWriteOffsets: GPUBuffer;
  cellGroupSums: GPUBuffer;
  cellGroupOffsets: GPUBuffer;
  maxCells: number;
  particleCapacity: number;
  scanBlockSize: number;
  scanGroupCount: number;
};

export type MaterialPreset = "mixed" | "granular" | "liquid" | "separated";

type MaterialRow = {
  color: [number, number, number, number];
  dynamics: [number, number, number, number];
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
    reset(device, count, worldWidth, worldHeight) {
      const particleData = createInitialParticleData(count, worldWidth, worldHeight);
      device.queue.writeBuffer(buffers[0], 0, particleData);
      device.queue.writeBuffer(buffers[1], 0, particleData);
      this.activeIndex = 0;
    },
    swap() {
      this.activeIndex = 1 - this.activeIndex;
    }
  };
}

export function createUniformBuffer(device: GPUDevice): GPUBuffer {
  return device.createBuffer({
    label: "Simulation parameters",
    size: SIM_PARAMS_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  });
}

export function createMaterialBuffer(device: GPUDevice, preset: MaterialPreset = "mixed"): GPUBuffer {
  const materialData = createMaterialData(preset);
  const materialBuffer = device.createBuffer({
    label: "Material parameters",
    size: materialData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  });

  device.queue.writeBuffer(materialBuffer, 0, materialData);
  return materialBuffer;
}

export function writeMaterialParams(device: GPUDevice, materialBuffer: GPUBuffer, preset: MaterialPreset) {
  device.queue.writeBuffer(materialBuffer, 0, createMaterialData(preset));
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
    maxCells: MAX_GRID_CELLS,
    particleCapacity: GRID_PARTICLE_CAPACITY,
    scanBlockSize: SCAN_BLOCK_SIZE,
    scanGroupCount: MAX_SCAN_GROUPS
  };
}

function createMaterialData(preset: MaterialPreset): ArrayBuffer {
  const data = new ArrayBuffer(MATERIAL_COUNT * MATERIAL_STRIDE_BYTES);
  const view = new DataView(data);
  const materials = MATERIAL_PRESETS[preset];

  for (let i = 0; i < MATERIAL_COUNT; i += 1) {
    const offset = i * MATERIAL_STRIDE_BYTES;
    const material = materials[i];

    for (let channel = 0; channel < 4; channel += 1) {
      view.setFloat32(offset + channel * Float32Array.BYTES_PER_ELEMENT, material.color[channel], true);
      view.setFloat32(offset + 16 + channel * Float32Array.BYTES_PER_ELEMENT, material.dynamics[channel], true);
    }
  }

  return data;
}

const MATERIAL_PRESETS: Record<MaterialPreset, MaterialRow[]> = {
  mixed: [
    {
      color: [0.28, 0.78, 0.95, 1.0],
      dynamics: [0.92, 1.25, 0.72, 0.24]
    },
    {
      color: [0.98, 0.70, 0.25, 1.0],
      dynamics: [1.18, 0.58, 0.48, 0.18]
    },
    {
      color: [0.54, 0.92, 0.48, 1.0],
      dynamics: [0.76, 1.55, 0.86, 0.28]
    },
    {
      color: [0.92, 0.46, 0.73, 1.0],
      dynamics: [1.04, 1.0, 0.62, 0.22]
    }
  ],
  granular: [
    {
      color: [0.32, 0.82, 0.95, 1.0],
      dynamics: [1.38, 0.12, 0.08, 0.04]
    },
    {
      color: [0.98, 0.76, 0.30, 1.0],
      dynamics: [1.5, 0.08, 0.06, 0.03]
    },
    {
      color: [0.58, 0.92, 0.52, 1.0],
      dynamics: [1.28, 0.16, 0.1, 0.04]
    },
    {
      color: [0.94, 0.50, 0.76, 1.0],
      dynamics: [1.44, 0.1, 0.08, 0.03]
    }
  ],
  liquid: [
    {
      color: [0.18, 0.74, 1.0, 1.0],
      dynamics: [0.54, 2.25, 1.2, 0.52]
    },
    {
      color: [1.0, 0.64, 0.20, 1.0],
      dynamics: [0.62, 1.85, 1.05, 0.42]
    },
    {
      color: [0.46, 0.95, 0.46, 1.0],
      dynamics: [0.48, 2.45, 1.32, 0.58]
    },
    {
      color: [0.98, 0.38, 0.76, 1.0],
      dynamics: [0.58, 2.1, 1.12, 0.48]
    }
  ],
  separated: [
    {
      color: [0.20, 0.76, 1.0, 1.0],
      dynamics: [0.72, 2.4, 1.45, -0.45]
    },
    {
      color: [1.0, 0.68, 0.20, 1.0],
      dynamics: [0.82, 1.95, 1.22, -0.38]
    },
    {
      color: [0.50, 0.96, 0.42, 1.0],
      dynamics: [0.68, 2.55, 1.5, -0.5]
    },
    {
      color: [0.96, 0.42, 0.78, 1.0],
      dynamics: [0.78, 2.25, 1.34, -0.42]
    }
  ]
};

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
  view.setFloat32(40, settings.particleRepulsion, true);
  view.setFloat32(44, settings.maxSpeed, true);
  view.setFloat32(48, grid.cellSize, true);
  view.setUint32(52, grid.columns, true);
  view.setUint32(56, grid.rows, true);
  view.setUint32(60, GRID_PARTICLE_CAPACITY, true);
  view.setFloat32(64, settings.cohesion, true);
  view.setFloat32(68, 0, true);
  view.setFloat32(72, 0, true);
  view.setFloat32(76, 0, true);

  device.queue.writeBuffer(uniformBuffer, 0, data);
}

function createInitialParticleData(count: number, worldWidth: number, worldHeight: number): ArrayBuffer {
  const data = new ArrayBuffer(count * PARTICLE_STRIDE_BYTES);
  const view = new DataView(data);
  const rng = mulberry32(0x5eed1234);
  const columns = Math.ceil(Math.sqrt(count * (worldWidth / Math.max(1, worldHeight))));
  const rows = Math.ceil(count / columns);
  const spawnWidth = worldWidth * 0.72;
  const spawnHeight = worldHeight * 0.58;
  const startX = (worldWidth - spawnWidth) * 0.5;
  const startY = (worldHeight - spawnHeight) * 0.28;
  const cellWidth = spawnWidth / Math.max(1, columns);
  const cellHeight = spawnHeight / Math.max(1, rows);
  const radius = Math.max(1.35, Math.min(3.25, Math.min(cellWidth, cellHeight) * 0.32));

  for (let i = 0; i < count; i += 1) {
    const column = i % columns;
    const row = Math.floor(i / columns);
    const jitterX = (rng() - 0.5) * cellWidth * 0.55;
    const jitterY = (rng() - 0.5) * cellHeight * 0.55;
    const x = startX + (column + 0.5) * cellWidth + jitterX;
    const y = startY + (row + 0.5) * cellHeight + jitterY;
    const angle = rng() * Math.PI * 2;
    const speed = rng() * 18;
    const materialId = i % 4;
    const offset = i * PARTICLE_STRIDE_BYTES;

    view.setFloat32(offset, x, true);
    view.setFloat32(offset + 4, y, true);
    view.setFloat32(offset + 8, Math.cos(angle) * speed, true);
    view.setFloat32(offset + 12, Math.sin(angle) * speed, true);
    view.setUint32(offset + 16, materialId, true);
    view.setUint32(offset + 20, 0, true);
    view.setFloat32(offset + 24, radius, true);
    view.setFloat32(offset + 28, radius * radius, true);
  }

  return data;
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
