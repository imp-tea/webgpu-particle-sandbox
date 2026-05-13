export const WORKGROUP_SIZE = 128;
export const SCAN_BLOCK_SIZE = 256;
export const PARTICLE_STRIDE_BYTES = 32;
export const BODY_STRIDE_BYTES = 32;
export const MATERIAL_COUNT = 4;
export const MATERIAL_STRIDE_BYTES = 32;
export const SIM_PARAMS_BYTES = 96;
export const DEBUG_COUNTER_BYTES = 16;
export const GRID_CELL_SIZE = 18;
export const MAX_GRID_COLUMNS = 256;
export const MAX_GRID_ROWS = 144;
export const MAX_GRID_CELLS = MAX_GRID_COLUMNS * MAX_GRID_ROWS;
export const GRID_PARTICLE_CAPACITY = 65_536;
export const MAX_BODIES = 96;
export const MAX_SCAN_GROUPS = Math.ceil(MAX_GRID_CELLS / SCAN_BLOCK_SIZE);

export const DEFAULT_CONFIG = {
  initialParticles: 0,
  maxParticles: 50_000,
  gravityY: 280,
  damping: 0.25,
  substeps: 2,
  bondIterations: 4,
  softBodyStrength: 2_600,
  viscosity: 4,
  mouseForce: 60_000,
  particleRepulsion: 180,
  cohesion: 0.08,
  maxSpeed: 1_500
} as const;

export type SimulationSettings = {
  particleCount: number;
  gravityY: number;
  damping: number;
  substeps: number;
  bondIterations: number;
  softBodyStrength: number;
  viscosity: number;
  mouseForce: number;
  particleRepulsion: number;
  cohesion: number;
  maxSpeed: number;
};

export function defaultSettings(): SimulationSettings {
  return {
    particleCount: DEFAULT_CONFIG.initialParticles,
    gravityY: DEFAULT_CONFIG.gravityY,
    damping: DEFAULT_CONFIG.damping,
    substeps: DEFAULT_CONFIG.substeps,
    bondIterations: DEFAULT_CONFIG.bondIterations,
    softBodyStrength: DEFAULT_CONFIG.softBodyStrength,
    viscosity: DEFAULT_CONFIG.viscosity,
    mouseForce: DEFAULT_CONFIG.mouseForce,
    particleRepulsion: DEFAULT_CONFIG.particleRepulsion,
    cohesion: DEFAULT_CONFIG.cohesion,
    maxSpeed: DEFAULT_CONFIG.maxSpeed
  };
}
