override WORKGROUP_SIZE: u32 = 128u;

struct SimParams {
  gravity: vec2<f32>,
  worldSize: vec2<f32>,
  mousePosition: vec2<f32>,
  mouseForce: f32,
  deltaTime: f32,
  damping: f32,
  particleCount: u32,
  particleRepulsion: f32,
  maxSpeed: f32,
  gridCellSize: f32,
  gridColumns: u32,
  gridRows: u32,
  gridParticleCapacity: u32,
  cohesion: f32,
  padding0: f32,
  padding1: f32,
  padding2: f32,
};

@group(0) @binding(0) var<storage, read_write> cellCounts: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> cellStarts: array<u32>;
@group(0) @binding(2) var<storage, read_write> cellWriteOffsets: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> params: SimParams;

// Clears per-cell histogram and scatter state before the grid is rebuilt.
@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let cellIndex = globalId.x;
  let activeCellCount = params.gridColumns * params.gridRows;

  if (cellIndex >= activeCellCount) {
    return;
  }

  atomicStore(&cellCounts[cellIndex], 0u);
  cellStarts[cellIndex] = 0u;
  atomicStore(&cellWriteOffsets[cellIndex], 0u);
}
