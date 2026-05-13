const SCAN_BLOCK_SIZE: u32 = 256u;

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
  softBodyStrength: f32,
  viscosity: f32,
  restColumns: u32,
  restRows: u32,
  bondIterations: u32,
  restCellSize: vec2<f32>,
};

@group(0) @binding(0) var<storage, read_write> cellCounts: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> cellStarts: array<u32>;
@group(0) @binding(2) var<storage, read_write> cellGroupSums: array<u32>;
@group(0) @binding(3) var<uniform> params: SimParams;

var<workgroup> scanScratch: array<u32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) localId: vec3<u32>,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  let localIndex = localId.x;
  let cellIndex = workgroupId.x * SCAN_BLOCK_SIZE + localIndex;
  let activeCellCount = params.gridColumns * params.gridRows;

  var value = 0u;
  if (cellIndex < activeCellCount) {
    value = atomicLoad(&cellCounts[cellIndex]);
  }

  scanScratch[localIndex] = value;
  workgroupBarrier();

  for (var offset = 1u; offset < SCAN_BLOCK_SIZE; offset = offset * 2u) {
    var addend = 0u;
    if (localIndex >= offset) {
      addend = scanScratch[localIndex - offset];
    }
    workgroupBarrier();
    scanScratch[localIndex] = scanScratch[localIndex] + addend;
    workgroupBarrier();
  }

  if (cellIndex < activeCellCount) {
    cellStarts[cellIndex] = scanScratch[localIndex] - value;
  }

  if (localIndex == SCAN_BLOCK_SIZE - 1u) {
    cellGroupSums[workgroupId.x] = scanScratch[localIndex];
  }
}
