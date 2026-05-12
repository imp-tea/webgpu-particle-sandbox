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
  padding0: f32,
  padding1: f32,
  padding2: f32,
};

@group(0) @binding(0) var<storage, read> cellGroupSums: array<u32>;
@group(0) @binding(1) var<storage, read_write> cellGroupOffsets: array<u32>;
@group(0) @binding(2) var<uniform> params: SimParams;

var<workgroup> scanScratch: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) localId: vec3<u32>) {
  let groupIndex = localId.x;
  let activeCellCount = params.gridColumns * params.gridRows;
  let activeGroupCount = (activeCellCount + SCAN_BLOCK_SIZE - 1u) / SCAN_BLOCK_SIZE;

  var value = 0u;
  if (groupIndex < activeGroupCount) {
    value = cellGroupSums[groupIndex];
  }

  scanScratch[groupIndex] = value;
  workgroupBarrier();

  for (var offset = 1u; offset < SCAN_BLOCK_SIZE; offset = offset * 2u) {
    var addend = 0u;
    if (groupIndex >= offset) {
      addend = scanScratch[groupIndex - offset];
    }
    workgroupBarrier();
    scanScratch[groupIndex] = scanScratch[groupIndex] + addend;
    workgroupBarrier();
  }

  if (groupIndex < activeGroupCount) {
    cellGroupOffsets[groupIndex] = scanScratch[groupIndex] - value;
  }
}
