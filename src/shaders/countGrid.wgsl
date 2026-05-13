override WORKGROUP_SIZE: u32 = 128u;

struct Particle {
  position: vec2<f32>,
  velocity: vec2<f32>,
  materialId: u32,
  flags: u32,
  radius: f32,
  mass: f32,
};

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

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> cellCounts: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> params: SimParams;
@group(0) @binding(3) var<storage, read_write> debugCounters: array<atomic<u32>>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let particleIndex = globalId.x;
  if (particleIndex >= params.particleCount) {
    return;
  }

  let particle = particles[particleIndex];
  if (particle.radius <= 0.0) {
    return;
  }

  let cell = particleCell(particle.position);
  let cellIndex = cell.y * params.gridColumns + cell.x;
  let occupancy = atomicAdd(&cellCounts[cellIndex], 1u) + 1u;
  atomicMax(&debugCounters[0], occupancy);
}

fn particleCell(position: vec2<f32>) -> vec2<u32> {
  let raw = vec2<i32>(floor(position / params.gridCellSize));
  let clamped = clamp(raw, vec2<i32>(0), vec2<i32>(i32(params.gridColumns) - 1, i32(params.gridRows) - 1));
  return vec2<u32>(clamped);
}
