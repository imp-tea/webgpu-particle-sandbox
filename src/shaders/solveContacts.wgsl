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
  padding0: f32,
  maxSpeed: f32,
  gridCellSize: f32,
  gridColumns: u32,
  gridRows: u32,
  gridParticleCapacity: u32,
  padding1: f32,
  softBodyStrength: f32,
  viscosity: f32,
  contactIterations: u32,
  padding2: u32,
  bondIterations: u32,
  restCellSize: vec2<f32>,
};

@group(0) @binding(0) var<storage, read> particlesIn: array<Particle>;
@group(0) @binding(1) var<storage, read_write> particlesOut: array<Particle>;
@group(0) @binding(2) var<uniform> params: SimParams;
@group(0) @binding(3) var<storage, read_write> cellCounts: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> cellStarts: array<u32>;
@group(0) @binding(5) var<storage, read> pairValues: array<u32>;

const BODY_ID_MASK: u32 = 0x0000ffffu;
const CONTACT_SLOP: f32 = 0.015;
const CONTACT_STIFFNESS: f32 = 0.88;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let index = globalId.x;
  if (index >= params.particleCount) {
    return;
  }

  var particle = particlesIn[index];
  if (!particleActive(particle)) {
    particlesOut[index] = particle;
    return;
  }

  var correction = vec2<f32>(0.0);
  var contactCount = 0u;
  let baseCell = vec2<i32>(particleCell(particle.position));

  for (var y = -1; y <= 1; y += 1) {
    for (var x = -1; x <= 1; x += 1) {
      let cell = baseCell + vec2<i32>(x, y);
      if (!cellInBounds(cell)) {
        continue;
      }

      let cellIndex = u32(cell.y) * params.gridColumns + u32(cell.x);
      let start = cellStarts[cellIndex];
      let count = atomicLoad(&cellCounts[cellIndex]);
      if (count == 0u) {
        continue;
      }

      for (var slot = 0u; slot < count; slot += 1u) {
        let otherIndex = pairValues[start + slot];
        if (otherIndex != index && otherIndex < params.particleCount) {
          let other = particlesIn[otherIndex];
          let nextCorrection = contactCorrection(particle, other);
          if (dot(nextCorrection, nextCorrection) > 0.0) {
            correction += nextCorrection;
            contactCount += 1u;
          }
        }
      }
    }
  }

  if (contactCount > 0u) {
    let averagedCorrection = correction / f32(contactCount);
    let maxCorrection = particle.radius * 0.85;
    let correctionLength = length(averagedCorrection);
    var limitedCorrection = averagedCorrection;
    if (correctionLength > maxCorrection && correctionLength > 0.0001) {
      limitedCorrection = averagedCorrection / correctionLength * maxCorrection;
    }
    let position = particle.position + limitedCorrection * CONTACT_STIFFNESS;
    particle.position = clampToWorld(position, particle.radius);

    if (params.deltaTime > 0.000001) {
      particle.velocity += limitedCorrection * CONTACT_STIFFNESS / params.deltaTime;
      particle.velocity = clampVelocity(particle.velocity);
    }
  }

  particlesOut[index] = particle;
}

fn contactCorrection(particle: Particle, other: Particle) -> vec2<f32> {
  if (!particleActive(other) || particleBodyId(particle) == particleBodyId(other)) {
    return vec2<f32>(0.0);
  }

  let delta = other.position - particle.position;
  let distanceSq = dot(delta, delta);
  if (distanceSq < 0.0001) {
    return vec2<f32>(0.0);
  }

  let minDistance = particle.radius + other.radius;
  let distance = sqrt(distanceSq);
  let penetration = minDistance - distance - CONTACT_SLOP;
  if (penetration <= 0.0) {
    return vec2<f32>(0.0);
  }

  let normal = delta / distance;
  let invMass = 1.0 / max(particle.mass, 0.001);
  let otherInvMass = 1.0 / max(other.mass, 0.001);
  let invMassSum = invMass + otherInvMass;
  if (invMassSum <= 0.0) {
    return vec2<f32>(0.0);
  }

  return -normal * penetration * (invMass / invMassSum);
}

fn clampToWorld(position: vec2<f32>, radius: f32) -> vec2<f32> {
  return clamp(position, vec2<f32>(radius), params.worldSize - vec2<f32>(radius));
}

fn clampVelocity(velocity: vec2<f32>) -> vec2<f32> {
  let speedSq = dot(velocity, velocity);
  let maxSpeedSq = params.maxSpeed * params.maxSpeed;
  if (speedSq > maxSpeedSq) {
    return normalize(velocity) * params.maxSpeed;
  }

  return velocity;
}

fn particleActive(particle: Particle) -> bool {
  return particle.radius > 0.0;
}

fn particleBodyId(particle: Particle) -> u32 {
  return particle.flags & BODY_ID_MASK;
}

fn particleCell(position: vec2<f32>) -> vec2<u32> {
  let raw = vec2<i32>(floor(position / params.gridCellSize));
  let clamped = clamp(raw, vec2<i32>(0), vec2<i32>(i32(params.gridColumns) - 1, i32(params.gridRows) - 1));
  return vec2<u32>(clamped);
}

fn cellInBounds(cell: vec2<i32>) -> bool {
  return cell.x >= 0 && cell.y >= 0 && cell.x < i32(params.gridColumns) && cell.y < i32(params.gridRows);
}
