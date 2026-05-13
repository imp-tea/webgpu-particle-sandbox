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
  wallBounce: f32,
  maxSpeed: f32,
  gridCellSize: f32,
  gridColumns: u32,
  gridRows: u32,
  gridParticleCapacity: u32,
  padding1: f32,
  softBodyStrength: f32,
  viscosity: f32,
  contactIterations: u32,
  selectedParticleIndex: u32,
  bondIterations: u32,
  jointCount: u32,
  jointIterations: u32,
  solverPhase: u32,
  padding2: vec3<u32>,
};

struct BodyParams {
  startIndex: u32,
  particleCount: u32,
  padding0: vec2<u32>,
  softBodyStrength: f32,
  viscosity: f32,
  friction: f32,
  padding1: f32,
};

struct Joint {
  bodyA: u32,
  bodyB: u32,
  jointType: u32,
  enabled: u32,
  localAnchorA: vec2<f32>,
  localAnchorB: vec2<f32>,
  restLength: f32,
  stiffness: f32,
  influenceRadius: f32,
  padding: f32,
};

@group(0) @binding(0) var<storage, read> particlesIn: array<Particle>;
@group(0) @binding(1) var<storage, read_write> particlesOut: array<Particle>;
@group(0) @binding(2) var<uniform> params: SimParams;
@group(0) @binding(3) var<storage, read_write> cellCounts: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> cellStarts: array<u32>;
@group(0) @binding(5) var<storage, read> pairValues: array<u32>;
@group(0) @binding(6) var<storage, read> bodies: array<BodyParams>;
@group(0) @binding(7) var<storage, read> joints: array<Joint>;

const BODY_ID_MASK: u32 = 0x0000ffffu;
const MAX_JOINTS: u32 = 64u;
const CONTACT_SLOP: f32 = 0.015;
const CONTACT_STIFFNESS: f32 = 0.72;
const CONTACT_VELOCITY_FEEDBACK: f32 = 0.10;
const CONTACT_NORMAL_DAMPING: f32 = 0.55;

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
  var normalSum = vec2<f32>(0.0);
  var velocityDelta = vec2<f32>(0.0);
  var frictionSum = 0.0;
  var contactCount = 0u;
  let bodyFriction = bodies[particleBodyId(particle)].friction;
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
            let normal = normalize(nextCorrection);
            let relativeVelocity = particle.velocity - other.velocity;
            let approachSpeed = min(dot(relativeVelocity, normal), 0.0);
            let otherFriction = bodies[particleBodyId(other)].friction;
            correction += nextCorrection;
            normalSum += normal;
            velocityDelta += -normal * approachSpeed * CONTACT_NORMAL_DAMPING;
            frictionSum += sqrt(max(0.0, bodyFriction) * max(0.0, otherFriction));
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
      particle.velocity += limitedCorrection * CONTACT_STIFFNESS * CONTACT_VELOCITY_FEEDBACK / params.deltaTime;
      particle.velocity += velocityDelta / f32(contactCount);
      let normalLengthSq = dot(normalSum, normalSum);
      if (normalLengthSq > 0.0001) {
        let contactNormal = normalSum / sqrt(normalLengthSq);
        let normalVelocity = contactNormal * dot(particle.velocity, contactNormal);
        let tangentVelocity = particle.velocity - normalVelocity;
        let contactFriction = clamp(frictionSum / f32(contactCount), 0.0, 0.95);
        particle.velocity = normalVelocity + tangentVelocity * (1.0 - contactFriction);
      }
      particle.velocity = clampVelocity(particle.velocity);
    }
  }

  particlesOut[index] = particle;
}

fn contactCorrection(particle: Particle, other: Particle) -> vec2<f32> {
  if (!particleActive(other)) {
    return vec2<f32>(0.0);
  }

  let bodyId = particleBodyId(particle);
  let otherBodyId = particleBodyId(other);
  if (bodyId == otherBodyId || bodiesAreJointed(bodyId, otherBodyId)) {
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

fn bodiesAreJointed(bodyA: u32, bodyB: u32) -> bool {
  let jointCount = min(params.jointCount, MAX_JOINTS);
  for (var jointIndex = 0u; jointIndex < jointCount; jointIndex += 1u) {
    let joint = joints[jointIndex];
    if (joint.enabled == 0u) {
      continue;
    }

    let forwardMatch = joint.bodyA == bodyA && joint.bodyB == bodyB;
    let reverseMatch = joint.bodyA == bodyB && joint.bodyB == bodyA;
    if (forwardMatch || reverseMatch) {
      return true;
    }
  }

  return false;
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
