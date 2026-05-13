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
  restRows: u32,
  bondIterations: u32,
  restCellSize: vec2<f32>,
};

struct BodyParams {
  startIndex: u32,
  particleCount: u32,
  padding0: vec2<u32>,
  padding1: vec4<f32>,
};

struct Bond {
  neighborIndex: u32,
  restLength: f32,
  weight: f32,
  padding: f32,
};

struct RestShape {
  localPosition: vec2<f32>,
  weight: f32,
  padding: f32,
};

@group(0) @binding(0) var<storage, read> particlesIn: array<Particle>;
@group(0) @binding(1) var<storage, read_write> particlesOut: array<Particle>;
@group(0) @binding(2) var<uniform> params: SimParams;
@group(0) @binding(3) var<storage, read> bodies: array<BodyParams>;
@group(0) @binding(4) var<storage, read> bonds: array<Bond>;
@group(0) @binding(5) var<storage, read> restShapes: array<RestShape>;

const BODY_ID_MASK: u32 = 0x0000ffffu;
const BOND_SLOT_COUNT: u32 = 8u;
const INVALID_BOND_INDEX: u32 = 0xffffffffu;
const SHAPE_MATCH_SAMPLE_COUNT: u32 = 24u;

// Integrates particles from a source state buffer into a destination state buffer.
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

  var acceleration = params.gravity;

  if (params.mouseForce != 0.0) {
    let toMouse = params.mousePosition - particle.position;
    let distanceSq = dot(toMouse, toMouse);
    if (distanceSq > 1.0) {
      let distance = sqrt(distanceSq);
      acceleration += normalize(toMouse) * params.mouseForce / max(distance, 40.0);
    }
  }

  var velocity = particle.velocity + acceleration * params.deltaTime;
  velocity *= max(0.0, 1.0 - params.damping * params.deltaTime);
  velocity = dampSoftBodyVelocity(index, velocity);

  let speedSq = dot(velocity, velocity);
  let maxSpeedSq = params.maxSpeed * params.maxSpeed;
  if (speedSq > maxSpeedSq) {
    velocity = normalize(velocity) * params.maxSpeed;
  }

  var position = particle.position + velocity * params.deltaTime;
  position = relaxSoftBodyPosition(index, position);
  position = applyShapeMemory(index, position);

  if (params.deltaTime > 0.000001) {
    velocity = (position - particle.position) / params.deltaTime;
  }

  let correctedSpeedSq = dot(velocity, velocity);
  if (correctedSpeedSq > maxSpeedSq) {
    velocity = normalize(velocity) * params.maxSpeed;
    position = particle.position + velocity * params.deltaTime;
  }

  let bounce = 0.72;
  let r = particle.radius;

  if (position.x < r) {
    position.x = r;
    velocity.x = abs(velocity.x) * bounce;
  }
  if (position.x > params.worldSize.x - r) {
    position.x = params.worldSize.x - r;
    velocity.x = -abs(velocity.x) * bounce;
  }
  if (position.y < r) {
    position.y = r;
    velocity.y = abs(velocity.y) * bounce;
  }
  if (position.y > params.worldSize.y - r) {
    position.y = params.worldSize.y - r;
    velocity.y = -abs(velocity.y) * bounce;
  }

  particle.position = position;
  particle.velocity = velocity;
  particlesOut[index] = particle;
}

fn relaxSoftBodyPosition(index: u32, predictedPosition: vec2<f32>) -> vec2<f32> {
  if (params.softBodyStrength <= 0.0 || params.bondIterations == 0u) {
    return predictedPosition;
  }

  let bodyId = particleBodyId(particlesIn[index]);
  let body = bodies[bodyId];
  if (body.particleCount == 0u || index < body.startIndex) {
    return predictedPosition;
  }

  let localIndex = index - body.startIndex;
  if (localIndex >= body.particleCount) {
    return predictedPosition;
  }

  let iterations = min(params.bondIterations, 8u);
  let targetStiffness = clamp(params.softBodyStrength / 6000.0, 0.0, 1.0);
  let iterationStiffness = 1.0 - pow(1.0 - targetStiffness, 1.0 / f32(iterations));
  var position = predictedPosition;

  for (var iteration = 0u; iteration < iterations; iteration += 1u) {
    for (var slot = 0u; slot < BOND_SLOT_COUNT; slot += 1u) {
      let bond = bonds[index * BOND_SLOT_COUNT + slot];
      position = relaxBond(position, bodyId, bond, iterationStiffness);
    }
  }

  return position;
}

fn applyShapeMemory(index: u32, predictedPosition: vec2<f32>) -> vec2<f32> {
  if (params.deltaTime <= 0.000001) {
    return predictedPosition;
  }

  let bodyId = particleBodyId(particlesIn[index]);
  let body = bodies[bodyId];
  if (body.particleCount < 2u || index < body.startIndex) {
    return predictedPosition;
  }

  let localIndex = index - body.startIndex;
  if (localIndex >= body.particleCount) {
    return predictedPosition;
  }

  let restShape = restShapes[index];
  if (restShape.weight <= 0.0) {
    return predictedPosition;
  }

  let sampleCount = min(body.particleCount, SHAPE_MATCH_SAMPLE_COUNT);
  var restCenter = vec2<f32>(0.0);
  var currentCenter = vec2<f32>(0.0);
  var validSamples = 0.0;

  for (var sample = 0u; sample < sampleCount; sample += 1u) {
    let sampleLocalIndex = min(body.particleCount - 1u, (sample * body.particleCount) / sampleCount);
    let sampleIndex = body.startIndex + sampleLocalIndex;
    let particle = particlesIn[sampleIndex];
    if (particleActive(particle) && particleBodyId(particle) == bodyId) {
      restCenter += restShapes[sampleIndex].localPosition;
      currentCenter += particle.position + particle.velocity * params.deltaTime;
      validSamples += 1.0;
    }
  }

  if (validSamples <= 1.0) {
    return predictedPosition;
  }

  restCenter /= validSamples;
  currentCenter /= validSamples;

  var cosineSum = 0.0;
  var sineSum = 0.0;

  for (var sample = 0u; sample < sampleCount; sample += 1u) {
    let sampleLocalIndex = min(body.particleCount - 1u, (sample * body.particleCount) / sampleCount);
    let sampleIndex = body.startIndex + sampleLocalIndex;
    let particle = particlesIn[sampleIndex];
    if (particleActive(particle) && particleBodyId(particle) == bodyId) {
      let restDelta = restShapes[sampleIndex].localPosition - restCenter;
      let currentDelta = particle.position + particle.velocity * params.deltaTime - currentCenter;
      cosineSum += dot(restDelta, currentDelta);
      sineSum += restDelta.x * currentDelta.y - restDelta.y * currentDelta.x;
    }
  }

  let rotationLength = sqrt(cosineSum * cosineSum + sineSum * sineSum);
  if (rotationLength <= 0.000001) {
    return predictedPosition;
  }

  let rotation = vec2<f32>(cosineSum, sineSum) / rotationLength;
  let localTarget = restShape.localPosition - restCenter;
  let rotatedTarget = vec2<f32>(
    rotation.x * localTarget.x - rotation.y * localTarget.y,
    rotation.y * localTarget.x + rotation.x * localTarget.y
  );
  let targetPosition = currentCenter + rotatedTarget;
  let frameStrength = 0.18;
  let stepStrength = 1.0 - pow(1.0 - frameStrength, clamp(params.deltaTime * 60.0, 0.0, 1.0));
  let weightedStrength = clamp(stepStrength * restShape.weight, 0.0, 0.35);

  return predictedPosition + (targetPosition - predictedPosition) * weightedStrength;
}

struct VelocitySample {
  delta: vec2<f32>,
  weight: f32,
};

fn dampSoftBodyVelocity(index: u32, velocity: vec2<f32>) -> vec2<f32> {
  if (params.viscosity <= 0.0 || params.deltaTime <= 0.000001) {
    return velocity;
  }

  let bodyId = particleBodyId(particlesIn[index]);
  let body = bodies[bodyId];
  if (body.particleCount == 0u || index < body.startIndex) {
    return velocity;
  }

  let localIndex = index - body.startIndex;
  if (localIndex >= body.particleCount) {
    return velocity;
  }

  var delta = vec2<f32>(0.0);
  var weightSum = 0.0;

  for (var slot = 0u; slot < BOND_SLOT_COUNT; slot += 1u) {
    let bond = bonds[index * BOND_SLOT_COUNT + slot];
    let sample = sampleNeighborVelocity(bodyId, bond, velocity);
    delta += sample.delta;
    weightSum += sample.weight;
  }

  if (weightSum <= 0.0) {
    return velocity;
  }

  let blend = clamp(params.viscosity * params.deltaTime * 0.18, 0.0, 0.85);
  return velocity + delta / weightSum * blend;
}

fn sampleNeighborVelocity(bodyId: u32, bond: Bond, velocity: vec2<f32>) -> VelocitySample {
  if (bond.neighborIndex == INVALID_BOND_INDEX || bond.neighborIndex >= params.particleCount || bond.weight <= 0.0) {
    return VelocitySample(vec2<f32>(0.0), 0.0);
  }

  let other = particlesIn[bond.neighborIndex];
  if (!particleActive(other) || particleBodyId(other) != bodyId) {
    return VelocitySample(vec2<f32>(0.0), 0.0);
  }

  return VelocitySample((other.velocity - velocity) * bond.weight, bond.weight);
}

fn relaxBond(position: vec2<f32>, bodyId: u32, bond: Bond, iterationStiffness: f32) -> vec2<f32> {
  if (
    bond.neighborIndex == INVALID_BOND_INDEX ||
    bond.neighborIndex >= params.particleCount ||
    bond.restLength <= 0.0 ||
    bond.weight <= 0.0
  ) {
    return position;
  }

  let other = particlesIn[bond.neighborIndex];
  if (!particleActive(other) || particleBodyId(other) != bodyId) {
    return position;
  }

  let otherPosition = other.position + other.velocity * params.deltaTime;
  let delta = position - otherPosition;
  let distanceSq = dot(delta, delta);
  if (distanceSq < 0.0001) {
    return position;
  }

  let distance = sqrt(distanceSq);
  let direction = delta / distance;
  let safeRestDistance = max(bond.restLength, 0.01);
  let stretch = clamp(distance - safeRestDistance, -safeRestDistance * 0.5, safeRestDistance * 0.5);
  let correction = stretch * 0.5 * iterationStiffness * bond.weight;

  return position - direction * correction;
}

fn particleActive(particle: Particle) -> bool {
  return particle.radius > 0.0;
}

fn particleBodyId(particle: Particle) -> u32 {
  return particle.flags & BODY_ID_MASK;
}
