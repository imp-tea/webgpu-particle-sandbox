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
  motorTargetAngularVelocity: f32,
  motorStrength: f32,
  softBodyStrength: f32,
  viscosity: f32,
  friction: f32,
  padding1: f32,
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
const FLOOR_TANGENT_SLEEP_SPEED: f32 = 5.0;

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

  var acceleration = params.gravity + dragAcceleration(index, particle) + motorAcceleration(index, particle);

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

  let body = bodies[particleBodyId(particle)];
  let bounce = clamp(params.wallBounce, 0.0, 1.0);
  let wallTangentRetain = clamp(1.0 - body.friction, 0.0, 1.0);
  let r = particle.radius;

  if (position.x < r) {
    position.x = r;
    velocity.x = abs(velocity.x) * bounce;
    velocity.y *= wallTangentRetain;
  }
  if (position.x > params.worldSize.x - r) {
    position.x = params.worldSize.x - r;
    velocity.x = -abs(velocity.x) * bounce;
    velocity.y *= wallTangentRetain;
  }
  if (position.y < r) {
    position.y = r;
    velocity.y = abs(velocity.y) * bounce;
    velocity.x *= wallTangentRetain;
  }
  if (position.y > params.worldSize.y - r) {
    position.y = params.worldSize.y - r;
    velocity.y = -abs(velocity.y) * bounce;
    velocity.x *= wallTangentRetain;
    if (abs(velocity.x) < FLOOR_TANGENT_SLEEP_SPEED) {
      velocity.x = 0.0;
    }
  }

  particle.position = position;
  particle.velocity = velocity;
  particlesOut[index] = particle;
}

fn relaxSoftBodyPosition(index: u32, predictedPosition: vec2<f32>) -> vec2<f32> {
  let bodyId = particleBodyId(particlesIn[index]);
  let body = bodies[bodyId];
  let softBodyStrength = body.softBodyStrength;
  if (softBodyStrength <= 0.0 || params.bondIterations == 0u) {
    return predictedPosition;
  }

  if (body.particleCount == 0u || index < body.startIndex) {
    return predictedPosition;
  }

  let localIndex = index - body.startIndex;
  if (localIndex >= body.particleCount) {
    return predictedPosition;
  }

  let iterations = min(params.bondIterations, 8u);
  let targetStiffness = clamp(softBodyStrength / 6000.0, 0.0, 1.0);
  let iterationStiffness = 1.0 - pow(1.0 - targetStiffness, 1.0 / f32(iterations));
  var position = predictedPosition;

  let reverseSlots = (params.solverPhase & 1u) == 1u;
  for (var iteration = 0u; iteration < iterations; iteration += 1u) {
    for (var slot = 0u; slot < BOND_SLOT_COUNT; slot += 1u) {
      let orderedSlot = select(slot, BOND_SLOT_COUNT - 1u - slot, reverseSlots);
      let bond = bonds[index * BOND_SLOT_COUNT + orderedSlot];
      position = relaxBond(position, bodyId, bond, iterationStiffness);
    }
  }

  return position;
}

fn dragAcceleration(index: u32, particle: Particle) -> vec2<f32> {
  if (params.mouseForce == 0.0 || params.selectedParticleIndex >= params.particleCount) {
    return vec2<f32>(0.0);
  }

  let grabbed = particlesIn[params.selectedParticleIndex];
  if (!particleActive(grabbed) || particleBodyId(grabbed) != particleBodyId(particle)) {
    return vec2<f32>(0.0);
  }

  let dragDelta = params.mousePosition - grabbed.position;
  if (dot(dragDelta, dragDelta) <= 1.0) {
    return vec2<f32>(0.0);
  }

  let distanceFromGrab = distance(particle.position, grabbed.position);
  let influenceRadius = max(grabbed.radius * 14.0, 96.0);
  let normalizedDistance = distanceFromGrab / influenceRadius;
  let falloff = max(0.18, 1.0 / (1.0 + normalizedDistance * normalizedDistance));
  let selectedBoost = select(1.0, 1.2, index == params.selectedParticleIndex);

  return dragDelta * params.mouseForce * falloff * selectedBoost / max(particle.mass, 0.001);
}

fn motorAcceleration(index: u32, particle: Particle) -> vec2<f32> {
  if (params.deltaTime <= 0.000001) {
    return vec2<f32>(0.0);
  }

  let bodyId = particleBodyId(particle);
  let body = bodies[bodyId];
  if (abs(body.motorTargetAngularVelocity) <= 0.0001 || body.motorStrength <= 0.0 || body.particleCount < 2u) {
    return vec2<f32>(0.0);
  }

  let center = bodyCenter(bodyId);
  let radial = particle.position - center;
  let radialLengthSq = dot(radial, radial);
  if (radialLengthSq <= 0.0001) {
    return vec2<f32>(0.0);
  }

  let radialLength = sqrt(radialLengthSq);
  let tangent = vec2<f32>(-radial.y, radial.x) / radialLength;
  let currentTangentialVelocity = tangent * dot(particle.velocity, tangent);
  let targetTangentialVelocity = tangent * body.motorTargetAngularVelocity * radialLength;
  let motorDelta = targetTangentialVelocity - currentTangentialVelocity;
  let rimFactor = smoothstep(particle.radius * 2.0, particle.radius * 8.0, radialLength);

  return motorDelta * body.motorStrength * rimFactor / max(particle.mass, 0.001);
}

fn bodyCenter(bodyId: u32) -> vec2<f32> {
  let body = bodies[bodyId];
  let sampleCount = min(body.particleCount, SHAPE_MATCH_SAMPLE_COUNT);
  var center = vec2<f32>(0.0);
  var validSamples = 0.0;

  for (var sample = 0u; sample < sampleCount; sample += 1u) {
    let sampleLocalIndex = phasedSampleLocalIndex(sample, sampleCount, body.particleCount);
    let sampleIndex = body.startIndex + sampleLocalIndex;
    let sampleParticle = particlesIn[sampleIndex];
    if (particleActive(sampleParticle) && particleBodyId(sampleParticle) == bodyId) {
      center += sampleParticle.position;
      validSamples += 1.0;
    }
  }

  if (validSamples <= 0.0) {
    return vec2<f32>(0.0);
  }

  return center / validSamples;
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
    let sampleLocalIndex = phasedSampleLocalIndex(sample, sampleCount, body.particleCount);
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
    let sampleLocalIndex = phasedSampleLocalIndex(sample, sampleCount, body.particleCount);
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
  let correction = targetPosition - predictedPosition;
  let correctionDistance = length(correction);
  if (correctionDistance <= 0.000001) {
    return predictedPosition;
  }

  let sourceParticle = particlesIn[index];
  let deformationGate = smoothstep(sourceParticle.radius * 0.25, sourceParticle.radius * 1.4, correctionDistance);
  if (deformationGate <= 0.000001) {
    return predictedPosition;
  }

  let wallClearance = min(
    min(predictedPosition.x - sourceParticle.radius, params.worldSize.x - sourceParticle.radius - predictedPosition.x),
    min(predictedPosition.y - sourceParticle.radius, params.worldSize.y - sourceParticle.radius - predictedPosition.y)
  );
  let wallFactor = mix(0.45, 1.0, smoothstep(0.0, sourceParticle.radius * 3.0, wallClearance));
  let speedFactor = mix(0.6, 1.0, smoothstep(10.0, 90.0, length(sourceParticle.velocity)));
  let contactQuietFactor = min(wallFactor * speedFactor, 1.0);
  let maxCorrection = mix(sourceParticle.radius * 0.08, sourceParticle.radius * 0.22, deformationGate) *
    clamp(params.deltaTime * 60.0, 0.25, 1.0);
  let frameStrength = 0.18;
  let stepStrength = 1.0 - pow(1.0 - frameStrength, clamp(params.deltaTime * 60.0, 0.0, 1.0));
  let weightedStrength = clamp(stepStrength * restShape.weight * deformationGate * contactQuietFactor, 0.0, 0.28);
  let correctionStep = min(correctionDistance * weightedStrength, maxCorrection);

  return predictedPosition + correction / correctionDistance * correctionStep;
}

struct VelocitySample {
  delta: vec2<f32>,
  weight: f32,
};

fn dampSoftBodyVelocity(index: u32, velocity: vec2<f32>) -> vec2<f32> {
  let bodyId = particleBodyId(particlesIn[index]);
  let body = bodies[bodyId];
  let viscosity = body.viscosity;
  if (viscosity <= 0.0 || params.deltaTime <= 0.000001) {
    return velocity;
  }

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

  let blend = clamp(viscosity * params.deltaTime * 0.18, 0.0, 0.85);
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

fn phasedSampleLocalIndex(sample: u32, sampleCount: u32, particleCount: u32) -> u32 {
  if (sampleCount == 0u || particleCount == 0u) {
    return 0u;
  }

  let base = (sample * particleCount) / sampleCount;
  let phaseOffset = params.solverPhase % particleCount;
  let mirrored = (sample & 1u) == 1u;
  let phased = (base + phaseOffset) % particleCount;
  return select(phased, particleCount - 1u - phased, mirrored);
}

fn particleActive(particle: Particle) -> bool {
  return particle.radius > 0.0;
}

fn particleBodyId(particle: Particle) -> u32 {
  return particle.flags & BODY_ID_MASK;
}
