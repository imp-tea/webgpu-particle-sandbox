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

struct RestShape {
  localPosition: vec2<f32>,
  weight: f32,
  padding: f32,
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

struct BodyTransform {
  center: vec2<f32>,
  restCenter: vec2<f32>,
  rotation: vec2<f32>,
  valid: f32,
};

@group(0) @binding(0) var<storage, read> particlesIn: array<Particle>;
@group(0) @binding(1) var<storage, read_write> particlesOut: array<Particle>;
@group(0) @binding(2) var<uniform> params: SimParams;
@group(0) @binding(3) var<storage, read> bodies: array<BodyParams>;
@group(0) @binding(4) var<storage, read> restShapes: array<RestShape>;
@group(0) @binding(5) var<storage, read> joints: array<Joint>;

const BODY_ID_MASK: u32 = 0x0000ffffu;
const MAX_JOINTS: u32 = 64u;
const SHAPE_MATCH_SAMPLE_COUNT: u32 = 24u;
const FLOOR_TANGENT_SLEEP_SPEED: f32 = 5.0;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let index = globalId.x;
  if (index >= params.particleCount) {
    return;
  }

  var particle = particlesIn[index];
  if (!particleActive(particle) || params.jointCount == 0u) {
    particlesOut[index] = particle;
    return;
  }

  let bodyId = particleBodyId(particle);
  let restShape = restShapes[index];
  var correction = vec2<f32>(0.0);

  let jointCount = min(params.jointCount, MAX_JOINTS);
  for (var jointIndex = 0u; jointIndex < jointCount; jointIndex += 1u) {
    let joint = joints[jointIndex];
    if (joint.enabled == 0u || (bodyId != joint.bodyA && bodyId != joint.bodyB)) {
      continue;
    }

    let transformA = bodyTransform(joint.bodyA);
    let transformB = bodyTransform(joint.bodyB);
    if (transformA.valid <= 0.0 || transformB.valid <= 0.0) {
      continue;
    }

    let anchorA = worldAnchor(transformA, joint.localAnchorA);
    let anchorB = worldAnchor(transformB, joint.localAnchorB);
    let delta = anchorB - anchorA;
    let distanceSq = dot(delta, delta);
    if (distanceSq <= 0.0001) {
      continue;
    }

    let anchorSeparation = sqrt(distanceSq);
    let normal = delta / anchorSeparation;
    let error = anchorSeparation - max(0.0, joint.restLength);
    let targetStiffness = clamp(joint.stiffness, 0.0, 1.0);
    let iterations = max(1.0, f32(max(params.jointIterations, 1u)));
    let iterationStiffness = 1.0 - pow(1.0 - targetStiffness, 1.0 / iterations);
    let anchorCorrection = normal * error * 0.5 * iterationStiffness;
    let localAnchor = select(joint.localAnchorB, joint.localAnchorA, bodyId == joint.bodyA);
    let signedCorrection = select(-anchorCorrection, anchorCorrection, bodyId == joint.bodyA);
    let influenceRadius = max(joint.influenceRadius, particle.radius * 4.0);
    let anchorDistance = length(restShape.localPosition - localAnchor);
    let normalizedDistance = anchorDistance / influenceRadius;
    let influence = max(0.18, 1.0 / (1.0 + normalizedDistance * normalizedDistance));
    correction += signedCorrection * influence;
  }

  if (dot(correction, correction) > 0.000001) {
    let maxCorrection = max(particle.radius * 2.0, 10.0);
    let correctionLength = length(correction);
    if (correctionLength > maxCorrection) {
      correction = correction / correctionLength * maxCorrection;
    }

    let previousPosition = particle.position;
    particle.position = clamp(previousPosition + correction, vec2<f32>(particle.radius), params.worldSize - vec2<f32>(particle.radius));
    if (params.deltaTime > 0.000001) {
      let appliedCorrection = particle.position - previousPosition;
      particle.velocity += appliedCorrection / params.deltaTime;
      if (particle.position.y >= params.worldSize.y - particle.radius && abs(particle.velocity.x) < FLOOR_TANGENT_SLEEP_SPEED) {
        particle.velocity.x = 0.0;
      }
      particle.velocity = clampVelocity(particle.velocity);
    }
  }

  particlesOut[index] = particle;
}

fn bodyTransform(bodyId: u32) -> BodyTransform {
  let body = bodies[bodyId];
  if (body.particleCount < 2u) {
    return BodyTransform(vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(1.0, 0.0), 0.0);
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
      currentCenter += particle.position;
      validSamples += 1.0;
    }
  }

  if (validSamples <= 1.0) {
    return BodyTransform(currentCenter, restCenter, vec2<f32>(1.0, 0.0), 0.0);
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
      let currentDelta = particle.position - currentCenter;
      cosineSum += dot(restDelta, currentDelta);
      sineSum += restDelta.x * currentDelta.y - restDelta.y * currentDelta.x;
    }
  }

  let rotationLength = sqrt(cosineSum * cosineSum + sineSum * sineSum);
  if (rotationLength <= 0.000001) {
    return BodyTransform(currentCenter, restCenter, vec2<f32>(1.0, 0.0), 1.0);
  }

  return BodyTransform(currentCenter, restCenter, vec2<f32>(cosineSum, sineSum) / rotationLength, 1.0);
}

fn worldAnchor(transform: BodyTransform, localAnchor: vec2<f32>) -> vec2<f32> {
  let local = localAnchor - transform.restCenter;
  let rotated = vec2<f32>(
    transform.rotation.x * local.x - transform.rotation.y * local.y,
    transform.rotation.y * local.x + transform.rotation.x * local.y
  );
  return transform.center + rotated;
}

fn clampVelocity(velocity: vec2<f32>) -> vec2<f32> {
  let speedSq = dot(velocity, velocity);
  let maxSpeedSq = params.maxSpeed * params.maxSpeed;
  if (speedSq > maxSpeedSq) {
    return normalize(velocity) * params.maxSpeed;
  }

  return velocity;
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
