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

struct MaterialParams {
  color: vec4<f32>,
  dynamics: vec4<f32>,
};

struct BodyParams {
  startIndex: u32,
  particleCount: u32,
  columns: u32,
  rows: u32,
  restCellSize: vec2<f32>,
  padding: vec2<f32>,
};

@group(0) @binding(0) var<storage, read> particlesIn: array<Particle>;
@group(0) @binding(1) var<storage, read_write> particlesOut: array<Particle>;
@group(0) @binding(2) var<uniform> params: SimParams;
@group(0) @binding(3) var<storage, read_write> cellCounts: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> cellStarts: array<u32>;
@group(0) @binding(5) var<storage, read> pairValues: array<u32>;
@group(0) @binding(6) var<storage, read> materials: array<MaterialParams>;
@group(0) @binding(7) var<storage, read> bodies: array<BodyParams>;

const BODY_ID_MASK: u32 = 0x0000ffffu;

// Integrates particles from a source state buffer into a destination state buffer.
// Neighbor forces are sampled from the spatial grid built earlier in the frame.
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

  acceleration += neighborForces(index, particle);

  var velocity = particle.velocity + acceleration * params.deltaTime;
  velocity *= max(0.0, 1.0 - params.damping * params.deltaTime);

  let speedSq = dot(velocity, velocity);
  let maxSpeedSq = params.maxSpeed * params.maxSpeed;
  if (speedSq > maxSpeedSq) {
    velocity = normalize(velocity) * params.maxSpeed;
  }

  var position = particle.position + velocity * params.deltaTime;
  position = relaxSoftBodyPosition(index, position);

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

fn neighborForces(index: u32, particle: Particle) -> vec2<f32> {
  var force = vec2<f32>(0.0);
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
          force += pairForce(particle, particlesIn[otherIndex]);
        }
      }
    }
  }

  return force;
}

fn pairForce(particle: Particle, other: Particle) -> vec2<f32> {
  if (!particleActive(other)) {
    return vec2<f32>(0.0);
  }

  let delta = particle.position - other.position;
  let distanceSq = dot(delta, delta);
  if (distanceSq < 0.0001) {
    return vec2<f32>(0.0);
  }

  let distance = sqrt(distanceSq);
  let restDistance = particle.radius + other.radius;
  let differentBody = particleBodyId(particle) != particleBodyId(other);
  let interactionDistance = select(restDistance * 4.0, restDistance * 6.0, differentBody);

  if (distance >= interactionDistance) {
    return vec2<f32>(0.0);
  }

  let material = materials[particle.materialId % 4u];
  let otherMaterial = materials[other.materialId % 4u];
  let direction = delta / distance;
  let overlap = max(0.0, restDistance - distance) / restDistance;
  let repulsionScale = (material.dynamics.x + otherMaterial.dynamics.x) * 0.5;
  let contactDistance = select(restDistance, restDistance * 2.65, differentBody);
  let nearContact = max(0.0, contactDistance - distance) / contactDistance;
  let bodyBarrier = select(0.0, nearContact * nearContact * params.particleRepulsion * 12.0, differentBody);
  let repulsion = overlap * params.particleRepulsion * repulsionScale + bodyBarrier;

  let normalizedDistance = clamp(distance / interactionDistance, 0.0, 1.0);
  let attractionBand = smoothstep(0.30, 0.62, normalizedDistance) * (1.0 - smoothstep(0.62, 1.0, normalizedDistance));
  let cohesionScale = (material.dynamics.y + otherMaterial.dynamics.y) * 0.5;
  let sameAffinity = (material.dynamics.z + otherMaterial.dynamics.z) * 0.5;
  let crossAffinity = (material.dynamics.w + otherMaterial.dynamics.w) * 0.5;
  let materialAffinity = select(crossAffinity, sameAffinity, particle.materialId == other.materialId);
  let attraction = select(
    attractionBand * params.particleRepulsion * materialAffinity * cohesionScale * params.cohesion,
    0.0,
    differentBody
  );

  let proximity = 1.0 - normalizedDistance;
  let velocityBlend = (other.velocity - particle.velocity) * params.viscosity * proximity * 0.35;
  return (direction * (repulsion - attraction) + velocityBlend) / max(particle.mass, 0.01);
}

fn relaxSoftBodyPosition(index: u32, predictedPosition: vec2<f32>) -> vec2<f32> {
  if (params.softBodyStrength <= 0.0 || params.bondIterations == 0u) {
    return predictedPosition;
  }

  let bodyId = particleBodyId(particlesIn[index]);
  let body = bodies[bodyId];
  if (body.particleCount == 0u || body.columns == 0u || body.rows == 0u || index < body.startIndex) {
    return predictedPosition;
  }

  let localIndex = index - body.startIndex;
  if (localIndex >= body.particleCount) {
    return predictedPosition;
  }

  let column = localIndex % body.columns;
  let row = localIndex / body.columns;
  let iterations = min(params.bondIterations, 8u);
  let targetStiffness = clamp(params.softBodyStrength / 6000.0, 0.0, 1.0);
  let iterationStiffness = 1.0 - pow(1.0 - targetStiffness, 1.0 / f32(iterations));
  let diagonalRestDistance = length(body.restCellSize);
  var position = predictedPosition;

  for (var iteration = 0u; iteration < iterations; iteration += 1u) {
    if (column > 0u) {
      position = relaxBond(position, bodyId, index - 1u, body.restCellSize.x, iterationStiffness, 1.0);
    }
    if (column + 1u < body.columns) {
      position = relaxBond(position, bodyId, index + 1u, body.restCellSize.x, iterationStiffness, 1.0);
    }
    if (row > 0u) {
      position = relaxBond(position, bodyId, index - body.columns, body.restCellSize.y, iterationStiffness, 1.0);
    }
    if (row + 1u < body.rows) {
      position = relaxBond(position, bodyId, index + body.columns, body.restCellSize.y, iterationStiffness, 1.0);
    }

    if (column > 0u && row > 0u) {
      position = relaxBond(position, bodyId, index - body.columns - 1u, diagonalRestDistance, iterationStiffness, 0.75);
    }
    if (column + 1u < body.columns && row > 0u) {
      position = relaxBond(position, bodyId, index - body.columns + 1u, diagonalRestDistance, iterationStiffness, 0.75);
    }
    if (column > 0u && row + 1u < body.rows) {
      position = relaxBond(position, bodyId, index + body.columns - 1u, diagonalRestDistance, iterationStiffness, 0.75);
    }
    if (column + 1u < body.columns && row + 1u < body.rows) {
      position = relaxBond(position, bodyId, index + body.columns + 1u, diagonalRestDistance, iterationStiffness, 0.75);
    }

    if (column > 1u) {
      position = relaxBond(position, bodyId, index - 2u, body.restCellSize.x * 2.0, iterationStiffness, 0.35);
    }
    if (column + 2u < body.columns) {
      position = relaxBond(position, bodyId, index + 2u, body.restCellSize.x * 2.0, iterationStiffness, 0.35);
    }
    if (row > 1u) {
      position = relaxBond(position, bodyId, index - body.columns * 2u, body.restCellSize.y * 2.0, iterationStiffness, 0.35);
    }
    if (row + 2u < body.rows) {
      position = relaxBond(position, bodyId, index + body.columns * 2u, body.restCellSize.y * 2.0, iterationStiffness, 0.35);
    }
  }

  return position;
}

fn relaxBond(
  position: vec2<f32>,
  bodyId: u32,
  otherIndex: u32,
  restDistance: f32,
  iterationStiffness: f32,
  weight: f32
) -> vec2<f32> {
  if (otherIndex >= params.particleCount) {
    return position;
  }

  let other = particlesIn[otherIndex];
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
  let safeRestDistance = max(restDistance, 0.01);
  let stretch = clamp(distance - safeRestDistance, -safeRestDistance * 0.5, safeRestDistance * 0.5);
  let correction = stretch * 0.5 * iterationStiffness * weight;

  return position - direction * correction;
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
