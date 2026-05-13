struct Particle {
  position: vec2<f32>,
  velocity: vec2<f32>,
  materialId: u32,
  flags: u32,
  radius: f32,
  mass: f32,
};

struct RenderParams {
  gravity: vec2<f32>,
  worldSize: vec2<f32>,
  mousePosition: vec2<f32>,
  mouseForce: f32,
  deltaTime: f32,
  damping: f32,
  particleCount: u32,
  wallBounce: f32,
  maxSpeed: f32,
};

struct MaterialParams {
  color: vec4<f32>,
};

struct VertexOut {
  @builtin(position) position: vec4<f32>,
  @location(0) local: vec2<f32>,
  @location(1) @interpolate(flat) materialId: u32,
  @location(2) speed: f32,
};

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: RenderParams;
@group(0) @binding(2) var<storage, read> materials: array<MaterialParams>;

@vertex
fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> VertexOut {
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(1.0, 1.0)
  );

  let particle = particles[instanceIndex];
  let local = corners[vertexIndex];
  let pixel = particle.position + local * particle.radius;
  let clip = vec2<f32>(
    pixel.x / params.worldSize.x * 2.0 - 1.0,
    1.0 - pixel.y / params.worldSize.y * 2.0
  );

  var out: VertexOut;
  out.position = vec4<f32>(clip, 0.0, 1.0);
  out.local = local;
  out.materialId = particle.materialId;
  out.speed = length(particle.velocity);
  return out;
}

@fragment
fn fragmentMain(in: VertexOut) -> @location(0) vec4<f32> {
  let distanceSq = dot(in.local, in.local);
  if (distanceSq > 1.0) {
    discard;
  }

  let base = materials[in.materialId % 4u].color.rgb;
  let edge = smoothstep(1.0, 0.72, distanceSq);
  let speedGlow = clamp(in.speed / 700.0, 0.0, 1.0);
  let color = base * (0.68 + edge * 0.36) + vec3<f32>(0.22, 0.34, 0.42) * speedGlow;
  let alpha = smoothstep(1.0, 0.82, distanceSq);
  return vec4<f32>(color, alpha);
}
