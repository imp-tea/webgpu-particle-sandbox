# WebGPU Particle Sandbox

A browser-based 2D particle and soft-body physics prototype built with TypeScript, WebGPU, and WGSL. The app starts with an empty canvas; use the controls to spawn soft bodies, tune their constraints, and interact with them in real time.

## Run

```bash
npm install
npm run dev
```

Open the Vite URL, usually `http://127.0.0.1:5173`.

On Windows PowerShell, if script execution policy blocks `npm`, use `npm.cmd install` and `npm.cmd run dev`.

Build a production bundle with:

```bash
npm run build
```

## Requirements

- A browser with WebGPU enabled.
- A GPU adapter exposed through `navigator.gpu`.

## Controls

- Choose a body shape, size, and particle radius, then click **Add body** to spawn a square, circle, or triangle soft body.
- Left mouse drag attracts active particles.
- Right mouse drag or Alt-drag repels active particles.
- **Pause** toggles GPU simulation dispatch.
- **Clear** removes all bodies and resets the active particle count to zero.
- Sliders adjust gravity, damping, simulation substeps, contact iterations, bond iterations, elasticity, viscosity, and mouse force.
- The debug line reports substeps, GPU-counted max cell occupancy, scatter overflow, and CPU-side frame/simulation command-encoding timing.

## File Structure

```text
src/main.ts                    App bootstrap, UI binding, render loop, pass orchestration
src/config.ts                  Tunables, limits, and shared byte sizes
src/input.ts                   Pointer input mapped to simulation uniforms
src/gpu/initWebGPU.ts          Adapter/device/context setup and resize handling
src/gpu/buffers.ts             Particle/grid/color/body/bond buffer allocation and CPU-side body spawning
src/gpu/pipelines.ts           Compute/render pipelines and bind groups
src/shaders/clearGrid.wgsl     Clears active spatial grid counters, offsets, and debug counters
src/shaders/countGrid.wgsl     Counts particles per active grid cell
src/shaders/scanCellStarts.wgsl Prefix-scans cell counts within fixed-size blocks
src/shaders/scanGroupOffsets.wgsl Prefix-scans block totals
src/shaders/addCellOffsets.wgsl Adds scanned block offsets to each cell start
src/shaders/scatterGrid.wgsl   Writes particle ids into contiguous per-cell ranges
src/shaders/simulate.wgsl      GPU integration, body-local velocity damping, soft-body bonds, wall collisions
src/shaders/solveContacts.wgsl Cross-body contact projection
src/shaders/render.wgsl        Instanced particle quad rendering from storage buffers
```

## Simulation Pipeline

The prototype uses two GPU storage buffers for particle state. Each simulation or contact pass reads from the active source buffer, writes the next state into the other buffer, then swaps them. This avoids CPU readback and avoids read/write hazards when compute shaders sample neighboring particles.

Each animation frame writes the latest UI and pointer state into a uniform buffer. If the app is not paused and at least one body exists, each active substep first runs:

```text
simulate          Integrate gravity, pointer force, body-local viscosity, bonds, and walls.
```

If at least two bodies exist, the app then runs the configured number of contact iterations. Each contact iteration rebuilds the spatial grid from the latest particle buffer and runs:

```text
clearGrid         Clear active cell counts, starts, scatter offsets, and debug counters.
countGrid         Count active particles per active grid cell with atomics.
scanCellStarts    Prefix-scan counts inside 256-cell blocks.
scanGroupOffsets  Prefix-scan the per-block totals.
addCellOffsets    Add block offsets to produce exact global cell starts.
scatterGrid       Scatter particle ids into contiguous per-cell ranges.
solveContacts     Push overlapping particles from different bodies apart.
```

Rendering draws six vertices per particle as instanced quads and shades circular sprites from the current particle buffer and fixed per-body colors. Soft-body bonds are stored explicitly in a separate GPU storage buffer, so body shapes are not constrained to square lattice indexing.

## GPU Data

Particle layout is 32 bytes:

```text
position:    vec2<f32>
velocity:    vec2<f32>
material id: u32
flags:       u32  active/body id bits
radius:      f32
mass:        f32
```

Body layout is 32 bytes:

```text
start index: u32
count:       u32
padding:     24 bytes
```

Bond layout is 16 bytes per slot, with 8 fixed slots per particle:

```text
neighbor id: u32
rest length: f32
weight:      f32
padding:     f32
```

Rest-shape layout is 16 bytes per particle:

```text
local pos:   vec2<f32>
weight:      f32  perimeter particles use stronger restore
padding:     f32
```

Color layout is 16 bytes:

```text
color:       vec4<f32>
```

CPU code only creates or clears particle, body, and bond data when the user adds bodies or clicks **Clear**. Normal simulation and rendering stay GPU-side.

## Behavior Notes

- Same-body particles preserve local soft-body shape through explicit per-particle neighbor bonds.
- Different bodies interact through the contact projection pass.
- Spawned bodies use shape-specific point sets for squares, circles, and triangles. Squares and triangles emit exact corner particles and evenly spaced edge particles first; circles emit evenly spaced particles exactly on the circumference. The interior starts from a hexagonal fill, rejects candidates too close to the fixed outline particles, then runs a few CPU-side spacing relaxation passes before each particle builds up to 8 nearby neighbor bonds. Boundary-to-boundary bonds receive a modestly higher weight.
- A low-strength shape-memory correction samples each body's current transform and nudges particles toward their rest-local positions under that transform. Perimeter particles receive the strongest correction so collapsed outlines can spring back without making the whole body rigid.
- Elasticity scales the explicit bond solve. Set it to `0` to return toward particle-fluid behavior.
- Bond iterations repeat the local shape solve inside each integration pass. Raise this for stiffer, less fabric-like bodies.
- Contact iterations run separate cross-body overlap projection passes. Raise this when bodies visibly interpenetrate.
- Viscosity damps relative motion across the same explicit neighbor slots without using the spatial grid. Higher values make soft bodies less jiggly but more sluggish.
- Higher substep, contact-iteration, and bond-iteration counts improve stability but multiply compute work.

## Current Limits

- Maximum particles: `50,000`.
- Maximum bodies: `96`.
- Spatial grid: up to `256 x 144` cells with 18-pixel cells.
- Prefix scan capacity: up to 256 scan groups; the current maximum grid uses 144 groups.
- Grid particle id capacity: `65,536`, which covers the current particle target.
- Particle radius is clamped to `1..8` pixels and body size is clamped to `40..260` pixels in the spawn path.
- The debug timing readout is CPU-side command encode/submit timing, not GPU timestamp-query timing.

## Implementation Status

Implemented:

1. WebGPU render loop with ping-pong particle buffers.
2. CPU-side spawning for square, circle, and triangle soft bodies.
3. GPU spatial grid build using count, prefix scan, and scatter passes.
4. Cross-body contact sampling from each particle's own cell and adjacent cells.
5. Body-local velocity damping and soft-body shape retention.
6. Explicit per-particle soft-body bond constraints.
7. Separate cross-body contact projection iterations.
8. Fixed per-body colors through a GPU storage buffer.
9. GPU debug counters for max cell occupancy and scatter overflow during contact grid builds.

Potential next work:

1. Add density limiting or position relaxation for calmer high-pressure same-body contacts.
2. Add optional GPU timestamp queries where supported.
