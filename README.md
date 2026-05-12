# WebGPU Particle Sandbox

A minimal browser-based 2D particle physics prototype built with TypeScript, WebGPU, and WGSL. It is inspired by the kind of interactive particle sandbox workflow seen in Simulario-like tools, but it does not use Unity or CPU physics libraries.

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

## Controls

- Left mouse drag attracts particles.
- Right mouse drag or Alt-drag repels particles.
- Pause toggles GPU simulation dispatch.
- Reset rewrites the initial particle state into both GPU ping-pong buffers.
- Sliders adjust particle count, gravity, damping, mouse force, short-range repulsion, and cohesion.
- The material preset selector rewrites the GPU material buffer live: Mixed, Granular, Liquid, and Separated.

## File Structure

```text
src/main.ts                  App bootstrap, UI binding, render loop, pass orchestration
src/config.ts                Tunables and shared byte sizes
src/input.ts                 Pointer input mapped to simulation uniforms
src/gpu/initWebGPU.ts        Adapter/device/context setup and resize handling
src/gpu/buffers.ts           Particle/grid/material buffer allocation, initial data, uniform packing
src/gpu/pipelines.ts         Compute/render pipelines and bind groups
src/shaders/clearGrid.wgsl   Clears active spatial grid counters and offsets
src/shaders/countGrid.wgsl   Counts particles per active grid cell
src/shaders/scanCellStarts.wgsl Prefix-scans cell counts within fixed-size blocks
src/shaders/scanGroupOffsets.wgsl Prefix-scans block totals
src/shaders/addCellOffsets.wgsl Adds scanned block offsets to each cell start
src/shaders/scatterGrid.wgsl Writes particle ids into contiguous per-cell ranges
src/shaders/simulate.wgsl    GPU integration, range-based neighbor forces, wall collisions
src/shaders/render.wgsl      Instanced particle quad rendering from storage buffers
```

## Current GPU Passes

The prototype uses two GPU storage buffers for particle state. Each simulation frame reads from the active source buffer and writes the next state into the other buffer, then swaps them. This avoids CPU readback and avoids read/write hazards when the compute shader samples other particles.

Particle layout is 32 bytes:

```text
position:    vec2<f32>
velocity:    vec2<f32>
material id: u32
flags:       u32
radius:      f32
mass:        f32
```

Material layout is 32 bytes:

```text
color:       vec4<f32>
dynamics:    vec4<f32>  repulsion scale, cohesion scale, same-material affinity, cross-material affinity
```

The simulation now uses seven compute phases per active frame:

```text
clearGrid         Clear active cell counts, starts, and scatter offsets.
countGrid         Count particles per active grid cell with atomics.
scanCellStarts    Prefix-scan counts inside 256-cell blocks.
scanGroupOffsets  Prefix-scan the per-block totals.
addCellOffsets    Add block offsets to produce exact global cell starts.
scatterGrid       Scatter particle ids into contiguous per-cell ranges.
simulate          Sample the current cell and 8 neighbors, then integrate into the next buffer.
```

The grid build uses a histogram/prefix-bin path instead of sorting `(cellId, particleId)` pairs every frame. The simulation pass applies gravity, damping, pointer attraction or repulsion, wall collisions, velocity clamping, short-range repulsion, and a material-aware cohesion band driven by the material parameter buffer. Rendering draws six vertices per particle as instanced quads and shades circular sprites from the current GPU particle buffer and material colors.

## Limitations

- Cohesion intentionally makes particles form blobs. Keep it near `0` for granular behavior and raise it for fluid/blob behavior.
- The prefix scan assumes the configured grid stays within 256 scan groups. The current 256x144 grid uses 144 groups.
- The grid particle id buffer is sized for 65,536 lanes, which covers the current 50,000-particle target.
- Material parameters are GPU buffer data and can be switched live with the material preset selector.
- Particle data is CPU-generated only on reset. Normal simulation and rendering stay GPU-side.
- The prototype targets clarity first. It does not yet expose multiple simulation substeps.

## Spatial Hashing Status

Implemented:

1. Compute each particle's cell coordinate from position and radius.
2. Count active particles per grid cell.
3. Prefix-scan cell counts into exact cell start offsets.
4. Scatter particle ids into contiguous per-cell ranges.
5. For each particle, sample only its own cell and adjacent cells.
6. Add short-range repulsion and a tunable Lennard-Jones-like cohesion band.
7. Read per-material color and dynamics parameters from a GPU storage buffer.
8. Switch material parameter presets live from the UI.

Next:

1. Add optional substeps for stability under high force and high density.
2. Add GPU debug counters for max cell occupancy and timing.

The fixed-cell overflow issue is removed, the dispatch-heavy bitonic sort has been replaced by a histogram/prefix-bin grid build, and material behavior now comes from a shared GPU parameter table.
