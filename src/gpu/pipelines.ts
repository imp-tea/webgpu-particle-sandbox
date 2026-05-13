import { WORKGROUP_SIZE } from "../config";
import addCellOffsetsShader from "../shaders/addCellOffsets.wgsl?raw";
import clearGridShader from "../shaders/clearGrid.wgsl?raw";
import countGridShader from "../shaders/countGrid.wgsl?raw";
import renderShader from "../shaders/render.wgsl?raw";
import scanCellStartsShader from "../shaders/scanCellStarts.wgsl?raw";
import scanGroupOffsetsShader from "../shaders/scanGroupOffsets.wgsl?raw";
import scatterGridShader from "../shaders/scatterGrid.wgsl?raw";
import simulateShader from "../shaders/simulate.wgsl?raw";
import solveContactsShader from "../shaders/solveContacts.wgsl?raw";
import solveJointsShader from "../shaders/solveJoints.wgsl?raw";
import type { GridBuffers } from "./buffers";

export type Pipelines = {
  clearGridPipeline: GPUComputePipeline;
  countGridPipeline: GPUComputePipeline;
  scanCellStartsPipeline: GPUComputePipeline;
  scanGroupOffsetsPipeline: GPUComputePipeline;
  addCellOffsetsPipeline: GPUComputePipeline;
  scatterGridPipeline: GPUComputePipeline;
  simulatePipeline: GPUComputePipeline;
  solveJointsPipeline: GPUComputePipeline;
  solveContactsPipeline: GPUComputePipeline;
  renderPipeline: GPURenderPipeline;
  clearGridBindGroup: GPUBindGroup;
  countGridBindGroups: [GPUBindGroup, GPUBindGroup];
  scanCellStartsBindGroup: GPUBindGroup;
  scanGroupOffsetsBindGroup: GPUBindGroup;
  addCellOffsetsBindGroup: GPUBindGroup;
  scatterGridBindGroups: [GPUBindGroup, GPUBindGroup];
  simulateBindGroups: [GPUBindGroup, GPUBindGroup];
  solveJointBindGroups: [GPUBindGroup, GPUBindGroup];
  solveContactBindGroups: [GPUBindGroup, GPUBindGroup];
  renderBindGroups: [GPUBindGroup, GPUBindGroup];
};

export function createPipelines(
  device: GPUDevice,
  format: GPUTextureFormat,
  particleBuffers: [GPUBuffer, GPUBuffer],
  uniformBuffer: GPUBuffer,
  gridBuffers: GridBuffers,
  materialBuffer: GPUBuffer,
  bodyBuffer: GPUBuffer,
  bondBuffer: GPUBuffer,
  restShapeBuffer: GPUBuffer,
  jointBuffer: GPUBuffer
): Pipelines {
  const clearGridBindGroupLayout = device.createBindGroupLayout({
    label: "Clear grid bind group layout",
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" }
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" }
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" }
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" }
      },
      {
        binding: 4,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" }
      }
    ]
  });

  const countGridBindGroupLayout = device.createBindGroupLayout({
    label: "Count grid bind group layout",
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" }
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" }
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" }
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" }
      }
    ]
  });

  const scanCellStartsBindGroupLayout = device.createBindGroupLayout({
    label: "Scan cell starts bind group layout",
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" }
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" }
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" }
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" }
      }
    ]
  });

  const scanGroupOffsetsBindGroupLayout = device.createBindGroupLayout({
    label: "Scan group offsets bind group layout",
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" }
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" }
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" }
      }
    ]
  });

  const addCellOffsetsBindGroupLayout = device.createBindGroupLayout({
    label: "Add cell offsets bind group layout",
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" }
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" }
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" }
      }
    ]
  });

  const scatterGridBindGroupLayout = device.createBindGroupLayout({
    label: "Scatter grid bind group layout",
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" }
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" }
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" }
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" }
      },
      {
        binding: 4,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" }
      },
      {
        binding: 5,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" }
      }
    ]
  });

  const simulateBindGroupLayout = device.createBindGroupLayout({
    label: "Simulation bind group layout",
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" }
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" }
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" }
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" }
      },
      {
        binding: 4,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" }
      },
      {
        binding: 5,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" }
      }
    ]
  });

  const solveContactBindGroupLayout = device.createBindGroupLayout({
    label: "Contact solve bind group layout",
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" }
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" }
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" }
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" }
      },
      {
        binding: 4,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" }
      },
      {
        binding: 5,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" }
      },
      {
        binding: 6,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" }
      },
      {
        binding: 7,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" }
      }
    ]
  });

  const solveJointBindGroupLayout = device.createBindGroupLayout({
    label: "Joint solve bind group layout",
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" }
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" }
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" }
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" }
      },
      {
        binding: 4,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" }
      },
      {
        binding: 5,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" }
      }
    ]
  });

  const renderBindGroupLayout = device.createBindGroupLayout({
    label: "Particle render bind group layout",
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: "read-only-storage" }
      },
      {
        binding: 1,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: "uniform" }
      },
      {
        binding: 2,
        visibility: GPUShaderStage.FRAGMENT,
        buffer: { type: "read-only-storage" }
      }
    ]
  });

  const clearGridPipeline = device.createComputePipeline({
    label: "Clear spatial grid pipeline",
    layout: device.createPipelineLayout({
      label: "Clear grid pipeline layout",
      bindGroupLayouts: [clearGridBindGroupLayout]
    }),
    compute: {
      module: device.createShaderModule({
        label: "Clear grid shader",
        code: clearGridShader
      }),
      entryPoint: "main",
      constants: {
        WORKGROUP_SIZE
      }
    }
  });

  const countGridPipeline = device.createComputePipeline({
    label: "Count spatial grid pipeline",
    layout: device.createPipelineLayout({
      label: "Count grid pipeline layout",
      bindGroupLayouts: [countGridBindGroupLayout]
    }),
    compute: {
      module: device.createShaderModule({
        label: "Count grid shader",
        code: countGridShader
      }),
      entryPoint: "main",
      constants: {
        WORKGROUP_SIZE
      }
    }
  });

  const scanCellStartsPipeline = device.createComputePipeline({
    label: "Scan cell starts pipeline",
    layout: device.createPipelineLayout({
      label: "Scan cell starts pipeline layout",
      bindGroupLayouts: [scanCellStartsBindGroupLayout]
    }),
    compute: {
      module: device.createShaderModule({
        label: "Scan cell starts shader",
        code: scanCellStartsShader
      }),
      entryPoint: "main"
    }
  });

  const scanGroupOffsetsPipeline = device.createComputePipeline({
    label: "Scan group offsets pipeline",
    layout: device.createPipelineLayout({
      label: "Scan group offsets pipeline layout",
      bindGroupLayouts: [scanGroupOffsetsBindGroupLayout]
    }),
    compute: {
      module: device.createShaderModule({
        label: "Scan group offsets shader",
        code: scanGroupOffsetsShader
      }),
      entryPoint: "main"
    }
  });

  const addCellOffsetsPipeline = device.createComputePipeline({
    label: "Add cell offsets pipeline",
    layout: device.createPipelineLayout({
      label: "Add cell offsets pipeline layout",
      bindGroupLayouts: [addCellOffsetsBindGroupLayout]
    }),
    compute: {
      module: device.createShaderModule({
        label: "Add cell offsets shader",
        code: addCellOffsetsShader
      }),
      entryPoint: "main",
      constants: {
        WORKGROUP_SIZE
      }
    }
  });

  const scatterGridPipeline = device.createComputePipeline({
    label: "Scatter spatial grid pipeline",
    layout: device.createPipelineLayout({
      label: "Scatter grid pipeline layout",
      bindGroupLayouts: [scatterGridBindGroupLayout]
    }),
    compute: {
      module: device.createShaderModule({
        label: "Scatter grid shader",
        code: scatterGridShader
      }),
      entryPoint: "main",
      constants: {
        WORKGROUP_SIZE
      }
    }
  });

  const simulatePipeline = device.createComputePipeline({
    label: "Particle simulation pipeline",
    layout: device.createPipelineLayout({
      label: "Simulation pipeline layout",
      bindGroupLayouts: [simulateBindGroupLayout]
    }),
    compute: {
      module: device.createShaderModule({
        label: "Simulation shader",
        code: simulateShader
      }),
      entryPoint: "main",
      constants: {
        WORKGROUP_SIZE
      }
    }
  });

  const solveJointsPipeline = device.createComputePipeline({
    label: "Joint solve pipeline",
    layout: device.createPipelineLayout({
      label: "Joint solve pipeline layout",
      bindGroupLayouts: [solveJointBindGroupLayout]
    }),
    compute: {
      module: device.createShaderModule({
        label: "Joint solve shader",
        code: solveJointsShader
      }),
      entryPoint: "main",
      constants: {
        WORKGROUP_SIZE
      }
    }
  });

  const renderPipeline = device.createRenderPipeline({
    label: "Particle render pipeline",
    layout: device.createPipelineLayout({
      label: "Render pipeline layout",
      bindGroupLayouts: [renderBindGroupLayout]
    }),
    vertex: {
      module: device.createShaderModule({
        label: "Particle render shader",
        code: renderShader
      }),
      entryPoint: "vertexMain"
    },
    fragment: {
      module: device.createShaderModule({
        label: "Particle render shader",
        code: renderShader
      }),
      entryPoint: "fragmentMain",
      targets: [
        {
          format,
          blend: {
            color: {
              srcFactor: "src-alpha",
              dstFactor: "one-minus-src-alpha",
              operation: "add"
            },
            alpha: {
              srcFactor: "one",
              dstFactor: "one-minus-src-alpha",
              operation: "add"
            }
          }
        }
      ]
    },
    primitive: {
      topology: "triangle-list"
    }
  });

  const clearGridBindGroup = device.createBindGroup({
    label: "Clear grid bind group",
    layout: clearGridBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: gridBuffers.cellCounts } },
      { binding: 1, resource: { buffer: gridBuffers.cellStarts } },
      { binding: 2, resource: { buffer: gridBuffers.cellWriteOffsets } },
      { binding: 3, resource: { buffer: uniformBuffer } },
      { binding: 4, resource: { buffer: gridBuffers.debugCounters } }
    ]
  });

  const solveContactsPipeline = device.createComputePipeline({
    label: "Contact solve pipeline",
    layout: device.createPipelineLayout({
      label: "Contact solve pipeline layout",
      bindGroupLayouts: [solveContactBindGroupLayout]
    }),
    compute: {
      module: device.createShaderModule({
        label: "Contact solve shader",
        code: solveContactsShader
      }),
      entryPoint: "main",
      constants: {
        WORKGROUP_SIZE
      }
    }
  });

  const countGridBindGroups: [GPUBindGroup, GPUBindGroup] = [
    createCountGridBindGroup(
      device,
      countGridBindGroupLayout,
      particleBuffers[0],
      gridBuffers.cellCounts,
      uniformBuffer,
      gridBuffers.debugCounters,
      "A"
    ),
    createCountGridBindGroup(
      device,
      countGridBindGroupLayout,
      particleBuffers[1],
      gridBuffers.cellCounts,
      uniformBuffer,
      gridBuffers.debugCounters,
      "B"
    )
  ];

  const scanCellStartsBindGroup = device.createBindGroup({
    label: "Scan cell starts bind group",
    layout: scanCellStartsBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: gridBuffers.cellCounts } },
      { binding: 1, resource: { buffer: gridBuffers.cellStarts } },
      { binding: 2, resource: { buffer: gridBuffers.cellGroupSums } },
      { binding: 3, resource: { buffer: uniformBuffer } }
    ]
  });

  const scanGroupOffsetsBindGroup = device.createBindGroup({
    label: "Scan group offsets bind group",
    layout: scanGroupOffsetsBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: gridBuffers.cellGroupSums } },
      { binding: 1, resource: { buffer: gridBuffers.cellGroupOffsets } },
      { binding: 2, resource: { buffer: uniformBuffer } }
    ]
  });

  const addCellOffsetsBindGroup = device.createBindGroup({
    label: "Add cell offsets bind group",
    layout: addCellOffsetsBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: gridBuffers.cellStarts } },
      { binding: 1, resource: { buffer: gridBuffers.cellGroupOffsets } },
      { binding: 2, resource: { buffer: uniformBuffer } }
    ]
  });

  const scatterGridBindGroups: [GPUBindGroup, GPUBindGroup] = [
    createScatterGridBindGroup(
      device,
      scatterGridBindGroupLayout,
      particleBuffers[0],
      gridBuffers.cellStarts,
      gridBuffers.cellWriteOffsets,
      gridBuffers.pairValues,
      uniformBuffer,
      gridBuffers.debugCounters,
      "A"
    ),
    createScatterGridBindGroup(
      device,
      scatterGridBindGroupLayout,
      particleBuffers[1],
      gridBuffers.cellStarts,
      gridBuffers.cellWriteOffsets,
      gridBuffers.pairValues,
      uniformBuffer,
      gridBuffers.debugCounters,
      "B"
    )
  ];

  const simulateBindGroups: [GPUBindGroup, GPUBindGroup] = [
    createSimulateBindGroup(
      device,
      simulateBindGroupLayout,
      particleBuffers[0],
      particleBuffers[1],
      uniformBuffer,
      bodyBuffer,
      bondBuffer,
      restShapeBuffer,
      "A to B"
    ),
    createSimulateBindGroup(
      device,
      simulateBindGroupLayout,
      particleBuffers[1],
      particleBuffers[0],
      uniformBuffer,
      bodyBuffer,
      bondBuffer,
      restShapeBuffer,
      "B to A"
    )
  ];

  const solveJointBindGroups: [GPUBindGroup, GPUBindGroup] = [
    createSolveJointBindGroup(
      device,
      solveJointBindGroupLayout,
      particleBuffers[0],
      particleBuffers[1],
      uniformBuffer,
      bodyBuffer,
      restShapeBuffer,
      jointBuffer,
      "A to B"
    ),
    createSolveJointBindGroup(
      device,
      solveJointBindGroupLayout,
      particleBuffers[1],
      particleBuffers[0],
      uniformBuffer,
      bodyBuffer,
      restShapeBuffer,
      jointBuffer,
      "B to A"
    )
  ];

  const solveContactBindGroups: [GPUBindGroup, GPUBindGroup] = [
    createSolveContactBindGroup(
      device,
      solveContactBindGroupLayout,
      particleBuffers[0],
      particleBuffers[1],
      uniformBuffer,
      gridBuffers.cellCounts,
      gridBuffers.cellStarts,
      gridBuffers.pairValues,
      bodyBuffer,
      jointBuffer,
      "A to B"
    ),
    createSolveContactBindGroup(
      device,
      solveContactBindGroupLayout,
      particleBuffers[1],
      particleBuffers[0],
      uniformBuffer,
      gridBuffers.cellCounts,
      gridBuffers.cellStarts,
      gridBuffers.pairValues,
      bodyBuffer,
      jointBuffer,
      "B to A"
    )
  ];

  const renderBindGroups: [GPUBindGroup, GPUBindGroup] = [
    createRenderBindGroup(device, renderBindGroupLayout, particleBuffers[0], uniformBuffer, materialBuffer, "A"),
    createRenderBindGroup(device, renderBindGroupLayout, particleBuffers[1], uniformBuffer, materialBuffer, "B")
  ];

  return {
    clearGridPipeline,
    countGridPipeline,
    scanCellStartsPipeline,
    scanGroupOffsetsPipeline,
    addCellOffsetsPipeline,
    scatterGridPipeline,
    simulatePipeline,
    solveJointsPipeline,
    solveContactsPipeline,
    renderPipeline,
    clearGridBindGroup,
    countGridBindGroups,
    scanCellStartsBindGroup,
    scanGroupOffsetsBindGroup,
    addCellOffsetsBindGroup,
    scatterGridBindGroups,
    simulateBindGroups,
    solveJointBindGroups,
    solveContactBindGroups,
    renderBindGroups
  };
}

function createCountGridBindGroup(
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  particles: GPUBuffer,
  cellCounts: GPUBuffer,
  uniformBuffer: GPUBuffer,
  debugCounters: GPUBuffer,
  label: string
) {
  return device.createBindGroup({
    label: `Count grid bind group ${label}`,
    layout,
    entries: [
      { binding: 0, resource: { buffer: particles } },
      { binding: 1, resource: { buffer: cellCounts } },
      { binding: 2, resource: { buffer: uniformBuffer } },
      { binding: 3, resource: { buffer: debugCounters } }
    ]
  });
}

function createScatterGridBindGroup(
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  particles: GPUBuffer,
  cellStarts: GPUBuffer,
  cellWriteOffsets: GPUBuffer,
  pairValues: GPUBuffer,
  uniformBuffer: GPUBuffer,
  debugCounters: GPUBuffer,
  label: string
) {
  return device.createBindGroup({
    label: `Scatter grid bind group ${label}`,
    layout,
    entries: [
      { binding: 0, resource: { buffer: particles } },
      { binding: 1, resource: { buffer: cellStarts } },
      { binding: 2, resource: { buffer: cellWriteOffsets } },
      { binding: 3, resource: { buffer: pairValues } },
      { binding: 4, resource: { buffer: uniformBuffer } },
      { binding: 5, resource: { buffer: debugCounters } }
    ]
  });
}

function createSimulateBindGroup(
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  source: GPUBuffer,
  destination: GPUBuffer,
  uniformBuffer: GPUBuffer,
  bodyBuffer: GPUBuffer,
  bondBuffer: GPUBuffer,
  restShapeBuffer: GPUBuffer,
  label: string
) {
  return device.createBindGroup({
    label: `Simulation bind group ${label}`,
    layout,
    entries: [
      { binding: 0, resource: { buffer: source } },
      { binding: 1, resource: { buffer: destination } },
      { binding: 2, resource: { buffer: uniformBuffer } },
      { binding: 3, resource: { buffer: bodyBuffer } },
      { binding: 4, resource: { buffer: bondBuffer } },
      { binding: 5, resource: { buffer: restShapeBuffer } }
    ]
  });
}

function createSolveJointBindGroup(
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  source: GPUBuffer,
  destination: GPUBuffer,
  uniformBuffer: GPUBuffer,
  bodyBuffer: GPUBuffer,
  restShapeBuffer: GPUBuffer,
  jointBuffer: GPUBuffer,
  label: string
) {
  return device.createBindGroup({
    label: `Joint solve bind group ${label}`,
    layout,
    entries: [
      { binding: 0, resource: { buffer: source } },
      { binding: 1, resource: { buffer: destination } },
      { binding: 2, resource: { buffer: uniformBuffer } },
      { binding: 3, resource: { buffer: bodyBuffer } },
      { binding: 4, resource: { buffer: restShapeBuffer } },
      { binding: 5, resource: { buffer: jointBuffer } }
    ]
  });
}

function createSolveContactBindGroup(
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  source: GPUBuffer,
  destination: GPUBuffer,
  uniformBuffer: GPUBuffer,
  cellCounts: GPUBuffer,
  cellStarts: GPUBuffer,
  pairValues: GPUBuffer,
  bodyBuffer: GPUBuffer,
  jointBuffer: GPUBuffer,
  label: string
) {
  return device.createBindGroup({
    label: `Contact solve bind group ${label}`,
    layout,
    entries: [
      { binding: 0, resource: { buffer: source } },
      { binding: 1, resource: { buffer: destination } },
      { binding: 2, resource: { buffer: uniformBuffer } },
      { binding: 3, resource: { buffer: cellCounts } },
      { binding: 4, resource: { buffer: cellStarts } },
      { binding: 5, resource: { buffer: pairValues } },
      { binding: 6, resource: { buffer: bodyBuffer } },
      { binding: 7, resource: { buffer: jointBuffer } }
    ]
  });
}

function createRenderBindGroup(
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  particleBuffer: GPUBuffer,
  uniformBuffer: GPUBuffer,
  materialBuffer: GPUBuffer,
  label: string
) {
  return device.createBindGroup({
    label: `Render bind group ${label}`,
    layout,
    entries: [
      { binding: 0, resource: { buffer: particleBuffer } },
      { binding: 1, resource: { buffer: uniformBuffer } },
      { binding: 2, resource: { buffer: materialBuffer } }
    ]
  });
}
