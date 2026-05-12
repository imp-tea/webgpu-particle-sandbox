export type WebGPUContext = {
  adapter: GPUAdapter;
  device: GPUDevice;
  context: GPUCanvasContext;
  format: GPUTextureFormat;
  canvas: HTMLCanvasElement;
  resize: () => { width: number; height: number; changed: boolean };
  getWorldSize: () => { width: number; height: number };
};

export async function initWebGPU(canvas: HTMLCanvasElement): Promise<WebGPUContext> {
  if (!navigator.gpu) {
    throw new Error("WebGPU is not available in this browser.");
  }

  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance"
  });
  if (!adapter) {
    throw new Error("No WebGPU adapter was found.");
  }

  const device = await adapter.requestDevice();
  const context = canvas.getContext("webgpu");
  if (!context) {
    throw new Error("Could not create a WebGPU canvas context.");
  }

  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device,
    format,
    alphaMode: "opaque"
  });

  let cssWidth = 1;
  let cssHeight = 1;

  const resize = () => {
    const rect = canvas.getBoundingClientRect();
    const nextCssWidth = Math.max(1, rect.width);
    const nextCssHeight = Math.max(1, rect.height);
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const nextWidth = Math.max(1, Math.floor(nextCssWidth * dpr));
    const nextHeight = Math.max(1, Math.floor(nextCssHeight * dpr));
    const changed = canvas.width !== nextWidth || canvas.height !== nextHeight;

    if (changed) {
      canvas.width = nextWidth;
      canvas.height = nextHeight;
    }

    cssWidth = nextCssWidth;
    cssHeight = nextCssHeight;
    return { width: nextCssWidth, height: nextCssHeight, changed };
  };

  resize();

  return {
    adapter,
    device,
    context,
    format,
    canvas,
    resize,
    getWorldSize: () => ({ width: cssWidth, height: cssHeight })
  };
}
