export type PointerState = {
  x: number;
  y: number;
  active: boolean;
  selectedParticleIndex: number;
  selectedBodyId: number;
};

export class PointerInput {
  onPointerDown: ((position: { x: number; y: number }) => void) | undefined;
  onPointerUp: (() => void) | undefined;

  readonly state: PointerState = {
    x: 0,
    y: 0,
    active: false,
    selectedParticleIndex: 0xffffffff,
    selectedBodyId: 0xffffffff
  };

  constructor(private readonly canvas: HTMLCanvasElement) {
    canvas.addEventListener("contextmenu", (event) => event.preventDefault());
    canvas.addEventListener("pointerdown", this.handlePointerDown);
    canvas.addEventListener("pointermove", this.handlePointerMove);
    canvas.addEventListener("pointerup", this.handlePointerUp);
    canvas.addEventListener("pointercancel", this.handlePointerUp);
    canvas.addEventListener("pointerleave", this.handlePointerUp);
  }

  private readonly handlePointerDown = (event: PointerEvent) => {
    this.canvas.setPointerCapture(event.pointerId);
    this.updatePosition(event);
    this.state.active = true;
    this.onPointerDown?.({ x: this.state.x, y: this.state.y });
  };

  private readonly handlePointerMove = (event: PointerEvent) => {
    this.updatePosition(event);
    if (event.buttons !== 0) {
      this.state.active = true;
    }
  };

  private readonly handlePointerUp = (event: PointerEvent) => {
    if (this.canvas.hasPointerCapture(event.pointerId)) {
      this.canvas.releasePointerCapture(event.pointerId);
    }
    this.updatePosition(event);
    this.state.active = false;
    this.state.selectedParticleIndex = 0xffffffff;
    this.onPointerUp?.();
  };

  private updatePosition(event: PointerEvent) {
    const rect = this.canvas.getBoundingClientRect();
    this.state.x = event.clientX - rect.left;
    this.state.y = event.clientY - rect.top;
  }
}
