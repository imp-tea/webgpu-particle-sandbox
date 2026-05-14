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
  onPointerMove: ((position: { x: number; y: number }) => void) | undefined;
  onPlace: ((position: { x: number; y: number }) => void) | undefined;
  onDelete: ((position: { x: number; y: number }) => void) | undefined;
  onAddJoint: ((position: { x: number; y: number }) => void) | undefined;

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
    this.updatePosition(event);
    const modifier = event.ctrlKey || event.metaKey;
    if (modifier && event.button === 0) {
      event.preventDefault();
      this.onPlace?.({ x: this.state.x, y: this.state.y });
      return;
    }
    if (modifier && event.button === 2) {
      event.preventDefault();
      this.onDelete?.({ x: this.state.x, y: this.state.y });
      return;
    }
    if (event.shiftKey && event.button === 0) {
      event.preventDefault();
      this.onAddJoint?.({ x: this.state.x, y: this.state.y });
      return;
    }
    if (event.button !== 0) {
      return;
    }
    this.canvas.setPointerCapture(event.pointerId);
    this.state.active = true;
    this.onPointerDown?.({ x: this.state.x, y: this.state.y });
  };

  private readonly handlePointerMove = (event: PointerEvent) => {
    this.updatePosition(event);
    if (event.buttons !== 0) {
      this.state.active = true;
    }
    this.onPointerMove?.({ x: this.state.x, y: this.state.y });
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
