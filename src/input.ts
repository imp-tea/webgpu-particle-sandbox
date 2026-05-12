export type PointerState = {
  x: number;
  y: number;
  active: boolean;
  forceSign: 1 | -1;
};

export class PointerInput {
  readonly state: PointerState = {
    x: 0,
    y: 0,
    active: false,
    forceSign: 1
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
    this.state.forceSign = this.pickForceSign(event);
  };

  private readonly handlePointerMove = (event: PointerEvent) => {
    this.updatePosition(event);
    if (event.buttons !== 0) {
      this.state.active = true;
      this.state.forceSign = this.pickForceSign(event);
    }
  };

  private readonly handlePointerUp = (event: PointerEvent) => {
    if (this.canvas.hasPointerCapture(event.pointerId)) {
      this.canvas.releasePointerCapture(event.pointerId);
    }
    this.updatePosition(event);
    this.state.active = false;
  };

  private updatePosition(event: PointerEvent) {
    const rect = this.canvas.getBoundingClientRect();
    this.state.x = event.clientX - rect.left;
    this.state.y = event.clientY - rect.top;
  }

  private pickForceSign(event: PointerEvent): 1 | -1 {
    const rightButton = event.button === 2 || (event.buttons & 2) === 2;
    return rightButton || event.altKey ? -1 : 1;
  }
}
