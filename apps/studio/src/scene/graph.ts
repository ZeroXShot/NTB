// The canvas. One three.js scene for both 2D and 3D (ADR 0006): 2D is this
// scene through an orthographic camera with the Z axis locked, 3D is the same
// scene through a perspective camera you can orbit. Picking, dragging and
// selection have one implementation, so the two modes cannot drift apart.
//
// Blocks are one InstancedMesh, links one LineSegments with per-vertex colour,
// labels plain DOM over the canvas. React never touches any of it: the scene
// owns its own frame loop.

import {
  BoxGeometry,
  BufferAttribute,
  BufferGeometry,
  Color,
  GridHelper,
  InstancedMesh,
  LineBasicMaterial,
  LineSegments,
  Matrix4,
  MeshLambertMaterial,
  DirectionalLight,
  AmbientLight,
  Object3D,
  OrthographicCamera,
  PerspectiveCamera,
  Plane,
  Raycaster,
  Scene,
  Vector2,
  Vector3,
  WebGLRenderer,
} from "three";

export type NodeStatus = "ok" | "warning" | "error";
export type BlockKind = "node" | "instance" | "generated";
export type LinkKind = "edge" | "rule" | "chain";
export type ViewMode = "2d" | "3d";

export interface SceneNode {
  id: string;
  label: string;
  pos: [number, number, number];
  extent: [number, number, number];
  status: NodeStatus;
  kind: BlockKind;
}

export interface SceneEdge {
  id: string;
  from: string;
  to: string;
  kind: LinkKind;
}

export interface GraphCallbacks {
  onSelect(nodeId: string | null, additive: boolean): void;
  onMove(nodeId: string, pos: [number, number, number]): void;
}

const BLOCK_COLORS: Record<BlockKind, number> = {
  node: 0x4c6ef5,
  instance: 0x1098ad,
  generated: 0x7048e8,
};
const STATUS_COLORS: Record<NodeStatus, number | null> = {
  ok: null,
  warning: 0xf08c00,
  error: 0xe03131,
};
export const LINK_COLORS: Record<LinkKind, number> = {
  edge: 0x8d99ae,
  rule: 0x22b8cf,
  chain: 0x748ffc,
};
const SELECTED = 0xffd43b;
const GRID_SNAP = 0.25;
const MAX_NODES = 20000;

export class GraphView {
  private readonly renderer: WebGLRenderer;
  private readonly scene = new Scene();
  private readonly flat: OrthographicCamera;
  private readonly solid: PerspectiveCamera;
  private readonly raycaster = new Raycaster();
  private readonly blocks: InstancedMesh;
  private readonly wires: LineSegments;
  private readonly grid: GridHelper;
  private readonly labels: HTMLDivElement;
  private readonly dummy = new Object3D();
  private readonly plane = new Plane(new Vector3(0, 0, 1), 0);

  private mode: ViewMode = "2d";
  private nodes: SceneNode[] = [];
  private edges: SceneEdge[] = [];
  private selection = new Set<string>();
  private dragging: { id: string; offset: Vector3; vertical: boolean } | null = null;
  private orbiting: { x: number; y: number; pan: boolean } | null = null;
  private frame = 0;
  private zoom = 40;
  // Spherical orbit around a target, which is also the 2D camera's centre.
  private target = new Vector3(0, 0, 0);
  private radius = 24;
  private theta = Math.PI / 4;
  private phi = Math.PI / 3;

  constructor(
    private readonly container: HTMLElement,
    private readonly callbacks: GraphCallbacks,
  ) {
    this.renderer = new WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(this.renderer.domElement);

    this.flat = new OrthographicCamera(-1, 1, 1, -1, 0.1, 4000);
    this.flat.position.set(0, 0, 100);
    this.flat.lookAt(0, 0, 0);
    this.solid = new PerspectiveCamera(50, 1, 0.1, 4000);
    this.solid.up.set(0, 0, 1);

    this.grid = new GridHelper(200, 200, 0x33383f, 0x22262b);
    this.grid.rotation.x = Math.PI / 2;
    this.scene.add(this.grid);
    this.scene.add(new AmbientLight(0xffffff, 1.6));
    const sun = new DirectionalLight(0xffffff, 1.1);
    sun.position.set(4, -6, 10);
    this.scene.add(sun);

    this.blocks = new InstancedMesh(
      new BoxGeometry(1, 1, 1),
      new MeshLambertMaterial({ transparent: true, opacity: 0.92 }),
      MAX_NODES,
    );
    this.blocks.count = 0;
    this.scene.add(this.blocks);

    this.wires = new LineSegments(
      new BufferGeometry(),
      new LineBasicMaterial({ vertexColors: true }),
    );
    this.scene.add(this.wires);

    this.labels = document.createElement("div");
    this.labels.className = "labels";
    container.appendChild(this.labels);

    this.bind();
    this.resize();
    this.loop();
  }

  setGraph(nodes: SceneNode[], edges: SceneEdge[]): void {
    this.nodes = nodes.slice(0, MAX_NODES);
    this.edges = edges;
    this.rebuild();
  }

  setSelection(ids: Iterable<string>): void {
    this.selection = new Set(ids);
    this.paint();
  }

  setMode(mode: ViewMode): void {
    if (mode === this.mode) return;
    this.mode = mode;
    this.grid.rotation.x = Math.PI / 2;
    this.resize();
  }

  /** Fit every block in view. */
  frameAll(): void {
    if (this.nodes.length === 0) return;
    const low = [Infinity, Infinity, Infinity];
    const high = [-Infinity, -Infinity, -Infinity];
    for (const node of this.nodes) {
      for (let axis = 0; axis < 3; axis += 1) {
        low[axis] = Math.min(low[axis]!, node.pos[axis]!);
        high[axis] = Math.max(high[axis]!, node.pos[axis]!);
      }
    }
    this.target.set(
      (low[0]! + high[0]!) / 2,
      (low[1]! + high[1]!) / 2,
      (low[2]! + high[2]!) / 2,
    );
    const span = Math.max(high[0]! - low[0]!, high[1]! - low[1]!, high[2]! - low[2]!, 4) + 6;
    this.zoom = Math.min(this.container.clientWidth, this.container.clientHeight) / span;
    this.radius = span * 1.6;
    this.resize();
  }

  dispose(): void {
    cancelAnimationFrame(this.frame);
    this.unbind();
    this.renderer.dispose();
    this.renderer.domElement.remove();
    this.labels.remove();
  }

  private get camera(): OrthographicCamera | PerspectiveCamera {
    return this.mode === "2d" ? this.flat : this.solid;
  }

  // --- interaction -------------------------------------------------------

  private readonly onPointerDown = (event: PointerEvent): void => {
    const hit = this.pick(event);
    if (hit === null) {
      if (event.button === 0) this.callbacks.onSelect(null, false);
      this.orbiting = {
        x: event.clientX,
        y: event.clientY,
        pan: this.mode === "2d" || event.button !== 0 || event.shiftKey,
      };
      this.renderer.domElement.setPointerCapture(event.pointerId);
      return;
    }
    this.callbacks.onSelect(hit.id, event.shiftKey);
    // Generated blocks belong to a generator; move the generator, not one of
    // its repetitions.
    if (event.button === 0 && !event.shiftKey && hit.kind !== "generated") {
      const world = this.toWorld(event, hit.pos, false);
      this.dragging = {
        id: hit.id,
        offset: new Vector3(hit.pos[0] - world.x, hit.pos[1] - world.y, 0),
        vertical: false,
      };
      this.renderer.domElement.setPointerCapture(event.pointerId);
    }
  };

  private readonly onPointerMove = (event: PointerEvent): void => {
    if (this.dragging) {
      const node = this.node(this.dragging.id);
      if (!node) return;
      // Alt lifts a block off the ground plane instead of sliding along it,
      // which is the only way to author a Z coordinate by hand.
      const vertical = event.altKey;
      const world = this.toWorld(event, node.pos, vertical);
      node.pos = vertical
        ? [node.pos[0], node.pos[1], snap(world.z)]
        : [
            snap(world.x + this.dragging.offset.x),
            snap(world.y + this.dragging.offset.y),
            node.pos[2],
          ];
      this.rebuild();
      return;
    }
    if (!this.orbiting) return;

    const dx = event.clientX - this.orbiting.x;
    const dy = event.clientY - this.orbiting.y;
    this.orbiting = { ...this.orbiting, x: event.clientX, y: event.clientY };
    if (this.orbiting.pan) {
      this.pan(dx, dy);
    } else {
      this.theta -= dx * 0.008;
      this.phi = Math.min(Math.PI - 0.05, Math.max(0.05, this.phi - dy * 0.008));
    }
  };

  private readonly onPointerUp = (event: PointerEvent): void => {
    if (this.dragging) {
      const node = this.node(this.dragging.id);
      // Placement is part of the model (ADR 0002), so a drag is an edit and
      // goes to the server as one command when the pointer comes up.
      if (node) this.callbacks.onMove(node.id, node.pos);
    }
    this.dragging = null;
    this.orbiting = null;
    if (this.renderer.domElement.hasPointerCapture(event.pointerId)) {
      this.renderer.domElement.releasePointerCapture(event.pointerId);
    }
  };

  private readonly onWheel = (event: WheelEvent): void => {
    event.preventDefault();
    const factor = Math.exp(-event.deltaY * 0.001);
    if (this.mode === "2d") this.zoom = clamp(this.zoom * factor, 4, 400);
    else this.radius = clamp(this.radius / factor, 2, 2000);
    this.resize();
  };

  private readonly onResize = (): void => this.resize();

  private pan(dx: number, dy: number): void {
    if (this.mode === "2d") {
      this.target.x -= dx / this.zoom;
      this.target.y += dy / this.zoom;
      return;
    }
    const right = new Vector3().setFromMatrixColumn(this.solid.matrix, 0);
    const up = new Vector3().setFromMatrixColumn(this.solid.matrix, 1);
    const scale = this.radius / this.container.clientHeight;
    this.target.addScaledVector(right, -dx * scale).addScaledVector(up, dy * scale);
  }

  private bind(): void {
    const canvas = this.renderer.domElement;
    canvas.addEventListener("pointerdown", this.onPointerDown);
    canvas.addEventListener("pointermove", this.onPointerMove);
    canvas.addEventListener("pointerup", this.onPointerUp);
    canvas.addEventListener("wheel", this.onWheel, { passive: false });
    canvas.addEventListener("contextmenu", preventDefault);
    window.addEventListener("resize", this.onResize);
  }

  private unbind(): void {
    const canvas = this.renderer.domElement;
    canvas.removeEventListener("pointerdown", this.onPointerDown);
    canvas.removeEventListener("pointermove", this.onPointerMove);
    canvas.removeEventListener("pointerup", this.onPointerUp);
    canvas.removeEventListener("wheel", this.onWheel);
    canvas.removeEventListener("contextmenu", preventDefault);
    window.removeEventListener("resize", this.onResize);
  }

  private pointer(event: PointerEvent): Vector2 {
    const rect = this.renderer.domElement.getBoundingClientRect();
    return new Vector2(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1,
    );
  }

  private pick(event: PointerEvent): SceneNode | null {
    this.raycaster.setFromCamera(this.pointer(event), this.camera);
    const hits = this.raycaster.intersectObject(this.blocks);
    const index = hits[0]?.instanceId;
    return index === undefined ? null : (this.nodes[index] ?? null);
  }

  private toWorld(
    event: PointerEvent,
    at: [number, number, number],
    vertical: boolean,
  ): Vector3 {
    this.raycaster.setFromCamera(this.pointer(event), this.camera);
    // Sliding uses the ground plane through the block; lifting uses an upright
    // plane facing the camera, so the pointer tracks the block either way.
    const normal = vertical
      ? new Vector3(this.camera.position.x - at[0], this.camera.position.y - at[1], 0).normalize()
      : new Vector3(0, 0, 1);
    if (!Number.isFinite(normal.x) || normal.lengthSq() === 0) normal.set(0, -1, 0);
    this.plane.setFromNormalAndCoplanarPoint(normal, new Vector3(at[0], at[1], at[2]));
    const point = new Vector3();
    return this.raycaster.ray.intersectPlane(this.plane, point) ?? new Vector3(...at);
  }

  private node(id: string): SceneNode | undefined {
    return this.nodes.find((n) => n.id === id);
  }

  // --- rendering ---------------------------------------------------------

  private rebuild(): void {
    this.blocks.count = this.nodes.length;
    const matrix = new Matrix4();
    this.nodes.forEach((node, index) => {
      this.dummy.position.set(node.pos[0], node.pos[1], node.pos[2]);
      this.dummy.scale.set(node.extent[0], node.extent[1], node.extent[2]);
      this.dummy.updateMatrix();
      this.blocks.setMatrixAt(index, matrix.copy(this.dummy.matrix));
    });
    this.blocks.instanceMatrix.needsUpdate = true;
    this.blocks.computeBoundingSphere();
    this.paint();
    this.wire();
    this.label();
  }

  private paint(): void {
    const colour = new Color();
    this.nodes.forEach((node, index) => {
      const status = STATUS_COLORS[node.status];
      const hex = this.selection.has(node.id)
        ? SELECTED
        : (status ?? BLOCK_COLORS[node.kind]);
      this.blocks.setColorAt(index, colour.setHex(hex));
    });
    if (this.blocks.instanceColor) this.blocks.instanceColor.needsUpdate = true;
  }

  private wire(): void {
    const positions: number[] = [];
    const colours: number[] = [];
    const colour = new Color();
    const byId = new Map(this.nodes.map((node) => [node.id, node]));
    for (const edge of this.edges) {
      const from = byId.get(edge.from);
      const to = byId.get(edge.to);
      if (!from || !to) continue;
      positions.push(...centre(from), ...centre(to));
      colour.setHex(LINK_COLORS[edge.kind]);
      colours.push(colour.r, colour.g, colour.b, colour.r, colour.g, colour.b);
    }
    this.wires.geometry.setAttribute(
      "position",
      new BufferAttribute(new Float32Array(positions), 3),
    );
    this.wires.geometry.setAttribute("color", new BufferAttribute(new Float32Array(colours), 3));
    this.wires.geometry.computeBoundingSphere();
  }

  private label(): void {
    while (this.labels.childElementCount > this.nodes.length) {
      this.labels.lastElementChild?.remove();
    }
    while (this.labels.childElementCount < this.nodes.length) {
      this.labels.appendChild(document.createElement("span"));
    }
    this.nodes.forEach((node, index) => {
      const element = this.labels.children[index] as HTMLSpanElement;
      element.textContent = node.label;
      element.dataset.node = node.id;
    });
    this.place();
  }

  private place(): void {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    const point = new Vector3();
    this.nodes.forEach((node, index) => {
      const element = this.labels.children[index] as HTMLSpanElement | undefined;
      if (!element) return;
      point.set(node.pos[0], node.pos[1], node.pos[2]).project(this.camera);
      element.style.transform = `translate(-50%, -50%) translate(${
        ((point.x + 1) / 2) * width
      }px, ${((1 - point.y) / 2) * height}px)`;
      const visible = Math.abs(point.x) < 1.2 && point.z < 1;
      element.style.visibility = visible ? "visible" : "hidden";
    });
  }

  private resize(): void {
    const width = this.container.clientWidth || 1;
    const height = this.container.clientHeight || 1;
    this.renderer.setSize(width, height, false);

    this.flat.left = -width / (2 * this.zoom);
    this.flat.right = width / (2 * this.zoom);
    this.flat.top = height / (2 * this.zoom);
    this.flat.bottom = -height / (2 * this.zoom);
    this.flat.position.set(this.target.x, this.target.y, 100);
    this.flat.updateProjectionMatrix();

    this.solid.aspect = width / height;
    this.solid.updateProjectionMatrix();
  }

  private loop = (): void => {
    this.frame = requestAnimationFrame(this.loop);
    if (this.mode === "2d") {
      this.flat.position.set(this.target.x, this.target.y, 100);
    } else {
      const sin = Math.sin(this.phi);
      this.solid.position.set(
        this.target.x + this.radius * sin * Math.cos(this.theta),
        this.target.y + this.radius * sin * Math.sin(this.theta),
        this.target.z + this.radius * Math.cos(this.phi),
      );
      this.solid.lookAt(this.target);
      this.solid.updateMatrix();
    }
    this.place();
    this.renderer.render(this.scene, this.camera);
  };
}

function centre(node: SceneNode): [number, number, number] {
  return [node.pos[0], node.pos[1], node.pos[2]];
}

function snap(value: number): number {
  return Math.round(value / GRID_SNAP) * GRID_SNAP;
}

function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value));
}

function preventDefault(event: Event): void {
  event.preventDefault();
}
