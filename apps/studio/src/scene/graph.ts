// The canvas. One three.js scene for both 2D and 3D (ADR 0006): 2D is this
// scene seen through an orthographic camera with rotation switched off, so
// picking, dragging and selection have a single implementation.
//
// Nodes are one InstancedMesh, edges one LineSegments, labels plain DOM over
// the canvas. React never touches any of it: the scene owns its own frame loop.

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
  MeshBasicMaterial,
  Object3D,
  OrthographicCamera,
  Plane,
  Raycaster,
  Scene,
  Vector2,
  Vector3,
  WebGLRenderer,
} from "three";

export type NodeStatus = "ok" | "warning" | "error";

export interface SceneNode {
  id: string;
  label: string;
  pos: [number, number, number];
  extent: [number, number, number];
  status: NodeStatus;
}

export interface SceneEdge {
  id: string;
  from: string;
  to: string;
}

export interface GraphCallbacks {
  onSelect(nodeId: string | null, additive: boolean): void;
  onMove(nodeId: string, pos: [number, number, number]): void;
}

const COLORS: Record<NodeStatus, number> = {
  ok: 0x4c6ef5,
  warning: 0xf08c00,
  error: 0xe03131,
};
const SELECTED = 0xffd43b;
const GRID_SNAP = 0.25;
const MAX_NODES = 20000;

export class GraphView {
  private readonly renderer: WebGLRenderer;
  private readonly scene = new Scene();
  private readonly camera: OrthographicCamera;
  private readonly raycaster = new Raycaster();
  private readonly blocks: InstancedMesh;
  private readonly wires: LineSegments;
  private readonly labels: HTMLDivElement;
  private readonly dummy = new Object3D();
  private readonly plane = new Plane(new Vector3(0, 0, 1), 0);

  private nodes: SceneNode[] = [];
  private edges: SceneEdge[] = [];
  private selection = new Set<string>();
  private dragging: { id: string; offset: Vector3 } | null = null;
  private panning: { x: number; y: number } | null = null;
  private frame = 0;
  private zoom = 40;

  constructor(
    private readonly container: HTMLElement,
    private readonly callbacks: GraphCallbacks,
  ) {
    this.renderer = new WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(this.renderer.domElement);

    this.camera = new OrthographicCamera(-1, 1, 1, -1, 0.1, 2000);
    this.camera.position.set(0, 0, 100);
    this.camera.lookAt(0, 0, 0);

    const grid = new GridHelper(200, 200, 0x33383f, 0x22262b);
    grid.rotation.x = Math.PI / 2;
    this.scene.add(grid);

    this.blocks = new InstancedMesh(
      new BoxGeometry(1, 1, 1),
      new MeshBasicMaterial({ transparent: true, opacity: 0.92 }),
      MAX_NODES,
    );
    this.blocks.count = 0;
    this.scene.add(this.blocks);

    this.wires = new LineSegments(
      new BufferGeometry(),
      new LineBasicMaterial({ color: 0x8d99ae }),
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

  /** Fit every node in view. Called on open and on demand. */
  frameAll(): void {
    if (this.nodes.length === 0) return;
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (const node of this.nodes) {
      minX = Math.min(minX, node.pos[0]);
      maxX = Math.max(maxX, node.pos[0]);
      minY = Math.min(minY, node.pos[1]);
      maxY = Math.max(maxY, node.pos[1]);
    }
    this.camera.position.x = (minX + maxX) / 2;
    this.camera.position.y = (minY + maxY) / 2;
    const span = Math.max(maxX - minX, maxY - minY, 4) + 6;
    this.zoom = Math.min(this.container.clientWidth, this.container.clientHeight) / span;
    this.resize();
  }

  dispose(): void {
    cancelAnimationFrame(this.frame);
    this.unbind();
    this.renderer.dispose();
    this.renderer.domElement.remove();
    this.labels.remove();
  }

  // --- interaction -------------------------------------------------------

  private readonly onPointerDown = (event: PointerEvent): void => {
    const hit = this.pick(event);
    if (hit === null) {
      if (event.button === 0) this.callbacks.onSelect(null, false);
      this.panning = { x: event.clientX, y: event.clientY };
      this.renderer.domElement.setPointerCapture(event.pointerId);
      return;
    }
    this.callbacks.onSelect(hit.id, event.shiftKey);
    if (event.button === 0 && !event.shiftKey) {
      const world = this.toWorld(event, hit.pos[2]);
      this.dragging = {
        id: hit.id,
        offset: new Vector3(hit.pos[0] - world.x, hit.pos[1] - world.y, 0),
      };
      this.renderer.domElement.setPointerCapture(event.pointerId);
    }
  };

  private readonly onPointerMove = (event: PointerEvent): void => {
    if (this.dragging) {
      const node = this.node(this.dragging.id);
      if (!node) return;
      const world = this.toWorld(event, node.pos[2]);
      node.pos = [
        snap(world.x + this.dragging.offset.x),
        snap(world.y + this.dragging.offset.y),
        node.pos[2],
      ];
      this.rebuild();
      return;
    }
    if (this.panning) {
      this.camera.position.x -= (event.clientX - this.panning.x) / this.zoom;
      this.camera.position.y += (event.clientY - this.panning.y) / this.zoom;
      this.panning = { x: event.clientX, y: event.clientY };
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
    this.panning = null;
    if (this.renderer.domElement.hasPointerCapture(event.pointerId)) {
      this.renderer.domElement.releasePointerCapture(event.pointerId);
    }
  };

  private readonly onWheel = (event: WheelEvent): void => {
    event.preventDefault();
    const factor = Math.exp(-event.deltaY * 0.001);
    this.zoom = Math.min(400, Math.max(4, this.zoom * factor));
    this.resize();
  };

  private readonly onResize = (): void => this.resize();

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

  private pick(event: PointerEvent): SceneNode | null {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const pointer = new Vector2(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1,
    );
    this.raycaster.setFromCamera(pointer, this.camera);
    const hits = this.raycaster.intersectObject(this.blocks);
    const index = hits[0]?.instanceId;
    return index === undefined ? null : (this.nodes[index] ?? null);
  }

  private toWorld(event: PointerEvent, z: number): Vector3 {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const pointer = new Vector2(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1,
    );
    this.raycaster.setFromCamera(pointer, this.camera);
    this.plane.constant = -z;
    const point = new Vector3();
    this.raycaster.ray.intersectPlane(this.plane, point);
    return point;
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
      const selected = this.selection.has(node.id);
      this.blocks.setColorAt(index, colour.setHex(selected ? SELECTED : COLORS[node.status]));
    });
    if (this.blocks.instanceColor) this.blocks.instanceColor.needsUpdate = true;
  }

  private wire(): void {
    const positions: number[] = [];
    const byId = new Map(this.nodes.map((node) => [node.id, node]));
    for (const edge of this.edges) {
      const from = byId.get(edge.from);
      const to = byId.get(edge.to);
      if (!from || !to) continue;
      positions.push(
        from.pos[0] + from.extent[0] / 2,
        from.pos[1],
        from.pos[2],
        to.pos[0] - to.extent[0] / 2,
        to.pos[1],
        to.pos[2],
      );
    }
    this.wires.geometry.setAttribute(
      "position",
      new BufferAttribute(new Float32Array(positions), 3),
    );
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
      element.style.visibility = Math.abs(point.x) < 1.2 ? "visible" : "hidden";
    });
  }

  private resize(): void {
    const width = this.container.clientWidth || 1;
    const height = this.container.clientHeight || 1;
    this.renderer.setSize(width, height, false);
    this.camera.left = -width / (2 * this.zoom);
    this.camera.right = width / (2 * this.zoom);
    this.camera.top = height / (2 * this.zoom);
    this.camera.bottom = -height / (2 * this.zoom);
    this.camera.updateProjectionMatrix();
  }

  private loop = (): void => {
    this.frame = requestAnimationFrame(this.loop);
    this.place();
    this.renderer.render(this.scene, this.camera);
  };
}

function snap(value: number): number {
  return Math.round(value / GRID_SNAP) * GRID_SNAP;
}

function preventDefault(event: Event): void {
  event.preventDefault();
}
