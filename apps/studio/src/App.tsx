// The studio window: palette, canvas, inspector, problems and generated code.
//
// Every edit here is a command sent to the server, and every pixel is drawn
// from the snapshot it sends back.

import { useCallback, useEffect, useMemo } from "react";
import type { JSX } from "react";
import {
  addNode,
  connectNodes,
  moveNode,
  removeNode,
  renameNode,
  setAttrs,
} from "./commands";
import { Inspector } from "./panels/Inspector";
import { Output } from "./panels/Output";
import { Palette } from "./panels/Palette";
import { Canvas } from "./scene/Canvas";
import type { SceneEdge, SceneNode } from "./scene/graph";
import { findNode, rootModule, useStudio } from "./store";
import type { OpInfo } from "./types";

export function App(): JSX.Element {
  const {
    snapshot,
    ops,
    connected,
    error,
    selection,
    linkFrom,
    connect,
    send,
    run,
    select,
    clearError,
    setLinkFrom,
  } = useStudio();

  useEffect(connect, [connect]);

  const module = rootModule(snapshot);
  const opsByName = useMemo(() => new Map(ops.map((op) => [op.name, op])), [ops]);
  const selected = findNode(snapshot, selection[0] ?? null);

  const status = useMemo(() => {
    const worst = new Map<string, "warning" | "error">();
    for (const diagnostic of snapshot?.derived.diagnostics ?? []) {
      if (!diagnostic.node) continue;
      if (diagnostic.severity === "error") worst.set(diagnostic.node, "error");
      else if (diagnostic.severity === "warning" && !worst.has(diagnostic.node)) {
        worst.set(diagnostic.node, "warning");
      }
    }
    return worst;
  }, [snapshot]);

  const nodes: SceneNode[] = useMemo(
    () =>
      (module?.nodes ?? []).map((node) => ({
        id: node.id,
        label: node.name || node.id,
        pos: node.placement?.pos ?? [0, 0, 0],
        extent: node.placement?.extent ?? [1, 1, 1],
        status: status.get(node.id) ?? "ok",
      })),
    [module, status],
  );

  const edges: SceneEdge[] = useMemo(
    () => (module?.edges ?? []).map((edge) => ({ id: edge.id, from: edge.src.node, to: edge.dst.node })),
    [module],
  );

  const place = useCallback(
    (op: OpInfo) => {
      if (!module) return;
      const count = module.nodes?.length ?? 0;
      run(addNode(module, op, [count * 3, 0, 0]));
    },
    [module, run],
  );

  const onSelect = useCallback(
    (nodeId: string | null, additive: boolean) => {
      if (nodeId && linkFrom && linkFrom !== nodeId && module) {
        const from = findNode(snapshot, linkFrom);
        const to = findNode(snapshot, nodeId);
        setLinkFrom(null);
        if (from && to) {
          const command = connectNodes(module, from, to, opsByName);
          if (typeof command === "string") useStudio.setState({ error: command });
          else run(command);
        }
        return;
      }
      select(nodeId, additive);
    },
    [linkFrom, module, opsByName, run, select, setLinkFrom, snapshot],
  );

  const remove = useCallback(() => {
    if (module && selection.length > 0) {
      for (const id of selection) run(removeNode(module, id));
    }
  }, [module, run, selection]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)) return;
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
        event.preventDefault();
        send({ type: event.shiftKey ? "redo" : "undo" });
      } else if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
        event.preventDefault();
        save();
      } else if (event.key === "Delete" || event.key === "Backspace") {
        remove();
      } else if (event.key.toLowerCase() === "c" && selection[0]) {
        setLinkFrom(selection[0]);
      } else if (event.key === "Escape") {
        setLinkFrom(null);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  function save(): void {
    if (snapshot?.path) send({ type: "save" });
    else saveAs();
  }

  function saveAs(): void {
    const path = window.prompt("Save the document as", snapshot?.path ?? "model.ntb");
    if (path) send({ type: "save", path });
  }

  function open(): void {
    const path = window.prompt("Open a .ntb file", snapshot?.path ?? "examples/mlp.ntb");
    if (path) send({ type: "open", path });
  }

  return (
    <div className="studio">
      <header className="toolbar">
        <strong>NTB Studio</strong>
        <span className="file">
          {snapshot?.path ?? "untitled"}
          {snapshot?.dirty && <em>•</em>}
        </span>
        <button onClick={() => send({ type: "new" })}>New</button>
        <button onClick={open}>Open</button>
        <button onClick={save}>Save</button>
        <button onClick={saveAs}>Save as</button>
        <span className="spacer" />
        <button disabled={!snapshot?.canUndo} onClick={() => send({ type: "undo" })}>
          Undo
        </button>
        <button disabled={!snapshot?.canRedo} onClick={() => send({ type: "redo" })}>
          Redo
        </button>
        <span className="spacer" />
        <span className={connected ? "link on" : "link"}>{connected ? "live" : "offline"}</span>
      </header>

      <Palette ops={ops} onPick={place} />

      <main className="stage">
        <Canvas
          nodes={nodes}
          edges={edges}
          selection={selection}
          onSelect={onSelect}
          onMove={(id, pos) => module && run(moveNode(module, id, pos))}
        />
        <div className="hint">
          {linkFrom
            ? `connecting from ${linkFrom} — click the target block, escape to cancel`
            : "drag to move · c to start a connection · del to remove"}
        </div>
      </main>

      <Inspector
        node={selected}
        op={selected ? opsByName.get(selected.op) : undefined}
        snapshot={snapshot}
        onRename={(name) => module && selected && run(renameNode(module, selected.id, name))}
        onAttrs={(attrs) => module && selected && run(setAttrs(module, selected.id, attrs))}
        onDelete={remove}
      />

      <Output snapshot={snapshot} onSelect={(nodeId) => select(nodeId, false)} />

      {error && (
        <div className="toast" role="alert" onClick={clearError}>
          {error}
        </div>
      )}
    </div>
  );
}
