// The studio window: palette, canvas, inspector, problems and generated code.
//
// Every edit here is a command sent to the server, and every pixel is drawn
// from the snapshot it sends back.

import { useCallback, useEffect, useMemo, useState } from "react";
import type { JSX } from "react";
import {
  addGenerator,
  bindPort,
  addNode,
  addRule,
  connectNodes,
  moveNode,
  removeGenerator,
  removeNode,
  removeRule,
  renameNode,
  setAttrs,
  updateGenerator,
  updateRule,
} from "./commands";
import { Inspector } from "./panels/Inspector";
import { Output } from "./panels/Output";
import { Palette } from "./panels/Palette";
import { Runs } from "./panels/Runs";
import { Spatial } from "./panels/Spatial";
import { Canvas } from "./scene/Canvas";
import type { SceneEdge, SceneNode, ViewMode } from "./scene/graph";
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
    runs,
    curves,
    refreshRuns,
    startRun,
    actOnRun,
  } = useStudio();
  const [mode, setMode] = useState<ViewMode>("2d");

  useEffect(connect, [connect]);

  const module = rootModule(snapshot);
  const opsByName = useMemo(() => new Map(ops.map((op) => [op.name, op])), [ops]);
  const selected = findNode(snapshot, selection[0] ?? null);

  const status = useMemo(() => {
    const worst = new Map<string, "warning" | "error">();
    for (const diagnostic of snapshot?.derived.diagnostics ?? []) {
      // Colour the block the problem is inside, which for a generated cell is
      // the repetition, not the module it was stamped from.
      const key = diagnostic.block ?? diagnostic.node;
      if (!key) continue;
      if (diagnostic.severity === "error") worst.set(key, "error");
      else if (diagnostic.severity === "warning" && !worst.has(key)) worst.set(key, "warning");
    }
    return worst;
  }, [snapshot]);

  // The scene draws the server's spatial preview, not the raw node list: a
  // generator's repetitions and a rule's edges exist only there.
  const nodes: SceneNode[] = useMemo(
    () =>
      (snapshot?.derived.blocks ?? []).map((block) => ({
        id: block.key,
        label: block.label,
        pos: block.pos,
        extent: block.extent,
        status: status.get(block.key) ?? "ok",
        kind: block.kind,
      })),
    [snapshot, status],
  );

  const edges: SceneEdge[] = useMemo(
    () =>
      (snapshot?.derived.links ?? []).map((link, index) => ({
        id: `${link.kind}-${index}`,
        from: link.src,
        to: link.dst,
        kind: link.kind,
      })),
    [snapshot],
  );

  const derivedCount = edges.filter((edge) => edge.kind !== "edge").length;

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
        <div className="modes">
          {(["2d", "3d"] as const).map((option) => (
            <button
              key={option}
              className={mode === option ? "on" : ""}
              onClick={() => setMode(option)}
            >
              {option.toUpperCase()}
            </button>
          ))}
        </div>
        <span className={connected ? "link on" : "link"}>{connected ? "live" : "offline"}</span>
      </header>

      <Palette ops={ops} onPick={place} />

      <main className="stage">
        <Canvas
          nodes={nodes}
          edges={edges}
          selection={selection}
          mode={mode}
          onSelect={onSelect}
          onMove={(id, pos) => module && run(moveNode(module, id, pos))}
        />
        <div className="hint">
          {linkFrom
            ? `connecting from ${linkFrom} — click the target block, escape to cancel`
            : mode === "2d"
              ? "drag to move · c to connect · del to remove"
              : "drag the background to orbit · alt-drag a block to lift it along Z"}
        </div>
        <div className="legend">
          <span className="swatch edge" /> drawn
          <span className="swatch rule" /> from a rule
          <span className="swatch chain" /> from a generator
          {derivedCount > 0 && <em>{derivedCount} edges nobody drew</em>}
        </div>
      </main>

      <Inspector
        node={selected}
        block={(snapshot?.derived.blocks ?? []).find((b) => b.key === selection[0])}
        op={selected ? opsByName.get(selected.op) : undefined}
        snapshot={snapshot}
        onRename={(name) => module && selected && run(renameNode(module, selected.id, name))}
        onAttrs={(attrs) => module && selected && run(setAttrs(module, selected.id, attrs))}
        onMove={(pos) => module && selected && run(moveNode(module, selected.id, pos))}
        onDelete={remove}
      />

      <Output
        snapshot={snapshot}
        onSelect={(nodeId) => select(nodeId, false)}
        spatial={
          <Spatial
            module={module}
            modules={snapshot?.document.modules ?? []}
            onGenerator={(generator) => module && run(updateGenerator(module, generator))}
            onRule={(rule) => module && run(updateRule(module, rule))}
            onAddGenerator={(generator) => module && run(addGenerator(module, generator))}
            onAddRule={(rule) => module && run(addRule(module, rule))}
            onRemoveGenerator={(id) => module && run(removeGenerator(module, id))}
            onRemoveRule={(id) => module && run(removeRule(module, id))}
            onBind={(direction, port, endpoint) =>
              module && run(bindPort(module, direction, port, endpoint))
            }
          />
        }
        runs={
          <Runs
            runs={runs}
            curves={curves}
            savedPath={snapshot?.path ?? null}
            onStart={startRun}
            onStop={(id) => actOnRun(id, "stop")}
            onResume={(id) => actOnRun(id, "resume")}
            onRefresh={refreshRuns}
          />
        }
      />

      {error && (
        <div className="toast" role="alert" onClick={clearError}>
          {error}
        </div>
      )}
    </div>
  );
}
