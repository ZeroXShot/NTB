// React owns the panels; the scene owns the canvas. This component only feeds
// the scene the current graph and hands its events back up.

import { useEffect, useRef } from "react";
import type { JSX } from "react";
import { GraphView, type SceneEdge, type SceneNode, type ViewMode } from "./graph";

interface Props {
  nodes: SceneNode[];
  edges: SceneEdge[];
  selection: string[];
  mode: ViewMode;
  onSelect: (nodeId: string | null, additive: boolean) => void;
  onMove: (nodeId: string, pos: [number, number, number]) => void;
}

export function Canvas({ nodes, edges, selection, mode, onSelect, onMove }: Props): JSX.Element {
  const host = useRef<HTMLDivElement>(null);
  const view = useRef<GraphView | null>(null);
  const framed = useRef(false);
  const handlers = useRef({ onSelect, onMove });
  handlers.current = { onSelect, onMove };

  useEffect(() => {
    if (!host.current) return undefined;
    const graph = new GraphView(host.current, {
      onSelect: (id, additive) => handlers.current.onSelect(id, additive),
      onMove: (id, pos) => handlers.current.onMove(id, pos),
    });
    view.current = graph;
    return () => {
      graph.dispose();
      view.current = null;
    };
  }, []);

  useEffect(() => {
    view.current?.setGraph(nodes, edges);
    view.current?.setSelection(selection);
    if (!framed.current && nodes.length > 0) {
      view.current?.frameAll();
      framed.current = true;
    }
  }, [nodes, edges, selection]);

  useEffect(() => {
    view.current?.setMode(mode);
  }, [mode]);

  return <div className="canvas" ref={host} />;
}
