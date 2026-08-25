// The client mirror of the server's session.
//
// The client is never authoritative: it sends commands and redraws whatever
// snapshot comes back. That is what keeps undo, validation and the generated
// code consistent with the document the server would save.

import { create } from "zustand";
import type { Command, Module, Node, OpInfo, RunRecord, Snapshot } from "./types";

const RECONNECT_MS = 1500;

interface Studio {
  snapshot: Snapshot | null;
  ops: OpInfo[];
  connected: boolean;
  error: string | null;
  selection: string[];
  linkFrom: string | null;
  runs: RunRecord[];
  /** Loss values per run, appended as the worker reports them. */
  curves: Record<string, number[]>;
  refreshRuns: () => void;
  startRun: (config: Record<string, unknown>) => void;
  actOnRun: (runId: string, action: "stop" | "resume") => void;
  connect: () => void;
  send: (message: Record<string, unknown>) => void;
  run: (command: Command) => void;
  select: (nodeId: string | null, additive: boolean) => void;
  clearError: () => void;
  setLinkFrom: (nodeId: string | null) => void;
}

let socket: WebSocket | null = null;

export const useStudio = create<Studio>((set, get) => ({
  snapshot: null,
  ops: [],
  connected: false,
  error: null,
  selection: [],
  linkFrom: null,
  runs: [],
  curves: {},

  refreshRuns: () => {
    void fetch("api/runs")
      .then((response) => response.json())
      .then((runs: RunRecord[]) => set({ runs }))
      .catch(() => undefined);
  },

  startRun: (config) => {
    void fetch("api/runs", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(config),
    })
      .then(async (response) => {
        if (!response.ok) throw new Error((await response.json()).detail ?? "could not start");
        get().refreshRuns();
      })
      .catch((reason: Error) => set({ error: reason.message }));
  },

  actOnRun: (runId, action) => {
    void fetch(`api/runs/${runId}/${action}`, { method: "POST" })
      .then(async (response) => {
        if (!response.ok) throw new Error((await response.json()).detail ?? action + " failed");
        get().refreshRuns();
      })
      .catch((reason: Error) => set({ error: reason.message }));
  },

  connect: () => {
    void fetch("api/ops")
      .then((response) => response.json())
      .then((ops: OpInfo[]) => set({ ops }))
      .catch(() => set({ error: "could not read the op catalogue" }));
    open(set, get);
  },

  send: (message) => {
    if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(message));
    else set({ error: "not connected to the studio server" });
  },

  run: (command) => get().send({ type: "command", command }),

  select: (nodeId, additive) =>
    set((state) => {
      if (nodeId === null) return { selection: [] };
      if (!additive) return { selection: [nodeId] };
      return state.selection.includes(nodeId)
        ? { selection: state.selection.filter((id) => id !== nodeId) }
        : { selection: [...state.selection, nodeId] };
    }),

  clearError: () => set({ error: null }),
  setLinkFrom: (nodeId) => set({ linkFrom: nodeId }),
}));

function open(set: (partial: Partial<Studio>) => void, get: () => Studio): void {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  socket = new WebSocket(`${protocol}//${window.location.host}/ws`);

  socket.onopen = () => set({ connected: true });
  socket.onclose = () => {
    set({ connected: false });
    window.setTimeout(() => open(set, get), RECONNECT_MS);
  };
  socket.onmessage = (event: MessageEvent<string>) => {
    const message = JSON.parse(event.data) as { type: string } & Record<string, unknown>;
    if (message.type === "state") {
      const snapshot = message as unknown as Snapshot;
      const alive = blockKeys(snapshot);
      set({
        snapshot,
        selection: get().selection.filter((id) => alive.has(id)),
        linkFrom: get().linkFrom && alive.has(get().linkFrom as string) ? get().linkFrom : null,
      });
    } else if (message.type === "run") {
      const runId = String(message.runId);
      if (message.event === "metric") {
        const curves = { ...get().curves };
        curves[runId] = [...(curves[runId] ?? []), Number(message.value)];
        set({ curves });
      }
      // Anything else changed the run's own state; ask for the new record.
      if (message.event !== "metric") get().refreshRuns();
    } else if (message.type === "error") {
      set({ error: String(message.message) });
    }
  };
}

/**
 * Everything on the canvas that can be selected, by key.
 *
 * Block keys, not node ids. A generator's repetition (`stack-0`) is a block and
 * is not an authored node, so filtering a selection against the node list
 * dropped it on the very next broadcast.
 */
export function blockKeys(snapshot: Snapshot | null): Set<string> {
  return new Set((snapshot?.derived?.blocks ?? []).map((block) => block.key));
}

export function rootModule(snapshot: Snapshot | null): Module | undefined {
  if (!snapshot) return undefined;
  return snapshot.document.modules?.find((module) => module.id === snapshot.document.root);
}

export function findNode(snapshot: Snapshot | null, nodeId: string | null): Node | undefined {
  if (!nodeId) return undefined;
  return rootModule(snapshot)?.nodes?.find((node) => node.id === nodeId);
}

/** A node id nobody is using yet, derived from the op name so it reads well. */
export function freshNodeId(module: Module | undefined, op: string): string {
  const stem = op.replace(/^ntb\./, "").replace(/[^A-Za-z0-9_]/g, "_");
  const taken = new Set(module?.nodes?.map((node) => node.id) ?? []);
  if (!taken.has(stem)) return stem;
  for (let index = 2; ; index += 1) {
    if (!taken.has(`${stem}${index}`)) return `${stem}${index}`;
  }
}

export function freshEdgeId(module: Module | undefined, from: string, to: string): string {
  const stem = `${from}__${to}`;
  const taken = new Set(module?.edges?.map((edge) => edge.id) ?? []);
  if (!taken.has(stem)) return stem;
  for (let index = 2; ; index += 1) {
    if (!taken.has(`${stem}_${index}`)) return `${stem}_${index}`;
  }
}
