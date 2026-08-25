// The pure half of the store: what survives a snapshot, and how ids are picked.
//
// Anything needing a GL context is not tested here. That arrives with the scene
// refactor, where there is something worth testing that a counter can measure.

import { describe, expect, it } from "vitest";
import { blockKeys, freshEdgeId, freshNodeId, rootModule } from "./store";
import type { Module, Snapshot } from "./types";

function snapshot(blocks: { key: string }[], nodes: { id: string }[] = []): Snapshot {
  return {
    revision: 1,
    document: { root: "m", modules: [{ id: "m", nodes }] },
    path: null,
    dirty: false,
    canUndo: false,
    canRedo: false,
    derived: { blocks, links: [], diagnostics: [], types: {}, code: "", codeError: "" },
  } as unknown as Snapshot;
}

describe("blockKeys", () => {
  it("keeps a generator's repetitions, which are not authored nodes", () => {
    // The bug: `stack-0` is selectable and is not in module.nodes, so filtering
    // the selection against the node list dropped it on the next broadcast.
    const state = snapshot([{ key: "fc" }, { key: "stack-0" }, { key: "stack-1" }], [{ id: "fc" }]);
    expect(blockKeys(state).has("stack-0")).toBe(true);
    expect(blockKeys(state).size).toBe(3);
  });

  it("is empty rather than undefined for a session with no snapshot", () => {
    expect(blockKeys(null).size).toBe(0);
  });
});

describe("rootModule", () => {
  it("finds the module the document points at", () => {
    expect(rootModule(snapshot([], [{ id: "a" }]))?.id).toBe("m");
  });

  it("is undefined without a snapshot", () => {
    expect(rootModule(null)).toBeUndefined();
  });
});

describe("freshNodeId", () => {
  const module = (ids: string[]): Module =>
    ({ id: "m", nodes: ids.map((id) => ({ id })) }) as unknown as Module;

  it("reads as the op it came from", () => {
    expect(freshNodeId(module([]), "ntb.conv2d")).toBe("conv2d");
  });

  it("counts up rather than colliding", () => {
    expect(freshNodeId(module(["conv2d"]), "ntb.conv2d")).toBe("conv2d2");
    expect(freshNodeId(module(["conv2d", "conv2d2"]), "ntb.conv2d")).toBe("conv2d3");
  });

  it("keeps a plugin's namespace, since only ntb. is stripped", () => {
    expect(freshNodeId(module([]), "example.softsign")).toBe("example_softsign");
  });
});

describe("freshEdgeId", () => {
  const module = (ids: string[]): Module =>
    ({ id: "m", edges: ids.map((id) => ({ id })) }) as unknown as Module;

  it("names an edge after both ends", () => {
    expect(freshEdgeId(module([]), "a", "b")).toBe("a__b");
  });

  it("counts up when the same pair is joined twice", () => {
    expect(freshEdgeId(module(["a__b"]), "a", "b")).toBe("a__b_2");
  });
});
