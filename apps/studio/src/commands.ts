// Command builders. Every edit in the UI becomes one of these and goes to the
// server; nothing mutates the local document.

import type { Endpoint, Generator, SpatialRule } from "./ir.gen";
import type { Command, Module, Node, OpInfo } from "./types";
import { freshEdgeId, freshNodeId } from "./store";

export function addNode(
  module: Module,
  op: OpInfo,
  pos: [number, number, number],
): Command {
  const node: Node = {
    id: freshNodeId(module, op.name),
    op: op.name,
    attrs: defaults(op),
    placement: { pos, extent: [2, 1, 1] },
  };
  return { kind: "add_node", module: module.id, node: node as unknown as Record<string, unknown> };
}

export function moveNode(module: Module, nodeId: string, pos: [number, number, number]): Command {
  return { kind: "move_node", module: module.id, node: nodeId, placement: { pos } };
}

export function removeNode(module: Module, nodeId: string): Command {
  return { kind: "remove_node", module: module.id, node: nodeId };
}

export function setAttrs(
  module: Module,
  nodeId: string,
  attrs: Record<string, unknown>,
): Command {
  return { kind: "set_attrs", module: module.id, node: nodeId, attrs };
}

export function renameNode(module: Module, nodeId: string, name: string): Command {
  return { kind: "rename_node", module: module.id, node: nodeId, name };
}

/** Connect two nodes on the first output and the first free input. */
export function connectNodes(
  module: Module,
  from: Node,
  to: Node,
  ops: Map<string, OpInfo>,
): Command | string {
  const source = ops.get(from.op);
  const target = ops.get(to.op);
  if (!source || !target) return "one of these nodes is not a known op";
  const out = source.outputs[0];
  if (!out) return `${from.op} produces nothing to connect`;
  const taken = new Set(
    (module.edges ?? [])
      .filter((edge) => edge.dst.node === to.id)
      .map((edge) => edge.dst.port ?? "in"),
  );
  const free = target.inputs.find((port) => port.variadic || !taken.has(port.name));
  if (!free) return `every input of ${to.id} is already connected`;
  return {
    kind: "connect",
    module: module.id,
    edge: {
      id: freshEdgeId(module, from.id, to.id),
      src: { node: from.id, port: out.name },
      dst: { node: to.id, port: free.name },
    },
  };
}

export function disconnect(module: Module, edgeId: string): Command {
  return { kind: "disconnect", module: module.id, edge: edgeId };
}

export function updateGenerator(module: Module, generator: Generator): Command {
  return { kind: "update_generator", module: module.id, generator: generator as unknown as Record<string, unknown> };
}

export function addGenerator(module: Module, generator: Generator): Command {
  return { kind: "add_generator", module: module.id, generator: generator as unknown as Record<string, unknown> };
}

export function removeGenerator(module: Module, id: string): Command {
  return { kind: "remove_generator", module: module.id, generator: id };
}

export function updateRule(module: Module, rule: SpatialRule): Command {
  return { kind: "update_rule", module: module.id, rule: rule as unknown as Record<string, unknown> };
}

export function addRule(module: Module, rule: SpatialRule): Command {
  return { kind: "add_rule", module: module.id, rule: rule as unknown as Record<string, unknown> };
}

export function removeRule(module: Module, id: string): Command {
  return { kind: "remove_rule", module: module.id, rule: id };
}

export function bindPort(
  module: Module,
  direction: "in" | "out",
  port: string,
  endpoint: Endpoint | null,
): Command {
  return { kind: "bind_port", module: module.id, direction, port, endpoint };
}

function defaults(op: OpInfo): Record<string, unknown> {
  const attrs: Record<string, unknown> = {};
  for (const attr of op.attrs) {
    if (attr.default !== null && attr.default !== undefined) attrs[attr.name] = attr.default;
  }
  return attrs;
}
