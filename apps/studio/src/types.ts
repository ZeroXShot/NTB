// What the server sends. The IR half is generated from the JSON Schema; only
// the session envelope is declared here.

import type { Document, Edge, Module, Node } from "./ir.gen";

export type { Document, Edge, Module, Node };

export interface Diagnostic {
  code: string;
  severity: "error" | "warning" | "info";
  message: string;
  module: string | null;
  node: string | null;
  port: string | null;
  edge: string | null;
  /** The root-module block this belongs to: which repetition, not just which op. */
  block: string | null;
  path: string | null;
  text: string;
}

export interface Block {
  key: string;
  label: string;
  op: string;
  kind: "node" | "instance" | "generated";
  pos: [number, number, number];
  extent: [number, number, number];
  source: string;
  index: number | null;
}

export interface Link {
  src: string;
  dst: string;
  kind: "edge" | "rule" | "chain";
  source: string;
}

export interface Derived {
  diagnostics: Diagnostic[];
  types: Record<string, string>;
  code: string;
  codeError: string;
  /** What the root module looks like in space, generated blocks included. */
  blocks: Block[];
  links: Link[];
  layoutProblems: string[];
}

export interface Snapshot {
  revision: number;
  document: Document;
  path: string | null;
  dirty: boolean;
  canUndo: boolean;
  canRedo: boolean;
  derived: Derived;
}

export interface AttrInfo {
  name: string;
  type: string;
  doc: string;
  default: unknown;
  required: boolean;
  minimum: number | null;
  choices: unknown[] | null;
  default_from: string | null;
}

export interface PortInfo {
  name: string;
  doc: string;
  optional: boolean;
  variadic: boolean;
}

export interface OpInfo {
  name: string;
  category: string;
  summary: string;
  doc: string;
  inputs: PortInfo[];
  outputs: PortInfo[];
  attrs: AttrInfo[];
  backends: string[];
}

export interface RunRecord {
  id: string;
  document: string;
  status: "running" | "done" | "failed" | "stopped";
  config: Record<string, unknown>;
  startedAt: number;
  endedAt: number | null;
  error: string | null;
  parameters: number | null;
  totalSteps: number | null;
  lastStep: number;
  checkpoint: string | null;
}

/** A command as the bus expects it. The server is the one that validates it. */
export type Command = { kind: string } & Record<string, unknown>;
