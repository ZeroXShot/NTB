// The inspector: a node's label, its attributes and the types NTB inferred on
// its ports. Attribute editors are built from the registry's declaration, so an
// op with a new attribute is editable the moment it exists.

import { useEffect, useState } from "react";
import type { JSX } from "react";
import type { AttrInfo, Node, OpInfo, Snapshot } from "../types";

interface Props {
  node: Node | undefined;
  op: OpInfo | undefined;
  snapshot: Snapshot | null;
  onRename: (name: string) => void;
  onAttrs: (attrs: Record<string, unknown>) => void;
  onDelete: () => void;
}

export function Inspector({
  node,
  op,
  snapshot,
  onRename,
  onAttrs,
  onDelete,
}: Props): JSX.Element {
  const [draft, setDraft] = useState("");
  useEffect(() => setDraft(node?.name ?? ""), [node?.id, node?.name]);

  if (!node) {
    return (
      <aside className="panel inspector">
        <h2>Inspector</h2>
        <p className="empty">Select a block.</p>
      </aside>
    );
  }

  const attrs = (node.attrs ?? {}) as Record<string, unknown>;
  const types = snapshot?.derived.types ?? {};

  return (
    <aside className="panel inspector">
      <h2>Inspector</h2>
      <div className="field">
        <label htmlFor="node-id">id</label>
        <input id="node-id" value={node.id} readOnly />
      </div>
      <div className="field">
        <label htmlFor="node-op">op</label>
        <input id="node-op" value={node.op} readOnly />
      </div>
      <div className="field">
        <label htmlFor="node-name">label</label>
        <input
          id="node-name"
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onBlur={() => draft !== (node.name ?? "") && onRename(draft)}
        />
      </div>

      {op && op.attrs.length > 0 && (
        <>
          <h3>attributes</h3>
          {op.attrs.map((attr) => (
            <AttrField
              key={attr.name}
              attr={attr}
              value={attrs[attr.name]}
              onChange={(value) => onAttrs({ ...attrs, [attr.name]: value })}
            />
          ))}
        </>
      )}

      <h3>types</h3>
      <table className="types">
        <tbody>
          {[...(op?.inputs ?? []), ...(op?.outputs ?? [])].map((port) => (
            <tr key={port.name}>
              <td>{port.name}</td>
              <td>{types[`${node.id}.${port.name}`] ?? "-"}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <p className="placement">
        at ({(node.placement?.pos ?? [0, 0, 0]).map((v) => v.toFixed(2)).join(", ")})
      </p>
      <button className="danger" onClick={onDelete}>
        Delete block
      </button>
    </aside>
  );
}

interface AttrProps {
  attr: AttrInfo;
  value: unknown;
  onChange: (value: unknown) => void;
}

function AttrField({ attr, value, onChange }: AttrProps): JSX.Element {
  const id = `attr-${attr.name}`;
  const label = (
    <label htmlFor={id} title={attr.doc}>
      {attr.name}
      {attr.required && <span className="required">*</span>}
    </label>
  );

  if (attr.type === "bool") {
    return (
      <div className="field">
        {label}
        <input
          id={id}
          type="checkbox"
          checked={Boolean(value)}
          onChange={(event) => onChange(event.target.checked)}
        />
      </div>
    );
  }

  if (attr.choices) {
    return (
      <div className="field">
        {label}
        <select id={id} value={String(value ?? "")} onChange={(e) => onChange(e.target.value)}>
          {attr.choices.map((choice) => (
            <option key={String(choice)} value={String(choice)}>
              {String(choice)}
            </option>
          ))}
        </select>
      </div>
    );
  }

  // int, float, and the list attributes, which are typed as JSON so a kernel
  // size can be "3" or "[3, 3]" without a bespoke widget per op.
  return (
    <div className="field">
      {label}
      <input
        id={id}
        value={render(value)}
        onChange={(event) => onChange(parse(event.target.value, attr.type))}
      />
    </div>
  );
}

function render(value: unknown): string {
  if (value === null || value === undefined) return "";
  return typeof value === "object" ? JSON.stringify(value) : String(value);
}

function parse(text: string, type: string): unknown {
  if (text.trim() === "") return null;
  if (type === "int" || type === "float") {
    const numeric = Number(text);
    return Number.isNaN(numeric) ? text : numeric;
  }
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}
