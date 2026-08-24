// The op palette. Everything in it comes from the server's registry, so an op
// added in Python shows up here without touching the frontend.

import { useMemo, useState } from "react";
import type { JSX } from "react";
import type { OpInfo } from "../types";

interface Props {
  ops: OpInfo[];
  onPick: (op: OpInfo) => void;
}

export function Palette({ ops, onPick }: Props): JSX.Element {
  const [query, setQuery] = useState("");

  const groups = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const grouped = new Map<string, OpInfo[]>();
    for (const op of ops) {
      if (needle && !op.name.includes(needle) && !op.summary.toLowerCase().includes(needle)) {
        continue;
      }
      const bucket = grouped.get(op.category) ?? [];
      bucket.push(op);
      grouped.set(op.category, bucket);
    }
    return [...grouped.entries()];
  }, [ops, query]);

  return (
    <aside className="panel palette">
      <h2>Ops</h2>
      <input
        className="search"
        placeholder="filter"
        value={query}
        onChange={(event) => setQuery(event.target.value)}
      />
      {groups.map(([category, entries]) => (
        <section key={category}>
          <h3>{category}</h3>
          {entries.map((op) => (
            <button key={op.name} className="op" title={op.doc} onClick={() => onPick(op)}>
              <span className="op-name">{op.name.replace(/^ntb\./, "")}</span>
              <span className="op-summary">{op.summary}</span>
            </button>
          ))}
        </section>
      ))}
      {groups.length === 0 && <p className="empty">nothing matches</p>}
    </aside>
  );
}
