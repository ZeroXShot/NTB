// The two things that tell you whether the model is real: what validation says
// about it, and the torch module it would become. Both come from the server, so
// what you read here is exactly what `ntb emit` would write.

import { useState } from "react";
import type { JSX } from "react";
import type { Diagnostic, Snapshot } from "../types";

interface Props {
  snapshot: Snapshot | null;
  onSelect: (nodeId: string) => void;
}

type Tab = "diagnostics" | "code";

export function Output({ snapshot, onSelect }: Props): JSX.Element {
  const [tab, setTab] = useState<Tab>("diagnostics");
  const derived = snapshot?.derived;
  const diagnostics = derived?.diagnostics ?? [];
  const errors = diagnostics.filter((d) => d.severity === "error").length;

  return (
    <section className="panel output">
      <nav className="tabs">
        <button className={tab === "diagnostics" ? "on" : ""} onClick={() => setTab("diagnostics")}>
          Problems{diagnostics.length > 0 && <span className="count">{diagnostics.length}</span>}
        </button>
        <button className={tab === "code" ? "on" : ""} onClick={() => setTab("code")}>
          torch
        </button>
      </nav>

      {tab === "diagnostics" ? (
        <div className="scroll">
          {diagnostics.length === 0 && <p className="ok">No problems. {errors === 0 && "✓"}</p>}
          {diagnostics.map((diagnostic, index) => (
            <Problem key={index} diagnostic={diagnostic} onSelect={onSelect} />
          ))}
        </div>
      ) : (
        <div className="scroll">
          {derived?.code ? (
            <pre className="code">{derived.code}</pre>
          ) : (
            <p className="empty">{derived?.codeError || "nothing to emit yet"}</p>
          )}
        </div>
      )}
    </section>
  );
}

function Problem({
  diagnostic,
  onSelect,
}: {
  diagnostic: Diagnostic;
  onSelect: (nodeId: string) => void;
}): JSX.Element {
  return (
    <button
      className={`diagnostic ${diagnostic.severity}`}
      onClick={() => diagnostic.node && onSelect(diagnostic.node)}
    >
      <span className="where">{diagnostic.node ?? diagnostic.module ?? "document"}</span>
      <span className="what">{diagnostic.message}</span>
      <span className="code-tag">{diagnostic.code}</span>
    </button>
  );
}
