// Generators and spatial rules: the two objects that turn geometry into a
// network. Editing one here re-derives the whole topology on the server, so the
// canvas and the generated code follow immediately.

import type { JSX } from "react";
import type { Generator, Module, SpatialRule } from "../ir.gen";

const KINDS = ["vertical_stack", "axis_projection", "neighborhood", "lattice"] as const;
const AXES = ["x", "y", "z"] as const;

interface Props {
  module: Module | undefined;
  modules: Module[];
  onGenerator: (generator: Generator) => void;
  onRule: (rule: SpatialRule) => void;
  onAddGenerator: (generator: Generator) => void;
  onAddRule: (rule: SpatialRule) => void;
  onRemoveGenerator: (id: string) => void;
  onRemoveRule: (id: string) => void;
}

export function Spatial(props: Props): JSX.Element {
  const { module, modules } = props;
  const generators = module?.generators ?? [];
  const rules = module?.spatial_rules ?? [];
  const repeatable = modules.filter((candidate) => candidate.id !== module?.id);

  return (
    <div className="scroll spatial">
      <header>
        <h3>generators</h3>
        <button
          disabled={repeatable.length === 0}
          title={
            repeatable.length === 0
              ? "a generator repeats another module; this document has only one"
              : "repeat a module through space"
          }
          onClick={() =>
            props.onAddGenerator({
              id: fresh("gen", generators),
              module: repeatable[0]!.id,
              count: 4,
              axis: "z",
              origin: [0, 0, 0],
              step: 1,
              chain: true,
              attr_bindings: {},
              label: "",
            })
          }
        >
          + generator
        </button>
      </header>
      {generators.length === 0 && <p className="empty">None. A generator repeats a module.</p>}
      {generators.map((generator) => (
        <GeneratorRow
          key={generator.id}
          generator={generator}
          modules={repeatable}
          onChange={props.onGenerator}
          onRemove={() => props.onRemoveGenerator(generator.id)}
        />
      ))}

      <header>
        <h3>spatial rules</h3>
        <button
          onClick={() =>
            props.onAddRule({
              id: fresh("rule", rules),
              kind: "vertical_stack",
              members: [],
              axis: "z",
              radius: null,
              output_port: "out",
              input_port: "in",
              bidirectional: false,
              label: "",
            })
          }
        >
          + rule
        </button>
      </header>
      {rules.length === 0 && (
        <p className="empty">None. A rule derives edges from where blocks sit.</p>
      )}
      {rules.map((rule) => (
        <RuleRow
          key={rule.id}
          rule={rule}
          onChange={props.onRule}
          onRemove={() => props.onRemoveRule(rule.id)}
        />
      ))}
    </div>
  );
}

function GeneratorRow({
  generator,
  modules,
  onChange,
  onRemove,
}: {
  generator: Generator;
  modules: Module[];
  onChange: (generator: Generator) => void;
  onRemove: () => void;
}): JSX.Element {
  const set = (patch: Partial<Generator>) => onChange({ ...generator, ...patch });
  return (
    <article className="card">
      <h4>
        {generator.id}
        <button className="tiny danger" onClick={onRemove}>
          remove
        </button>
      </h4>
      <div className="field">
        <label>repeats</label>
        <select value={generator.module} onChange={(e) => set({ module: e.target.value })}>
          {modules.map((candidate) => (
            <option key={candidate.id} value={candidate.id}>
              {candidate.id}
            </option>
          ))}
        </select>
      </div>
      <div className="field">
        <label>count</label>
        <input
          type="number"
          min={1}
          value={generator.count}
          onChange={(e) => set({ count: Math.max(1, Number(e.target.value)) })}
        />
      </div>
      <div className="field">
        <label>along</label>
        <select value={generator.axis ?? "z"} onChange={(e) => set({ axis: pickAxis(e) })}>
          {AXES.map((axis) => (
            <option key={axis} value={axis}>
              {axis}
            </option>
          ))}
        </select>
      </div>
      <div className="field">
        <label>step</label>
        <input
          type="number"
          step="0.25"
          value={generator.step ?? 1}
          onChange={(e) => set({ step: Number(e.target.value) })}
        />
      </div>
      <Triple
        label="origin"
        value={generator.origin ?? [0, 0, 0]}
        onChange={(origin) => set({ origin })}
      />
      <div className="field">
        <label title="Wire each repetition into the next">chain</label>
        <input
          type="checkbox"
          checked={generator.chain ?? true}
          onChange={(e) => set({ chain: e.target.checked })}
        />
      </div>
    </article>
  );
}

function RuleRow({
  rule,
  onChange,
  onRemove,
}: {
  rule: SpatialRule;
  onChange: (rule: SpatialRule) => void;
  onRemove: () => void;
}): JSX.Element {
  const set = (patch: Partial<SpatialRule>) => onChange({ ...rule, ...patch });
  const neighborhood = rule.kind === "neighborhood";
  const ordered = rule.kind === "vertical_stack" || rule.kind === "axis_projection";
  return (
    <article className="card">
      <h4>
        {rule.id}
        <button className="tiny danger" onClick={onRemove}>
          remove
        </button>
      </h4>
      <div className="field">
        <label>kind</label>
        <select
          value={rule.kind}
          onChange={(e) => {
            const kind = KINDS.find((k) => k === e.target.value) ?? "vertical_stack";
            // radius belongs to neighborhood alone, and the ordered kinds cannot
            // be bidirectional; the server would refuse either combination.
            set({
              kind,
              radius: kind === "neighborhood" ? (rule.radius ?? 1.5) : null,
              bidirectional:
                kind === "vertical_stack" || kind === "axis_projection"
                  ? false
                  : rule.bidirectional,
            });
          }}
        >
          {KINDS.map((kind) => (
            <option key={kind} value={kind}>
              {kind.replace("_", " ")}
            </option>
          ))}
        </select>
      </div>
      <div className="field">
        <label>along</label>
        <select value={rule.axis ?? "z"} onChange={(e) => set({ axis: pickAxis(e) })}>
          {AXES.map((axis) => (
            <option key={axis} value={axis}>
              {axis}
            </option>
          ))}
        </select>
      </div>
      {neighborhood && (
        <div className="field">
          <label>radius</label>
          <input
            type="number"
            step="0.25"
            min={0.25}
            value={rule.radius ?? 1.5}
            onChange={(e) => set({ radius: Number(e.target.value) })}
          />
        </div>
      )}
      <div className="field">
        <label title="Node ids, or a generator id for all of its repetitions">members</label>
        <input
          value={(rule.members ?? []).join(", ")}
          onChange={(e) =>
            set({
              members: e.target.value
                .split(",")
                .map((name) => name.trim())
                .filter(Boolean),
            })
          }
        />
      </div>
      <div className="field">
        <label>from port</label>
        <input
          value={rule.output_port ?? "out"}
          onChange={(e) => set({ output_port: e.target.value })}
        />
      </div>
      <div className="field">
        <label>to port</label>
        <input
          value={rule.input_port ?? "in"}
          onChange={(e) => set({ input_port: e.target.value })}
        />
      </div>
      {!ordered && (
        <div className="field">
          <label title="Both directions. Ordered kinds would cycle">both ways</label>
          <input
            type="checkbox"
            checked={rule.bidirectional ?? false}
            onChange={(e) => set({ bidirectional: e.target.checked })}
          />
        </div>
      )}
    </article>
  );
}

function Triple({
  label,
  value,
  onChange,
}: {
  label: string;
  value: [number, number, number];
  onChange: (value: [number, number, number]) => void;
}): JSX.Element {
  return (
    <div className="field axes">
      <label>{label}</label>
      <div>
        {[0, 1, 2].map((axis) => (
          <input
            key={axis}
            type="number"
            step="0.25"
            value={value[axis]}
            onChange={(event) => {
              const next: [number, number, number] = [...value];
              next[axis] = Number(event.target.value);
              onChange(next);
            }}
          />
        ))}
      </div>
    </div>
  );
}

function pickAxis(event: { target: { value: string } }): "x" | "y" | "z" {
  return AXES.find((axis) => axis === event.target.value) ?? "z";
}

function fresh(stem: string, existing: { id: string }[]): string {
  const taken = new Set(existing.map((item) => item.id));
  for (let index = 1; ; index += 1) {
    const candidate = `${stem}${index}`;
    if (!taken.has(candidate)) return candidate;
  }
}
