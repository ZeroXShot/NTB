// Training runs. The work happens in a process of its own (ADR 11); this panel
// only starts one and draws what it says.

import { useEffect, useState } from "react";
import type { JSX } from "react";
import type { RunRecord } from "../types";

interface Props {
  runs: RunRecord[];
  curves: Record<string, number[]>;
  savedPath: string | null;
  onStart: (config: Record<string, unknown>) => void;
  onStop: (runId: string) => void;
  onResume: (runId: string) => void;
  onRefresh: () => void;
}

const OPTIMISERS = ["adam", "adamw", "sgd"] as const;
const LOSSES = ["mse", "cross_entropy", "bce"] as const;

export function Runs({
  runs,
  curves,
  savedPath,
  onStart,
  onStop,
  onResume,
  onRefresh,
}: Props): JSX.Element {
  const [epochs, setEpochs] = useState(1);
  const [steps, setSteps] = useState(50);
  const [batch, setBatch] = useState(32);
  const [lr, setLr] = useState(0.001);
  const [optimiser, setOptimiser] = useState<string>("adam");
  const [loss, setLoss] = useState<string>("mse");

  useEffect(onRefresh, [onRefresh]);

  return (
    <div className="scroll runs">
      <header>
        <h3>train</h3>
        <button
          disabled={!savedPath}
          title={savedPath ? "train this model" : "save the document first"}
          onClick={() =>
            onStart({
              epochs,
              steps_per_epoch: steps,
              batch_size: batch,
              learning_rate: lr,
              optimiser,
              loss,
            })
          }
        >
          Start
        </button>
      </header>
      {!savedPath && (
        <p className="empty">
          A run trains the file on disk, so the document has to be saved before one can start.
        </p>
      )}

      <div className="field">
        <label>epochs</label>
        <input
          type="number"
          min={1}
          value={epochs}
          onChange={(e) => setEpochs(Math.max(1, Number(e.target.value)))}
        />
      </div>
      <div className="field">
        <label title="Synthetic data only">steps / epoch</label>
        <input
          type="number"
          min={1}
          value={steps}
          onChange={(e) => setSteps(Math.max(1, Number(e.target.value)))}
        />
      </div>
      <div className="field">
        <label>batch</label>
        <input
          type="number"
          min={1}
          value={batch}
          onChange={(e) => setBatch(Math.max(1, Number(e.target.value)))}
        />
      </div>
      <div className="field">
        <label>learning rate</label>
        <input
          type="number"
          step="0.0001"
          min={0}
          value={lr}
          onChange={(e) => setLr(Number(e.target.value))}
        />
      </div>
      <div className="field">
        <label>optimiser</label>
        <select value={optimiser} onChange={(e) => setOptimiser(e.target.value)}>
          {OPTIMISERS.map((name) => (
            <option key={name}>{name}</option>
          ))}
        </select>
      </div>
      <div className="field">
        <label>loss</label>
        <select value={loss} onChange={(e) => setLoss(e.target.value)}>
          {LOSSES.map((name) => (
            <option key={name}>{name}</option>
          ))}
        </select>
      </div>
      <p className="empty">
        Data is synthetic: random tensors shaped like the model's own inputs. It answers whether
        the architecture trains and how fast, not whether it is any good.
      </p>

      <header>
        <h3>runs</h3>
        <button className="tiny" onClick={onRefresh}>
          refresh
        </button>
      </header>
      {runs.length === 0 && <p className="empty">None yet.</p>}
      {runs.map((run) => (
        <article key={run.id} className={`card run ${run.status}`}>
          <h4>
            <span>
              {run.id} <em>{run.status}</em>
            </span>
            {run.status === "running" ? (
              <button className="tiny danger" onClick={() => onStop(run.id)}>
                stop
              </button>
            ) : (
              run.checkpoint && (
                <button className="tiny" onClick={() => onResume(run.id)}>
                  resume
                </button>
              )
            )}
          </h4>
          <p className="empty">
            {run.lastStep} steps
            {run.totalSteps ? ` of ${run.totalSteps}` : ""}
            {run.parameters ? ` · ${run.parameters.toLocaleString()} parameters` : ""}
          </p>
          {run.error && <p className="run-error">{run.error}</p>}
          <Curve values={curves[run.id] ?? []} />
        </article>
      ))}
    </div>
  );
}

/** The loss so far, as a plain polyline. No chart library for one line. */
function Curve({ values }: { values: number[] }): JSX.Element | null {
  if (values.length < 2) return null;
  const width = 240;
  const height = 48;
  const low = Math.min(...values);
  const high = Math.max(...values);
  const span = high - low || 1;
  const points = values
    .map((value, index) => {
      const x = (index / (values.length - 1)) * width;
      const y = height - ((value - low) / span) * height;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");

  return (
    <svg className="curve" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
      <polyline points={points} />
      <title>{`loss ${high.toFixed(4)} to ${low.toFixed(4)} over ${values.length} steps`}</title>
    </svg>
  );
}
