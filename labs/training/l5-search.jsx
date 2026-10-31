import { useState, useMemo, useCallback } from "react";
import { THEME, MONO } from "./common";

// Lesson 5 - Choosing hyperparameters: random vs grid search, coarse-to-fine (Practical Issues, slides 46-64).
// Aha: when one hyperparameter matters much more than the others (and you don't know which), a k x k
// grid tries only k values of it, while k^2 random trials try k^2 different values. Search in log
// space, then zoom in around the best, and distrust a best value sitting on the edge of the box.

// @sim-start
const BOX5 = { lr: [-6, 0], wd: [-7, -1] };
// validation accuracy of the ETA-slot classifier (10 slots, so 10% is chance) after 5 epochs
function acc5(a, b) {
  const d = (a + 3.55) / 0.35;
  const e = (b + 4.6) / 2.2;
  return 0.86 * Math.exp(-0.5 * d * d) * (1 - 0.06 * Math.min(1, e * e)) + 0.1;
}
function trialNoise(k, seed) {
  let h = (k * 374761393 + seed * 668265263) >>> 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177) >>> 0;
  return (h / 4294967296 - 0.5) * 0.01;
}
// n trials in box; layout grid (k x k with k = round(sqrt n)) or random
function trials5(box, n, layout, seed) {
  const out = [];
  if (layout === "grid") {
    const k = Math.max(1, Math.round(Math.sqrt(n)));
    for (let i = 0; i < k; i++)
      for (let j = 0; j < k; j++) {
        const a = box.lr[0] + ((box.lr[1] - box.lr[0]) * (i + 0.5)) / k;
        const b = box.wd[0] + ((box.wd[1] - box.wd[0]) * (j + 0.5)) / k;
        out.push({ a, b, acc: acc5(a, b) + trialNoise(out.length, seed) });
      }
  } else {
    const r = rng(seed * 7919 + 13);
    for (let q = 0; q < n; q++) {
      const a = box.lr[0] + (box.lr[1] - box.lr[0]) * r();
      const b = box.wd[0] + (box.wd[1] - box.wd[0]) * r();
      out.push({ a, b, acc: acc5(a, b) + trialNoise(q, seed) });
    }
  }
  let best = 0;
  for (let i = 1; i < out.length; i++) if (out[i].acc > out[best].acc) best = i;
  const lrs = new Set(out.map((t) => t.a.toFixed(4)));
  return { list: out, best, distinctLr: lrs.size };
}
const N5C = 10;
const GOAL5 = 0.94;
function onBorder(t, box) {
  const fa = (t.a - box.lr[0]) / (box.lr[1] - box.lr[0]);
  const fb = (t.b - box.wd[0]) / (box.wd[1] - box.wd[0]);
  return fa < 0.06 || fa > 0.94 || fb < 0.06 || fb > 0.94;
}
// @sim-end

const PLANE = 400;
const LAYOUT_OPTS = [
  ["grid", "grid"],
  ["random", "random"],
];
function pctTxt(v) {
  return fmt(v * 100, 1) + "%";
}

// hyperparameter plane: log10 lr across, log10 weight decay up; trials as dots, best as a diamond
function Plane({ sets, boxes, reveal }) {
  const X = (a) => ((a - BOX5.lr[0]) / (BOX5.lr[1] - BOX5.lr[0])) * PLANE;
  const Y = (b) => PLANE - ((b - BOX5.wd[0]) / (BOX5.wd[1] - BOX5.wd[0])) * PLANE;
  const ticksA = [-6, -5, -4, -3, -2, -1, 0];
  const ticksB = [-7, -6, -5, -4, -3, -2, -1];
  const band = reveal ? [X(-3.55 - 0.35 * 1.2), X(-3.55 + 0.35 * 1.2)] : null;
  return (
    <svg viewBox={"0 0 " + PLANE + " " + PLANE} style={{ width: "100%", maxWidth: 520, display: "block", margin: "0 auto", border: "1px solid " + THEME.grid, background: THEME.bg }} role="img" aria-label="Hyperparameter plane with trial runs">
      {ticksA.map((t) => (
        <g key={"a" + t}>
          <line x1={X(t)} x2={X(t)} y1={0} y2={PLANE} stroke={THEME.grid} strokeWidth={0.6} />
          <text x={X(t) + 3} y={PLANE - 22} fontFamily={MONO} fontSize={10} fill={THEME.muted}>
            1e{t}
          </text>
        </g>
      ))}
      {ticksB.map((t) => (
        <g key={"b" + t}>
          <line x1={0} x2={PLANE} y1={Y(t)} y2={Y(t)} stroke={THEME.grid} strokeWidth={0.6} />
          <text x={3} y={Y(t) - 3} fontFamily={MONO} fontSize={10} fill={THEME.muted}>
            1e{t}
          </text>
        </g>
      ))}
      {band && (
        <g>
          <rect x={band[0]} y={0} width={band[1] - band[0]} height={PLANE} fill={rgba(THEME.ok, 0.16)} stroke={THEME.ok} strokeDasharray="4 3" />
          <text x={band[1] + 4} y={40} fontFamily={MONO} fontSize={10.5} fill={THEME.ok} stroke={THEME.bg} strokeWidth={3} paintOrder="stroke">
            accuracy &gt; 80% only here
          </text>
        </g>
      )}
      {(boxes || []).map((b, i) => (
        <rect key={"bx" + i} x={X(b.lr[0])} y={Y(b.wd[1])} width={X(b.lr[1]) - X(b.lr[0])} height={Y(b.wd[0]) - Y(b.wd[1])} fill="none" stroke={b.color} strokeWidth={1.6} strokeDasharray={b.dash} />
      ))}
      {sets.map((s, si) =>
        s.t.list.map((t, i) => {
          const isBest = i === s.t.best;
          const col = t.acc > 0.8 ? THEME.highlight : t.acc > 0.4 ? THEME.accent : THEME.muted;
          return <Marker key={si + "-" + i} x={X(t.a)} y={Y(t.b)} shape={isBest ? "diamond" : s.shape} color={col} r={isBest ? 6.5 : 4} />;
        })
      )}
      <g fontFamily={MONO} fontSize={11} fill={THEME.text} stroke={THEME.bg} strokeWidth={3} paintOrder="stroke">
        <text x={PLANE - 6} y={PLANE - 8} textAnchor="end">
          learning rate (log) →
        </text>
        <text x={8} y={16}>
          ↑ weight decay (log)
        </text>
        <text x={8} y={PLANE - 8}>
          ◆ best run · bright = high accuracy
        </text>
      </g>
    </svg>
  );
}
// the same trials seen from the side: accuracy against learning rate only
function Side({ sets, reveal }) {
  const xs = [];
  const ys = [];
  for (let i = 0; i <= 200; i++) {
    const a = BOX5.lr[0] + ((BOX5.lr[1] - BOX5.lr[0]) * i) / 200;
    xs.push(a);
    ys.push(acc5(a, -4.6));
  }
  const marks = [];
  sets.forEach((s) => s.t.list.forEach((t, i) => marks.push({ x: t.a, y: t.acc, shape: i === s.t.best ? "diamond" : s.shape, color: t.acc > 0.8 ? THEME.highlight : THEME.accent })));
  return (
    <Plot
      series={reveal ? [{ xs, ys, color: THEME.ok, dash: "5 3", width: 1.4 }] : []}
      marks={marks}
      xr={BOX5.lr}
      yr={[0, 1]}
      xl="learning rate (log10) →"
      label={"validation accuracy of each run, seen along the learning-rate axis" + (reveal ? " · dashed = true curve" : "")}
      height={170}
    />
  );
}

export default function SearchLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [layout, setLayout] = useState("grid");
  const [n, setN] = useState(9);
  const [seed, setSeed] = useState(1);
  const [reveal, setReveal] = useState(false);
  const [lrLo, setLrLo] = useState(-6);
  const [lrHi, setLrHi] = useState(0);
  const [wdLo, setWdLo] = useState(-7);
  const [wdHi, setWdHi] = useState(-1);
  const nNow = screen === 0 ? 9 : n;
  const t = useMemo(() => trials5(BOX5, nNow, layout, seed), [nNow, layout, seed]);
  const other = useMemo(() => trials5(BOX5, nNow, layout === "grid" ? "random" : "grid", seed), [nNow, layout, seed]);
  const reseed = useCallback(() => setSeed((s) => s + 1), []);
  const changeN = useCallback((v) => setN(Math.round(v)), []);
  const r1 = useMemo(() => trials5(BOX5, N5C, "random", 110), []);
  const box2 = useMemo(() => ({ lr: [Math.min(lrLo, lrHi - 0.05), Math.max(lrHi, lrLo + 0.05)], wd: [Math.min(wdLo, wdHi - 0.05), Math.max(wdHi, wdLo + 0.05)] }), [lrLo, lrHi, wdLo, wdHi]);
  const r2 = useMemo(() => trials5(box2, N5C, "random", 206), [box2]);
  const best2 = r2.list[r2.best];
  const border2 = onBorder(best2, box2);
  const okC = best2.acc >= GOAL5 && !border2;
  const revealNow = screen === 2 ? true : screen === 3 ? false : reveal;

  const rows = (tt) => [
    ["runs", String(tt.list.length)],
    ["different learning rates tried", String(tt.distinctLr)],
    ["best validation accuracy", pctTxt(tt.list[tt.best].acc)],
  ];
  const sets = [{ t, shape: layout === "grid" ? "square" : "circle" }];

  let question, stage, side;
  if (screen === 0) {
    question = (
      <>
        The ETA model now predicts one of 10 delivery slots. You can afford 9 training runs of 5 epochs each to choose the learning rate and the weight decay, both searched in log space. Lay the 9 runs out as a 3 × 3 grid, or draw them at random. Which finds the better model? Redraw the random runs a few times.
      </>
    );
    stage = (
      <>
        <Plane sets={sets} reveal={false} />
        <Side sets={sets} reveal={false} />
      </>
    );
    side = (
      <>
        <Panel>
          <Pills options={LAYOUT_OPTS} value={layout} onPick={setLayout} />
          <Btn onClick={reseed} disabled={layout === "grid"}>
            redraw random runs
          </Btn>
          <div style={{ height: 8 }} />
          <Readout rows={rows(t)} />
        </Panel>
        <Note>The lower chart shows the same runs from the side: accuracy against learning rate alone. What pattern do the dots make?</Note>
      </>
    );
  } else if (screen === 1) {
    question = (
      <>
        Change the budget. With 16 or 25 runs, how many different learning rates does the grid try, and how many does random search try? Reveal the true map: why does one of the two hyperparameters decide almost everything here?
      </>
    );
    stage = (
      <>
        <Plane sets={sets} reveal={revealNow} />
        <Side sets={sets} reveal={revealNow} />
      </>
    );
    side = (
      <>
        <Panel>
          <Pills options={LAYOUT_OPTS} value={layout} onPick={setLayout} />
          <Slider id="l5-n" label="budget (runs)" value={n} min={4} max={36} stepSize={1} onChange={changeN} display={String(layout === "grid" ? Math.pow(Math.round(Math.sqrt(n)), 2) : n)} />
          <Btn onClick={reseed} disabled={layout === "grid"}>
            redraw random runs
          </Btn>
          <Toggle label="reveal the true accuracy band" value={reveal} onChange={setReveal} />
          <Readout rows={rows(t).concat([["the other layout would get", pctTxt(other.list[other.best].acc)]])} />
        </Panel>
      </>
    );
  } else if (screen === 2) {
    question = (
      <>
        Name it: <i>random search</i> (Bergstra & Bengio, 2012). Some hyperparameters matter much more than others and you don't know which in advance. A k × k grid spends k² runs on only k values of the important one; k² random runs try k² values. Sample on a log scale for rates and strengths.
      </>
    );
    stage = (
      <>
        <Plane sets={sets} reveal />
        <Side sets={sets} reveal />
      </>
    );
    side = (
      <>
        <Panel>
          <Pills options={LAYOUT_OPTS} value={layout} onPick={setLayout} />
          <Readout rows={rows(t)} />
        </Panel>
        <Note>
          The slides' recipe: (1) check the initial loss (ln 10 ≈ 2.3 for 10 slots, regularization off); (2) overfit ~20 examples to 100%; (3) find an η that drops the loss within ~100 steps (try 1e-1 … 1e-4); (4) coarse grid of η and weight decay (1e-4, 1e-5, 0) for 1–5 epochs; (5) refine around the best, train 10–20 epochs; (6) read the loss and accuracy curves (big train/val gap: more regularization; no gap: bigger model); (7) go back to 5. One validation set is enough; babysit one model or train many in parallel.
        </Note>
      </>
    );
  } else {
    question = (
      <>
        Coarse to fine. Round 1 (circles) spent {N5C} random runs on the full box. Now place the box for round 2 (squares, {N5C} more runs) using what round 1 shows. Reach {GOAL5 * 100}% validation accuracy, and make sure the best run is not sitting on the edge of your box.
      </>
    );
    const sets2 = [
      { t: r1, shape: "circle" },
      { t: r2, shape: "square" },
    ];
    stage = (
      <>
        <Plane sets={sets2} boxes={[{ lr: box2.lr, wd: box2.wd, color: THEME.highlight, dash: "5 3" }]} reveal={false} />
        <Side sets={sets2} reveal={false} />
      </>
    );
    side = (
      <>
        <Panel edge={okC ? THEME.ok : THEME.grid}>
          <Slider id="l5-lrlo" label="round 2: lowest lr (log10)" value={lrLo} min={-6} max={0} stepSize={0.01} onChange={setLrLo} />
          <Slider id="l5-lrhi" label="round 2: highest lr (log10)" value={lrHi} min={-6} max={0} stepSize={0.01} onChange={setLrHi} />
          <Slider id="l5-wdlo" label="round 2: lowest weight decay (log10)" value={wdLo} min={-7} max={-1} stepSize={0.01} onChange={setWdLo} />
          <Slider id="l5-wdhi" label="round 2: highest weight decay (log10)" value={wdHi} min={-7} max={-1} stepSize={0.01} onChange={setWdHi} />
          <Readout
            rows={[
              ["round 1 best", pctTxt(r1.list[r1.best].acc) + " at lr 1e" + fmt(r1.list[r1.best].a, 2)],
              ["round 2 best", pctTxt(best2.acc) + " at lr 1e" + fmt(best2.a, 2)],
              ["round 2 best on the box edge?", border2 ? "yes" : "no", border2 ? THEME.warn : undefined],
            ]}
          />
          <Goal met={best2.acc >= GOAL5}>round 2 reaches {GOAL5 * 100}%</Goal>
          <Goal met={!border2}>best run is not on the edge of the box</Goal>
          <Hint
            hints={[
              "Look at round 1 from the side (lower chart). Which learning rates gave the highest accuracy? Did weight decay seem to matter?",
              "Narrow the learning-rate range to about one power of ten around round 1's best, and keep weight decay wide. If the best lands on an edge, move the box so it sits inside.",
            ]}
          />
        </Panel>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="Random beats grid" question={question} stage={stage} side={side} />;
}
