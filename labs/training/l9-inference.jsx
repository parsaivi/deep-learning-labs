import { useState, useMemo, useCallback } from "react";
import { THEME, MONO } from "./common";

// Lesson 9 - Batch norm at test time (Batch Normalization, slides 38-40).
// Aha: at test time there is no batch. Normalizing a single order by its own "batch" gives u = 0 for
// every order, and small test batches make a prediction depend on the other customers. So training
// keeps a running average of each unit's batch mean and variance, and the test uses those fixed
// numbers. The averaging window is a trade-off: too short and it is noisy, too long and it lags
// behind statistics that drift while the weights train.

// @sim-start
const T9 = 400;
const NOISE9 = 0.4;
const drift9 = (t) => 1.6 - 1.4 * Math.exp(-t / 90);
const VAR9 = 0.64;
const STREAM9 = (function () {
  const r = rng(7);
  const mu = [];
  const v = [];
  for (let t = 0; t < T9; t++) {
    mu.push(drift9(t) + NOISE9 * gauss(r));
    const e = gauss(r);
    v.push(VAR9 * Math.max(0.2, 1 + 0.35 * e));
  }
  return { mu, v };
})();
function running9(beta) {
  let m = 0;
  let v = 1;
  const ms = [];
  let err = 0;
  for (let t = 0; t < T9; t++) {
    m = beta * m + (1 - beta) * STREAM9.mu[t];
    v = beta * v + (1 - beta) * STREAM9.v[t];
    ms.push(m);
    if (t >= T9 - 150) err += Math.abs(m - drift9(t)) / 150;
  }
  return { ms, m, v, err };
}
const GOAL9 = 0.07;
// 12 test orders from the final distribution
const TEST9 = (function () {
  const r = rng(91);
  const z = [];
  for (let i = 0; i < 12; i++) z.push(drift9(T9) + Math.sqrt(VAR9) * gauss(r));
  return z.sort((a, b) => a - b);
})();
const eta9 = (u) => 30 + 8 * u; // minutes, from the layer after batch norm
// test-time normalisation of TEST9: by test batches of size k, or by running statistics
function predict9(mode, k, run, seed) {
  if (mode === "running") return TEST9.map((z) => eta9((z - run.m) / Math.sqrt(run.v + 1e-5)));
  const r = rng(seed);
  const idx = TEST9.map((_, i) => i);
  for (let i = idx.length - 1; i > 0; i--) {
    const j = Math.floor(r() * (i + 1));
    const t = idx[i];
    idx[i] = idx[j];
    idx[j] = t;
  }
  const out = new Array(TEST9.length);
  for (let s = 0; s < idx.length; s += k) {
    const grp = idx.slice(s, s + k);
    const bn = bnForward(grp.map((i) => TEST9[i]));
    grp.forEach((i, q) => (out[i] = eta9(bn.u[q])));
  }
  return out;
}
// @sim-end

const TS9 = Array.from({ length: T9 }, (_, t) => t);
const TRUE9 = TS9.map(drift9);
const K_OPTS = [
  [1, "1 order"],
  [2, "2"],
  [4, "4"],
  [12, "all 12"],
];
const MODE9_OPTS = [
  ["batch", "test-batch statistics"],
  ["running", "running averages from training"],
];
const IDEAL9 = TEST9.map((z) => eta9((z - drift9(T9)) / Math.sqrt(VAR9)));

function TrainPanel({ run, show }) {
  return (
    <Plot
      series={[
        { xs: TS9, ys: STREAM9.mu, color: THEME.muted, width: 1 },
        { xs: TS9, ys: TRUE9, color: THEME.ok, dash: "6 3", width: 1.6 },
      ].concat(show ? [{ xs: TS9, ys: run.ms, color: THEME.highlight, width: 2.4 }] : [])}
      legend={[
        { color: THEME.muted, text: "each batch's mean μ_B" },
        { color: THEME.ok, dash: "6 3", text: "true mean of all orders" },
      ].concat(show ? [{ color: THEME.highlight, width: 2.4, text: "running average" }] : [])}
      xr={[0, T9]}
      yr={[-0.5, 3]}
      xl="training step →"
      label="one unit's mean z while training"
      height={190}
    />
  );
}
function TestPanel({ pred }) {
  return (
    <Plot
      series={[
        { xs: TEST9, ys: IDEAL9, color: THEME.ok, dash: "6 3", width: 1.4 },
        { xs: TEST9, ys: pred, color: THEME.highlight, width: 2.2 },
      ]}
      marks={TEST9.map((z, i) => ({ x: z, y: pred[i], shape: "circle", color: THEME.highlight }))}
      legend={[
        { color: THEME.highlight, width: 2.2, text: "predicted ETA (● orders)" },
        { color: THEME.ok, dash: "6 3", text: "with the true statistics" },
      ]}
      xr={[0, 3.2]}
      yr={[10, 50]}
      xl="test order's z (longer, busier orders to the right) →"
      label="12 test orders: predicted ETA in minutes"
      height={180}
    />
  );
}

export default function InferenceLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [k, setK] = useState(1);
  const [mode, setMode] = useState("batch");
  const [win, setWin] = useState(10);
  const [seed, setSeed] = useState(4);
  const [winC, setWinC] = useState(500);
  const modeNow = screen === 0 ? "batch" : mode;
  const beta = 1 - 1 / win;
  const run = useMemo(() => running9(beta), [beta]);
  const pred = useMemo(() => predict9(modeNow, k, run, seed), [modeNow, k, run, seed]);
  const betaC = 1 - 1 / winC;
  const runC = useMemo(() => running9(betaC), [betaC]);
  const predC = useMemo(() => predict9("running", 1, runC, 1), [runC]);
  const reshuffle = useCallback(() => setSeed((s) => s + 1), []);
  const okC = runC.err <= GOAL9;
  const spreadTxt = fmt(Math.max.apply(null, pred) - Math.min.apply(null, pred), 1) + " min";
  const winSlider = (id, val, set) => <LogSlider id={id} label={"averaging window ≈ 1/(1 − β) steps"} value={val} lo={0.3} hi={3} onChange={set} digits={0} />;

  let question, stage, side;
  if (screen === 0) {
    question = (
      <>
        Training is over. In the app, orders arrive one at a time, and the ETA layer after batch norm turns u into minutes (30 + 8u). If batch norm still uses the statistics of the batch it is given, what ETA does a single order get? Try test batches of 2 and 4 and reshuffle: does an order's ETA depend on who else ordered at the same moment?
      </>
    );
    stage = (
      <>
        <TrainPanel run={run} show={false} />
        <TestPanel pred={pred} />
      </>
    );
    side = (
      <>
        <Panel>
          <Label>test batch size</Label>
          <Pills options={K_OPTS} value={k} onPick={setK} />
          <Btn onClick={reshuffle} disabled={k === 1 || k === 12}>
            reshuffle who arrives together
          </Btn>
          <div style={{ height: 8 }} />
          <Readout rows={[["range of predicted ETAs", spreadTxt]]} />
        </Panel>
      </>
    );
  } else if (screen === 1) {
    question = (
      <>
        Fix it: during training, keep a running average of each batch's mean (and variance), and normalize test orders with those fixed numbers. Each step: m ← β·m + (1 − β)·μ_B. Change the averaging window. Why does a window of 3 steps give a jumpy estimate, and why does 500 steps miss the true mean at the end?
      </>
    );
    stage = (
      <>
        <TrainPanel run={run} show />
        <TestPanel pred={pred} />
      </>
    );
    side = (
      <>
        <Panel>
          <Pills options={MODE9_OPTS} value={mode} onPick={setMode} />
          {mode === "batch" && <Pills options={K_OPTS} value={k} onPick={setK} />}
          {winSlider("l9-win", win, setWin)}
          <Readout
            rows={[
              ["β", fmt(beta, 4)],
              ["running mean at the end", fmt(run.m, 3)],
              ["true mean at the end", fmt(drift9(T9 - 1), 3)],
            ]}
          />
        </Panel>
      </>
    );
  } else if (screen === 2) {
    question = (
      <>
        Name it. At test time batch norm uses fixed statistics μ_BN and σ²_BN, estimated during training, usually by an exponential running average. Then BN is the same fixed linear map for every input, and an order's prediction no longer depends on the rest of the batch.
      </>
    );
    stage = (
      <>
        <TrainPanel run={run} show />
        <TestPanel pred={pred} />
      </>
    );
    side = (
      <>
        <Panel>
          <Eq size={12.5}>
            train: μ<sup>[l]</sup> ← β μ<sup>[l]</sup> + (1 − β) μ_B (same for σ²)
            <br />
            or: μ_BN = mean over batches of μ_B
            <br />
            σ²_BN = B/(B − 1) · mean over batches of σ_B²
            <br />
            test: u = (z − μ_BN) / √(σ²_BN + ε), ẑ = γu + β
          </Eq>
          <Pills options={MODE9_OPTS} value={mode} onPick={setMode} />
          {winSlider("l9-win2", win, setWin)}
        </Panel>
        <Note>The slides' version averages the batch statistics of the final, converged network; B/(B − 1) makes the variance estimate unbiased. In practice the running average is used (PyTorch's momentum 0.1 means β = 0.9). Statistics are kept per unit.</Note>
      </>
    );
  } else {
    question = (
      <>
        Same training run. The test-time mean must track the true mean closely over the last 150 steps (whenever the model is saved, it should be ready to ship): average error at most {GOAL9}. Choose the averaging window.
      </>
    );
    stage = (
      <>
        <TrainPanel run={runC} show />
        <TestPanel pred={predC} />
      </>
    );
    side = (
      <>
        <Panel edge={okC ? THEME.ok : THEME.grid}>
          {winSlider("l9-winc", winC, setWinC)}
          <Readout
            rows={[
              ["β", fmt(betaC, 4)],
              ["avg |running − true|, last 150 steps", fmt(runC.err, 3)],
            ]}
          />
          <Goal met={okC}>tracking error ≤ {GOAL9}</Goal>
          <Hint
            hints={[
              "Sweep the window from short to long and watch the error: it falls, then rises again. Look at the plot at each end to see why.",
              "A short window averages too few noisy batches; a long one remembers the early steps, when the mean was still far lower. The best window is a few dozen steps.",
            ]}
          />
        </Panel>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="Test time: no batch to lean on" question={question} stage={stage} side={side} />;
}
