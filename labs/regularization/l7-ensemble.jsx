import { useState, useMemo, useCallback } from "react";
import { THEME, MONO } from "./common";

// Lesson 7 - Bagging and model ensembles (Regularization deck, slides 20-24 and 56-59).
// Aha: each overfit model is wrong in its own random way. Average several and the shared shape stays
// while the private wiggles cancel. Dropout is a cheap version of this (2^M thinned networks sharing
// weights); averaging the weights along one SGD run (Polyak averaging) is another.

// @sim-start
const TR7 = makeETA(1, 10, 1.6);
const VA7 = makeETA(901, 60, 1.6);
const K7 = 12;
const MODELS7 = [];
for (let k = 0; k < K7; k++) MODELS7.push(ridgeFit(bootstrap(TR7, 100 + k), 9, 1e-3));
function avgW(ws) {
  return ws[0].map((_, i) => ws.reduce((s, w) => s + w[i], 0) / ws.length);
}
const ENS7 = [];
for (let k = 1; k <= K7; k++) ENS7.push(rmseOf(avgW(MODELS7.slice(0, k)), VA7));
const SINGLE7 = MODELS7.map((w) => rmseOf(w, VA7));
// challenge: one SGD run with a constant step on 40 orders (degree 4); average the weights or not
const TRC7 = makeETA(3, 40, 1.6);
const VAC7 = makeETA(903, 60, 1.6);
const STEPS7 = 3000;
const SGD7 = (function () {
  const r = rng(7);
  const D = 5;
  const P = TRC7.km.map((k) => legendre(kmX(k), 4));
  const w = new Array(D).fill(0);
  w[0] = 20;
  const hist = [];
  for (let t = 0; t < STEPS7; t++) {
    const i = Math.floor(r() * TRC7.n);
    let p = 0;
    for (let k = 0; k < D; k++) p += w[k] * P[i][k];
    const e = p - TRC7.y[i];
    for (let k = 0; k < D; k++) w[k] -= 0.3 * 2 * e * P[i][k];
    hist.push(w.slice());
  }
  return hist;
})();
const LS7 = rmseOf(ridgeFit(TRC7, 4, 0), VAC7);
function polyak7(beta) {
  const e = SGD7[0].slice();
  const va = [];
  const raw = [];
  for (let t = 0; t < STEPS7; t++) {
    for (let k = 0; k < e.length; k++) e[k] = beta * e[k] + (1 - beta) * SGD7[t][k];
    if (t % 15 === 0 || t === STEPS7 - 1) {
      va.push(rmseOf(e, VAC7));
      raw.push(rmseOf(SGD7[t], VAC7));
    }
  }
  return { w: e, va, raw, final: rmseOf(e, VAC7), last: rmseOf(SGD7[STEPS7 - 1], VAC7) };
}
const GOALC7 = 1.95;
// @sim-end

function EnsPlot({ k }) {
  const xs = [];
  for (let i = 1; i <= K7; i++) xs.push(i);
  return (
    <Plot
      series={[
        { xs, ys: SINGLE7.map((v) => Math.min(v, 11)), color: THEME.muted, dash: "2 3", label: "model k alone" },
        { xs, ys: ENS7.map((v) => Math.min(v, 11)), color: THEME.highlight, label: "average of first k ▲", ldy: -6 },
      ]}
      xr={[0, 12]}
      yr={[0, 12]}
      xl="models averaged k →"
      label="miss on new orders (minutes) · clipped at 11"
      height={170}
      marks={[{ x: k, y: Math.min(ENS7[k - 1], 11), shape: "triangle", color: THEME.highlight }]}
      vlines={[{ x: k, color: THEME.muted }]}
    />
  );
}

export default function EnsembleLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [showAll, setShowAll] = useState(false);
  const [k, setK] = useState(1);
  const [kind, setKind] = useState("bag");
  const [beta, setBeta] = useState(0);
  const kk = screen === 0 ? (showAll ? 6 : 1) : k;
  const avg = useMemo(() => avgW(MODELS7.slice(0, kk)), [kk]);
  const pol = useMemo(() => polyak7(beta), [beta]);
  const okC = pol.final < GOALC7;

  const curves = [];
  const nFaint = screen === 0 ? (showAll ? 6 : 1) : kk;
  for (let i = 0; i < nFaint; i++) curves.push({ w: MODELS7[i], color: THEME.muted, width: 1.2, opacity: 0.55, dash: "3 3" });
  if (kk > 1) curves.push({ w: avg, color: THEME.highlight, width: 2.6, label: "average of " + kk });
  else curves.push({ w: MODELS7[0], color: THEME.accent, label: "model 1" });
  const plot = (
    <EtaPlot
      sets={[
        { data: VA7, shape: "triangle", color: THEME.highlight, faint: true },
        { data: TR7, shape: "circle", color: THEME.accent },
      ]}
      curves={curves}
      label="dashed = single models · solid = their average · ▲ new orders"
      height={230}
    />
  );

  let question, stage, side;
  if (screen === 0) {
    question = (
      <>
        Model 1 is a degree-9 fit trained on a resample of the ten orders (some orders drawn twice, some left out). It is badly wrong on new orders. Train five more the same way, each on its own resample. They are all wiggly. Show them, and their average. How can six wrong curves average to a good one?
      </>
    );
    stage = plot;
    side = (
      <Panel>
        <Toggle label="train 5 more and average all 6" value={showAll} onChange={setShowAll} />
        <Readout
          rows={[
            ["model 1 alone, new-order miss", fmt(SINGLE7[0], 2) + " min"],
            ["average of 6, new-order miss", showAll ? fmt(ENS7[5], 2) + " min" : "—", THEME.highlight],
          ]}
        />
        <Note>Each model uses a tiny penalty (λ = 0.001), so on its own it still overfits.</Note>
      </Panel>
    );
  } else if (screen === 1) {
    question = (
      <>
        Add models one at a time. Does the average keep improving, and does it matter much how bad any single model is? Where do the wiggles of different models agree, and where do they disagree?
      </>
    );
    stage = (
      <>
        {plot}
        <EnsPlot k={k} />
      </>
    );
    side = (
      <Panel>
        <Slider id="r7-k" label="models averaged" value={k} min={1} max={K7} stepSize={1} onChange={setK} display={String(k)} />
        <Readout
          rows={[
            ["model " + k + " alone", fmt(SINGLE7[k - 1], 2) + " min"],
            ["average of " + k, fmt(ENS7[k - 1], 2) + " min", THEME.highlight],
          ]}
        />
        <Note>The models agree near the orders (that's the city's shape) and disagree between them (that's each one's noise). Averaging keeps what they agree on.</Note>
      </Panel>
    );
  } else if (screen === 2) {
    const kinds = {
      bag: ["Bagging", "resample the training set, train one model per resample, average (or vote)."],
      init: ["Same model, different inits", "train after picking hyperparameters; each run lands in a different minimum."],
      snap: ["Snapshot ensemble", "save several checkpoints of one run (e.g. at the ends of a cyclic learning-rate schedule) and average them."],
      drop: ["Dropout", "every mask is one thinned network; 2^M of them share weights, and the test-time scaling approximates their average."],
      polyak: ["Polyak averaging", "keep a running average of the weights during training and use that at test time."],
    };
    question = (
      <>
        Name it. A <b>model ensemble</b>: train several models and average their predictions at test time. Independent errors partly cancel, which usually buys a few percent. <b>Bagging</b> (Breiman) makes the models different by resampling the data. Pick a variation to see where its diversity comes from.
      </>
    );
    stage = (
      <>
        {plot}
        <EnsPlot k={k} />
      </>
    );
    side = (
      <Panel>
        <Eq size={13}>
          ŷ(x) = (1/K) Σ<sub>k</sub> f<sub>k</sub>(x)
          <br />
          squared error of ŷ ≤ average squared error of the f<sub>k</sub>
        </Eq>
        <Pills options={Object.keys(kinds).map((q) => [q, kinds[q][0]])} value={kind} onPick={setKind} />
        <Note>
          <b style={{ color: THEME.text }}>{kinds[kind][0]}:</b> {kinds[kind][1]}
        </Note>
        <Slider id="r7-k2" label="models averaged" value={k} min={1} max={K7} stepSize={1} onChange={setK} display={String(k)} />
      </Panel>
    );
  } else {
    question = (
      <>
        No time to train several models. One SGD run on 40 orders with a constant step size keeps bouncing around the minimum, so the weights you happen to stop with are noisy. Keep a running average of the weights instead, w̄ ← β·w̄ + (1 − β)·w, and ship w̄. Get the new-order miss of the shipped weights at step {STEPS7} below {GOALC7} min.
      </>
    );
    const xs = pol.va.map((_, i) => Math.min(i * 15, STEPS7 - 1));
    stage = (
      <>
        <EtaPlot
          sets={[
            { data: VAC7, shape: "triangle", color: THEME.highlight, faint: true },
            { data: TRC7, shape: "circle", color: THEME.accent },
          ]}
          curves={[
            { w: SGD7[STEPS7 - 1], color: THEME.muted, dash: "4 3", label: "last SGD weights", ldy: 14 },
            { w: pol.w, color: THEME.highlight, label: "averaged weights w̄" },
          ]}
          label="40 training orders, degree 4 · ▲ new orders"
          height={220}
        />
        <Plot
          series={[
            { xs, ys: pol.raw, color: THEME.muted, dash: "4 3", width: 1.2, label: "SGD weights" },
            { xs, ys: pol.va, color: THEME.highlight, label: "averaged w̄", ldy: 14 },
          ]}
          xr={[0, STEPS7]}
          yr={[1.5, 3.5]}
          xl="SGD step →"
          label="miss on new orders (minutes)"
          height={160}
          hlines={[{ y: GOALC7, color: THEME.ok, label: "goal" }]}
        />
      </>
    );
    side = (
      <Panel edge={okC ? THEME.ok : THEME.grid}>
        <Slider id="r7-b" label="averaging β" value={beta} min={0} max={0.999} stepSize={0.001} onChange={setBeta} display={fmt(beta, 3)} />
        <Readout
          rows={[
            ["last SGD weights", fmt(pol.last, 2) + " min"],
            ["shipped averaged weights", fmt(pol.final, 2) + " min", THEME.highlight],
            ["best possible (exact fit)", fmt(LS7, 2) + " min"],
          ]}
        />
        <Goal met={okC}>shipped miss below {GOALC7} min</Goal>
        <Hint
          hints={[
            "β = 0 ships the last noisy weights. What does a larger β average over?",
            "β = 0.9 averages roughly the last 10 steps; 0.99 roughly the last 100. Too close to 1 and w̄ still remembers the start of training.",
          ]}
        />
      </Panel>
    );
  }
  return <LessonFrame screen={screen} onScreen={setScreen} title="Many wrong models, one good average" question={question} stage={stage} side={side} />;
}
