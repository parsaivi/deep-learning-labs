import { useState, useMemo } from "react";
import { THEME, MONO } from "./common";

// Lesson 2 - L2 regularization / weight decay (Regularization deck, slides 7-10).
// Aha: the wild curve needs huge weights that cancel each other. Charging the fit lambda * ||w||^2
// keeps all 10 weights but makes big ones expensive, so the fit trades a little training error for a
// much calmer curve. In gradient descent the charge becomes "shrink every weight by (1 - 2*alpha*lambda)
// before each step": weight decay.

// @sim-start
const TR2 = makeETA(1, 10, 1.6);
const VA2 = makeETA(901, 60, 1.6);
const D2 = 9;
function fit2(lam, tr, va) {
  const w = ridgeFit(tr, D2, lam);
  let l = 0;
  for (let i = 0; i < tr.n; i++) l += Math.pow(polyAt(w, tr.km[i]) - tr.y[i], 2) / tr.n;
  const n2 = Math.pow(wNorm(w), 2);
  return { w, tr: rmseOf(w, tr), va: rmseOf(w, va), L: l, R: n2, J: l + lam * n2 };
}
const SWEEP2 = [];
for (let e = -6; e <= 0.5001; e += 0.05) SWEEP2.push({ e, f: fit2(Math.pow(10, e), TR2, VA2) });
// challenge city
const TRC2 = makeETA(6, 10, 1.6);
const VAC2 = makeETA(906, 60, 1.6);
const SWEEPC2 = [];
for (let e = -6; e <= 0.5001; e += 0.05) SWEEPC2.push({ e, f: fit2(Math.pow(10, e), TRC2, VAC2) });
const GOALC2 = 2.25;
// @sim-end

function SweepPlot({ sweep, lam, goal }) {
  const e = Math.log10(lam);
  const cur = fit2(lam, sweep === SWEEP2 ? TR2 : TRC2, sweep === SWEEP2 ? VA2 : VAC2);
  const hl = goal ? [{ y: goal, color: THEME.ok, label: "goal" }] : [];
  return (
    <Plot
      series={[
        { xs: sweep.map((s) => s.e), ys: sweep.map((s) => Math.min(s.f.tr, 11)), color: THEME.accent, label: "training ●" },
        { xs: sweep.map((s) => s.e), ys: sweep.map((s) => Math.min(s.f.va, 11)), color: THEME.highlight, dash: "6 3", label: "new orders ▲", ldy: -4 },
      ]}
      xr={[-6, 0.5]}
      yr={[0, 12]}
      xl="log₁₀ λ (penalty strength) →"
      label="typical miss (minutes) · clipped at 11"
      height={180}
      hlines={hl}
      vlines={[{ x: e, color: THEME.muted }]}
      marks={[
        { x: e, y: Math.min(cur.tr, 11), shape: "circle", color: THEME.accent },
        { x: e, y: Math.min(cur.va, 11), shape: "triangle", color: THEME.highlight },
      ]}
    />
  );
}

export default function DecayLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [pick, setPick] = useState("a");
  const [lam, setLam] = useState(1e-6);
  const [lamC, setLamC] = useState(1e-5);
  const [alpha, setAlpha] = useState(0.1);
  const lamUsed = screen === 0 ? (pick === "a" ? 1e-6 : Math.pow(10, -1.75)) : lam;
  const cur = useMemo(() => fit2(lamUsed, TR2, VA2), [lamUsed]);
  const curC = useMemo(() => fit2(lamC, TRC2, VAC2), [lamC]);
  const okC = curC.va < GOALC2;
  const maxW = Math.max.apply(null, cur.w.slice(1).map(Math.abs));

  const plot = (
    <EtaPlot
      sets={[
        { data: VA2, shape: "triangle", color: THEME.highlight, faint: true },
        { data: TR2, shape: "circle", color: THEME.accent },
      ]}
      curves={[{ w: cur.w, color: THEME.accent, label: screen === 0 ? (pick === "a" ? "fit A" : "fit B") : "λ = " + (lamUsed < 1e-3 ? lamUsed.toExponential(0) : fmt(lamUsed, 3)) }]}
      label="degree 9, all 10 weights kept · ● training · ▲ new orders"
      height={230}
    />
  );
  const bars = <WeightBars w={cur.w} cap={40} />;
  const rows = [
    ["largest |weight|", fmt(maxW, 1)],
    ["miss on training orders", fmt(cur.tr, 2) + " min"],
    ["miss on new orders", fmt(cur.va, 2) + " min", THEME.highlight],
  ];

  let question, stage, side;
  if (screen === 0) {
    question = (
      <>
        Two fits, both degree 9, both with 10 weights. A threads all ten orders. B misses each one by about a minute. Flip between them: look at the size of the weights under each curve. What does A need its weights to do to hit every dot?
      </>
    );
    stage = (
      <>
        {plot}
        {bars}
      </>
    );
    side = (
      <Panel>
        <Pills options={[["a", "fit A"], ["b", "fit B"]]} value={pick} onPick={setPick} />
        <Readout rows={rows} />
        <Note>The bars are the weights w1…w9 of the curve (w0, the average ETA, is not shown). ▲ on a bar: clipped.</Note>
      </Panel>
    );
  } else if (screen === 1) {
    question = (
      <>
        Add a charge for big weights: the fit now minimises (training error) + λ·(sum of squared weights). Slide λ from almost nothing to large. When does the curve calm down? What happens to the training miss? Where is the new-order miss lowest?
      </>
    );
    stage = (
      <>
        {plot}
        <SweepPlot sweep={SWEEP2} lam={lam} />
      </>
    );
    side = (
      <Panel>
        <LogSlider id="r2-l" label="penalty λ" value={lam} lo={-6} hi={0.5} onChange={setLam} digits={4} />
        <Readout rows={rows} />
        {bars}
        <Note>Large λ: the cheapest curve is nearly flat, so it underfits. Somewhere between is the sweet spot.</Note>
      </Panel>
    );
  } else if (screen === 2) {
    const decay = 1 - 2 * alpha * lam;
    question = (
      <>
        Name it. This is <b>L2 regularization</b>: J(W) = L(W) + λ‖W‖². Take one gradient step on J and the penalty's gradient 2λW turns into a shrink: every weight is multiplied by (1 − 2αλ) before the usual step. So it is also called <b>weight decay</b>. Small weights keep each tanh or sigmoid unit in its straight middle, so the network can only make gentle bends.
      </>
    );
    stage = (
      <>
        {plot}
        <SweepPlot sweep={SWEEP2} lam={lam} />
      </>
    );
    side = (
      <Panel>
        <Eq>
          J(W) = (1/N) Σ L<sub>n</sub>(W) + λ ‖W‖²
          <br />
          W ← W − α(∇L + 2λW)
          <br />W ← W·(1 − 2αλ) − α∇L
        </Eq>
        <LogSlider id="r2-l2" label="penalty λ" value={lam} lo={-6} hi={0.5} onChange={setLam} digits={4} />
        <Slider id="r2-a" label="learning rate α" value={alpha} min={0.01} max={0.5} stepSize={0.005} onChange={setAlpha} />
        <Readout
          rows={[
            ["L (training error)", fmt(cur.L, 2)],
            ["‖W‖²", fmt(cur.R, 1)],
            ["J = L + λ‖W‖²", fmt(cur.J, 2)],
            ["shrink per step (1 − 2αλ)", fmt(decay, 5), THEME.highlight],
          ]}
        />
        <Note>Bigger λ means a worse fit on the training data in exchange for a network that generalizes. λ is a hyperparameter: pick it on validation data, never on training data.</Note>
      </Panel>
    );
  } else {
    question = (
      <>
        A new city, same degree-9 model, ten fresh orders. Choose λ so the miss on 60 new orders drops below {GOALC2} minutes.
      </>
    );
    const wC = curC.w;
    stage = (
      <>
        <EtaPlot
          sets={[
            { data: VAC2, shape: "triangle", color: THEME.highlight, faint: true },
            { data: TRC2, shape: "circle", color: THEME.accent },
          ]}
          curves={[{ w: wC, color: THEME.accent, label: "your fit" }]}
          label="new city · ● training · ▲ new orders"
          height={230}
        />
        <WeightBars w={wC} cap={40} />
      </>
    );
    side = (
      <Panel edge={okC ? THEME.ok : THEME.grid}>
        <LogSlider id="r2-lc" label="penalty λ" value={lamC} lo={-6} hi={0.5} onChange={setLamC} digits={4} />
        <Readout
          rows={[
            ["miss on training orders", fmt(curC.tr, 2) + " min"],
            ["miss on new orders", fmt(curC.va, 2) + " min", THEME.highlight],
          ]}
        />
        <Goal met={okC}>new-order miss below {GOALC2} min</Goal>
        <Hint
          hints={[
            "Start tiny and raise λ slowly. Watch the weight bars, not just the curve: when do they stop being clipped?",
            "Too much λ flattens the S-bend that real traffic makes. The sweet spot keeps the bend but drops the wiggles: λ around a few hundredths.",
          ]}
        />
      </Panel>
    );
  }
  return <LessonFrame screen={screen} onScreen={setScreen} title="Charge the fit for big weights" question={question} stage={stage} side={side} />;
}
