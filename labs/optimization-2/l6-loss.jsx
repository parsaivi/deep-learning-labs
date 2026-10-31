import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import { THEME, MONO } from "./common";

// Lesson 6 - The shape of the loss decides how gradient descent behaves (Optimization II, slides 66-71, wrap-up 72-76).
// Aha: with a sigmoid output, L2 multiplies the error by sigma'(z), which dies when the model is confidently
// wrong, so the worse the model, the weaker the push: a plateau. Cross-entropy cancels that factor: the push
// is the error itself, steep far from the optimum and gentle near it.
// Setup as on the slides: one sigmoid unit, 2-D input, no bias, 100 random examples, loss over the two weights.

// @sim-start
function sigm(z) {
  return 1 / (1 + Math.exp(-z));
}
function softplus(z) {
  return z > 0 ? z + Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z));
}
function makeLate(seed, n, wt) {
  const r = rng(seed);
  const X = [];
  const Y = [];
  for (let i = 0; i < n; i++) {
    const x = [gauss(r), gauss(r)];
    X.push(x);
    Y.push(r() < sigm(wt[0] * x[0] + wt[1] * x[1]) ? 1 : 0);
  }
  return { X, Y, n };
}
const LATE = makeLate(8, 100, [-2, 3.5]);
function lossFn(kind) {
  return function (w) {
    let s = 0;
    for (let i = 0; i < LATE.n; i++) {
      const z = w[0] * LATE.X[i][0] + w[1] * LATE.X[i][1];
      if (kind === "l2") {
        const e = sigm(z) - LATE.Y[i];
        s += e * e;
      } else s += LATE.Y[i] ? softplus(-z) : softplus(z);
    }
    return s / LATE.n;
  };
}
const F_L2 = lossFn("l2");
const F_CE = lossFn("ce");
function accuracy(w) {
  let c = 0;
  for (let i = 0; i < LATE.n; i++) c += (w[0] * LATE.X[i][0] + w[1] * LATE.X[i][1] > 0 ? 1 : 0) === LATE.Y[i] ? 1 : 0;
  return c / LATE.n;
}
const STEPS6 = 60;
const STEPS6C = 40;
const ACC = 0.85;
const STARTS_C = [
  [1, 1],
  [5, -5],
  [12, -12],
];
const ETA_CAP = 30;
function train6(kind, start, eta, steps) {
  const r = runOpt(kind === "l2" ? F_L2 : F_CE, start, { kind: "gd", eta }, steps, 80);
  const acc = r.pts.map(accuracy);
  const reach = acc.findIndex((a) => a >= ACC);
  const held = !r.diverged && acc.length === steps + 1 && acc.slice(-10).every((a) => a >= ACC);
  return { r, acc, reach, held };
}
// @sim-end

const XR6 = [-14, 14];
const SURF_L2 = { f: F_L2, xr: XR6, yr: XR6, xl: "w₁ (slack)", yl: "w₂ (rain)", floor: 0.004 };
const SURF_CE = { f: F_CE, xr: XR6, yr: XR6, xl: "w₁ (slack)", yl: "w₂ (rain)", floor: 0.02 };
const LOSSES = [
  ["l2", "L2: (σ(z) − y)²"],
  ["ce", "cross-entropy"],
];
const START_SHAPES = ["circle", "square", "triangle"];

function gradSize(f, w) {
  const g = grad2(f, w);
  return Math.hypot(g[0], g[1]);
}
function MapCard({ title, surf, res, start, onHandle }) {
  return (
    <div style={{ flex: "1 1 240px", minWidth: 0 }}>
      <Label>{title}</Label>
      <LossMap surf={surf} target={null} handle={start} onHandle={onHandle} maxWidth={340} paths={[{ key: "p", pts: res.r.pts, color: THEME.highlight, shape: "circle", width: 1.6, label: res.held ? "≥ 85%" : fmt(res.acc[res.acc.length - 1] * 100, 0) + "%" }]} ariaLabel={title + " loss map with the gradient-descent path"} />
    </div>
  );
}
function OneOrder({ z }) {
  const zs = [];
  const l2 = [];
  const ce = [];
  const g2 = [];
  const gc = [];
  for (let v = -8; v <= 8.001; v += 0.08) {
    const s = sigm(v);
    zs.push(v);
    l2.push((s - 1) * (s - 1));
    ce.push(softplus(-v));
    g2.push(Math.abs(2 * (s - 1) * s * (1 - s)));
    gc.push(Math.abs(s - 1));
  }
  const s = sigm(z);
  return (
    <>
      <Plot
        series={[
          { xs: zs, ys: ce, color: THEME.accent, width: 2.2 },
          { xs: zs, ys: l2, color: THEME.highlight, dash: "6 3", width: 2.2 },
        ]}
        xr={[-8, 8]}
        yr={[0, 4]}
        xl="score z for a late order (y = 1) →"
        label="loss of one order"
        legend={[{ color: THEME.accent, text: "● cross-entropy −log σ(z)" }, { color: THEME.highlight, dash: "6 3", text: "■ L2 (σ(z) − 1)²" }]}
        height={170}
        vlines={[{ x: z, color: THEME.text, label: "z = " + fmt(z, 1) }]}
        marks={[
          { x: z, y: softplus(-z), shape: "circle", color: THEME.accent },
          { x: z, y: (s - 1) * (s - 1), shape: "square", color: THEME.highlight },
        ]}
      />
      <Plot
        series={[
          { xs: zs, ys: gc, color: THEME.accent, width: 2.2 },
          { xs: zs, ys: g2, color: THEME.highlight, dash: "6 3", width: 2.2 },
        ]}
        xr={[-8, 8]}
        yr={[-4, 0.3]}
        logY
        xl="score z →"
        label="how hard that order pushes the weights (log scale)"
        legend={[{ color: THEME.accent, text: "|dCE/dz| = |σ − y|" }, { color: THEME.highlight, dash: "6 3", text: "|dL2/dz|" }]}
        height={160}
        vlines={[{ x: z, color: THEME.text }]}
      />
    </>
  );
}

export default function LossLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [start, setStart] = useState([5, -5]);
  const [eta, setEta] = useState(2);
  const [z, setZ] = useState(-1);
  const [kindC, setKindC] = useState("l2");
  const [etaC, setEtaC] = useState(2);
  const onHandle = useCallback((x, y) => setStart([x, y]), []);
  const rL2 = useMemo(() => train6("l2", start, eta, STEPS6), [start, eta]);
  const rCE = useMemo(() => train6("ce", start, eta, STEPS6), [start, eta]);
  const runsC = useMemo(() => STARTS_C.map((s) => train6(kindC, s, etaC, STEPS6C)), [kindC, etaC]);
  const okC = runsC.every((r) => r.held);

  const maps = (
    <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
      <MapCard title="L2 loss" surf={SURF_L2} res={rL2} start={start} onHandle={onHandle} />
      <MapCard title="cross-entropy loss" surf={SURF_CE} res={rCE} start={start} onHandle={onHandle} />
    </div>
  );
  const compare = (
    <Readout
      rows={[
        ["gradient size at start: L2", gradSize(F_L2, start).toExponential(1)],
        ["gradient size at start: cross-entropy", gradSize(F_CE, start).toExponential(1)],
        ["L2: step reaching 85% accuracy", rL2.reach >= 0 ? rL2.reach : "not in " + STEPS6],
        ["cross-entropy: step reaching 85%", rCE.reach >= 0 ? rCE.reach : "not in " + STEPS6],
      ]}
    />
  );

  let question, stage, side;
  if (screen === 0) {
    question = (
      <>
        Will an order arrive late? One sigmoid unit reads two numbers per order: minutes of slack before the promised time (x₁) and rain (x₂). It outputs σ(w₁x₁ + w₂x₂), using 100 past orders. Two copies train from the same start with the same η: one minimizes L2, the other cross-entropy. Drag the start into the far corners. Why does one of them crawl?
      </>
    );
    stage = maps;
    side = (
      <>
        <Panel>
          <LogSlider id="l6-eta" label="learning rate η (both)" value={eta} lo={-1} hi={1.5} onChange={setEta} digits={2} />
          {compare}
        </Panel>
        <Note>The bottom-right corner means "more slack → later, rain → earlier": confidently wrong. Compare the gradient size there with the size near the middle.</Note>
      </>
    );
  } else if (screen === 1) {
    const s = sigm(z);
    question = (
      <>
        Zoom in on a single late order (y = 1) and the score z the unit gives it. Slide z from right (confidently correct) to left (confidently wrong). As the prediction gets <i>worse</i>, what happens to each loss's push on the weights?
      </>
    );
    stage = <OneOrder z={z} />;
    side = (
      <>
        <Panel>
          <Slider id="l6-z" label="score z of the late order" value={z} min={-8} max={8} stepSize={0.01} onChange={setZ} display={fmt(z, 2)} />
          <Readout
            rows={[
              ["predicted P(late) = σ(z)", fmt(s, 4)],
              ["● cross-entropy push |σ − y|", fmt(Math.abs(s - 1), 4)],
              ["■ L2 push |2(σ − y) σ(1 − σ)|", Math.abs(2 * (s - 1) * s * (1 - s)).toExponential(2)],
            ]}
          />
        </Panel>
        <Note>Where is the L2 push largest? What happens to it as z goes past that point toward −8?</Note>
      </>
    );
  } else if (screen === 2) {
    question = (
      <>
        Name it. A good loss is steep far from the optimum, so a bad start still gets a strong push, and gentle near it, ideally like a quadratic bowl, so the steps do not overshoot. L2 on a sigmoid breaks the first rule. The chain rule puts in σ′(z) = σ(1 − σ), which vanishes exactly when the unit is confidently wrong.
      </>
    );
    stage = maps;
    side = (
      <>
        <Panel>
          <Eq size={12.5}>
            L2: &nbsp;&nbsp;∂L/∂z = 2(σ − y) · σ(1 − σ) &nbsp;→ 0 when σ saturates
            <br />
            CE: &nbsp;&nbsp;∂L/∂z = σ − y &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;stays ≈ 1 when wrong
            <br />
            CE = −y log σ − (1 − y) log(1 − σ)
          </Eq>
          <LogSlider id="l6-etan" label="learning rate η (both)" value={eta} lo={-1} hi={1.5} onChange={setEta} digits={2} />
          {compare}
        </Panel>
        <Note>
          On the maps: L2 is not convex here (flat shelves far out), while cross-entropy is convex; both have a unique minimum. The usual pairing is L2 for regression outputs and cross-entropy (KL) for sigmoid or softmax outputs.
        </Note>
        <Panel>
          <Label>wrap-up: the deck's last slides</Label>
          <Note>
            Second-order steps (Newton, part I) need no learning rate but cost O(n²) memory and O(n³) time for n weights. L-BFGS approximates them cheaply and works well in full-batch, noise-free training, but not with mini-batches. In practice: Adam is a good default. SGD + momentum can beat it, with more tuning of η and its schedule.
          </Note>
        </Panel>
      </>
    );
  } else {
    stage = (
      <LossMap
        surf={kindC === "l2" ? SURF_L2 : SURF_CE}
        target={null}
        maxWidth={460}
        paths={runsC.map((r, i) => ({ key: "c" + i, pts: r.r.pts, color: r.held ? THEME.ok : THEME.highlight, shape: START_SHAPES[i], width: 1.5, label: ["near", "wrong", "very wrong"][i] + (r.held ? " ✓" : "") }))}
        ariaLabel="Loss map with three training runs from three starts"
      />
    );
    question = (
      <>
        Your unit is one layer of a bigger network, and steps larger than η = {ETA_CAP} wreck the other layers. Training must work from any start: near the answer, wrong, and very wrong. Choose the loss and one η ≤ {ETA_CAP} so that all three runs reach 85% accuracy and hold it through the last 10 of {STEPS6C} steps.
      </>
    );
    side = (
      <>
        <Panel edge={okC ? THEME.ok : THEME.grid}>
          <Pills options={LOSSES} value={kindC} onPick={setKindC} />
          <LogSlider id="l6-etac" label={"learning rate η (max " + ETA_CAP + ")"} value={etaC} lo={-1} hi={Math.log10(ETA_CAP)} onChange={setEtaC} digits={2} />
          <Readout
            rows={runsC.map((r, i) => [
              ["● near", "■ wrong", "▲ very wrong"][i] + " start [" + STARTS_C[i].join(", ") + "]",
              r.r.diverged ? "diverged" : fmt(r.acc[r.acc.length - 1] * 100, 0) + "%" + (r.held ? " held" : ""),
              r.held ? THEME.ok : undefined,
            ])}
          />
          <Goal met={okC}>all three starts reach and hold 85%</Goal>
          <Hint
            hints={[
              "With L2, push η to its cap and look at the very-wrong start. Is its gradient big enough for any η you are allowed?",
              "Switch loss. The push no longer fades with how wrong the unit is, so one moderate η works for every start. Then check that η is not so large that the near start overshoots.",
            ]}
          />
        </Panel>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="Steep far, gentle near" question={question} stage={stage} side={side} />;
}
