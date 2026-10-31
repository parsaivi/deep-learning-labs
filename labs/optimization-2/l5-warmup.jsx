import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import { THEME, MONO } from "./common";

// Lesson 5 - Learning-rate schedules and warmup (Optimization II, slides 59-65).
// Aha: a step is safe only while eta x sharpness < 2. A fresh network sits in a sharp region, so a big
// eta explodes at once; a short warmup lets the first small steps fix the sharp directions, after which
// a big eta is safe. At the end, the mini-batch noise floor grows with eta, so eta must come back down.
// Model: 60 directions, curvature 100 down to 0.1. The 3 sharpest directions are extra sharp
// (x (1 + s * their remaining loss share)) until they are fixed. We plot the exact expected loss.

// @sim-start
const D5 = 60;
const LMAX5 = 100;
const TOP5 = 3;
const NZ5 = 0.5;
function makeNet5(s, share) {
  const lam = [];
  const q0 = [];
  for (let i = 0; i < D5; i++) lam.push(LMAX5 * Math.pow(10, (-3 * i) / (D5 - 1)));
  for (let i = 0; i < D5; i++) q0.push((2 * (i < TOP5 ? (2.0 * share) / TOP5 : (2.0 * (1 - share)) / (D5 - TOP5))) / lam[i]);
  let b0 = 0;
  for (let i = 0; i < TOP5; i++) b0 += lam[i] * q0[i];
  return { lam, q0, b0, s };
}
const NET5 = makeNet5(6, 0.6);
const NET5C = makeNet5(12, 0.6);
function schedule(kind, peak, warm, T) {
  const w = Math.round(warm * T);
  return function (k) {
    if (k < w) return (peak * (k + 1)) / w;
    const t = (k - w) / Math.max(1, T - w);
    if (kind === "cos") return peak * 0.5 * (1 + Math.cos(Math.PI * t));
    if (kind === "lin") return peak * (1 - t);
    if (kind === "isqrt") return peak / Math.sqrt(1 + (k - w) / 10);
    if (kind === "step") return peak * (t < 0.5 ? 1 : t < 0.75 ? 0.1 : 0.01);
    return peak;
  };
}
// expected loss: mean part q (shrinks deterministically) and noise part v, per direction
function train5(net, sched, T) {
  const q = net.q0.slice();
  const v = net.q0.map(() => 0);
  const loss = [];
  const etas = [];
  const margin = [];
  for (let k = 0; k < T; k++) {
    let A = 0;
    let B = 0;
    for (let i = 0; i < D5; i++) {
      A += net.lam[i] * (q[i] + v[i]);
      if (i < TOP5) B += net.lam[i] * q[i];
    }
    const L = 0.3 + A / 2;
    if (!isFinite(L) || L > 40) {
      loss.push(60);
      return { loss, etas, margin, nanAt: k };
    }
    loss.push(L);
    const m = 1 + (net.s * B) / net.b0;
    const e = sched(k);
    etas.push(e);
    margin.push(e * LMAX5 * m);
    for (let i = 0; i < D5; i++) {
      const f = Math.pow(1 - e * net.lam[i] * (i < TOP5 ? m : 1), 2);
      q[i] = f * q[i];
      v[i] = f * v[i] + e * e * net.lam[i] * NZ5 * NZ5;
    }
  }
  return { loss, etas, margin, nanAt: -1 };
}
function final5(t) {
  return t.nanAt >= 0 ? Infinity : t.loss[t.loss.length - 1];
}
const T5 = 300;
const TARGET5 = 0.5;
const T5C = 200;
const TARGET5C = 0.56;
// @sim-end

const KINDS5 = [
  ["const", "constant"],
  ["step", "step ×0.1"],
  ["cos", "cosine"],
  ["lin", "linear"],
  ["isqrt", "1/√t"],
];

function Charts({ t, T, target, compact }) {
  const ks = Array.from({ length: T }, (_, k) => k);
  return (
    <>
      <Plot
        series={[{ xs: ks.slice(0, t.loss.length), ys: t.loss, color: THEME.highlight, label: t.nanAt >= 0 ? "NaN at step " + t.nanAt : "loss", width: 2.2 }]}
        xr={[0, T]}
        yr={[-0.5, 0.6]}
        logY
        xl="iteration →"
        label="expected training loss (log scale)"
        height={compact ? 160 : 180}
        hlines={[{ y: target, color: THEME.ok, label: "target " + target }]}
      />
      <Plot
        series={[{ xs: ks.slice(0, t.margin.length), ys: t.margin, color: THEME.accent, label: "η × sharpness", width: 2 }]}
        xr={[0, T]}
        yr={[0, 3]}
        xl="iteration →"
        label="η × sharpest curvature: steps are stable only below 2"
        height={130}
        hlines={[{ y: 2, color: THEME.warn, dash: "6 3", label: "2: unstable above" }]}
      />
      <Plot series={[{ xs: ks.slice(0, t.etas.length), ys: t.etas, color: THEME.text, dash: "5 3", label: "ηₜ" }]} xr={[0, T]} yr={[0, 0.03]} xl="iteration →" label="learning rate schedule" height={110} />
    </>
  );
}

export default function WarmupLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [etaConst, setEtaConst] = useState(0.002);
  const [kind, setKind] = useState("const");
  const [warm, setWarm] = useState(0.05);
  const [peak, setPeak] = useState(0.01);
  const [kindC, setKindC] = useState("const");
  const [warmC, setWarmC] = useState(0);
  const [peakC, setPeakC] = useState(0.002);
  const tP = useMemo(() => train5(NET5, () => etaConst, T5), [etaConst]);
  const t = useMemo(() => train5(NET5, schedule(kind, peak, warm, T5), T5), [kind, peak, warm]);
  const tC = useMemo(() => train5(NET5C, schedule(kindC, peakC, warmC, T5C), T5C), [kindC, peakC, warmC]);
  const okP = final5(tP) < TARGET5;
  const ok = final5(t) < TARGET5;
  const okC = final5(tC) < TARGET5C;
  const status = (tt, target, good) => (
    <>
      <Readout rows={[["loss at the end", tt.nanAt >= 0 ? "NaN at step " + tt.nanAt : fmt(final5(tt), 3), tt.nanAt >= 0 ? THEME.warn : undefined]]} />
      <Goal met={tt.nanAt < 0}>no NaN</Goal>
      <Goal met={good}>final loss below {target}</Goal>
    </>
  );
  const controls = (k, setK, pk, setPk, w, setW, id) => (
    <>
      <Pills options={KINDS5} value={k} onPick={setK} />
      <LogSlider id={id + "-peak"} label="peak learning rate" value={pk} lo={-3} hi={-1.55} onChange={setPk} digits={4} />
      <Slider id={id + "-warm"} label="warmup (share of training)" value={w} min={0} max={0.3} stepSize={0.005} onChange={setW} display={fmt(w * 100, 1) + " %"} />
    </>
  );

  let question, stage, side;
  if (screen === 0) {
    question = (
      <>
        A fresh network, 300 steps. Its loss is very sharp at the start: the first predictions are wildly off. The middle chart multiplies η by the sharpest curvature at each step. Find a single fixed η that ends below {TARGET5}.
      </>
    );
    stage = <Charts t={tP} T={T5} target={TARGET5} />;
    side = (
      <>
        <Panel edge={okP ? THEME.ok : THEME.grid}>
          <LogSlider id="l5-const" label="fixed learning rate η" value={etaConst} lo={-3} hi={-1.55} onChange={setEtaConst} digits={4} />
          {status(tP, TARGET5, okP)}
        </Panel>
        <Note>Find the largest η that survives. Where does η × sharpness sit at step 0, and where does it sit after 50 steps of training?</Note>
      </>
    );
  } else if (screen === 1) {
    question = (
      <>
        Now η can change during training. Start η at zero and raise it in a straight line over the warmup, then follow a decay shape down to the end. Get below {TARGET5}. Which part of the curve does the warmup fix, and which part does the decay fix?
      </>
    );
    stage = <Charts t={t} T={T5} target={TARGET5} />;
    side = (
      <>
        <Panel edge={ok ? THEME.ok : THEME.grid}>
          {controls(kind, setKind, peak, setPeak, warm, setWarm, "l5")}
          {status(t, TARGET5, ok)}
        </Panel>
        <Note>With a 5% warmup, compare "constant" with "cosine" at the same peak. Then set warmup to 0 and watch the start.</Note>
      </>
    );
  } else if (screen === 2) {
    question = (
      <>
        Name it. Warmup keeps η × sharpness under 2 while the first steps fix the sharp directions. Decay lowers the noise floor at the end, because the floor grows with η. The common shapes, with α₀ the peak, t the step after warmup and T the steps left:
      </>
    );
    stage = <Charts t={t} T={T5} target={TARGET5} compact />;
    side = (
      <>
        <Panel>
          <Eq size={12.5}>
            step: &nbsp;&nbsp;α₀, then ×0.1 at fixed points (e.g. ResNet: epochs 30, 60, 90)
            <br />
            cosine: αₜ = ½ α₀ (1 + cos(tπ/T))
            <br />
            linear: αₜ = α₀ (1 − t/T)
            <br />
            1/√t: &nbsp;&nbsp;αₜ = α₀ / √t &nbsp;(here √(1 + t/10))
            <br />
            warmup: rise linearly from 0 (e.g. first ~5,000 iterations)
          </Eq>
          {controls(kind, setKind, peak, setPeak, warm, setWarm, "l5n")}
        </Panel>
        <Note>Rule of thumb from large-batch training: batch size × N ⇒ initial learning rate × N (lesson 3), and use warmup, because that larger η is exactly what explodes on a fresh network.</Note>
      </>
    );
  } else {
    question = (
      <>
        A bigger model: its start is twice as sharp, and you can afford only {T5C} steps. Design a schedule that ends below {TARGET5C} with no NaN.
      </>
    );
    stage = <Charts t={tC} T={T5C} target={TARGET5C} />;
    side = (
      <>
        <Panel edge={okC ? THEME.ok : THEME.grid}>
          {controls(kindC, setKindC, peakC, setPeakC, warmC, setWarmC, "l5c")}
          {status(tC, TARGET5C, okC)}
          <Hint
            hints={[
              "Watch the middle chart. Every NaN starts with the blue line going above 2. Where does it go above 2: during the warmup, right after it, or at the peak?",
              "The peak has to stay below 2/100 = 0.02 even once the network is calm. Use a warmup long enough that the blue line stays under 2 the whole way up, and a shape that ends near zero.",
            ]}
          />
        </Panel>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="Warm up, then cool down" question={question} stage={stage} side={side} />;
}
