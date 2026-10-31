import { useState, useMemo, useCallback } from "react";
import { THEME, MONO } from "./common";

// Lesson 3 - Constant initialization and symmetry (Practical Issues, slides 32-35).
// Aha: hidden units that start identical get identical gradients, so they stay identical forever:
// three units act as one. Random initial weights break the tie. Too little randomness and they stay
// near-copies for a long time; too much and they start saturated.

// @sim-start
const H3 = 3;
const N3 = 40;
const XS3 = Array.from({ length: N3 }, (_, i) => -2 + (4 * i) / (N3 - 1));
// relative ETA over standardised distance: a jump at the edge of the centre, a second at the ring road
const TGT3 = (x) => 0.9 * Math.tanh(3 * (x + 0.7)) + 0.7 * Math.tanh(3 * (x - 1.0));
const YS3 = XS3.map(TGT3);
const DIR3 = (function () {
  const r = rng(7);
  const d = [];
  for (let i = 0; i < 3 * H3; i++) d.push(gauss(r));
  return d;
})();
const STEPS3 = 1000;
const ETA3 = 0.3;
// full-batch gradient descent on 1 -> 3 tanh -> 1. Returns snapshots of the weights every 10 steps.
function train3(c, sd) {
  const w = [], b = [], v = [];
  for (let j = 0; j < H3; j++) {
    w.push(c + sd * DIR3[j]);
    b.push(c + sd * DIR3[H3 + j]);
    v.push(c + sd * DIR3[2 * H3 + j]);
  }
  let a = 0;
  const snaps = [];
  const losses = [];
  for (let k = 0; k <= STEPS3; k++) {
    const gw = [0, 0, 0], gb = [0, 0, 0], gv = [0, 0, 0];
    let ga = 0;
    let L = 0;
    for (let n = 0; n < N3; n++) {
      const h = [];
      let y = a;
      for (let j = 0; j < H3; j++) {
        h.push(Math.tanh(w[j] * XS3[n] + b[j]));
        y += v[j] * h[j];
      }
      const e = y - YS3[n];
      L += (0.5 * e * e) / N3;
      ga += e / N3;
      for (let j = 0; j < H3; j++) {
        gv[j] += (e * h[j]) / N3;
        const d = e * v[j] * (1 - h[j] * h[j]);
        gw[j] += (d * XS3[n]) / N3;
        gb[j] += d / N3;
      }
    }
    if (k % 10 === 0) {
      snaps.push({ w: w.slice(), b: b.slice(), v: v.slice(), a });
      losses.push(L);
    }
    if (k === STEPS3) break;
    for (let j = 0; j < H3; j++) {
      w[j] -= ETA3 * gw[j];
      b[j] -= ETA3 * gb[j];
      v[j] -= ETA3 * gv[j];
    }
    a -= ETA3 * ga;
  }
  return { snaps, losses };
}
// how many genuinely different units: pairs closer than tol in (w, b, v) count as copies
function distinct3(s) {
  let n = H3;
  const same = [];
  for (let i = 0; i < H3; i++)
    for (let j = i + 1; j < H3; j++) {
      const d = Math.abs(s.w[i] - s.w[j]) + Math.abs(s.b[i] - s.b[j]) + Math.abs(s.v[i] - s.v[j]);
      if (d < 0.02 && !same.includes(j)) {
        same.push(j);
        n--;
      }
    }
  return n;
}
const GOAL3 = 0.004;
// @sim-end

const UNIT_STYLE = [
  [THEME.ok, "6 3"],
  [THEME.warn, "2 3"],
  [THEME.text, "9 3 2 3"],
];
const GRID3 = Array.from({ length: 121 }, (_, i) => -2 + (4 * i) / 120);
const KS3 = Array.from({ length: STEPS3 / 10 + 1 }, (_, i) => i * 10);

function FitPanel({ snap, showUnits }) {
  const unitSeries = [0, 1, 2].map((j) => ({
    xs: GRID3,
    ys: GRID3.map((x) => snap.v[j] * Math.tanh(snap.w[j] * x + snap.b[j])),
    color: UNIT_STYLE[j][0],
    dash: UNIT_STYLE[j][1],
    width: 1.4,
  }));
  const fit = { xs: GRID3, ys: GRID3.map((x) => snap.a + [0, 1, 2].reduce((s, j) => s + snap.v[j] * Math.tanh(snap.w[j] * x + snap.b[j]), 0)), color: THEME.highlight, width: 2.6 };
  return (
    <Plot
      series={(showUnits ? unitSeries : []).concat([fit])}
      marks={XS3.map((x, i) => ({ x, y: YS3[i], shape: "circle", color: THEME.muted }))}
      legend={[{ color: THEME.highlight, width: 2.6, text: "network's ETA" }].concat(showUnits ? [0, 1, 2].map((j) => ({ color: UNIT_STYLE[j][0], dash: UNIT_STYLE[j][1], text: "unit " + (j + 1) + " part" })) : [])}
      xr={[-2, 2]}
      yr={[-2, 2]}
      xl="distance (standardised) →"
      label="● orders · extra minutes vs. average"
      height={230}
    />
  );
}
function WeightTrace({ snaps, upto }) {
  return (
    <Plot
      series={[0, 1, 2].map((j) => ({ xs: KS3.slice(0, upto + 1), ys: snaps.slice(0, upto + 1).map((s) => s.w[j]), color: UNIT_STYLE[j][0], dash: UNIT_STYLE[j][1], label: "w" + (j + 1), ldy: j * 11 }))}
      xr={[0, STEPS3]}
      yr={[-1, 4]}
      xl="training step →"
      label="input weight of each hidden unit"
      height={150}
    />
  );
}

export default function TwinsLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [c, setC] = useState(0.5);
  const [noise, setNoise] = useState(false);
  const [sd, setSd] = useState(1e-3);
  const [step, setStep] = useState(STEPS3);
  const [sdC, setSdC] = useState(1e-2);
  const sdNow = screen === 0 || !noise ? 0 : sd;
  const run = useMemo(() => train3(c, sdNow), [c, sdNow]);
  const runC = useMemo(() => train3(0, sdC), [sdC]);
  const upto = Math.round(step / 10);
  const snap = run.snaps[upto];
  const snapC = runC.snaps[runC.snaps.length - 1];
  const lossC = runC.losses[runC.losses.length - 1];
  const changeStep = useCallback((v) => setStep(Math.round(v / 10) * 10), []);

  const rows = (r, s, u) => [
    ["fit loss at step " + u * 10, fmt(r.losses[u], 4)],
    ["different hidden units", distinct3(s) + " of " + H3, distinct3(s) < H3 ? THEME.warn : undefined],
  ];
  const controls = (
    <>
      <Slider id="l3-c" label="starting value of every weight" value={c} min={-1} max={1} stepSize={0.01} onChange={setC} />
      <Toggle label="add random noise to the start" value={noise} onChange={setNoise} />
      {noise && <LogSlider id="l3-sd" label="noise std" value={sd} lo={-6} hi={0.8} onChange={setSd} />}
      <Slider id="l3-step" label="training step" value={step} min={0} max={STEPS3} stepSize={10} onChange={changeStep} display={String(step)} />
    </>
  );

  let question, stage, side;
  if (screen === 0) {
    question = (
      <>
        The ETA curve has two jumps: at the edge of the centre and at the ring road. A network with three hidden tanh units should fit it easily. Every weight starts at the same value. Train it (scrub the step slider) and watch each unit's part of the answer. Why can't three units fit two jumps?
      </>
    );
    stage = (
      <>
        <FitPanel snap={snap} showUnits />
        <WeightTrace snaps={run.snaps} upto={upto} />
      </>
    );
    side = (
      <>
        <Panel>
          <Slider id="l3-c0" label="starting value of every weight" value={c} min={-1} max={1} stepSize={0.01} onChange={setC} />
          <Slider id="l3-step0" label="training step" value={step} min={0} max={STEPS3} stepSize={10} onChange={changeStep} display={String(step)} />
          <Readout rows={rows(run, snap, upto)} />
        </Panel>
        <Note>The three dashed curves are each unit's contribution. Count how many you can actually see.</Note>
      </>
    );
  } else if (screen === 1) {
    question = (
      <>
        Add a little random noise to the starting weights. How small can the noise be and still let the units go separate ways within 1000 steps? Scrub through training: when do the three weight lines split, and what does the fit do at that moment?
      </>
    );
    stage = (
      <>
        <FitPanel snap={snap} showUnits />
        <WeightTrace snaps={run.snaps} upto={upto} />
      </>
    );
    side = (
      <>
        <Panel>
          {controls}
          <Readout rows={rows(run, snap, upto)} />
        </Panel>
        <Note>The noise is a fixed random direction scaled by the std, so the same std always gives the same run.</Note>
      </>
    );
  } else if (screen === 2) {
    question = (
      <>
        Name it: <i>symmetry</i>. Units with identical incoming and outgoing weights compute the same thing, receive the same gradient and take the same step, so they are still identical after any number of steps. All units of the layer do the same thing. Random initialization breaks the symmetry.
      </>
    );
    stage = (
      <>
        <FitPanel snap={snap} showUnits />
        <WeightTrace snaps={run.snaps} upto={upto} />
      </>
    );
    side = (
      <>
        <Panel>
          <Eq>
            if w<sub>1</sub> = w<sub>2</sub>, b<sub>1</sub> = b<sub>2</sub>, v<sub>1</sub> = v<sub>2</sub>
            <br />
            then h<sub>1</sub> = h<sub>2</sub> and ∂L/∂w<sub>1</sub> = v<sub>1</sub>·f′·x = ∂L/∂w<sub>2</sub>
          </Eq>
          {controls}
          <Readout rows={rows(run, snap, upto)} />
        </Panel>
        <Note>The slides' first fix is small random numbers, a Gaussian with mean 0 and std 0.01. That works for small networks; lesson 4 shows what it does to a deep one. Tiny noise breaks the tie in principle, but the split grows from that tiny difference, so it can take many steps to show.</Note>
      </>
    );
  } else {
    question = (
      <>
        A colleague played it safe and set every weight to 0. Now nothing moves at all: tanh(0) = 0, so every hidden gradient is 0. Keep the mean at 0 and choose the noise std so the network fits the curve with loss below {GOAL3} after {STEPS3} steps.
      </>
    );
    stage = (
      <>
        <FitPanel snap={snapC} showUnits />
        <WeightTrace snaps={runC.snaps} upto={runC.snaps.length - 1} />
      </>
    );
    side = (
      <>
        <Panel edge={lossC < GOAL3 ? THEME.ok : THEME.grid}>
          <LogSlider id="l3-sdc" label="noise std (mean 0)" value={sdC} lo={-6} hi={1.3} onChange={setSdC} />
          <Readout
            rows={[
              ["fit loss after " + STEPS3 + " steps", fmt(lossC, 4)],
              ["different hidden units", distinct3(snapC) + " of " + H3],
            ]}
          />
          <Goal met={distinct3(snapC) === H3}>three different units</Goal>
          <Goal met={lossC < GOAL3}>loss below {GOAL3}</Goal>
          <Hint
            hints={[
              "Sweep the std from very small to very large and watch the loss: it fails on both sides, for different reasons.",
              "With a tiny std the units are near-copies for most of training. With a huge std, look at the input weights: tanh(w·x + b) with large w is a hard step, flat almost everywhere. The working range sits around 1.",
            ]}
          />
        </Panel>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="Identical twins never part" question={question} stage={stage} side={side} />;
}
