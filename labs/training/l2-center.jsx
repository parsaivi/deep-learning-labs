import { useState, useMemo, useCallback } from "react";
import { THEME, MONO } from "./common";

// Lesson 2 - Zero-centred inputs and input normalization (Practical Issues, slides 11-13, 24-31).
// Aha: the gradient on a weight is (error) x (its input). If every input is positive, one order's
// gradient has the same sign on every weight, so a single step can only move all weights up or all
// down. Reaching a target that needs one up and one down takes a zig-zag. Subtracting the mean
// removes the restriction; dividing by the std stops one feature's units from dominating.

// @sim-start
const N2 = 120;
const W2 = [3, -2]; // minutes per km, minutes per kg (heavier parcels go by car: faster)
const DATA2 = (function () {
  const r = rng(777);
  const X = [];
  const Y = [];
  for (let i = 0; i < N2; i++) {
    const km = 0.5 + 7.5 * r();
    const kg = 0.5 + 5.5 * r();
    X.push([km, kg]);
    Y.push(W2[0] * km + W2[1] * kg + 1.0 * gauss(r));
  }
  return { X, Y };
})();
// mode: raw | center | std ; unit: 1 (kg) or 1000 (grams)
function prep2(mode, unit) {
  const x = DATA2.X.map((v) => [v[0], v[1] * unit]);
  const mu = [0, 0];
  const sd = [1, 1];
  let ym = 0;
  if (mode !== "raw") {
    for (const v of x) {
      mu[0] += v[0] / N2;
      mu[1] += v[1] / N2;
    }
    for (const y of DATA2.Y) ym += y / N2;
  }
  if (mode === "std") {
    sd[0] = 0;
    sd[1] = 0;
    for (const v of x) {
      sd[0] += (v[0] - mu[0]) * (v[0] - mu[0]) / N2;
      sd[1] += (v[1] - mu[1]) * (v[1] - mu[1]) / N2;
    }
    sd[0] = Math.sqrt(sd[0]);
    sd[1] = Math.sqrt(sd[1]);
  }
  const xs = x.map((v) => [(v[0] - mu[0]) / sd[0], (v[1] - mu[1]) / sd[1]]);
  const ys = DATA2.Y.map((y) => y - ym);
  // weights in the processed space -> minutes per km and per kg
  const toOrig = (w) => [w[0] / sd[0], (w[1] / sd[1]) * unit];
  const fromOrig = (w) => [w[0] * sd[0], (w[1] * sd[1]) / unit];
  let a = 0, bb = 0, c = 0, p = 0, q = 0, yy = 0;
  for (let i = 0; i < N2; i++) {
    a += (xs[i][0] * xs[i][0]) / N2;
    bb += (xs[i][0] * xs[i][1]) / N2;
    c += (xs[i][1] * xs[i][1]) / N2;
    p += (xs[i][0] * ys[i]) / N2;
    q += (xs[i][1] * ys[i]) / N2;
    yy += (ys[i] * ys[i]) / N2;
  }
  const det = a * c - bb * bb;
  const bestP = [(c * p - bb * q) / det, (a * q - bb * p) / det];
  const lossP = (w) => 0.5 * (yy + a * w[0] * w[0] + 2 * bb * w[0] * w[1] + c * w[1] * w[1] - 2 * p * w[0] - 2 * q * w[1]);
  return { xs, ys, toOrig, fromOrig, best: toOrig(bestP), loss: (wo) => lossP(fromOrig(wo)) };
}
// SGD from start (in min/km, min/kg). Returns path in original units and the sign pattern of each step.
function sgd2(P, start, eta, steps, b, seed) {
  const r = rng(seed || 3);
  let w = P.fromOrig(start);
  const pts = [P.toOrig(w)];
  const signs = { same: 0, mixed: 0 };
  let diverged = false;
  for (let k = 0; k < steps; k++) {
    let g0 = 0, g1 = 0;
    for (let j = 0; j < b; j++) {
      const i = Math.floor(r() * N2);
      const res = w[0] * P.xs[i][0] + w[1] * P.xs[i][1] - P.ys[i];
      g0 += (res * P.xs[i][0]) / b;
      g1 += (res * P.xs[i][1]) / b;
    }
    const d = [-eta * g0, -eta * g1];
    if (Math.abs(d[0]) > 1e-12 && Math.abs(d[1]) > 1e-12) {
      if (d[0] > 0 === d[1] > 0) signs.same++;
      else signs.mixed++;
    }
    w = [w[0] + d[0], w[1] + d[1]];
    const wo = P.toOrig(w);
    if (!isFinite(wo[0]) || !isFinite(wo[1]) || Math.abs(wo[0]) + Math.abs(wo[1]) > 1e4) {
      diverged = true;
      pts.push([clamp(wo[0] || 0, -50, 50), clamp(wo[1] || 0, -50, 50)]);
      break;
    }
    pts.push(wo);
  }
  const end = pts[pts.length - 1];
  return { pts, signs, diverged, err: diverged ? Infinity : Math.max(Math.abs(end[0] - P.best[0]), Math.abs(end[1] - P.best[1])) };
}
const STEPS2C = 60;
const TOL2 = 0.25;
// @sim-end

const SURF_BOX = { xr: [-1, 5], yr: [-4.5, 1.5], xl: "w_km (min per km)", yl: "w_kg (min per kg)" };
const MODE_OPTS = [
  ["raw", "raw"],
  ["center", "subtract mean"],
  ["std", "subtract mean, ÷ std"],
];
const B_OPTS = [
  [1, "1 order"],
  [8, "8 orders"],
  [32, "32 orders"],
];

// the wedge of directions one step can take: the cone spanned by ±(km, kg) of the orders
function coneOverlay(P0, centered, xsRaw) {
  return function (P) {
    const [cx, cy] = P(P0[0], P0[1]);
    if (centered) return null;
    let lo = Infinity, hi = -Infinity;
    for (const v of xsRaw) {
      const a = Math.atan2(v[1], v[0]);
      if (a < lo) lo = a;
      if (a > hi) hi = a;
    }
    const R = 120;
    const wedge = (a0, a1) => {
      let d = "M" + cx + "," + cy;
      for (let k = 0; k <= 12; k++) {
        const a = a0 + ((a1 - a0) * k) / 12;
        d += " L" + (cx + R * Math.cos(a)).toFixed(1) + "," + (cy - R * Math.sin(a)).toFixed(1);
      }
      return d + " Z";
    };
    return (
      <g pointerEvents="none">
        <path d={wedge(lo, hi)} fill={rgba(THEME.highlight, 0.14)} stroke={THEME.highlight} strokeDasharray="3 3" />
        <path d={wedge(lo + Math.PI, hi + Math.PI)} fill={rgba(THEME.highlight, 0.14)} stroke={THEME.highlight} strokeDasharray="3 3" />
        <text x={cx + 0.75 * R * Math.cos((lo + hi) / 2)} y={cy - 0.75 * R * Math.sin((lo + hi) / 2)} fontFamily={MONO} fontSize={10.5} fill={THEME.highlight} stroke={THEME.bg} strokeWidth={3} paintOrder="stroke" textAnchor="middle">
          one-order steps
        </text>
      </g>
    );
  };
}

export default function CenterLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [mode, setMode] = useState("raw");
  const [b, setB] = useState(1);
  const [eta, setEta] = useState(0.02);
  const [start, setStart] = useState([0, 0]);
  const [modeC, setModeC] = useState("raw");
  const [etaC, setEtaC] = useState(1e-7);
  const modeNow = screen === 0 ? "raw" : mode === "std" ? "center" : mode;
  const bNow = screen === 0 ? 1 : b;
  const P = useMemo(() => prep2(modeNow, 1), [modeNow]);
  const run = useMemo(() => sgd2(P, start, eta, 200, bNow, 3), [P, start, eta, bNow]);
  const PC = useMemo(() => prep2(modeC, 1000), [modeC]);
  const runC = useMemo(() => sgd2(PC, [0, 0], etaC, STEPS2C, 1, 5), [PC, etaC]);
  const surf = useMemo(() => Object.assign({ f: P.loss, floor: 0.05 }, SURF_BOX), [P]);
  const surfC = useMemo(() => Object.assign({ f: PC.loss, floor: 0.05 }, SURF_BOX), [PC]);
  const onHandle = useCallback((x, y) => setStart([x, y]), []);
  const overlay = useMemo(() => (bNow === 1 ? coneOverlay(start, modeNow !== "raw", DATA2.X) : null), [start, modeNow, bNow]);
  const end = run.pts[run.pts.length - 1];
  const path = { pts: run.pts, color: THEME.highlight, shape: "circle", label: "SGD, 200 steps" };
  const pathC = { pts: runC.pts, color: THEME.highlight, shape: "circle", label: runC.diverged ? "diverged" : STEPS2C + " steps" };
  const okC = !runC.diverged && runC.err <= TOL2;

  const rows = [
    ["steps moving both weights the same way", String(run.signs.same)],
    ["steps moving them opposite ways", String(run.signs.mixed), run.signs.mixed === 0 ? THEME.warn : undefined],
    ["end point (w_km, w_kg)", fmt(end[0], 2) + ", " + fmt(end[1], 2)],
  ];

  let question, stage, side;
  if (screen === 0) {
    question = (
      <>
        A linear ETA model: minutes = w_km · km + w_kg · kg. The best fit (+) needs w_km to go up and w_kg to go down. SGD takes one order per step, starting from the yellow handle. Look at the direction of every step. Does any single step ever move toward the cross directly?
      </>
    );
    stage = <LossMap surf={surf} paths={[path]} target={P.best} handle={start} onHandle={onHandle} overlay={overlay} ariaLabel="Loss over the two weights with the SGD path" />;
    side = (
      <>
        <Panel>
          <LogSlider id="l2-eta0" label="learning rate η" value={eta} lo={-3} hi={-1.4} onChange={setEta} />
          <Readout rows={rows} />
        </Panel>
        <Note>The shaded wedges show every direction one order's gradient step can point from the start. Drag the start to another place: do the wedges change?</Note>
      </>
    );
  } else if (screen === 1) {
    question = (
      <>
        km and kg are always positive. Subtract each feature's mean (and the ETA's mean) so the inputs have both signs. What happens to the wedges and the path? Then go back to raw inputs and use 8 or 32 orders per step: why does a mini-batch help even without centering?
      </>
    );
    stage = <LossMap surf={surf} paths={[path]} target={P.best} handle={start} onHandle={onHandle} overlay={overlay} ariaLabel="Loss over the two weights with the SGD path" />;
    side = (
      <>
        <Panel>
          <Label>inputs</Label>
          <Pills options={MODE_OPTS.slice(0, 2)} value={modeNow} onPick={setMode} />
          <Label>orders per step</Label>
          <Pills options={B_OPTS} value={b} onPick={setB} />
          <LogSlider id="l2-eta1" label="learning rate η" value={eta} lo={-3} hi={-1.4} onChange={setEta} />
          <Readout rows={rows} />
        </Panel>
        <Note>With centred inputs the map itself also changes shape. Compare how round the valley is in the two modes.</Note>
      </>
    );
  } else if (screen === 2) {
    question = (
      <>
        Name it. For one order, ∂L/∂w<sub>i</sub> = δ · x<sub>i</sub>, where δ is the error. If every x<sub>i</sub> &gt; 0, all components share the sign of δ: each step goes all-up or all-down, so the path zig-zags. That is why you want zero-mean data, and why sigmoid, whose outputs feed the next layer and are always positive, is worse than tanh.
      </>
    );
    stage = <LossMap surf={surf} paths={[path]} target={P.best} handle={start} onHandle={onHandle} overlay={overlay} ariaLabel="Loss over the two weights with the SGD path" />;
    side = (
      <>
        <Panel>
          <Eq>
            μ<sub>i</sub> = (1/N) Σ<sub>n</sub> x<sub>i</sub><sup>(n)</sup> &nbsp; σ<sub>i</sub>² = (1/N) Σ<sub>n</sub> (x<sub>i</sub><sup>(n)</sup> − μ<sub>i</sub>)²
            <br />x<sub>i</sub> ← (x<sub>i</sub> − μ<sub>i</sub>) / σ<sub>i</sub>
          </Eq>
          <Pills options={MODE_OPTS.slice(0, 2)} value={modeNow} onPick={setMode} />
          <Readout rows={rows} />
        </Panel>
        <Note>
          Compute μ and σ on the training set only, and use the same numbers on test orders. Dividing by σ fixes features measured in very different units: the valley gets round and one η suits every weight (Optimization Lab, lesson 2). For images, the usual step is only to subtract the mean image or the per-channel mean (and sometimes divide by the per-channel std). PCA and whitening are rare.
        </Note>
      </>
    );
  } else {
    question = (
      <>
        A partner courier sends parcel weight in <i>grams</i> (500 to 6000), not kg. Same model, same orders, {STEPS2C} SGD steps of one order each from (0, 0). Choose how to preprocess the two inputs and the learning rate so the model ends within {TOL2} min of the best fit on both weights. The map is drawn per kg so you can compare.
      </>
    );
    stage = <LossMap surf={surfC} paths={[pathC]} target={PC.best} ariaLabel="Loss map for the grams data with the SGD path" />;
    side = (
      <>
        <Panel edge={okC ? THEME.ok : THEME.grid}>
          <Label>preprocessing of km and grams</Label>
          <Pills options={MODE_OPTS} value={modeC} onPick={setModeC} />
          <LogSlider id="l2-etac" label="learning rate η" value={etaC} lo={-9} hi={0} onChange={setEtaC} />
          <Readout
            rows={[
              ["end (w_km, w_kg)", runC.diverged ? "diverged" : fmt(runC.pts[runC.pts.length - 1][0], 2) + ", " + fmt(runC.pts[runC.pts.length - 1][1], 2), runC.diverged ? THEME.warn : undefined],
              ["best fit", fmt(PC.best[0], 2) + ", " + fmt(PC.best[1], 2)],
              ["largest miss", runC.diverged ? "∞" : fmt(runC.err, 2) + " min"],
            ]}
          />
          <Goal met={!runC.diverged}>no divergence</Goal>
          <Goal met={okC}>both weights within {TOL2} of the best fit</Goal>
          <Hint
            hints={[
              "With grams, the gradient on the weight feature is about 1000 times larger than on km. What does that force η to be, and what does that η do to w_km?",
              "Centering fixes the sign problem but not the size problem: grams still vary by thousands. Rescale both inputs to the same spread, then find η again.",
            ]}
          />
        </Panel>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="Center the inputs" question={question} stage={stage} side={side} />;
}
