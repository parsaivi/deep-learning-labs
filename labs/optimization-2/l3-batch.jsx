import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import { THEME, MONO } from "./common";

// Lesson 3 - Mini-batches (Optimization II, slides 21-48 and the linear-scaling rule on 65).
// Aha: averaging b samples cuts the gradient's noise variance by b, which buys a b-times bigger step, so per
// sample read a batch is neither better nor worse, until eta hits the curvature limit. What a batch buys is
// hardware: a GPU handles b samples in the same time as one, up to its width.
// The race uses the exact expected loss of SGD on this quadratic: per eigen-direction,
// E[x^2] <- (1 - eta*lam)^2 E[x^2] + eta^2 lam s2 (N - b) / ((N - 1) b).

// @sim-start
const M3 = makeOrders(41, 2000, 0.9, 1.8, 0.6);
const S2 = 2 * M3.lossBest; // variance of one sample's residual at the best fit
const EIG3 = eig2([
  [M3.st.Sxx, M3.st.Sx],
  [M3.st.Sx, 1],
]);
const LAMS = [EIG3.l1, EIG3.l2];
const START3 = [2.2, 1.8];
const EPS3 = 2e-4;
const OVER = 0.5; // ms per update, whatever b is
function proj(start) {
  const d = [start[0] - M3.best[0], start[1] - M3.best[1]];
  return [Math.pow(d[0] * EIG3.v1[0] + d[1] * EIG3.v1[1], 2), Math.pow(d[0] * EIG3.v2[0] + d[1] * EIG3.v2[1], 2)];
}
const Q0 = proj(START3);
function model(b, eta) {
  const noise = (S2 * (M3.n - b)) / ((M3.n - 1) * b);
  const f = LAMS.map((l) => (1 - eta * l) * (1 - eta * l));
  const qInf = LAMS.map((l, i) => (f[i] < 1 ? (eta * eta * l * noise) / (1 - f[i]) : Infinity));
  const ex = (k) => 0.5 * LAMS.reduce((s, l, i) => s + l * (Math.pow(f[i], k) * (Q0[i] - qInf[i]) + qInf[i]), 0);
  const floor = 0.5 * LAMS.reduce((s, l, i) => s + l * qInf[i], 0);
  return { ex, floor, stable: f[0] < 1 && f[1] < 1 };
}
function stepsTo(b, eta, eps) {
  const m = model(b, eta);
  if (!m.stable || m.floor >= eps) return Infinity;
  if (m.ex(0) < eps) return 0;
  let lo = 0;
  let hi = 1;
  while (m.ex(hi) >= eps) {
    hi *= 2;
    if (hi > 1e8) return Infinity;
  }
  while (hi - lo > 1) {
    const mid = Math.floor((lo + hi) / 2);
    if (m.ex(mid) < eps) hi = mid;
    else lo = mid;
  }
  return hi;
}
function msPer(b, W) {
  return OVER + Math.ceil(b / W);
}
function bestEta(b, W) {
  let best = { t: Infinity, eta: 0.01, k: Infinity };
  for (let le = -4.5; le <= -0.25; le += 0.01) {
    const eta = Math.pow(10, le);
    const k = stepsTo(b, eta, EPS3);
    const t = k * msPer(b, W);
    if (t < best.t) best = { t, eta, k };
  }
  return best;
}
// @sim-end

const GPU = { W: 64, cap: 1024 };
const GPU_C = { W: 256, cap: 512 };
const BUDGET_C = 150;
const SURF3 = { f: M3.loss, xr: [-0.6, 2.6], yr: [1, 7.6], xl: "slope w₁", yl: "w₂ = prediction at x = 4" };
const NFAN = 16;
function bOf(e) {
  return Math.min(M3.n, Math.max(1, Math.round(Math.pow(2, e))));
}
function sampleGrad(w, b, r) {
  // mean gradient over b samples drawn without replacement
  const n = M3.n;
  const pick = new Set();
  let g0 = 0;
  let g1 = 0;
  if (b >= n) {
    for (let i = 0; i < n; i++) pick.add(i);
  } else {
    while (pick.size < b) pick.add(Math.floor(r() * n));
  }
  pick.forEach((i) => {
    const res = w[1] + w[0] * M3.x[i] - M3.y[i];
    g0 += res * M3.x[i];
    g1 += res;
  });
  return [g0 / b, g1 / b];
}
function curve(b, eta, W) {
  const m = model(b, eta);
  const tb = msPer(b, W);
  const ts = [];
  const rd = [];
  const ys = [];
  for (let e = 0; e <= 5.2; e += 0.02) {
    const k = Math.round(Math.pow(10, e));
    ts.push(k * tb);
    rd.push(k * b);
    ys.push(m.stable ? m.ex(k) : Infinity);
  }
  return { ts, rd, ys, m };
}
const REF1 = bestEta(1, GPU.W);
const REFN = bestEta(M3.n, GPU.W);
const BS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2000];
const SWEEP = BS.map((b) => Object.assign({ b }, bestEta(b, GPU.W)));

function fanOverlay(start, gTrue, draws) {
  return function (P) {
    const s = P(start[0], start[1]);
    const unit = 70 / Math.hypot(gTrue[0], gTrue[1]);
    const tip = (g) => {
      // the map's y axis is w2, so scale both components to pixels the same way P does
      const e = P(start[0] - g[0] * 0.01, start[1] - g[1] * 0.01);
      const dx = e[0] - s[0];
      const dy = e[1] - s[1];
      const L = Math.hypot(dx, dy) || 1;
      const len = Math.min(190, Math.hypot(g[0], g[1]) * unit);
      return [s[0] + (dx / L) * len, s[1] + (dy / L) * len];
    };
    return (
      <g>
        {draws.map((g, i) => {
          const t = tip(g);
          return <Arrow key={i} x1={s[0]} y1={s[1]} x2={t[0]} y2={t[1]} color={THEME.accent} width={1.2} />;
        })}
        {(() => {
          const t = tip(gTrue);
          return (
            <g>
              <Arrow x1={s[0]} y1={s[1]} x2={t[0]} y2={t[1]} color={THEME.highlight} width={3} dash="6 3" />
              <text x={t[0] + 6} y={t[1] - 6} fontFamily={MONO} fontSize={11} fill={THEME.highlight} stroke={THEME.bg} strokeWidth={3} paintOrder="stroke">
                all 2000 samples
              </text>
            </g>
          );
        })()}
      </g>
    );
  };
}

function Race({ b, eta, W, cap, budget }) {
  const fits = b <= cap;
  const c = useMemo(() => curve(b, eta, W), [b, eta, W]);
  const r1 = useMemo(() => curve(1, REF1.eta, W), [W]);
  const rN = useMemo(() => curve(M3.n, REFN.eta, W), [W]);
  const series = (key) => [
    { xs: r1[key], ys: r1.ys, color: THEME.muted, dash: "2 3", width: 1.4 },
    { xs: rN[key], ys: rN.ys, color: THEME.text, dash: "7 4", width: 1.4 },
  ].concat(fits ? [{ xs: c[key], ys: c.ys, color: THEME.highlight, width: 2.4 }] : []);
  const legend = [
    { color: THEME.highlight, width: 2.4, text: fits ? "yours: b = " + b : "yours: does not fit" },
    { color: THEME.muted, dash: "2 3", text: "b = 1, best η" },
    { color: THEME.text, dash: "7 4", text: "full batch, best η" },
  ];
  const vl = budget ? [{ x: budget, color: THEME.ok, label: "budget " + budget + " ms" }] : [];
  return (
    <>
      <Plot series={series("ts")} xr={[0, 5]} yr={[-4.5, 1]} logX logY xl="GPU time (ms) →" label="expected extra loss vs time" legend={legend} height={180} hlines={[{ y: EPS3, color: THEME.ok, label: "target" }]} vlines={vl} />
      <Plot series={series("rd")} xr={[0, 7]} yr={[-4.5, 1]} logX logY xl="samples read →" label="expected extra loss vs samples read" legend={legend} height={150} hlines={[{ y: EPS3, color: THEME.ok, label: "target" }]} />
    </>
  );
}

function raceStats(b, eta, W, cap) {
  if (b > cap) return { fits: false };
  const k = stepsTo(b, eta, EPS3);
  const m = model(b, eta);
  return { fits: true, k, t: k * msPer(b, W), reads: k * b, floor: m.stable ? m.floor : Infinity, stable: m.stable };
}
function StatRows({ s, b, W }) {
  if (!s.fits) return <Readout rows={[["batch", "does not fit in GPU memory", THEME.warn]]} />;
  return (
    <Readout
      rows={[
        ["time per update", fmt(msPer(b, W), 1) + " ms"],
        ["noise floor of the loss", !s.stable ? "diverges" : s.floor < 1e-6 ? "≈ 0" : s.floor.toExponential(1), s.floor >= EPS3 ? THEME.warn : undefined],
        ["updates to reach target", isFinite(s.k) ? s.k : "never"],
        ["samples read to target", isFinite(s.k) ? s.reads.toLocaleString() : "—"],
        ["time to target", isFinite(s.t) ? fmt(s.t, 0) + " ms" : "—"],
      ]}
    />
  );
}

export default function BatchLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [eB, setEB] = useState(0);
  const [start, setStart] = useState(START3);
  const [seed, setSeed] = useState(1);
  const [eB2, setEB2] = useState(4);
  const [eta, setEta] = useState(0.005);
  const [eBC, setEBC] = useState(6);
  const [etaC, setEtaC] = useState(0.02);
  const onHandle = useCallback((x, y) => setStart([x, y]), []);
  const resample = useCallback(() => setSeed((s) => s + 1), []);

  const b = bOf(eB);
  const fan = useMemo(() => {
    const r = rng(seed * 7919 + b);
    const gT = sampleGrad(start, M3.n, r);
    const draws = [];
    for (let i = 0; i < NFAN; i++) draws.push(sampleGrad(start, b, r));
    let s = 0;
    for (let i = 0; i < 200; i++) {
      const g = sampleGrad(start, b, r);
      s += (g[0] - gT[0]) * (g[0] - gT[0]) + (g[1] - gT[1]) * (g[1] - gT[1]);
    }
    return { gT, draws, spread: Math.sqrt(s / 200) / Math.hypot(gT[0], gT[1]) };
  }, [start, b, seed]);

  const b2 = bOf(eB2);
  const st2 = raceStats(b2, eta, GPU.W, GPU.cap);
  const bC = bOf(eBC);
  const stC = raceStats(bC, etaC, GPU_C.W, GPU_C.cap);
  const okC = stC.fits && stC.t <= BUDGET_C;

  let question, stage, side;
  if (screen === 0) {
    question = (
      <>
        <b>Where we are:</b> SGD uses one sample per update and batch uses all N. The deck's middle ground is the mini-batch: average b samples per update (slides 21–48). Two questions: how much does b reduce the noise, and what is a good b?
        <br />
        <b>The puzzle:</b> a dataset of 2000 samples. The thick dashed arrow is −∇E computed from all 2000. Each thin arrow uses only b samples picked at random. How large must b be to make the thin arrows <i>half</i> as spread out as with b = 1?
      </>
    );
    stage = <LossMap surf={SURF3} target={M3.best} handle={start} onHandle={onHandle} overlay={fanOverlay(start, fan.gT, fan.draws)} ariaLabel="Loss map with the full-data gradient and 16 mini-batch gradients" />;
    side = (
      <>
        <Panel>
          <Slider id="l3-b" label="batch size b" value={eB} min={0} max={Math.log2(M3.n)} stepSize={0.01} onChange={setEB} display={b} />
          <Readout
            rows={[
              ["spread of a batch gradient (rms, ÷ true size)", fmt(fan.spread, 3)],
              ["spread × √b", fmt(fan.spread * Math.sqrt(b), 2)],
            ]}
          />
          <Btn onClick={resample}>draw new batches</Btn>
        </Panel>
        <Note>Find the b where the spread is half its b = 1 value, then where it is a quarter. Every batch gradient is still right on average: the thin arrows scatter around the thick one, not off to one side.</Note>
      </>
    );
  } else if (screen === 1) {
    question = (
      <>
        Now race to a target loss (just above the best fit's) on a GPU that processes 64 samples in parallel: an update costs 0.5 ms plus 1 ms per group of 64. Pick b, then find the η that gets there fastest. Double b and find the best η again. What happens to the best η, the number of samples read and the time?
      </>
    );
    stage = <Race b={b2} eta={eta} W={GPU.W} cap={GPU.cap} />;
    side = (
      <>
        <Panel>
          <Slider id="l3-b2" label="batch size b" value={eB2} min={0} max={Math.log2(M3.n)} stepSize={0.01} onChange={setEB2} display={b2} />
          <LogSlider id="l3-eta" label="learning rate η" value={eta} lo={-4.5} hi={-0.3} onChange={setEta} digits={4} />
          <StatRows s={st2} b={b2} W={GPU.W} />
        </Panel>
        <Note>The grey dotted curve is b = 1 and the white dashed one is the full batch, each at its own best η. A step that is too large for the batch's noise raises the floor and the curve never reaches the target.</Note>
      </>
    );
  } else if (screen === 2) {
    question = (
      <>
        Name it. The batch loss is an unbiased estimate of the true loss, with variance var(loss)/b. A b-times smaller variance lets η grow about b times (the <b>linear-scaling rule</b>), so the number of samples read stays flat: per sample, the convergence rate is O(1/√(bk) + 1/k). The gain is in time: up to the GPU width, a bigger batch costs nothing extra.
      </>
    );
    const xs = SWEEP.map((s) => s.b);
    stage = (
      <>
        <Plot
          series={[
            { xs, ys: SWEEP.map((s) => s.t), color: THEME.highlight, label: "time to target (ms)", width: 2.2 },
            { xs, ys: SWEEP.map((s) => s.k * s.b), color: THEME.accent, dash: "6 3", label: "samples read", ldy: 14 },
          ]}
          xr={[0, 3.4]}
          yr={[1.5, 5]}
          logX
          logY
          xl="batch size b →"
          label="best case for each b (its best η), GPU width 64"
          height={200}
          vlines={[{ x: 64, color: THEME.text, label: "GPU width" }]}
          marks={SWEEP.map((s) => ({ x: s.b, y: s.t, shape: "circle", color: THEME.highlight }))}
        />
        <Plot series={[{ xs, ys: SWEEP.map((s) => s.eta), color: THEME.text, dash: "4 3", label: "best η" }]} xr={[0, 3.4]} yr={[-4, 0]} logX logY xl="batch size b →" label="best η grows with b, then hits the curvature limit" height={140} marks={SWEEP.map((s) => ({ x: s.b, y: s.eta, shape: "diamond", color: THEME.text }))} />
      </>
    );
    side = (
      <>
        <Panel>
          <Eq size={12.5}>
            BatchErr = (1/b) Σᵢ loss(f(xᵢ; W), yᵢ)
            <br />
            E[BatchErr] = E[loss] &nbsp;&nbsp;var(BatchErr) = var(loss)/b
            <br />
            batch × N &nbsp;⇒&nbsp; η × N (rule of thumb)
          </Eq>
          <Readout rows={SWEEP.filter((s) => [1, 16, 64, 256, 1024, 2000].indexOf(s.b) >= 0).map((s) => ["b = " + s.b + ": best η " + fmt(s.eta, 4), fmt(s.t, 0) + " ms"])} />
        </Panel>
        <Note>Three regimes: below the GPU width time falls about as 1/b; between the width and the curvature limit it barely changes; past the limit η cannot grow, so the extra samples are wasted. Typical choices are 64 to 1024, as large as fits in GPU memory. A small dataset (n &lt; 2000) can simply use full batch (slides 47–48). Next: how to tell from the loss curve whether η itself is right.</Note>
      </>
    );
  } else {
    question = (
      <>
        A new GPU: 256 samples in parallel, but memory for at most {GPU_C.cap} per batch. Reach the target in {BUDGET_C} ms or less.
      </>
    );
    stage = <Race b={bC} eta={etaC} W={GPU_C.W} cap={GPU_C.cap} budget={BUDGET_C} />;
    side = (
      <>
        <Panel edge={okC ? THEME.ok : THEME.grid}>
          <Slider id="l3-bc" label="batch size b" value={eBC} min={0} max={Math.log2(M3.n)} stepSize={0.01} onChange={setEBC} display={bC} />
          <LogSlider id="l3-etac" label="learning rate η" value={etaC} lo={-4.5} hi={-0.3} onChange={setEtaC} digits={4} />
          <StatRows s={stC} b={bC} W={GPU_C.W} />
          <Goal met={okC}>target reached within {BUDGET_C} ms</Goal>
          <Hint
            hints={[
              "On this GPU, a batch of 256 costs the same per update as a batch of 1. Use that.",
              "You found the best η for b = 64 in Explore. When you make the batch N times larger, make η about N times larger too, then fine-tune.",
            ]}
          />
        </Panel>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="How many samples per update?" question={question} stage={stage} side={side} />;
}
