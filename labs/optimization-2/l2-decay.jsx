import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import { THEME, MONO } from "./common";

// Lesson 2 - Why SGD with a fixed step never settles, and how fast eta must shrink (Optimization II, slides 17-20).
// Aha: even at the best fit, every single order still disagrees with it, so each update still kicks the
// weights. The kicks scale with eta, so the walker buzzes in a cloud forever. Shrinking eta shrinks the
// cloud, but if it shrinks too fast the walker runs out of total travel before it arrives.

// @sim-start
const O2 = makeOrders(11, 40, 0.8, 2.0, 0.5);
const C2 = makeOrders(31, 60, 1.1, 1.5, 0.6);
const START2 = [2.2, 1.8];
const START2C = [-0.6, 1.5];
const K2 = 1600;
const K2C = 2000;
const TAIL = 400;
const TOL2 = 0.025;
const TOL2C = 0.03;
const TAU = 50;
function decay(eta0, p) {
  return (k) => eta0 * Math.pow(1 + k / TAU, -p);
}
function settleRun(o, start, sched, steps) {
  const r = runSGD(o, start, { b: 1, sched, steps, order: "shuffle", seed: 3 });
  const d = r.pts.map((q) => dist(q, o.best));
  let tailMax = 0;
  for (let i = Math.max(0, d.length - TAIL); i < d.length; i++) tailMax = Math.max(tailMax, d[i]);
  return { r, d, tailMax: r.diverged ? Infinity : tailMax };
}
// @sim-end

function zoomSurf(o) {
  const z = 0.22;
  return { f: o.loss, xr: [o.best[0] - z, o.best[0] + z], yr: [o.best[1] - z, o.best[1] + z], xl: "rate (zoom)", yl: "fee (zoom)", floor: 0.0005 };
}
const SURF2 = { f: O2.loss, xr: [-0.6, 2.6], yr: [1, 7.6], xl: "rate €/km", yl: "fee of a 4 km order €" };
const ZOOM2 = zoomSurf(O2);
const SURF2C = { f: C2.loss, xr: [-1, 2.6], yr: [0.8, 8.2], xl: "rate €/km", yl: "fee of a 4 km order €" };
const ZOOM2C = zoomSurf(C2);

function tolRing(o, tol, ok) {
  return function (P) {
    const c = P(o.best[0], o.best[1]);
    const e = P(o.best[0] + tol, o.best[1]);
    return <circle cx={c[0]} cy={c[1]} r={Math.abs(e[0] - c[0])} fill="none" stroke={ok ? THEME.ok : THEME.highlight} strokeWidth={2} strokeDasharray="5 3" />;
  };
}
function sumInf(eta0, p, pow) {
  // sum over k >= 0 of (eta0 (1 + k/TAU)^-p)^pow, or Infinity when it diverges
  const q = p * pow;
  if (q <= 1) return Infinity;
  return Math.pow(eta0, pow) * (1 + TAU / (q - 1));
}

function Views({ o, surf, zoom, res, tol, ok, start }) {
  const tail = res.r.pts.slice(-TAIL);
  return (
    <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
      <div style={{ flex: "1 1 250px", minWidth: 0 }}>
        <LossMap surf={surf} target={o.best} paths={[{ key: "p", pts: res.r.pts, color: THEME.accent, shape: "circle", width: 1.2 }, { key: "s", pts: [start], color: THEME.highlight, shape: "square", label: "start" }]} ariaLabel="Whole loss map with the SGD path" />
      </div>
      <div style={{ flex: "1 1 250px", minWidth: 0 }}>
        <LossMap surf={zoom} target={o.best} overlay={tolRing(o, tol, ok)} paths={[{ key: "t", pts: tail, color: ok ? THEME.ok : THEME.accent, shape: "circle", width: 1, label: "last " + TAIL }]} ariaLabel="Zoom around the best fit with the last 400 updates and the tolerance ring" />
      </div>
    </div>
  );
}

export default function DecayLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [etaC0, setEtaC0] = useState(0.02);
  const [eta0, setEta0] = useState(0.04);
  const [p, setP] = useState(0);
  const [eta0C, setEta0C] = useState(0.02);
  const [pC, setPC] = useState(0);

  const constRes = useMemo(() => settleRun(O2, START2, () => etaC0, K2), [etaC0]);
  const res = useMemo(() => settleRun(O2, START2, decay(eta0, p), K2), [eta0, p]);
  const resC = useMemo(() => settleRun(C2, START2C, decay(eta0C, pC), K2C), [eta0C, pC]);
  const okConst = constRes.tailMax <= TOL2;
  const ok = res.tailMax <= TOL2;
  const okC = resC.tailMax <= TOL2C;

  const distPlot = (r, tol, K) => (
    <Plot
      series={[{ xs: r.r.ks, ys: r.d, color: THEME.accent, label: "distance to +" }]}
      xr={[0, K]}
      yr={[-3, 1]}
      logY
      xl="update k"
      label="distance from the best fit (log scale)"
      height={150}
      hlines={[{ y: tol, color: THEME.highlight, label: "ring" }]}
      vlines={[{ x: K - TAIL, color: THEME.muted, label: "last " + TAIL + " must stay in the ring" }]}
    />
  );
  const etaPlot = (e0, pp, K) => {
    const xs = [];
    const ys = [];
    const s = decay(e0, pp);
    for (let k = 0; k <= K; k += 8) {
      xs.push(k);
      ys.push(s(k));
    }
    return <Plot series={[{ xs, ys, color: THEME.text, dash: "5 3", label: "ηₖ" }]} xr={[0, K]} yr={[-4, -0.5]} logY xl="update k" label="learning rate over training (log scale)" height={120} />;
  };
  const status = (r, tol, good) => (
    <>
      <Readout
        rows={[
          ["distance after the last update", r.r.diverged ? "diverged" : fmt(r.d[r.d.length - 1], 4)],
          ["widest swing in the last " + TAIL, r.r.diverged ? "—" : fmt(r.tailMax, 4)],
        ]}
      />
      <Goal met={good}>the last {TAIL} updates all stay inside the ring (radius {fmt(tol, 3)})</Goal>
    </>
  );

  let question, stage, side;
  if (screen === 0) {
    question = (
      <>
        Back to the 40 orders, one order per update, {K2} updates. The right map zooms in on the best fit; the dashed ring has radius {TOL2}. Find a fixed η so that, by the end, the fitter has arrived and stays in the ring for the last {TAIL} updates.
      </>
    );
    stage = (
      <>
        <Views o={O2} surf={SURF2} zoom={ZOOM2} res={constRes} tol={TOL2} ok={okConst} start={START2} />
        {distPlot(constRes, TOL2, K2)}
      </>
    );
    side = (
      <>
        <Panel edge={okConst ? THEME.ok : THEME.grid}>
          <LogSlider id="l2-const" label="fixed learning rate η" value={etaC0} lo={-3} hi={-1} onChange={setEtaC0} digits={4} />
          {status(constRes, TOL2, okConst)}
        </Panel>
        <Note>Try a large η, then a tiny one. Each fails differently. What is the cloud in the zoom made of, given that the fitter is already next to the best fit?</Note>
      </>
    );
  } else if (screen === 1) {
    question = (
      <>
        Let η shrink: ηₖ = η₀ · (1 + k/{TAU})^(−p). With p = 0 it is constant. Find a pair (η₀, p) that lands in the ring and stays. Then push p up to 2 with a small η₀. What goes wrong now?
      </>
    );
    stage = (
      <>
        <Views o={O2} surf={SURF2} zoom={ZOOM2} res={res} tol={TOL2} ok={ok} start={START2} />
        {etaPlot(eta0, p, K2)}
        {distPlot(res, TOL2, K2)}
      </>
    );
    side = (
      <>
        <Panel edge={ok ? THEME.ok : THEME.grid}>
          <LogSlider id="l2-eta0" label="starting rate η₀" value={eta0} lo={-2.3} hi={-0.9} onChange={setEta0} digits={3} />
          <Slider id="l2-p" label="decay power p" value={p} min={0} max={2} stepSize={0.01} onChange={setP} display={fmt(p, 2)} />
          {status(res, TOL2, ok)}
        </Panel>
        <Note>Compare the η trace with the distance trace. When does the cloud start to shrink? With p = 2, where does the fitter stop?</Note>
      </>
    );
  } else if (screen === 2) {
    const s1 = sumInf(eta0, p, 1);
    const s2 = sumInf(eta0, p, 2);
    question = (
      <>
        Name it. At the best fit the <i>average</i> gradient is zero, but one order's gradient is not, so a step of size η still kicks the weights. The cloud's width grows with η. For SGD to settle, the steps must shrink, but slowly enough that their total is still unlimited:
      </>
    );
    stage = (
      <>
        <Views o={O2} surf={SURF2} zoom={ZOOM2} res={res} tol={TOL2} ok={ok} start={START2} />
        {etaPlot(eta0, p, K2)}
      </>
    );
    side = (
      <>
        <Panel>
          <Eq size={13}>
            Σₖ ηₖ = ∞ &nbsp;&nbsp;(total travel unlimited: can reach any point)
            <br />
            Σₖ ηₖ² &lt; ∞ &nbsp;(total kick energy limited: the cloud dies out)
          </Eq>
          <LogSlider id="l2-eta0n" label="starting rate η₀" value={eta0} lo={-2.3} hi={-0.9} onChange={setEta0} digits={3} />
          <Slider id="l2-pn" label="decay power p" value={p} min={0} max={2} stepSize={0.01} onChange={setP} display={fmt(p, 2)} />
          <Readout
            rows={[
              ["Σ ηₖ over all k", isFinite(s1) ? "≈ " + fmt(s1, 2) + " (finite: can stall)" : "∞", isFinite(s1) ? THEME.warn : THEME.ok],
              ["Σ ηₖ² over all k", isFinite(s2) ? "≈ " + fmt(s2, 4) + " (finite)" : "∞ (cloud never dies)", isFinite(s2) ? THEME.ok : THEME.warn],
            ]}
          />
          <Goal met={p > 0.5 && p <= 1}>both conditions hold: 0.5 &lt; p ≤ 1</Goal>
        </Panel>
        <Note>
          ηₖ ∝ 1/k (p = 1) is the fastest shrinking that keeps Σηₖ infinite, and it is the best rate for strongly convex losses. With these conditions SGD converges almost surely: to the optimum for a convex loss, to a stationary point otherwise. In practice the schedule is found by trial (lesson 5).
        </Note>
      </>
    );
  } else {
    question = (
      <>
        A busier city: 60 orders with noisier fees, a start far from the best fit, {K2C} updates. The ring has radius {TOL2C}. Design ηₖ = η₀ · (1 + k/{TAU})^(−p) so that the last {TAIL} updates stay inside it.
      </>
    );
    stage = (
      <>
        <Views o={C2} surf={SURF2C} zoom={ZOOM2C} res={resC} tol={TOL2C} ok={okC} start={START2C} />
        {distPlot(resC, TOL2C, K2C)}
      </>
    );
    side = (
      <>
        <Panel edge={okC ? THEME.ok : THEME.grid}>
          <LogSlider id="l2-eta0c" label="starting rate η₀" value={eta0C} lo={-2.3} hi={-0.9} onChange={setEta0C} digits={3} />
          <Slider id="l2-pc" label="decay power p" value={pC} min={0} max={2} stepSize={0.01} onChange={setPC} display={fmt(pC, 2)} />
          {status(resC, TOL2C, okC)}
          <Hint
            hints={[
              "Read the distance trace. If it flattens out above the ring, the steps shrank too early. If it reaches the ring but keeps bouncing out, they do not shrink enough.",
              "Keep p between about 0.75 and 1.5, where Σηₖ is large but ηₖ still dies away. Then raise η₀ until the far start is reached in time.",
            ]}
          />
        </Panel>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="Why SGD never settles" question={question} stage={stage} side={side} />;
}
