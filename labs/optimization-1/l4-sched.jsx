import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import { THEME, MONO } from "./common";

// Lesson 4 - Decaying learning rates on a rugged loss (Optimization I, slides 48-54, plus 3-8 on saddles).
// Aha: a sharp little dip has a tiny eta_opt, so a large eta cannot stay in it and bounces out. Start big to
// skip the ripples, then shrink eta so that late in training a dip near the bottom can hold you.

function ripple(c, A, om, ctr) {
  return function (w) {
    return 0.5 * c * (w - ctr) * (w - ctr) + A * (1 - Math.cos(om * (w - ctr)));
  };
}
const LOSS = ripple(0.1, 0.15, 4, 6);
const RIPPLE_CURV = 0.1 + 0.15 * 16; // curvature at the bottom of a dip
const LOSS_C = ripple(0.16, 0.12, 5, 3.5);
const N = 80;
function slope(f, w) {
  return (f(w + 1e-5) - f(w - 1e-5)) / 2e-5;
}
function makeSched(kind, eta0, alpha) {
  if (kind === "inv") return (k) => eta0 / (1 + alpha * k);
  if (kind === "inv2") return (k) => eta0 / ((1 + alpha * k) * (1 + alpha * k));
  if (kind === "exp") return (k) => eta0 * Math.exp(-alpha * k);
  if (kind === "step") return (k) => eta0 * Math.pow(0.1, Math.floor(k / 40));
  return () => eta0;
}
function walk(f, w0, sched) {
  const ws = [w0];
  const etas = [];
  let w = w0;
  for (let k = 0; k < N; k++) {
    const e = sched(k);
    etas.push(e);
    w = w - e * slope(f, w);
    if (!isFinite(w) || Math.abs(w) > 1e3) return { ws, etas, diverged: true };
    ws.push(w);
  }
  return { ws, etas, diverged: false };
}
function settled(ws) {
  if (ws.length < N + 1) return false;
  const last = ws.slice(-6);
  return Math.max(...last) - Math.min(...last) < 0.02;
}
const KINDS = [
  ["const", "constant"],
  ["inv", "η₀/(1+αk)"],
  ["inv2", "η₀/(1+αk)²"],
  ["exp", "η₀·e^(−αk)"],
  ["step", "×0.1 at step 40"],
];

export default function SchedLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [eta, setEta] = useState(0.2);
  const [kind, setKind] = useState("inv");
  const [eta0, setEta0] = useState(3);
  const [alpha, setAlpha] = useState(0);
  const [dims, setDims] = useState(10);
  const [kindC, setKindC] = useState("const");
  const [eta0C, setEta0C] = useState(0.3);
  const [alphaC, setAlphaC] = useState(0.05);

  const sched = screen === 0 ? makeSched("const", eta) : makeSched(kind, eta0, alpha);
  const res = useMemo(() => walk(LOSS, 0.5, sched), [screen, eta, kind, eta0, alpha]);
  const end = res.ws[res.ws.length - 1];
  const ok = !res.diverged && settled(res.ws) && Math.abs(end - 6) < 0.35;
  const resC = useMemo(() => walk(LOSS_C, 9, makeSched(kindC, eta0C, alphaC)), [kindC, eta0C, alphaC]);
  const endC = resC.ws[resC.ws.length - 1];
  const okC = !resC.diverged && settled(resC.ws) && Math.abs(endC - 3.5) < 0.35;
  const pickKind = useCallback((k) => setKind(k), []);
  const pickKindC = useCallback((k) => setKindC(k), []);

  const status = (r, e, good, center) => (
    <>
      <Readout
        rows={[
          ["final weight", r.diverged ? "diverged" : fmt(e, 2)],
          ["final loss", r.diverged ? "—" : fmt((center === 6 ? LOSS : LOSS_C)(e), 3)],
        ]}
      />
      <Goal met={!r.diverged && settled(r.ws)}>settled (last 6 steps move less than 0.02)</Goal>
      <Goal met={good}>settled in the deepest dip</Goal>
    </>
  );
  const curve = (f, r, extra) => <Curve1D f={f} xr={[0, 10]} yr={[0, 3]} pts={r.ws.slice(-40)} xl="weight w" label={"loss along one weight · last 40 of " + N + " steps shown"} vlines={extra} />;
  const wTrace = (r, ctr) => <Trace series={[{ vals: r.ws.map((w) => w - ctr), color: THEME.highlight, label: "w − best w" }]} n={N} yr={[-6, 6]} zero label="weight over training" height={130} />;

  let question, stage, side;
  if (screen === 0) {
    question = (
      <>
        A real network's loss is rugged. This is the loss along one weight of the delivery-time model: a wide valley covered in small dips, with the deepest dip at w = 6. Training starts at w = 0.5 and runs 80 steps. Is there any fixed η that ends up settled in the deepest dip?
      </>
    );
    stage = (
      <>
        {curve(LOSS, res)}
        {wTrace(res, 6)}
      </>
    );
    side = (
      <>
        <Panel edge={ok ? THEME.ok : THEME.grid}>
          <Slider id="l4-eta" label="fixed learning rate η" value={eta} min={0.02} max={6} stepSize={0.01} onChange={setEta} display={fmt(eta, 2)} />
          {status(res, end, ok, 6)}
        </Panel>
        <Note>Try a small η, then a large one. Each fails in a different way. Describe each failure in one sentence.</Note>
      </>
    );
  } else if (screen === 1) {
    question = (
      <>
        Let η change during training: η_k is the step size at step k. Pick a schedule and tune η₀ and α so the walker escapes the first dips, crosses the valley and then settles in the deepest dip. At about what η does the walker stop hopping between dips?
      </>
    );
    stage = (
      <>
        {curve(LOSS, res)}
        <Trace
          series={[
            { vals: res.etas, color: THEME.text, dash: "5 3", label: "η_k" },
          ]}
          n={N}
          yr={[-2, 1]}
          log
          label="learning rate over training (log scale)"
          height={120}
        />
        {wTrace(res, 6)}
      </>
    );
    side = (
      <>
        <Panel edge={ok ? THEME.ok : THEME.grid}>
          <Pills options={KINDS} value={kind} onPick={pickKind} />
          <Slider id="l4-eta0" label="starting η₀" value={eta0} min={0.05} max={6} stepSize={0.01} onChange={setEta0} display={fmt(eta0, 2)} />
          <Slider id="l4-alpha" label="decay α" value={alpha} min={0} max={0.4} stepSize={0.001} onChange={setAlpha} display={fmt(alpha, 3)} />
          {status(res, end, ok, 6)}
        </Panel>
        <Note>Compare the η trace with the weight trace. When does the weight stop jumping around?</Note>
      </>
    );
  } else if (screen === 2) {
    const etaHold = 2 / RIPPLE_CURV;
    const pAll = Math.pow(0.5, dims);
    question = (
      <>
        Name it. A dip with curvature c has η_opt = 1/c. Each dip here curves by about {fmt(RIPPLE_CURV, 1)}, so it can hold the walker only once η &lt; 2/c ≈ {fmt(etaHold, 2)}. The wide valley curves by just 0.1, so it stays stable up to η = 20. A big η sees the valley and ignores the dips. A shrinking η lets one dip catch you near the bottom.
      </>
    );
    stage = (
      <>
        {curve(LOSS, res)}
        <Trace
          series={[
            { vals: res.etas, color: THEME.text, dash: "5 3", label: "η_k" },
            { vals: res.etas.map(() => etaHold), color: THEME.warn, dash: "2 3", label: "2/c of a dip" },
          ]}
          n={N}
          yr={[-2, 1]}
          log
          label="learning rate over training (log scale)"
          height={130}
        />
      </>
    );
    side = (
      <>
        <Panel>
          <Pills options={KINDS} value={kind} onPick={pickKind} />
          <Slider id="l4-eta0n" label="starting η₀" value={eta0} min={0.05} max={6} stepSize={0.01} onChange={setEta0} display={fmt(eta0, 2)} />
          <Slider id="l4-alphan" label="decay α" value={alpha} min={0} max={0.4} stepSize={0.001} onChange={setAlpha} display={fmt(alpha, 3)} />
          <Eq size={12.5}>
            1/(1+αk) decay: η_k = η₀ / (1 + αk)
            <br />
            quadratic: η_k = η₀ / (1 + αk)²
            <br />
            exponential: η_k = η₀ · e^(−αk)
            <br />
            in practice: ×0.1 when the loss stalls
          </Eq>
        </Panel>
        <Panel>
          <Label>are bad minima even the problem? (slides 3-8)</Label>
          <Slider id="l4-dims" label="number of weights N" value={dims} min={1} max={60} stepSize={1} onChange={setDims} display={dims} />
          <Note>A flat point is a minimum only if the loss curves up in all N directions. If each direction were a coin flip, the chance would be</Note>
          <Readout rows={[["P(all N curve up) = 0.5^N", pAll < 1e-4 ? pAll.toExponential(1) : fmt(pAll, 4)]]} />
          <Note>So in large networks most flat points are saddles. Most local minima that do exist have loss close to the global one.</Note>
        </Panel>
      </>
    );
  } else {
    question = (
      <>
        A second weight, a new landscape: dips are tighter and the deepest one sits at w = 3.5. Training starts at w = 9. Design a schedule that ends settled in that dip after 80 steps.
      </>
    );
    stage = (
      <>
        {curve(LOSS_C, resC, okC ? [{ x: 3.5, color: THEME.ok, label: "deepest dip" }] : [])}
        <Trace series={[{ vals: resC.etas, color: THEME.text, dash: "5 3", label: "η_k" }]} n={N} yr={[-2, 1]} log label="learning rate (log scale)" height={110} />
        {wTrace(resC, 3.5)}
      </>
    );
    side = (
      <>
        <Panel edge={okC ? THEME.ok : THEME.grid}>
          <Pills options={KINDS} value={kindC} onPick={pickKindC} />
          <Slider id="l4-eta0c" label="starting η₀" value={eta0C} min={0.05} max={6} stepSize={0.01} onChange={setEta0C} display={fmt(eta0C, 2)} />
          <Slider id="l4-alphac" label="decay α" value={alphaC} min={0} max={0.4} stepSize={0.001} onChange={setAlphaC} display={fmt(alphaC, 3)} />
          {status(resC, endC, okC, 3.5)}
          <Hint
            hints={[
              "Look at how the walker fails. Stuck in the first dip means η starts too small. Never settling means η never drops below what one dip can hold.",
              "These dips curve by about 0.16 + 0.12·25 ≈ 3.2, so they hold you once η < 0.63. Start well above that (1 to 2), and pick α so η crosses 0.63 only in the last third of training.",
            ]}
          />
        </Panel>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="Big steps first, small steps later" question={question} stage={stage} side={side} />;
}
