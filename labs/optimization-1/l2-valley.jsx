import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import { THEME, MONO } from "./common";

// Lesson 2 - One learning rate for every direction (Optimization I, slides 17-24).
// Aha: the steepest direction sets the speed limit (eta < 2/lambda_max) and the flattest direction then crawls.
// How slow you are is set by the ratio lambda_max / lambda_min, the condition number.

// two-weight quadratic loss, curvature 4 along w1 and 0.25 along w2 (e.g. y_hat = w1*x1 + w2*x2 with mean x1^2 = 4, mean x2^2 = 0.25).
const W_STAR = [1.5, 3.0];
const START = [-1.0, 5.5];
const L1 = 4;
const L2 = 0.25;
const A_DIAG = [
  [L1, 0],
  [0, L2],
];
const A_TILT = rotA(L1, L2, -35);
const SURF_DIAG = { f: quad(A_DIAG, W_STAR), xr: [-1.5, 4.5], yr: [0, 6], xl: "w₁", yl: "w₂" };
const SURF_TILT = { f: quad(A_TILT, W_STAR), xr: [-1.5, 4.5], yr: [0, 6], xl: "w₁", yl: "w₂" };
const N = 40;
const TOL = 0.05;

function firstStay(vals, target, tol) {
  for (let k = 0; k < vals.length; k++) {
    let ok = true;
    for (let j = k; j < vals.length; j++)
      if (Math.abs(vals[j] - target) > tol) {
        ok = false;
        break;
      }
    if (ok) return k;
  }
  return -1;
}
function FactorBars({ eta, lams, labels }) {
  // |1 - eta*lambda| per direction: below 1 shrinks, above 1 grows
  const W = 600;
  const X = (v) => 140 + (clamp(v, 0, 2) / 2) * (W - 160);
  return (
    <svg viewBox={"0 0 " + W + " " + (26 + lams.length * 26)} style={{ width: "100%", display: "block" }} role="img" aria-label="Shrink factor per direction">
      <text x={X(1)} y={12} textAnchor="middle" fontFamily={MONO} fontSize={10.5} fill={THEME.muted}>
        1 = no progress
      </text>
      <line x1={X(1)} x2={X(1)} y1={16} y2={20 + lams.length * 26} stroke={THEME.text} strokeDasharray="3 3" />
      {lams.map((l, i) => {
        const v = Math.abs(1 - eta * l);
        const bad = v >= 1;
        return (
          <g key={i}>
            <text x={4} y={34 + i * 26} fontFamily={MONO} fontSize={11.5} fill={THEME.text}>
              {labels[i]}
            </text>
            <rect x={X(0)} y={22 + i * 26} width={X(2) - X(0)} height={16} fill={THEME.bg} stroke={THEME.grid} />
            <rect x={X(0)} y={24 + i * 26} width={X(v) - X(0)} height={12} fill={bad ? THEME.warn : THEME.highlight} opacity={0.85} />
            <text x={Math.min(X(v) + 6, W - 40)} y={34 + i * 26} fontFamily={MONO} fontSize={11} fill={THEME.text}>
              {fmt(v, 2)}
              {bad ? " ✕" : ""}
              {1 - eta * l < 0 ? " (flips)" : ""}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
function axesOverlay(A) {
  const e = eig2(A);
  return function (P) {
    const lines = [
      [e.v1, "steep axis"],
      [e.v2, "flat axis"],
    ];
    return (
      <g>
        {lines.map((l) => {
          const a = P(W_STAR[0] - 4 * l[0][0], W_STAR[1] - 4 * l[0][1]);
          const b = P(W_STAR[0] + 4 * l[0][0], W_STAR[1] + 4 * l[0][1]);
          const t = P(W_STAR[0] + 1.6 * l[0][0], W_STAR[1] + 1.6 * l[0][1]);
          return (
            <g key={l[1]}>
              <line x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} stroke={THEME.text} strokeDasharray="2 4" strokeWidth={1} opacity={0.7} />
              <text x={t[0] + 6} y={t[1] - 4} fontFamily={MONO} fontSize={10.5} fill={THEME.text} stroke={THEME.bg} strokeWidth={3} paintOrder="stroke">
                {l[1]}
              </text>
            </g>
          );
        })}
      </g>
    );
  };
}
const OVER_DIAG = axesOverlay(A_DIAG);
const OVER_TILT = axesOverlay(A_TILT);

export default function ValleyLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [eta, setEta] = useState(0.1);
  const [tilt, setTilt] = useState(false);
  const [start, setStart] = useState(START);
  const [s, setS] = useState(1);
  const [etaC, setEtaC] = useState(0.3);
  const [nC, setNC] = useState(5);

  const useTilt = screen === 1 || screen === 2 ? tilt : false;
  const surf = useTilt ? SURF_TILT : SURF_DIAG;
  const A = useTilt ? A_TILT : A_DIAG;
  const run = useMemo(() => runOpt(surf.f, start, { kind: "gd", eta }, N, 60), [surf, start, eta]);
  const w1s = run.pts.map((p) => p[0]);
  const w2s = run.pts.map((p) => p[1]);
  const k1 = run.diverged ? -1 : firstStay(w1s, W_STAR[0], TOL);
  const k2 = run.diverged ? -1 : firstStay(w2s, W_STAR[1], TOL);
  const onStart = useCallback((x, y) => setStart([x, y]), []);
  const kappa = L1 / L2;

  // challenge: input x2 multiplied by s, so the curvature along w2 grows by s^2
  const lam2 = L2 * s * s;
  const surfC = useMemo(() => ({ f: quad(A_DIAG.map((r, i) => r.map((v, j) => (i === 1 && j === 1 ? lam2 : v))), [W_STAR[0], W_STAR[1] / s]), xr: [-1.5, 4.5], yr: [W_STAR[1] / s - 3, W_STAR[1] / s + 3], xl: "w₁", yl: "w₂ (for input s·x₂)" }), [s, lam2]);
  const runC = useMemo(() => runOpt(surfC.f, [START[0], START[1] / s], { kind: "gd", eta: etaC }, nC, 60), [surfC, s, etaC, nC]);
  const endC = runC.pts[runC.pts.length - 1];
  const perKg = endC[1] * s;
  const okC = !runC.diverged && Math.abs(endC[0] - W_STAR[0]) < TOL && Math.abs(perKg - W_STAR[1]) < TOL;

  const traces = (
    <Trace
      series={[
        { vals: w1s.map((v) => v - W_STAR[0]), color: THEME.highlight, label: "w₁ − w₁*" },
        { vals: w2s.map((v) => v - W_STAR[1]), color: THEME.text, dash: "5 3", label: "w₂ − w₂*" },
      ]}
      n={N}
      yr={[-3, 3]}
      zero
      label="error in each weight after each step"
    />
  );
  const map = (
    <LossMap
      surf={surf}
      paths={[{ pts: run.pts, color: THEME.highlight, label: run.diverged ? "diverged" : "after " + N + " steps" }]}
      handle={start}
      onHandle={onStart}
      target={W_STAR}
      overlay={screen >= 1 ? (useTilt ? OVER_TILT : OVER_DIAG) : null}
      ariaLabel="Loss over the two weights with the gradient-descent path"
    />
  );
  const etaSlider = <Slider id={"l2-eta" + screen} label="learning rate η (same for both weights)" value={eta} min={0.005} max={0.6} stepSize={0.005} onChange={setEta} display={fmt(eta, 3)} />;
  const settle = (
    <>
      <Goal met={k1 >= 0}>w₁ settled within 0.05: {k1 >= 0 ? "step " + k1 : run.diverged ? "diverged" : "not in " + N}</Goal>
      <Goal met={k2 >= 0}>w₂ settled within 0.05: {k2 >= 0 ? "step " + k2 : run.diverged ? "diverged" : "not in " + N}</Goal>
    </>
  );

  let question, stage, side;
  if (screen === 0) {
    question = (
      <>
        <b>Where we are:</b> lesson 1 showed that on one weight the best η is 1/E'', set by the curvature. Real networks have many weights, and the loss curves differently in each direction (slides 17–24). Gradient descent still uses <i>one</i> η for all of them. What does that cost?
        <br />
        <b>The puzzle:</b> the loss below has two weights. It curves 16× more sharply along w₁ than along w₂. The cross marks the minimum. Find the η that gets both weights within 0.05 in the fewest steps. Why can't both settle quickly?
      </>
    );
    stage = (
      <>
        {map}
        {traces}
      </>
    );
    side = (
      <>
        <Panel>
          {etaSlider}
          {settle}
        </Panel>
        <Note>You can drag the start point. Watch which weight is still far off after the other has settled.</Note>
      </>
    );
  } else if (screen === 1) {
    question = (
      <>
        <b>Claim to test:</b> along each axis of the bowl, gradient descent is the 1-D walk from lesson 1. Each step multiplies that axis's error by |1 − ηλ|, where λ is the curvature along it. The bars show this factor. Push η up until one bar passes 1. Which axis hits the limit first? Just under that limit, how fast is the other axis shrinking? Then turn on the tilt (the two inputs are correlated) and see if the same holds along the rotated axes.
      </>
    );
    stage = (
      <>
        {map}
        <FactorBars eta={eta} lams={[L1, L2]} labels={["steep axis", "flat axis"]} />
      </>
    );
    side = (
      <>
        <Panel>
          {etaSlider}
          <Toggle label="correlated inputs (tilted bowl)" value={tilt} onChange={setTilt} />
          {settle}
        </Panel>
        <Note>With the tilt on, the steep and flat axes no longer line up with w₁ and w₂. The factors along the dotted axes stay the same, so the problem is the same, just rotated.</Note>
      </>
    );
  } else if (screen === 2) {
    const best = 2 / (L1 + L2);
    question = (
      <>
        <b>What you found:</b> the steepest direction sets the speed limit, η &lt; 2/λ_max. Under that limit the flattest direction can only crawl. How slow you are depends on the ratio κ = λ_max/λ_min, called the condition number. The λᵢ are the eigenvalues of the Hessian A. Set η to the best compromise below and count the steps.
      </>
    );
    stage = (
      <>
        {map}
        <FactorBars eta={eta} lams={[L1, L2]} labels={["λ₁ = " + L1, "λ₂ = " + L2]} />
      </>
    );
    side = (
      <>
        <Panel>
          {etaSlider}
          <Toggle label="tilted bowl" value={tilt} onChange={setTilt} />
          <Eq size={12.5}>
            E = ½(w − w*)ᵀA(w − w*), A = EΛEᵀ
            <br />
            per axis: error ← (1 − ηλᵢ) · error
            <br />
            no divergence: η &lt; 2/λ_max = {fmt(2 / L1, 3)}
            <br />
            κ = λ_max/λ_min = {fmt(kappa, 0)}
            <br />
            best single η = 2/(λ_max+λ_min) = {fmt(best, 3)}
            <br />
            then both shrink by (κ−1)/(κ+1) = {fmt((kappa - 1) / (kappa + 1), 3)}
          </Eq>
          <Note>Bigger κ means a longer, narrower valley and more steps. The gradient is perpendicular to the contours, so on a stretched bowl it points across the valley instead of toward the minimum (slides 19–23).</Note>
        </Panel>
      </>
    );
  } else {
    question = (
      <>
        You can't change the data, but you can rescale an input: multiply x₂ by s. That changes the curvature along w₂ to 0.25·s². Pick s and η so that both weights are within 0.05 of the minimum after at most {nC} steps (w₂ is reported in the original scale). Go for {nC === 1 ? "one step" : "fewer steps"} if you can.
      </>
    );
    stage = (
      <>
        <LossMap surf={surfC} paths={[{ pts: runC.pts, color: THEME.highlight, label: runC.diverged ? "diverged" : "after " + nC }]} target={[W_STAR[0], W_STAR[1] / s]} ariaLabel="Loss map with x₂ scaled by s" />
        <FactorBars eta={etaC} lams={[L1, lam2]} labels={["w₁ axis", "w₂ axis"]} />
      </>
    );
    side = (
      <>
        <Panel edge={okC ? THEME.ok : THEME.grid}>
          <Slider id="l2-s" label="input scale s (x₂ → s·x₂)" value={s} min={1} max={8} stepSize={0.01} onChange={setS} display={"s = " + fmt(s, 2)} />
          <Slider id="l2-etac" label="learning rate η" value={etaC} min={0.01} max={0.6} stepSize={0.001} onChange={setEtaC} display={fmt(etaC, 3)} />
          <Slider id="l2-nc" label="step budget" value={nC} min={1} max={5} stepSize={1} onChange={setNC} display={nC + " steps"} />
          <Readout
            rows={[
              ["w₁ after the steps", fmt(endC[0], 3)],
              ["w₂ after the steps (original scale)", fmt(perKg, 3)],
              ["curvature along w₂", fmt(lam2, 2)],
            ]}
          />
          <Goal met={okC}>both weights within 0.05 of the minimum</Goal>
          <Hint
            hints={[
              "Watch the shape of the bowl as you change s. Which shape lets one η suit both directions?",
              "The curvature along w₂ is 0.25·s². Make it equal to the curvature along w₁, which is 4. Then lesson 1 tells you which η lands in one step.",
            ]}
          />
        </Panel>
        <Note>Rescaling an input stretches one axis. That makes κ = 1 for an axis-aligned bowl. A tilted bowl needs a rotation as well. Next lesson: do both at once with the Hessian (Newton's method).</Note>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="One η, many directions" question={question} stage={stage} side={side} />;
}
