import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import { THEME, MONO } from "./common";

// Lesson 3 - Rescale and rotate until the bowl is round: Newton's method (Optimization I, slides 23-47).
// Aha: on a stretched bowl the gradient points across the valley, not at the minimum. Stretch space by
// A^(1/2) and the bowl turns into a circle, so the gradient points straight at the minimum. Undoing the
// stretch gives w <- w - A^-1 grad = Newton. With a saddle-shaped Hessian, Newton heads for the saddle.

const W_STAR = [1.5, 3.0];
const A = rotA(4, 0.25, -35);
const EIG = eig2(A);
const BOWL = { f: quad(A, W_STAR), xr: [-1.5, 4.5], yr: [0, 6], xl: "w₁", yl: "w₂" };
const START = [-0.6, 5.2];

// S = E diag(s1, s2) E^T, warped coordinates u = S (w - w*)
function stretch(s1, s2) {
  const v1 = EIG.v1;
  const v2 = EIG.v2;
  const M = (a, b) => [
    [a * v1[0] * v1[0] + b * v2[0] * v2[0], a * v1[0] * v1[1] + b * v2[0] * v2[1]],
    [a * v1[0] * v1[1] + b * v2[0] * v2[1], a * v1[1] * v1[1] + b * v2[1] * v2[1]],
  ];
  return { S: M(s1, s2), Si: M(1 / s1, 1 / s2) };
}
function mul(M, v) {
  return [M[0][0] * v[0] + M[0][1] * v[1], M[1][0] * v[0] + M[1][1] * v[1]];
}

// a curved valley that is not a quadratic (for Newton on a general loss)
function banana(w) {
  const x = w[0] - 1.5;
  const y = w[1] - 3;
  const t = y - 0.35 * x * x;
  return 0.3 * x * x + 1.5 * t * t;
}
const BANANA = { f: banana, xr: [-1.5, 4.5], yr: [0, 6], xl: "w₁", yl: "w₂" };
const B_START = [-1.0, 3.5];

// challenge: a loss with a saddle at the origin and minima at (0, ±1.22)
function saddle(w) {
  return 0.5 * w[0] * w[0] + 0.25 * Math.pow(w[1], 4) - 0.75 * w[1] * w[1];
}
const SADDLE = { f: saddle, xr: [-2.2, 2.2], yr: [-2.2, 2.2], xl: "w₁", yl: "w₂", floor: 0.05 };
const MINS = [
  [0, Math.sqrt(1.5)],
  [0, -Math.sqrt(1.5)],
];
const S_START = [1.8, 0.25];

function gradArrow(f, w, P, target) {
  const g = grad2(f, w);
  const n = Math.hypot(g[0], g[1]) || 1;
  const a = P(w[0], w[1]);
  const b = P(w[0] - (1.4 * g[0]) / n, w[1] - (1.4 * g[1]) / n);
  const t = P(target[0], target[1]);
  return (
    <g>
      <line x1={a[0]} y1={a[1]} x2={t[0]} y2={t[1]} stroke={THEME.text} strokeDasharray="4 4" strokeWidth={1.2} />
      <Arrow x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} color={THEME.warn} width={2.4} />
      <text x={b[0] + 6} y={b[1] + 4} fontFamily={MONO} fontSize={10.5} fill={THEME.warn} stroke={THEME.bg} strokeWidth={3} paintOrder="stroke">
        −gradient
      </text>
    </g>
  );
}
function angleTo(f, w, target) {
  const g = grad2(f, w);
  const d = [target[0] - w[0], target[1] - w[1]];
  const c = (-g[0] * d[0] - g[1] * d[1]) / ((Math.hypot(g[0], g[1]) || 1) * (Math.hypot(d[0], d[1]) || 1));
  return (Math.acos(clamp(c, -1, 1)) * 180) / Math.PI;
}
// ellipse of the local quadratic model at w (Hessian H, gradient g), passing through w
function modelEllipse(f, w, P) {
  const g = grad2(f, w);
  const H = hess2(f, w);
  const e = eig2(H);
  if (e.l2 <= 0) return null;
  const det = H[0][0] * H[1][1] - H[0][1] * H[0][1];
  const step = [-(H[1][1] * g[0] - H[0][1] * g[1]) / det, -(-H[0][1] * g[0] + H[0][0] * g[1]) / det];
  const m = [w[0] + step[0], w[1] + step[1]];
  const r2 = step[0] * (H[0][0] * step[0] + H[0][1] * step[1]) + step[1] * (H[0][1] * step[0] + H[1][1] * step[1]);
  const pts = [];
  for (let i = 0; i <= 64; i++) {
    const t = (i / 64) * 2 * Math.PI;
    const a = (Math.sqrt(r2 / e.l1) * Math.cos(t));
    const b = (Math.sqrt(r2 / e.l2) * Math.sin(t));
    const q = P(m[0] + a * e.v1[0] + b * e.v2[0], m[1] + a * e.v1[1] + b * e.v2[1]);
    pts.push(q[0].toFixed(1) + "," + q[1].toFixed(1));
  }
  const c = P(m[0], m[1]);
  return (
    <g>
      <polyline points={pts.join(" ")} fill="none" stroke={THEME.text} strokeWidth={1.4} strokeDasharray="5 4" />
      <Marker x={c[0]} y={c[1]} shape="diamond" color={THEME.text} r={4} />
      <text x={c[0] + 8} y={c[1] + 14} fontFamily={MONO} fontSize={10.5} fill={THEME.text} stroke={THEME.bg} strokeWidth={3} paintOrder="stroke">
        bottom of the fitted bowl
      </text>
    </g>
  );
}

export default function NewtonLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [probe, setProbe] = useState(START);
  const [s1, setS1] = useState(1);
  const [s2, setS2] = useState(1);
  const [etaU, setEtaU] = useState(0.2);
  const [kSel, setKSel] = useState(0);
  const [nParams, setNParams] = useState(5);
  const [mu, setMu] = useState(0);
  const [etaN, setEtaN] = useState(1);

  const onProbe = useCallback((x, y) => setProbe([x, y]), []);
  const ang = angleTo(BOWL.f, probe, W_STAR);

  // explore: gradient descent in the stretched space, drawn in both spaces
  const st = useMemo(() => stretch(s1, s2), [s1, s2]);
  const warped = useMemo(
    () => ({
      f: (u) => BOWL.f([W_STAR[0] + st.Si[0][0] * u[0] + st.Si[0][1] * u[1], W_STAR[1] + st.Si[1][0] * u[0] + st.Si[1][1] * u[1]]),
      xr: [-6, 6],
      yr: [-6, 6],
      xl: "ũ₁",
      yl: "ũ₂",
    }),
    [st]
  );
  const u0 = mul(st.S, [START[0] - W_STAR[0], START[1] - W_STAR[1]]);
  const runU = useMemo(() => runOpt(warped.f, u0, { kind: "gd", eta: etaU }, 12, 200), [warped, u0[0], u0[1], etaU]);
  const backW = runU.pts.map((u) => {
    const d = mul(st.Si, u);
    return [W_STAR[0] + d[0], W_STAR[1] + d[1]];
  });
  const endW = backW[backW.length - 1];
  const distEnd = Math.hypot(endW[0] - W_STAR[0], endW[1] - W_STAR[1]);
  const after1 = backW.length > 1 ? Math.hypot(backW[1][0] - W_STAR[0], backW[1][1] - W_STAR[1]) : 9;

  // name: Newton vs gradient descent on the curved valley
  const newtonB = useMemo(() => runOpt(banana, B_START, { kind: "newton", eta: 1 }, 8), []);
  const gdB = useMemo(() => runOpt(banana, B_START, { kind: "gd", eta: 0.25 }, 8), []);
  const kB = Math.min(kSel, newtonB.pts.length - 1);
  const wB = newtonB.pts[kB];
  const N = Math.pow(10, nParams);

  // challenge
  const runS = useMemo(() => runOpt(saddle, S_START, { kind: "newton", eta: etaN, mu }, 8, 50), [etaN, mu]);
  const endS = runS.pts[runS.pts.length - 1];
  const distMin = Math.min(...MINS.map((m) => Math.hypot(endS[0] - m[0], endS[1] - m[1])));
  const Hs = hess2(saddle, S_START);
  const eS = eig2([
    [Hs[0][0] + mu, Hs[0][1]],
    [Hs[0][1], Hs[1][1] + mu],
  ]);
  const okS = !runS.diverged && distMin < 0.05;

  let question, stage, side;
  if (screen === 0) {
    question = (
      <>
        <b>Where we are:</b> lesson 2 showed that one η can't suit a bowl that curves differently in different directions. Rescaling inputs only fixes an axis-aligned bowl. The deck's fix is to scale <i>and rotate</i> the axes until the bowl is round (slides 23–30). That leads to Newton's method.
        <br />
        <b>The puzzle:</b> this is the tilted bowl from lesson 2. The red arrow is −∇E, the direction gradient descent moves. The dashed line goes to the minimum. Drag the point. Where does −∇E point straight at the minimum? What do those places have in common?
      </>
    );
    stage = <LossMap surf={BOWL} handle={probe} onHandle={onProbe} target={W_STAR} overlay={(P) => gradArrow(BOWL.f, probe, P, W_STAR)} ariaLabel="Tilted bowl with the negative gradient arrow" />;
    side = (
      <>
        <Panel>
          <Readout rows={[["angle between −∇E and the direction to the minimum", fmt(ang, 1) + "°", ang < 3 ? THEME.ok : THEME.text]]} />
          <Goal met={ang < 3}>−∇E points at the minimum (under 3°)</Goal>
        </Panel>
        <Note>The gradient is always perpendicular to the contour line through your point. On a stretched oval, that is rarely toward the centre.</Note>
      </>
    );
  } else if (screen === 1) {
    question = (
      <>
        Stretch space. The right map is the same loss with the bowl's steep axis scaled by s₁ and its flat axis by s₂. Find s₁ and s₂ that make the contours round. Then run 12 plain gradient steps in the stretched space and map them back on the left. What η lands in one step?
      </>
    );
    stage = (
      <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
        <div style={{ flex: "1 1 260px", minWidth: 0 }}>
          <Label>original rates</Label>
          <LossMap surf={BOWL} paths={[{ pts: backW, color: THEME.highlight, label: "mapped back" }]} target={W_STAR} maxWidth={330} ariaLabel="Original space with the path mapped back" />
        </div>
        <div style={{ flex: "1 1 260px", minWidth: 0 }}>
          <Label>stretched space</Label>
          <LossMap surf={warped} paths={[{ pts: runU.pts, color: THEME.highlight, shape: "square", label: "GD here" }]} target={[0, 0]} maxWidth={330} ariaLabel="Stretched space with gradient descent" />
        </div>
      </div>
    );
    side = (
      <>
        <Panel>
          <Slider id="l3-s1" label="s₁ (steep axis)" value={s1} min={0.3} max={3} stepSize={0.01} onChange={setS1} display={fmt(s1, 2)} />
          <Slider id="l3-s2" label="s₂ (flat axis)" value={s2} min={0.3} max={3} stepSize={0.01} onChange={setS2} display={fmt(s2, 2)} />
          <Slider id="l3-etau" label="η in the stretched space" value={etaU} min={0.02} max={2.2} stepSize={0.01} onChange={setEtaU} display={fmt(etaU, 2)} />
          <Readout
            rows={[
              ["distance to minimum after 1 step", runU.diverged ? "diverged" : fmt(after1, 3)],
              ["distance after 12 steps", runU.diverged ? "diverged" : fmt(distEnd, 3)],
            ]}
          />
          <Goal met={!runU.diverged && after1 < 0.02}>one step lands on the minimum</Goal>
        </Panel>
        <Note>The bowl's curvatures are 4 along the steep axis and 0.25 along the flat one. What stretch turns both into the same curvature?</Note>
      </>
    );
  } else if (screen === 2) {
    question = (
      <>
        <b>What you found:</b> the contours became round at s₁ = 2 = √4 and s₂ = 0.5 = √0.25, so the stretch is S = A^½. On a round bowl the gradient points at the minimum, and η = 1 lands in one step. Mapped back, that step is w ← w − η·A⁻¹∇E. For a general loss, replace A with the Hessian H at the current point: this is Newton's method (slides 28–43). Step through it on a curved valley that is not a quadratic.
      </>
    );
    stage = (
      <LossMap
        surf={BANANA}
        paths={[
          { pts: gdB.pts, color: THEME.text, dash: "4 3", shape: "square", label: "GD η=0.25" },
          { pts: newtonB.pts.slice(0, kB + 1), color: THEME.highlight, label: "Newton" },
        ]}
        overlay={(P) => modelEllipse(banana, wB, P)}
        target={[1.5, 3]}
        ariaLabel="Newton's method on a curved valley"
      />
    );
    side = (
      <>
        <Panel>
          <Slider id="l3-k" label="Newton step" value={kSel} min={0} max={6} stepSize={1} onChange={setKSel} display={"k = " + kB} />
          <Eq size={12.5}>
            E(w) ≈ E(w_k) + ∇Eᵀ(w − w_k) + ½(w − w_k)ᵀH(w − w_k)
            <br />
            w_k+1 = w_k − η·H⁻¹∇E(w_k), η = 1
            <br />
            loss at step k: {fmt(banana(wB), 4)}
          </Eq>
          <Note>Each step fits a bowl to the loss at your point (dashed oval) and jumps to its bottom (diamond). On a true quadratic, η = 1 is exact, and η &gt; 2 diverges, just as in lesson 1.</Note>
        </Panel>
        <Panel>
          <Label>why nobody does this for a real network</Label>
          <Slider id="l3-n" label="number of weights" value={nParams} min={2} max={9} stepSize={0.01} onChange={setNParams} display={"10^" + fmt(nParams, 1)} />
          <Readout
            rows={[
              ["Hessian entries N²", N * N >= 1e5 ? (N * N).toExponential(1) : fmt(N * N, 0)],
              ["memory at 4 bytes each", N * N * 4 >= 1e12 ? fmt((N * N * 4) / 1e12, 1) + " TB" : N * N * 4 >= 1e9 ? fmt((N * N * 4) / 1e9, 1) + " GB" : fmt((N * N * 4) / 1e6, 2) + " MB"],
            ]}
          />
        </Panel>
      </>
    );
  } else {
    question = (
      <>
        Newton's problem 2 (slides 44–47): at a saddle the Hessian has a negative eigenvalue, and Newton moves toward the flat point, not the minimum. This loss has a saddle at the centre (curving up along w₁, down along w₂) and two minima, marked with crosses. Plain Newton from the start heads for the saddle. Using damped Newton, w ← w − η(H + μI)⁻¹∇E, reach a minimum (within 0.05) in 8 steps.
      </>
    );
    stage = (
      <LossMap
        surf={SADDLE}
        paths={[{ pts: runS.pts, color: THEME.highlight, label: runS.diverged ? "diverged" : "after 8 steps" }]}
        overlay={(P) => {
          const s = P(0, 0);
          const m2 = P(MINS[1][0], MINS[1][1]);
          return (
            <g>
              <Marker x={s[0]} y={s[1]} shape="diamond" color={THEME.warn} r={5} />
              <text x={s[0] + 9} y={s[1] + 16} fontFamily={MONO} fontSize={10.5} fill={THEME.warn} stroke={THEME.bg} strokeWidth={3} paintOrder="stroke">
                saddle
              </text>
              <line x1={m2[0] - 9} x2={m2[0] + 9} y1={m2[1]} y2={m2[1]} stroke={THEME.text} strokeWidth={1.5} />
              <line x1={m2[0]} x2={m2[0]} y1={m2[1] - 9} y2={m2[1] + 9} stroke={THEME.text} strokeWidth={1.5} />
            </g>
          );
        }}
        target={MINS[0]}
        ariaLabel="Loss with a saddle and two minima"
      />
    );
    side = (
      <>
        <Panel edge={okS ? THEME.ok : THEME.grid}>
          <Slider id="l3-mu" label="damping μ" value={mu} min={0} max={6} stepSize={0.01} onChange={setMu} display={fmt(mu, 2)} />
          <Slider id="l3-etan" label="η" value={etaN} min={0.1} max={1.5} stepSize={0.01} onChange={setEtaN} display={fmt(etaN, 2)} />
          <Readout
            rows={[
              ["eigenvalues of H + μI at start", fmt(eS.l1, 2) + ", " + fmt(eS.l2, 2), eS.l2 < 0 ? THEME.warn : THEME.text],
              ["distance to nearest minimum", runS.diverged ? "diverged" : fmt(distMin, 3)],
            ]}
          />
          <Goal met={okS}>reached a minimum within 8 steps</Goal>
          <Hint
            hints={[
              "Newton jumps to the flat point of the bowl it fits. Near the start that fitted shape curves down along w₂. What is the flat point of a saddle-shaped fit?",
              "Make every eigenvalue of H + μI positive, but not much more: a huge μ turns Newton into a tiny gradient step of size η/μ. A μ around 1 with η near 1 is worth trying.",
            ]}
          />
        </Panel>
        <Note>Adding μI is the "diagonal loading" behind Levenberg-Marquardt (slide 47). In high dimensions, saddles like this far outnumber bad minima (slides 3–8). Newton is also too costly for a real network, so next we stay with plain gradient steps and change η over time instead.</Note>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="Make the bowl round" question={question} stage={stage} side={side} />;
}
