import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import { THEME, MONO } from "./common";

// Lesson 1 - Step size against curvature (Optimization I, slides 12-16).
// Aha: the right step size is set by how sharply the loss curves, not by the data or the gradient.
// For a parabola, eta_opt = 1 / E'' lands in one step; past 2 eta_opt every step overshoots further.

// eight training points (x, y). Model: y_hat = w * x, squared-error loss. "4x" = the same data with every input scaled by 4.
const DIST = [1.2, 1.9, 2.4, 3.0, 3.5, 4.1, 4.6, 5.3];
const NOISE = [0.3, -0.2, 0.4, -0.3, 0.1, 0.5, -0.4, 0.2];
const FEE = DIST.map((d, i) => 1.6 * d + NOISE[i]);
const UNITS = { km: { s: 1, name: "x" }, blk: { s: 4, name: "4x" } };
function curvature(u) {
  let a = 0;
  for (const d of DIST) a += (d * UNITS[u].s) * (d * UNITS[u].s);
  return a / DIST.length;
}
function bestRate(u) {
  let num = 0;
  let den = 0;
  DIST.forEach((d, i) => {
    num += d * UNITS[u].s * FEE[i];
    den += (d * UNITS[u].s) * (d * UNITS[u].s);
  });
  return num / den;
}
function lossFor(u) {
  const s = UNITS[u].s;
  return function (r) {
    let L = 0;
    DIST.forEach((d, i) => {
      const e = FEE[i] - r * d * s;
      L += 0.5 * e * e;
    });
    return L / DIST.length;
  };
}
function slopeOf(f, r) {
  return (f(r + 1e-5) - f(r - 1e-5)) / 2e-5;
}
function walk(f, r0, eta, n) {
  const out = [r0];
  let r = r0;
  for (let k = 0; k < n; k++) {
    r = r - eta * slopeOf(f, r);
    if (!isFinite(r) || Math.abs(r) > 1e4) break;
    out.push(r);
  }
  return out;
}
const LOSS = { km: lossFor("km"), blk: lossFor("blk") };
const XR = [-0.6, 3.6];
const R0 = 0.3;

// challenge: inputs scaled by 1/1.609 and a log-cosh loss (not a parabola, so E'' changes with w)
const MILES = DIST.map((d) => d / 1.609);
const FEE_C = DIST.map((d, i) => 1.6 * d + 2 * NOISE[i]);
function robustLoss(r) {
  let L = 0;
  MILES.forEach((d, i) => {
    L += 64 * Math.log(Math.cosh((FEE_C[i] - r * d) / 8));
  });
  return L / MILES.length;
}
const R_STAR_C = (() => {
  let lo = 0;
  let hi = 6;
  for (let k = 0; k < 80; k++) {
    const m = (lo + hi) / 2;
    if (slopeOf(robustLoss, m) > 0) hi = m;
    else lo = m;
  }
  return (lo + hi) / 2;
})();

function EtaBar({ eta, etaOpt, max }) {
  // eta axis split into the three regimes
  const W = 600;
  const X = (e) => 10 + (clamp(e, 0, max) / max) * (W - 20);
  const zones = [
    [0, etaOpt, "creeps in", THEME.ok, "none"],
    [etaOpt, 2 * etaOpt, "zig-zags in", THEME.highlight, "5 3"],
    [2 * etaOpt, max, "blows up", THEME.warn, "2 3"],
  ];
  return (
    <svg viewBox={"0 0 " + W + " 64"} style={{ width: "100%", display: "block" }} role="img" aria-label="Learning-rate regimes">
      {zones.map((z) => (
        <g key={z[2]}>
          <rect x={X(z[0])} y={10} width={Math.max(0, X(z[1]) - X(z[0]))} height={18} fill={rgba(z[3], 0.22)} stroke={z[3]} strokeDasharray={z[4] === "none" ? undefined : z[4]} />
          <text x={(X(z[0]) + X(z[1])) / 2} y={23} textAnchor="middle" fontFamily={MONO} fontSize={11} fill={THEME.text}>
            {z[2]}
          </text>
        </g>
      ))}
      {[
        [etaOpt, "η_opt = 1/E''"],
        [2 * etaOpt, "2·η_opt"],
      ].map((t) => (
        <text key={t[1]} x={X(t[0])} y={42} textAnchor="middle" fontFamily={MONO} fontSize={10.5} fill={THEME.muted}>
          {t[1]}
        </text>
      ))}
      <polygon points={X(eta) + ",30 " + (X(eta) - 6) + ",40 " + (X(eta) + 6) + ",40"} fill={THEME.text} transform="translate(0,14)" />
      <text x={X(eta)} y={63} textAnchor="middle" fontFamily={MONO} fontSize={10.5} fill={THEME.text}>
        your η
      </text>
    </svg>
  );
}

export default function StepLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [unit, setUnit] = useState("km");
  const [shown, setShown] = useState(6);
  const [eta, setEta] = useState(0.05);
  const [path, setPath] = useState([1.0]);
  const [etaC, setEtaC] = useState(0.1);
  const [probe, setProbe] = useState(null);

  const f = LOSS[unit];
  const a = curvature(unit);
  const rStar = bestRate(unit);
  const usedEta = screen === 0 ? 0.05 : eta;
  const pts = useMemo(() => walk(f, R0, usedEta, 8), [f, usedEta]);
  const vis = pts.slice(0, shown + 1);
  const diverged = pts.length < 9 || Math.abs(pts[pts.length - 1] - rStar) > 50;
  const errs = pts.map((r) => r - rStar);
  const ratio = 1 - usedEta * a;
  const pickUnit = useCallback((u) => setUnit(u), []);

  const rc = path[path.length - 1];
  const stepsC = path.length - 1;
  const doProbe = useCallback(() => {
    setProbe({ r: rc, g0: slopeOf(robustLoss, rc), g1: slopeOf(robustLoss, rc + 0.1) });
  }, [rc]);
  const stepC = useCallback(() => {
    setPath((p) => p.concat([p[p.length - 1] - etaC * slopeOf(robustLoss, p[p.length - 1])]));
  }, [etaC]);
  const resetC = useCallback(() => {
    setPath([1.0]);
    setProbe(null);
  }, []);
  const solved = Math.abs(rc - R_STAR_C) < 0.01 && stepsC <= 3;

  const unitPills = (
    <Pills
      options={[
        ["km", "inputs x"],
        ["blk", "inputs 4x"],
      ]}
      value={unit}
      onPick={pickUnit}
    />
  );
  const curve = (
    <Curve1D
      f={f}
      xr={XR}
      yr={[0, 9]}
      pts={vis}
      xl="weight w"
      label={"loss E(w) = mean ½(y − w·" + UNITS[unit].name + ")²"}
      vlines={screen >= 1 ? [{ x: rStar, color: THEME.muted, label: "w*" }] : []}
    />
  );
  const rList = (
    <div style={{ fontFamily: MONO, fontSize: 12.5, lineHeight: 1.6, color: THEME.text }}>
      {pts.slice(0, shown + 1).map((r, k) => (
        <span key={k} style={{ marginRight: 10, whiteSpace: "nowrap" }}>
          w{k}={fmt(r, 2)}
        </span>
      ))}
      {diverged && <div style={{ color: THEME.warn }}>✕ each step lands farther from w*: diverging</div>}
    </div>
  );

  let question, stage, side;
  if (screen === 0) {
    question = (
      <>
        <b>Where we are:</b> the deck starts its tour of convergence with the simplest case, gradient descent on a one-weight quadratic loss (slides 12–16). Before anything harder we need one answer: for a given loss, which learning rates η converge, and which one is best?
        <br />
        <b>The puzzle:</b> we fit y ≈ w·x with squared error, so E(w) is a parabola. With η = 0.05 gradient descent settles. Now multiply every input by 4 (same data, different scale) and keep η = 0.05. Switch between the two and replay the steps. Why does one converge and the other blow up?
      </>
    );
    stage = curve;
    side = (
      <>
        <Panel>
          {unitPills}
          <Slider id="l1-shown" label="steps shown" value={shown} min={0} max={8} stepSize={1} onChange={setShown} display={shown + " of 8"} />
          <Readout rows={[["learning rate η", "0.05 (fixed)"]]} />
          {rList}
        </Panel>
        <Note>Both curves come from the same data. Compare how wide the two bowls are.</Note>
      </>
    );
  } else if (screen === 1) {
    question = (
      <>
        Now you set η. With inputs x, find three values: the η that reaches w* in exactly one step, the η where the steps start to zig-zag around w*, and the η where they blow up. How are the three related? Then switch to 4x and find them again.
      </>
    );
    stage = (
      <>
        {curve}
        <Trace series={[{ vals: errs, color: THEME.highlight, label: "w_k − w*" }]} n={8} yr={[-2, 2]} zero label="error w_k − w* after each step" />
      </>
    );
    side = (
      <>
        <Panel>
          {unitPills}
          <Slider id="l1-eta" label="learning rate η" value={eta} min={0.001} max={unit === "km" ? 0.2 : 0.015} stepSize={0.0005} onChange={setEta} display={fmt(eta, 4)} />
          <Slider id="l1-shown2" label="steps shown" value={shown} min={0} max={8} stepSize={1} onChange={setShown} display={shown + " of 8"} />
          <Readout
            rows={[
              ["(w₁ − w*) / (w₀ − w*)", fmt(ratio, 3), Math.abs(ratio) > 1 ? THEME.warn : THEME.text],
              ["(w₂ − w*) / (w₁ − w*)", pts.length > 2 ? fmt((pts[2] - rStar) / (pts[1] - rStar), 3) : "—"],
            ]}
          />
        </Panel>
        <Note>Each step multiplies the error w − w* by the same factor. When is that factor 0 (one step)? Negative (zig-zag)? Below −1 (blow-up)?</Note>
      </>
    );
  } else if (screen === 2) {
    const etaOpt = 1 / a;
    const T = (r) => f(pts[0]) + slopeOf(f, pts[0]) * (r - pts[0]) + 0.5 * a * (r - pts[0]) * (r - pts[0]);
    question = (
      <>
        <b>What you found:</b> on a quadratic, the best step size is set by the curvature E'', not by the data size or the gradient. η_opt = 1/E'' reaches the minimum in one step. Below it you creep in; between η_opt and 2η_opt you zig-zag but still converge; above 2η_opt you diverge (slides 13–15). Drag η and check the three zones.
      </>
    );
    stage = (
      <>
        <Curve1D f={f} xr={XR} yr={[0, 9]} pts={vis} xl="weight w" label="E(w) and its Taylor parabola at w₀ (dashed; exact here because E is a parabola)" curves={[{ fn: T, color: THEME.text, dash: "5 4", width: 1.3 }]} />
        <EtaBar eta={eta} etaOpt={etaOpt} max={3 * etaOpt} />
      </>
    );
    side = (
      <>
        <Panel>
          {unitPills}
          <Slider id="l1-eta3" label="learning rate η" value={eta} min={0.001} max={3 * etaOpt} stepSize={0.0005} onChange={setEta} display={fmt(eta, 4)} />
          <Eq size={13}>
            E(w) = E(w₀) + E'(w₀)(w − w₀) + ½E''(w − w₀)²
            <br />
            w_min = w₀ − E'(w₀) / E''
            <br />
            E'' = mean(x²) = <b style={{ color: THEME.highlight }}>{fmt(a, 2)}</b>
            <br />
            η_opt = 1 / E'' = <b style={{ color: THEME.highlight }}>{fmt(etaOpt, 4)}</b>
            <br />
            w_k+1 − w* = (1 − η/η_opt)(w_k − w*) = {fmt(1 - eta / etaOpt, 3)} × (w_k − w*)
          </Eq>
          <Note>Scaling inputs by 4 multiplies E'' by 16, so η_opt shrinks by 16. With η = 0.05 the x run was in the creeping zone and the 4x run was far past 2·η_opt. Next: with two or more weights, each direction has its own curvature, so one η can't be best for all of them.</Note>
        </Panel>
      </>
    );
  } else {
    question = (
      <>
        Real losses are not parabolas. Here E(w) is a log-cosh loss, so E'' changes with w and you aren't told its value. You can probe the slope E' at your current w and at w + 0.1. Starting from w = 1.00, get within 0.01 of the minimum in at most 3 steps.
      </>
    );
    stage = (
      <>
        <Curve1D f={robustLoss} xr={[0, 4]} yr={[0, 8]} pts={path} xl="weight w" label="log-cosh loss E(w)" vlines={solved ? [{ x: R_STAR_C, color: THEME.ok, label: "w*" }] : []} />
        <Trace series={[{ vals: path.map((r) => r - R_STAR_C), color: THEME.highlight, label: "w_k − w*" }]} n={Math.max(3, stepsC)} yr={[-1.8, 1.8]} zero label="error w_k − w*" />
      </>
    );
    side = (
      <>
        <Panel edge={solved ? THEME.ok : THEME.grid}>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 8 }}>
            <Btn onClick={doProbe}>probe slope here</Btn>
            <Btn onClick={resetC}>restart</Btn>
          </div>
          {probe && (
            <Readout
              rows={[
                ["E'(" + fmt(probe.r, 3) + ")", fmt(probe.g0, 3)],
                ["E'(" + fmt(probe.r + 0.1, 3) + ")", fmt(probe.g1, 3)],
              ]}
            />
          )}
          <Slider id="l1-etac" label="learning rate η" value={etaC} min={0.005} max={0.8} stepSize={0.001} onChange={setEtaC} display={fmt(etaC, 3)} />
          <Btn onClick={stepC} strong disabled={stepsC >= 6}>
            take one step: w ← w − η·E'(w)
          </Btn>
          <div style={{ marginTop: 8 }}>
            <Readout
              rows={[
                ["current w", fmt(rc, 4)],
                ["slope E'(w)", fmt(slopeOf(robustLoss, rc), 4)],
              ]}
            />
            <Goal met={stepsC <= 3}>steps used: {stepsC} / 3</Goal>
            <Goal met={solved}>w within 0.01 of the minimum w*</Goal>
          </div>
          <Hint
            hints={[
              "The two probes tell you how fast the slope changes as w moves. That rate of change is the curvature E''.",
              "Set η = 1/E'' with E'' ≈ (difference of the two slopes) / 0.1. The curve is not a parabola, so probe again after each step and update η.",
            ]}
          />
        </Panel>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="How big a step?" question={question} stage={stage} side={side} />;
}
