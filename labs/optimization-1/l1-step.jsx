import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import { THEME, MONO } from "./common";

// Lesson 1 - Step size against curvature (Optimization I, slides 12-16).
// Aha: the right step size is set by how sharply the loss curves, not by the data or the gradient.
// For a parabola, eta_opt = 1 / E'' lands in one step; past 2 eta_opt every step overshoots further.

// eight past deliveries: distance in km, fee in euro. Model: fee = r * distance.
const DIST = [1.2, 1.9, 2.4, 3.0, 3.5, 4.1, 4.6, 5.3];
const NOISE = [0.3, -0.2, 0.4, -0.3, 0.1, 0.5, -0.4, 0.2];
const FEE = DIST.map((d, i) => 1.6 * d + NOISE[i]);
const UNITS = { km: { s: 1, name: "km", per: "€/km" }, blk: { s: 4, name: "block (250 m)", per: "€/block" } };
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

// challenge: distances logged in miles, robust log-cosh loss (big misses count less than squared)
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
        ["km", "distance in km"],
        ["blk", "distance in blocks"],
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
      xl={"rate r (" + UNITS[unit].per + ")"}
      label={"mean squared fee error E(r), distances in " + UNITS[unit].name}
      vlines={screen >= 1 ? [{ x: rStar, color: THEME.muted, label: "best r" }] : []}
    />
  );
  const rList = (
    <div style={{ fontFamily: MONO, fontSize: 12.5, lineHeight: 1.6, color: THEME.text }}>
      {pts.slice(0, shown + 1).map((r, k) => (
        <span key={k} style={{ marginRight: 10, whiteSpace: "nowrap" }}>
          r{k}={fmt(r, 2)}
        </span>
      ))}
      {diverged && <div style={{ color: THEME.warn }}>✕ the steps grow every time: diverging</div>}
    </div>
  );

  let question, stage, side;
  if (screen === 0) {
    question = (
      <>
        The fee model is fee = r × distance, and gradient descent fits the rate r. Last month distances were logged in km and η = 0.05 worked nicely. The new dispatch app logs distances in city blocks (4 blocks per km). It's the same deliveries and the same η. Switch units and replay the steps. Why does one setting settle and the other blow up?
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
        <Note>Both curves are the same deliveries priced the same way. Look at the width of the two bowls, not only the numbers.</Note>
      </>
    );
  } else if (screen === 1) {
    question = (
      <>
        Now you control η. Use km first. Find the η that lands on the best rate in exactly one step. Then raise η until the walker starts to zig-zag, and again until it explodes. How are those three values related? Then switch to blocks and find them again.
      </>
    );
    stage = (
      <>
        {curve}
        <Trace series={[{ vals: errs, color: THEME.highlight, label: "r_k − best r" }]} n={8} yr={[-2, 2]} zero label="distance from the best rate after each step" />
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
              ["(r₁ − r*) / (r₀ − r*)", fmt(ratio, 3), Math.abs(ratio) > 1 ? THEME.warn : THEME.text],
              ["(r₂ − r*) / (r₁ − r*)", pts.length > 2 ? fmt((pts[2] - rStar) / (pts[1] - rStar), 3) : "—"],
            ]}
          />
        </Panel>
        <Note>Each step multiplies the distance to the best rate by the same factor. When is that factor 0? When is it negative? When is it below −1?</Note>
      </>
    );
  } else if (screen === 2) {
    const etaOpt = 1 / a;
    const T = (r) => f(pts[0]) + slopeOf(f, pts[0]) * (r - pts[0]) + 0.5 * a * (r - pts[0]) * (r - pts[0]);
    question = (
      <>
        Name it. A parabola is fully described by its slope and its curvature E'' at your current point. Jump to its bottom and you get Newton's step. Your "one-step η" was 1/E''. Drag η and check that the three regimes sit at η_opt and 2·η_opt.
      </>
    );
    stage = (
      <>
        <Curve1D f={f} xr={XR} yr={[0, 9]} pts={vis} xl={"rate r (" + UNITS[unit].per + ")"} label="E(r) and its Taylor parabola at r₀ (dashed: exact here, E is a parabola)" curves={[{ fn: T, color: THEME.text, dash: "5 4", width: 1.3 }]} />
        <EtaBar eta={eta} etaOpt={etaOpt} max={3 * etaOpt} />
      </>
    );
    side = (
      <>
        <Panel>
          {unitPills}
          <Slider id="l1-eta3" label="learning rate η" value={eta} min={0.001} max={3 * etaOpt} stepSize={0.0005} onChange={setEta} display={fmt(eta, 4)} />
          <Eq size={13}>
            E(r) = E(r₀) + E'(r₀)(r − r₀) + ½E''(r − r₀)²
            <br />
            r_min = r₀ − E'(r₀) / E''
            <br />
            E'' = mean(distance²) = <b style={{ color: THEME.highlight }}>{fmt(a, 2)}</b>
            <br />
            η_opt = 1 / E'' = <b style={{ color: THEME.highlight }}>{fmt(etaOpt, 4)}</b>
            <br />
            r_k+1 − r* = (1 − η/η_opt)(r_k − r*) = {fmt(1 - eta / etaOpt, 3)} × (r_k − r*)
          </Eq>
          <Note>Measuring in blocks multiplies every distance by 4, so E'' grows by 16 and η_opt shrinks by 16. With η = 0.05 the km run sat in the creeping zone. The blocks run was far past 2·η_opt (slides 14-16).</Note>
        </Panel>
      </>
    );
  } else {
    question = (
      <>
        A partner city logs distances in miles and fits a robust loss, so big misses count less than squared. Nobody tells you E''. You can probe the slope at your current r and 0.1 further on. Starting from 1.00 €/mile, get within 0.01 of the best rate in at most 3 steps.
      </>
    );
    stage = (
      <>
        <Curve1D f={robustLoss} xr={[0, 4]} yr={[0, 8]} pts={path} xl="rate r (€/mile)" label="robust fee loss E(r), miles" vlines={solved ? [{ x: R_STAR_C, color: THEME.ok, label: "best r" }] : []} />
        <Trace series={[{ vals: path.map((r) => r - R_STAR_C), color: THEME.highlight, label: "r_k − best r" }]} n={Math.max(3, stepsC)} yr={[-1.8, 1.8]} zero label="distance from the best rate" />
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
            take one step: r ← r − η·E'(r)
          </Btn>
          <div style={{ marginTop: 8 }}>
            <Readout
              rows={[
                ["current r", fmt(rc, 4) + " €/mile"],
                ["slope E'(r)", fmt(slopeOf(robustLoss, rc), 4)],
              ]}
            />
            <Goal met={stepsC <= 3}>steps used: {stepsC} / 3</Goal>
            <Goal met={solved}>slope flat enough that r is within 0.01 of the best rate</Goal>
          </div>
          <Hint
            hints={[
              "The two probes tell you how fast the slope changes per €/mile. That rate of change is the curvature E''.",
              "Set η = 1/E'' with E'' ≈ (difference of the two slopes) / 0.1. The curve is not a parabola, so probe again after each step and update η.",
            ]}
          />
        </Panel>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="How big a step?" question={question} stage={stage} side={side} />;
}
