import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import { THEME, MONO } from "./common";

// Lesson 3 - MLPs as universal approximators (slides 85-90).
// Aha: two threshold neurons with weights +h and -h make a pulse. A summing output unit stacks
// pulses, and narrower pulses trace any curve as closely as you like, so one hidden layer is enough.

// delivery fee (EUR) along the coast road, by distance (km)
function coastFee(d) {
  return 2 + 0.3 * d + 1.3 * Math.exp(-(((d - 6.5) / 0.8) ** 2)) - 0.9 * Math.exp(-(((d - 2.5) / 0.7) ** 2));
}
// challenge: the late-night tariff on the ring road
function nightFee(d) {
  return 2.2 + 0.2 * d + 1.2 * Math.exp(-(((d - 7.2) / 1.1) ** 2)) - 1.1 * Math.exp(-(((d - 2.2) / 0.9) ** 2)) + 0.5 * (d > 4.5 ? 1 : 0);
}

const PX0 = 46;
const PX1 = 606;
const PY0 = 228;
const PY1 = 18;
const YMAX = 6;
function sx(d) {
  return PX0 + (d / 10) * (PX1 - PX0);
}
function sy(v) {
  return PY0 - (v / YMAX) * (PY0 - PY1);
}
function inv(p) {
  return [clamp(((p.x - PX0) / (PX1 - PX0)) * 10, 0, 10), clamp(((PY0 - p.y) / (PY0 - PY1)) * YMAX, -1, YMAX)];
}
function poly(fn, n) {
  const pts = [];
  for (let i = 0; i <= (n || 300); i++) {
    const d = (10 * i) / (n || 300);
    pts.push(sx(d) + "," + sy(fn(d)));
  }
  return pts.join(" ");
}
function meanAbs(fa, fb) {
  let s = 0;
  for (let i = 0; i <= 400; i++) {
    const d = (10 * i) / 400;
    s += Math.abs(fa(d) - fb(d));
  }
  return s / 401;
}
function act(kind, z, sharp) {
  return kind === "step" ? (z >= 0 ? 1 : 0) : sig(z * sharp);
}

function Axes({ ylabel }) {
  const ticks = [];
  for (let d = 0; d <= 10; d += 2)
    ticks.push(
      <g key={"x" + d}>
        <line x1={sx(d)} x2={sx(d)} y1={PY0} y2={PY1} stroke={THEME.grid} strokeWidth={0.6} />
        <text x={sx(d)} y={PY0 + 15} textAnchor="middle" fontFamily={MONO} fontSize={10.5} fill={THEME.muted}>
          {d} km
        </text>
      </g>
    );
  for (let v = 0; v <= YMAX; v += 2)
    ticks.push(
      <g key={"y" + v}>
        <line x1={PX0} x2={PX1} y1={sy(v)} y2={sy(v)} stroke={THEME.grid} strokeWidth={0.6} />
        <text x={PX0 - 6} y={sy(v) + 4} textAnchor="end" fontFamily={MONO} fontSize={10.5} fill={THEME.muted}>
          €{v}
        </text>
      </g>
    );
  return (
    <g>
      <rect x={PX0} y={PY1} width={PX1 - PX0} height={PY0 - PY1} fill={THEME.bg} stroke={THEME.grid} />
      {ticks}
      {ylabel && (
        <text x={PX0 + 6} y={PY1 + 14} fontFamily={MONO} fontSize={11} fill={THEME.muted}>
          {ylabel}
        </text>
      )}
    </g>
  );
}
function ErrorBand({ fa, fb }) {
  // filled area between target and approximation, drawn as one path
  const top = [];
  const bot = [];
  for (let i = 0; i <= 300; i++) {
    const d = (10 * i) / 300;
    top.push(sx(d) + "," + sy(fa(d)));
    bot.push(sx(d) + "," + sy(fb(d)));
  }
  return <polygon points={top.concat(bot.reverse()).join(" ")} fill={rgba(THEME.warn, 0.28)} stroke="none" />;
}

// drag-aware plot: children get drawn, handles get dragged
function DragPlot({ children, handles, onMove, height, ariaLabel }) {
  const ref = useRef(null);
  const drag = useRef(null);
  const down = useCallback((e) => {
    const id = e.target.getAttribute && e.target.getAttribute("data-hid");
    if (!id) return;
    drag.current = id;
    ref.current.setPointerCapture(e.pointerId);
    e.preventDefault();
  }, []);
  const move = useCallback(
    (e) => {
      if (!drag.current) return;
      const [d, v] = inv(svgPoint(ref.current, e));
      onMove(drag.current, d, v);
    },
    [onMove]
  );
  const up = useCallback(() => {
    drag.current = null;
  }, []);
  return (
    <svg ref={ref} viewBox={"0 0 620 " + (height || 250)} style={{ width: "100%", display: "block", touchAction: "none" }} role="img" aria-label={ariaLabel} onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={up}>
      {children}
      {(handles || []).map((h) => (
        <g key={h.id} style={{ cursor: h.kind === "x" ? "ew-resize" : "ns-resize" }}>
          <circle data-hid={h.id} cx={h.x} cy={h.y} r={14} fill="transparent" />
          {h.kind === "x" ? (
            <polygon data-hid={h.id} points={h.x + "," + (h.y - 8) + " " + (h.x + 7) + "," + (h.y + 5) + " " + (h.x - 7) + "," + (h.y + 5)} fill={THEME.bg} stroke={h.color || THEME.accent} strokeWidth={2} />
          ) : (
            <rect data-hid={h.id} x={h.x - 6} y={h.y - 6} width={12} height={12} fill={THEME.bg} stroke={h.color || THEME.highlight} strokeWidth={2} />
          )}
        </g>
      ))}
    </svg>
  );
}
function MiniStep({ T, w, label }) {
  // one hidden neuron's contribution: w * step(d - T)
  const W = 190;
  const H = 70;
  const X = (d) => 8 + (d / 10) * (W - 16);
  const Y = (v) => H / 2 - v * 12;
  return (
    <div style={{ flex: "1 1 150px", minWidth: 0 }}>
      <div style={{ fontFamily: MONO, fontSize: 11.5, color: THEME.muted }}>{label}</div>
      <svg viewBox={"0 0 " + W + " " + H} style={{ width: "100%", display: "block" }} aria-hidden="true">
        <rect x="0" y="0" width={W} height={H} fill={THEME.bg} stroke={THEME.grid} />
        <line x1="8" x2={W - 8} y1={Y(0)} y2={Y(0)} stroke={THEME.grid} />
        <polyline points={X(0) + "," + Y(0) + " " + X(T) + "," + Y(0) + " " + X(T) + "," + Y(w) + " " + X(10) + "," + Y(w)} fill="none" stroke={w >= 0 ? THEME.accent : THEME.warn} strokeWidth={2} strokeDasharray={w >= 0 ? undefined : "5 3"} />
      </svg>
    </div>
  );
}

const STAIR_START = (function () {
  const T = [];
  const Lv = [2.4];
  for (let i = 1; i <= 10; i++) {
    T.push(i * 0.9);
    Lv.push(2.4);
  }
  return { T, L: Lv };
})();

export default function CurvesLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [t1, setT1] = useState(2.2);
  const [t2, setT2] = useState(7.4);
  const [a, setA] = useState(0.8);
  const [neg, setNeg] = useState(false);
  const [K, setK] = useState(4);
  const [kind, setKind] = useState("step");
  const [sharp, setSharp] = useState(6);
  const [inspect, setInspect] = useState(1);
  const [stair, setStair] = useState(STAIR_START);

  // screen 1: two neurons
  const lo = Math.min(t1, t2);
  const hi = Math.max(t1, t2);
  const w2 = neg ? -a : a;
  const twoOut = useCallback((d) => a * (d >= t1 ? 1 : 0) + w2 * (d >= t2 ? 1 : 0), [a, w2, t1, t2]);
  const pulseErr = meanAbs(twoOut, (d) => (d >= 4 && d < 6 ? 1.5 : 0));
  const movePulse = useCallback((id, d, v) => {
    if (id === "t1") setT1(Math.round(d * 20) / 20);
    else if (id === "t2") setT2(Math.round(d * 20) / 20);
    else if (id === "a") setA(clamp(Math.round(v * 20) / 20, 0, 3));
  }, []);

  // screens 2 and 3: K pulses, heights read off the curve at each pulse centre
  const pulses = useMemo(() => {
    const out = [];
    for (let k = 0; k < K; k++) {
      const T0 = (10 * k) / K;
      const T1 = (10 * (k + 1)) / K;
      out.push({ T0, T1, h: coastFee((T0 + T1) / 2) });
    }
    return out;
  }, [K]);
  const approx = useCallback(
    (d) => {
      let s = 0;
      for (const p of pulses) s += p.h * (act(kind, d - p.T0 + 1e-9, sharp) - act(kind, d - p.T1 + 1e-9, sharp));
      // the last pulse is closed at 10 km
      return s + (kind === "step" && d >= 10 - 1e-9 ? pulses[pulses.length - 1].h : 0);
    },
    [pulses, kind, sharp]
  );
  const err = meanAbs(approx, coastFee);
  const insp = pulses[Math.min(inspect, K) - 1];

  // screen 4: free staircase with ten neurons
  const stairOut = useCallback(
    (d) => {
      let v = stair.L[0];
      for (let i = 0; i < stair.T.length; i++) if (d >= stair.T[i]) v += stair.L[i + 1] - stair.L[i];
      return v;
    },
    [stair]
  );
  const stairErr = meanAbs(stairOut, nightFee);
  const moveStair = useCallback((id, d, v) => {
    setStair((prev) => {
      const T = prev.T.slice();
      const Lv = prev.L.slice();
      if (id[0] === "T") {
        const i = parseInt(id.slice(1), 10);
        const lo2 = i > 0 ? T[i - 1] + 0.05 : 0.05;
        const hi2 = i < T.length - 1 ? T[i + 1] - 0.05 : 9.95;
        T[i] = clamp(Math.round(d * 20) / 20, lo2, hi2);
      } else {
        const i = parseInt(id.slice(1), 10);
        Lv[i] = clamp(Math.round(v * 20) / 20, 0, YMAX);
      }
      return { T, L: Lv };
    });
  }, []);

  let question, stage, side;
  if (screen === 0) {
    question = (
      <>
        Fees along the coast road depend on distance. Before fitting a real fee curve, try two threshold neurons. Each one jumps from 0 to 1 at its own distance T. The output adds them with weights. Can you make a €1.50 surcharge that applies only between 4 and 6 km (the dashed box)?
      </>
    );
    stage = (
      <>
        <DragPlot
          ariaLabel="Output of two threshold neurons"
          onMove={movePulse}
          handles={[
            { id: "t1", kind: "x", x: sx(t1), y: PY0 + 2, color: THEME.accent },
            { id: "t2", kind: "x", x: sx(t2), y: PY0 + 2, color: THEME.text },
            { id: "a", kind: "y", x: sx(Math.min(9.6, lo + 0.25)), y: sy(a), color: THEME.highlight },
          ]}
          height={250}
        >
          <Axes ylabel="output = w₁·step(d − T₁) + w₂·step(d − T₂)" />
          <rect x={sx(4)} y={sy(1.5)} width={sx(6) - sx(4)} height={sy(0) - sy(1.5)} fill="none" stroke={THEME.text} strokeDasharray="5 4" />
          <polyline points={poly(twoOut, 600)} fill="none" stroke={THEME.highlight} strokeWidth={2.6} />
        </DragPlot>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginTop: 6 }}>
          <MiniStep T={t1} w={a} label={"neuron 1: T₁ = " + fmt(t1, 2) + " km, w₁ = " + sgn(a)} />
          <MiniStep T={t2} w={w2} label={"neuron 2: T₂ = " + fmt(t2, 2) + " km, w₂ = " + sgn(w2)} />
        </div>
      </>
    );
    side = (
      <>
        <Panel>
          <Note>Drag the triangles on the distance axis to move T₁ and T₂. Drag the yellow square to change the weight size.</Note>
          <Label>sign of the second weight</Label>
          <div style={{ display: "flex", gap: 6 }}>
            <Pill value={false} active={!neg} onPick={setNeg}>
              w₂ = +w₁
            </Pill>
            <Pill value={true} active={neg} onPick={setNeg}>
              w₂ = −w₁
            </Pill>
          </div>
        </Panel>
        <Panel edge={pulseErr < 0.02 ? THEME.accent : THEME.grid}>
          <Meter label="distance from the dashed box (mean |gap|)" value={pulseErr} max={1} good={0.02} unit=" €" />
        </Panel>
        <Note>With equal signs you get a staircase. What changes when the second neuron subtracts what the first one added?</Note>
      </>
    );
  } else if (screen === 1 || screen === 2) {
    question =
      screen === 1 ? (
        <>
          Now the real coast-road fee (white curve): a promo dip at the mall near 2.5 km, a bridge toll near 6.5 km. Each pulse is one pair of neurons, and its height is read off the curve. How many neurons until the average error drops below 5 cents? Does the kind of neuron matter?
        </>
      ) : (
        <>
          Name it. Pick a pulse to inspect and find its two neurons. Every bump in the output is one +h and −h pair, and the output unit only adds. Which part of the formula below is the pulse you are inspecting?
        </>
      );
    stage = (
      <>
        <DragPlot ariaLabel="Fee curve and its pulse approximation" onMove={movePulse} height={250}>
          <Axes ylabel="delivery fee on the coast road" />
          <ErrorBand fa={coastFee} fb={approx} />
          {screen === 2 && insp && <rect x={sx(insp.T0)} y={sy(insp.h)} width={sx(insp.T1) - sx(insp.T0)} height={sy(0) - sy(insp.h)} fill={rgba(THEME.accent, 0.18)} stroke={THEME.accent} strokeWidth={1.4} />}
          <polyline points={poly(coastFee)} fill="none" stroke={THEME.text} strokeWidth={1.6} strokeDasharray="6 4" />
          <polyline points={poly(approx, 600)} fill="none" stroke={THEME.highlight} strokeWidth={2.4} />
          <text x={sx(2.5)} y={sy(coastFee(2.5)) + 22} textAnchor="middle" fontFamily={MONO} fontSize={10.5} fill={THEME.muted}>
            mall promo
          </text>
          <text x={sx(6.5)} y={sy(coastFee(6.5)) - 10} textAnchor="middle" fontFamily={MONO} fontSize={10.5} fill={THEME.muted}>
            bridge toll
          </text>
        </DragPlot>
        <div style={{ display: "flex", gap: 14, flexWrap: "wrap", fontSize: 12.5, color: THEME.muted, marginTop: 4 }}>
          <span>
            <svg width="22" height="8" aria-hidden="true">
              <line x1="0" x2="22" y1="4" y2="4" stroke={THEME.text} strokeDasharray="5 3" strokeWidth="1.6" />
            </svg>{" "}
            real fee
          </span>
          <span>
            <svg width="22" height="8" aria-hidden="true">
              <line x1="0" x2="22" y1="4" y2="4" stroke={THEME.highlight} strokeWidth="2.4" />
            </svg>{" "}
            network output
          </span>
          <span>
            <svg width="14" height="10" aria-hidden="true">
              <rect width="14" height="10" fill={rgba(THEME.warn, 0.4)} />
            </svg>{" "}
            error
          </span>
        </div>
      </>
    );
    side =
      screen === 1 ? (
        <>
          <Panel>
            <Slider id="l3-k" label="pulses" value={K} min={1} max={60} stepSize={1} onChange={setK} display={K + " pulses = " + 2 * K + " hidden neurons"} />
            <Label>hidden neuron type</Label>
            <div style={{ display: "flex", gap: 6, marginBottom: 6 }}>
              <Pill value="step" active={kind === "step"} onPick={setKind}>
                threshold
              </Pill>
              <Pill value="sigmoid" active={kind === "sigmoid"} onPick={setKind}>
                sigmoid
              </Pill>
            </div>
            {kind === "sigmoid" && <Slider id="l3-s" label="sigmoid steepness" value={sharp} min={0.5} max={40} stepSize={0.1} onChange={setSharp} display={fmt(sharp, 1)} />}
          </Panel>
          <Panel edge={err < 0.05 ? THEME.accent : THEME.grid}>
            <Meter label="average error" value={err} max={0.6} good={0.05} unit=" €" />
          </Panel>
          <Note>With sigmoid neurons, try a narrow pulse and a gentle steepness. Neighbouring pulses blur into each other. That is fine, the sum still follows the curve.</Note>
        </>
      ) : (
        <>
          <Panel>
            <Slider id="l3-i" label="inspect pulse" value={Math.min(inspect, K)} min={1} max={K} stepSize={1} onChange={setInspect} display={"#" + Math.min(inspect, K) + " of " + K} />
            {insp && (
              <div style={{ fontFamily: MONO, fontSize: 13, lineHeight: 1.7 }}>
                <div style={{ color: THEME.accent }}>
                  neuron a: step(d − {fmt(insp.T0, 2)}), weight {sgn(insp.h)}
                </div>
                <div style={{ color: THEME.warn }}>
                  neuron b: step(d − {fmt(insp.T1, 2)}), weight −{fmt(insp.h)}
                </div>
              </div>
            )}
          </Panel>
          <Panel>
            <Label>The formula behind the picture</Label>
            <Eq size={13.5}>
              f(d) ≈ Σₖ hₖ · [ step(d − Tₖ) − step(d − Tₖ₊₁) ]
            </Eq>
            <Note>
              One hidden layer of 2K threshold neurons and one summing output. Narrower pulses (bigger K) push the error toward zero for any continuous fee curve. That is the universal approximation result on slides 85-88. In two dimensions the pulse becomes the cylinder from lesson 2.
            </Note>
            <Note>
              Caveat (slide 89): this argument needs a summing output, with no activation on the output unit. With an activated output (sigmoid, tanh) the net is still universal, but only for targets inside the activation's range.
            </Note>
          </Panel>
        </>
      );
  } else {
    const handles = [];
    stair.T.forEach((T, i) => handles.push({ id: "T" + i, kind: "x", x: sx(T), y: PY0 + 2, color: THEME.accent }));
    stair.L.forEach((Lv, i) => {
      const from = i === 0 ? 0 : stair.T[i - 1];
      const to = i < stair.T.length ? stair.T[i] : 10;
      handles.push({ id: "L" + i, kind: "y", x: sx((from + to) / 2), y: sy(Lv), color: THEME.highlight });
    });
    question = (
      <>
        Late-night tariff on the ring road (white curve). The budget is ten hidden threshold neurons and one bias, so no more pairs. Each neuron is one step in a staircase. Get the average error under 10 cents.
      </>
    );
    stage = (
      <>
        <DragPlot ariaLabel="Ten-step staircase fit" onMove={moveStair} handles={handles} height={250}>
          <Axes ylabel="late-night fee, ring road" />
          <ErrorBand fa={nightFee} fb={stairOut} />
          <polyline points={poly(nightFee)} fill="none" stroke={THEME.text} strokeWidth={1.6} strokeDasharray="6 4" />
          <polyline points={poly(stairOut, 800)} fill="none" stroke={THEME.highlight} strokeWidth={2.4} />
        </DragPlot>
        <Note>Triangles move where each neuron switches on. Squares set the level of each stretch of road.</Note>
      </>
    );
    side = (
      <>
        <Panel edge={stairErr < 0.1 ? THEME.accent : THEME.grid}>
          <Meter label="average error" value={stairErr} max={1.2} good={0.1} unit=" €" />
          <div style={{ fontFamily: MONO, fontSize: 12.5, color: THEME.muted, lineHeight: 1.6, marginTop: 6 }}>
            output weights (jump at each step):
            <br />
            {stair.L.slice(1).map((Lv, i) => sgn(Lv - stair.L[i]) + " ")}
          </div>
          <Hint
            hints={[
              "A pair per pulse would only buy five pulses. Here neighbouring stretches share a neuron, since each step only has to add the change in level.",
              "Put the steps close together where the curve is steep (around the dip, the jump at 4.5 km and the toll bump) and far apart where it is flat.",
            ]}
          />
        </Panel>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="Draw any fee curve" question={question} stage={stage} side={side} />;
}
