import { useState, useMemo, useCallback, useRef } from "react";
import { THEME, MONO } from "./common";

// Lesson 8 - Back-propagating through batch norm (Batch Normalization, slides 21-37).
// Aha: batch norm is one function of the whole mini-batch, so each order's gradient depends on all
// the others. Any part of the incoming gradient that would move every order the same way, or stretch
// the batch, is removed (mean and std are recomputed, so such a push changes nothing). The gradient
// reaching z always sums to 0, and you can push one order by pushing its batch-mates.

// @sim-start
const Z8 = [1.2, 2.5, 0.4, 3.1, 1.9, 2.2];
const BN8 = bnForward(Z8);
const LOCK8 = 3; // order #4 (0-based 3) in the challenge
function back8(g) {
  return bnBackward(g, BN8.u, BN8.sd);
}
const GOAL8 = 0.9;
// @sim-end

const NB = Z8.length;
const GMAX = 1.5;
const BAR_MID = 100;
const BAR_H = 60;
const DZ_MID = 272;
const COLW = (PX1 - PX0) / NB;

function sumOf(a) {
  return a.reduce((s, v) => s + v, 0);
}
// two rows of bars: upstream gradient ∂L/∂u (draggable) and the result ∂L/∂z
function GradStage({ g, dz, onSet, locked }) {
  const svgRef = useRef(null);
  const dragRef = useRef(-1);
  const Yg = (v) => BAR_MID - (clamp(v, -GMAX, GMAX) / GMAX) * BAR_H;
  const Yz = (v) => DZ_MID - (clamp(v, -GMAX, GMAX) / GMAX) * BAR_H;
  const setFrom = useCallback(
    (e, i) => {
      const p = svgPoint(svgRef.current, e);
      const v = clamp(((BAR_MID - p.y) / BAR_H) * GMAX, -1, 1);
      onSet(i, Math.round(v * 100) / 100);
    },
    [onSet]
  );
  const down = useCallback(
    (e) => {
      const p = svgPoint(svgRef.current, e);
      if (p.y > BAR_MID + BAR_H + 10) return;
      const i = Math.floor((p.x - PX0) / COLW);
      if (i < 0 || i >= NB || i === locked) return;
      dragRef.current = i;
      svgRef.current.setPointerCapture(e.pointerId);
      setFrom(e, i);
    },
    [setFrom, locked]
  );
  const move = useCallback(
    (e) => {
      if (dragRef.current >= 0) setFrom(e, dragRef.current);
    },
    [setFrom]
  );
  const up = useCallback(() => {
    dragRef.current = -1;
  }, []);
  return (
    <svg
      ref={svgRef}
      viewBox="0 0 620 396"
      style={{ width: "100%", display: "block", touchAction: "none", cursor: "ns-resize" }}
      role="img"
      aria-label="Drag the upstream gradient bars; the gradient reaching z is drawn below"
      onPointerDown={down}
      onPointerMove={move}
      onPointerUp={up}
      onPointerCancel={up}
    >
      <rect x={PX0} y={BAR_MID - BAR_H - 10} width={PX1 - PX0} height={2 * BAR_H + 20} fill={THEME.bg} stroke={THEME.grid} />
      <rect x={PX0} y={DZ_MID - BAR_H - 10} width={PX1 - PX0} height={2 * BAR_H + 20} fill={THEME.bg} stroke={THEME.grid} />
      <line x1={PX0} x2={PX1} y1={BAR_MID} y2={BAR_MID} stroke={THEME.grid} />
      <line x1={PX0} x2={PX1} y1={DZ_MID} y2={DZ_MID} stroke={THEME.grid} />
      <text x={PX0} y={BAR_MID - BAR_H - 16} fontFamily={MONO} fontSize={10.5} fill={THEME.accent}>
        push on u (= −∂L/∂u) the loss asks for (drag) · up = raise that ETA
      </text>
      <text x={PX0} y={DZ_MID - BAR_H - 16} fontFamily={MONO} fontSize={10.5} fill={THEME.highlight}>
        push that actually reaches each z (= −∂L/∂z)
      </text>
      {g.map((v, i) => {
        const cx = PX0 + COLW * (i + 0.5);
        const w = COLW * 0.42;
        const lockedHere = i === locked;
        return (
          <g key={i}>
            <rect x={cx - w / 2} y={Math.min(Yg(v), BAR_MID)} width={w} height={Math.max(1, Math.abs(Yg(v) - BAR_MID))} fill={rgba(THEME.accent, 0.55)} stroke={THEME.accent} strokeDasharray={lockedHere ? "3 2" : undefined} />
            <rect x={cx - w / 2} y={Math.min(Yz(dz[i]), DZ_MID)} width={w} height={Math.max(1, Math.abs(Yz(dz[i]) - DZ_MID))} fill={rgba(THEME.highlight, 0.55)} stroke={THEME.highlight} />
            <text x={cx} y={BAR_MID + BAR_H + 24} textAnchor="middle" fontFamily={MONO} fontSize={10} fill={THEME.text}>
              {lockedHere ? "locked 0" : fmt(v, 2)}
            </text>
            <text x={cx} y={DZ_MID + BAR_H + 24} textAnchor="middle" fontFamily={MONO} fontSize={10} fill={THEME.text}>
              {fmt(dz[i], 2)}
            </text>
            <text x={cx} y={BAR_MID - BAR_H + 6} textAnchor="middle" fontFamily={MONO} fontSize={10.5} fill={THEME.text}>
              #{i + 1}
            </text>
            <text x={cx} y={384} textAnchor="middle" fontFamily={MONO} fontSize={9.5} fill={THEME.muted}>
              z={fmt(Z8[i], 1)} u={sgn(BN8.u[i], 1)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

const PRESETS = {
  equal: [1, 1, 1, 1, 1, 1],
  stretch: BN8.u.map((u) => Math.round(clamp(u * 0.6, -1, 1) * 100) / 100),
  one: [0, 0, 1, 0, 0, 0],
};

export default function BackpropLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [g, setG] = useState(PRESETS.equal);
  const [gC, setGC] = useState([0, 0, 0, 0, 0, 0]);
  const dz = useMemo(() => back8(g), [g]);
  const dzC = useMemo(() => back8(gC), [gC]);
  const setOne = useCallback((i, v) => setG((a) => a.map((x, k) => (k === i ? v : x))), []);
  const setOneC = useCallback((i, v) => setGC((a) => a.map((x, k) => (k === i ? v : x))), []);
  const pEqual = useCallback(() => setG(PRESETS.equal), []);
  const pStretch = useCallback(() => setG(PRESETS.stretch), []);
  const pOne = useCallback(() => setG(PRESETS.one), []);
  const okC = dzC[LOCK8] >= GOAL8;
  const rows = (gg, d) => [
    ["Σ ∂L/∂z over the batch", fmt(sumOf(d), 3)],
    ["Σ u · ∂L/∂z", fmt(sumOf(d.map((v, i) => v * BN8.u[i])), 3)],
    ["Σ ∂L/∂u (what was asked)", fmt(sumOf(gg), 2)],
  ];
  const presets = (
    <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
      <Btn onClick={pEqual}>push all up</Btn>
      <Btn onClick={pStretch}>stretch the batch</Btn>
      <Btn onClick={pOne}>push #3 only</Btn>
    </div>
  );

  let question, stage, side;
  if (screen === 0) {
    question = (
      <>
        A batch of six orders passes through batch norm. The loss says every order's predicted ETA is too low: it pushes all six normalized values up by the same amount (top bars). What reaches the six z values (bottom bars)? Why?
      </>
    );
    stage = <GradStage g={g} dz={dz} onSet={setOne} locked={-1} />;
    side = (
      <>
        <Panel>
          {presets}
          <div style={{ height: 8 }} />
          <Readout rows={rows(g, dz)} />
        </Panel>
        <Note>Drag any top bar. Think about what happens to u if you add the same amount to every z and then subtract the batch mean again.</Note>
      </>
    );
  } else if (screen === 1) {
    question = (
      <>
        Try the other presets, then drag freely. Push only order #3 up: what happens to the other five? Does any pattern of top bars ever give bottom bars that do not sum to zero?
      </>
    );
    stage = <GradStage g={g} dz={dz} onSet={setOne} locked={-1} />;
    side = (
      <>
        <Panel>
          {presets}
          <div style={{ height: 8 }} />
          <Readout rows={rows(g, dz)} />
        </Panel>
        <Note>"Stretch the batch" pushes orders above the mean up and orders below it down, in proportion to u.</Note>
      </>
    );
  } else if (screen === 2) {
    question = (
      <>
        Name it. Batch norm is a vector function of the whole mini-batch: μ_B and σ_B² depend on every z, so ∂L/∂z<sup>(n)</sup> collects three paths, through u<sup>(n)</sup>, through σ_B² and through μ_B. The last two remove the part of the push that is the same for everyone and the part that only stretches the batch.
      </>
    );
    stage = <GradStage g={g} dz={dz} onSet={setOne} locked={-1} />;
    side = (
      <>
        <Panel>
          <Eq size={12.5}>
            ∂L/∂σ² = −½ (σ² + ε)^(−3/2) Σ<sub>n</sub> ∂L/∂u<sup>(n)</sup> (z<sup>(n)</sup> − μ)
            <br />
            ∂L/∂μ = −(σ² + ε)^(−1/2) Σ<sub>n</sub> ∂L/∂u<sup>(n)</sup>
            <br />
            ∂L/∂z<sup>(n)</sup> = ∂L/∂u<sup>(n)</sup> / √(σ² + ε) + ∂L/∂σ² · 2(z<sup>(n)</sup> − μ)/B + ∂L/∂μ / B
            <br />
            = (g<sub>n</sub> − mean(g) − u<sub>n</sub> · mean(g·u)) / √(σ² + ε)
          </Eq>
          {presets}
          <div style={{ height: 8 }} />
          <Readout rows={rows(g, dz)} />
        </Panel>
        <Note>Here g = ∂L/∂u (it already includes γ). The bars show pushes, minus the gradients; the map is linear, so the same formula holds. In ∂L/∂μ a second term has a factor Σ(z − μ) = 0, so it drops out, as on slide 33. The compact last line is the same formula: subtract the batch average of g and the part of g that lines up with u.</Note>
      </>
    );
  } else {
    question = (
      <>
        Order #4 is predicted too early and should go up, but its own push is locked at 0 (say its label is missing from this batch). Using only the other five top bars, each between −1 and +1, make the push reaching z₄ at least {GOAL8}.
      </>
    );
    stage = <GradStage g={gC} dz={dzC} onSet={setOneC} locked={LOCK8} />;
    side = (
      <>
        <Panel edge={okC ? THEME.ok : THEME.grid}>
          <Readout
            rows={[
              ["push reaching z₄", fmt(dzC[LOCK8], 3), okC ? THEME.ok : undefined],
              ["sum over the batch", fmt(sumOf(dzC), 3)],
            ]}
          />
          <Goal met={okC}>
            push on z₄ ≥ {GOAL8} without touching #4
          </Goal>
          <Hint
            hints={[
              "Mean subtraction gives every order minus the batch average of the top bars. What does pulling the others down do to z₄?",
              "There is a second effect, the batch's spread. #4 is the highest order (u₄ = " + sgn(BN8.u[LOCK8], 2) + ") and #3 sits far below everyone. Moving #3 down widens the batch and shrinks u₄; moving #3 up narrows it and stretches u₄ upward. Try each of the five bars one at a time and keep the direction that raises z₄.",
            ]}
          />
        </Panel>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="One order's gradient feels the batch" question={question} stage={stage} side={side} />;
}
