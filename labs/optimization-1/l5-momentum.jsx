import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import { THEME, MONO } from "./common";

// Lesson 5 - Momentum and Nesterov (Optimization I, slides 55-75).
// Aha: across a narrow valley the steps flip sign, along it they keep the same sign. A running average of
// past steps cancels the flips and adds up the steady part, so the walker speeds up along the valley floor.

const W_STAR = [1.5, 3.0];
const A = rotA(8, 0.25, 20);
const EIG = eig2(A);
const VALLEY = { f: quad(A, W_STAR), xr: [-1.5, 4.5], yr: [0, 6], xl: "w₁ €/km", yl: "w₂ €/kg" };
const START = [-1.2, 4.6];
const N = 60;
const TOL = 0.1;

function along(d, v) {
  return d[0] * v[0] + d[1] * v[1];
}
function StepTable({ steps, m }) {
  const rows = [];
  for (let k = 0; k < Math.min(8, steps.length); k++) {
    const lo = Math.max(0, k - m + 1);
    let a = 0;
    let b = 0;
    for (let j = lo; j <= k; j++) {
      a += along(steps[j], EIG.v1);
      b += along(steps[j], EIG.v2);
    }
    rows.push([k + 1, along(steps[k], EIG.v1), along(steps[k], EIG.v2), a / (k - lo + 1), b / (k - lo + 1)]);
  }
  const cell = { padding: "2px 8px", textAlign: "right", fontVariantNumeric: "tabular-nums" };
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ borderCollapse: "collapse", fontFamily: MONO, fontSize: 12.5, color: THEME.text, width: "100%" }}>
        <thead>
          <tr style={{ color: THEME.muted }}>
            <th style={cell}>step</th>
            <th style={cell}>across</th>
            <th style={cell}>along</th>
            <th style={cell}>avg across</th>
            <th style={cell}>avg along</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r[0]} style={{ borderTop: "1px solid " + THEME.grid }}>
              <td style={cell}>{r[0]}</td>
              <td style={cell}>{sgn(r[1], 2)}</td>
              <td style={cell}>{sgn(r[2], 2)}</td>
              <td style={Object.assign({}, cell, { color: THEME.highlight })}>{sgn(r[3], 2)}</td>
              <td style={Object.assign({}, cell, { color: THEME.highlight })}>{sgn(r[4], 2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
function WeightBars({ beta }) {
  // weight of the gradient from i steps ago in the running sum: beta^i
  const n = 25;
  const W = 600;
  const bw = (W - 60) / n;
  return (
    <svg viewBox={"0 0 " + W + " 110"} style={{ width: "100%", display: "block" }} role="img" aria-label="Weight of past gradients">
      <rect x={40} y={10} width={W - 50} height={76} fill={THEME.bg} stroke={THEME.grid} />
      {Array.from({ length: n }, (_, i) => {
        const h = Math.pow(beta, i) * 70;
        return <rect key={i} x={42 + i * bw} y={84 - h} width={bw - 3} height={h} fill={THEME.highlight} opacity={0.85} />;
      })}
      <text x={44} y={102} fontFamily={MONO} fontSize={10.5} fill={THEME.muted}>
        now
      </text>
      <text x={W - 12} y={102} textAnchor="end" fontFamily={MONO} fontSize={10.5} fill={THEME.muted}>
        {n - 1} steps ago
      </text>
      <text x={36} y={18} textAnchor="end" fontFamily={MONO} fontSize={10} fill={THEME.muted}>
        1
      </text>
    </svg>
  );
}

export default function MomentumLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [eta, setEta] = useState(0.2);
  const [m, setM] = useState(1);
  const [etaM, setEtaM] = useState(0.05);
  const [beta, setBeta] = useState(0.5);
  const [start, setStart] = useState(START);
  const [kind, setKind] = useState("gd");
  const [etaC, setEtaC] = useState(0.2);
  const [betaC, setBetaC] = useState(0.5);

  const onStart = useCallback((x, y) => setStart([x, y]), []);
  const pick = useCallback((k) => setKind(k), []);
  const gd = useMemo(() => runOpt(VALLEY.f, start, { kind: "gd", eta }, N, 60), [start, eta]);
  const mom = useMemo(() => runOpt(VALLEY.f, start, { kind: "momentum", eta: etaM, beta }, N, 60), [start, etaM, beta]);
  const nes = useMemo(() => runOpt(VALLEY.f, start, { kind: "nesterov", eta: etaM, beta }, N, 60), [start, etaM, beta]);
  const gdRef = useMemo(() => runOpt(VALLEY.f, start, { kind: "gd", eta: etaM }, N, 60), [start, etaM]);
  const cRun = useMemo(() => runOpt(VALLEY.f, START, { kind, eta: etaC, beta: betaC }, N, 60), [kind, etaC, betaC]);
  const sGd = settleStep(gd.pts, W_STAR, TOL);
  const sMom = settleStep(mom.pts, W_STAR, TOL);
  const sNes = settleStep(nes.pts, W_STAR, TOL);
  const sC = cRun.diverged ? -1 : settleStep(cRun.pts, W_STAR, TOL);
  const okC = sC >= 0 && sC <= 30;
  const shown = cRun.pts.slice(0, 31);

  const across = (r) => r.steps.map((d) => along(d, EIG.v1));
  const alongV = (r) => r.steps.map((d) => along(d, EIG.v2));
  const label = (s, r) => (r.diverged ? "diverged" : s >= 0 ? "settled at step " + s : "not settled in " + N);

  let question, stage, side;
  if (screen === 0) {
    question = (
      <>
        A narrow valley: per-km and per-kg rates fitted together, with the bowl tilted and stretched 32 to 1. Plain gradient descent zig-zags. The table splits each step into the part across the valley and the part along it. Which part keeps flipping sign? Now average the last few steps. What happens to each part?
      </>
    );
    stage = <LossMap surf={VALLEY} paths={[{ pts: gd.pts.slice(0, 21), color: THEME.highlight, label: "first 20 steps" }]} handle={start} onHandle={onStart} target={W_STAR} ariaLabel="Gradient descent zig-zagging down a narrow valley" />;
    side = (
      <>
        <Panel>
          <Slider id="l5-eta" label="learning rate η" value={eta} min={0.01} max={0.26} stepSize={0.001} onChange={setEta} display={fmt(eta, 3)} />
          <Slider id="l5-m" label="average over the last m steps" value={m} min={1} max={8} stepSize={1} onChange={setM} display={"m = " + m} />
          <StepTable steps={gd.steps} m={m} />
        </Panel>
        <Note>Stepping with that average instead of the raw step: would it zig-zag less? Would it move faster along the floor?</Note>
      </>
    );
  } else if (screen === 1) {
    question = (
      <>
        Momentum keeps a running sum of past steps and moves by that instead. Keep η small and raise β, the share of the old step kept each time. What happens across the valley and along it? Find a β that beats plain gradient descent (dashed) by a lot. What goes wrong when β is close to 1?
      </>
    );
    stage = (
      <>
        <LossMap
          surf={VALLEY}
          paths={[
            { pts: gdRef.pts, color: THEME.text, dash: "4 3", shape: "square", label: "GD, same η" },
            { pts: mom.pts, color: THEME.highlight, label: "momentum" },
          ]}
          handle={start}
          onHandle={onStart}
          target={W_STAR}
          ariaLabel="Momentum against gradient descent in the valley"
        />
        <Trace
          series={[
            { vals: across(mom), color: THEME.warn, label: "step across" },
            { vals: alongV(mom), color: THEME.highlight, dash: "5 3", label: "step along" },
          ]}
          n={40}
          yr={[-0.6, 0.6]}
          zero
          label="momentum's step, split across / along the valley"
          height={140}
        />
      </>
    );
    side = (
      <>
        <Panel>
          <Slider id="l5-etam" label="learning rate η" value={etaM} min={0.005} max={0.25} stepSize={0.001} onChange={setEtaM} display={fmt(etaM, 3)} />
          <Slider id="l5-beta" label="momentum β" value={beta} min={0} max={0.99} stepSize={0.005} onChange={setBeta} display={fmt(beta, 3)} />
          <Goal met={sMom >= 0 && (sGd < 0 || sMom < settleStep(gdRef.pts, W_STAR, TOL))}>momentum: {label(sMom, mom)}</Goal>
          <Goal met={settleStep(gdRef.pts, W_STAR, TOL) >= 0}>gradient descent: {label(settleStep(gdRef.pts, W_STAR, TOL), gdRef)}</Goal>
        </Panel>
        <Note>"Settled" means within 0.1 of the best pair and staying there.</Note>
      </>
    );
  } else if (screen === 2) {
    question = (
      <>
        Name it. The step ΔW is a running sum of gradients in which older ones fade by β each step. A gradient that keeps its sign adds up to 1/(1−β) times its size. One that flips sign cancels out. Nesterov changes the order: first coast with the old step, then take the gradient where you land. Compare the two.
      </>
    );
    stage = (
      <>
        <LossMap
          surf={VALLEY}
          paths={[
            { pts: mom.pts, color: THEME.highlight, label: "momentum" },
            { pts: nes.pts, color: THEME.text, dash: "5 3", shape: "triangle", label: "Nesterov" },
          ]}
          handle={start}
          onHandle={onStart}
          target={W_STAR}
          ariaLabel="Momentum against Nesterov"
        />
        <WeightBars beta={beta} />
      </>
    );
    side = (
      <>
        <Panel>
          <Slider id="l5-etan" label="learning rate η" value={etaM} min={0.005} max={0.25} stepSize={0.001} onChange={setEtaM} display={fmt(etaM, 3)} />
          <Slider id="l5-betan" label="momentum β" value={beta} min={0} max={0.99} stepSize={0.005} onChange={setBeta} display={fmt(beta, 3)} />
          <Eq size={12.5}>
            momentum: ΔW_k = βΔW_k−1 + η∇E(W_k−1)
            <br />
            Nesterov: ΔW_k = βΔW_k−1 + η∇E(W_k−1 − βΔW_k−1)
            <br />
            W_k = W_k−1 − ΔW_k
            <br />
            steady gradient g: ΔW → ηg/(1−β) = {fmt(1 / (1 - beta), 1)}·ηg
          </Eq>
          <Goal met={sMom >= 0}>momentum: {label(sMom, mom)}</Goal>
          <Goal met={sNes >= 0}>Nesterov: {label(sNes, nes)}</Goal>
          <Note>The bars show how much the gradient from i steps ago still counts: βⁱ. Nesterov's look-ahead feels the far wall before it gets there, so it overshoots less (slides 67-73).</Note>
        </Panel>
      </>
    );
  } else {
    question = (
      <>
        Same valley, fixed start. Get the rates to settle within 0.1 of the best pair in at most 30 steps. Plain gradient descent can't do it with any η: its best is about 53 steps. Pick a method and tune it.
      </>
    );
    stage = (
      <>
        <LossMap surf={VALLEY} paths={[{ pts: shown, color: THEME.highlight, label: cRun.diverged ? "diverged" : "after 30 steps" }]} target={W_STAR} ariaLabel="Challenge valley" />
        <Trace
          series={[
            { vals: cRun.pts.map((p) => Math.hypot(p[0] - W_STAR[0], p[1] - W_STAR[1])), color: THEME.highlight, label: "distance to best" },
            { vals: cRun.pts.map(() => TOL), color: THEME.text, dash: "3 3" },
          ]} n={30} yr={[-2, 1]} log label="distance to the best pair (log scale), dashed rule at 0.1" height={130} />
      </>
    );
    side = (
      <>
        <Panel edge={okC ? THEME.ok : THEME.grid}>
          <Pills
            options={[
              ["gd", "gradient descent"],
              ["momentum", "momentum"],
              ["nesterov", "Nesterov"],
            ]}
            value={kind}
            onPick={pick}
          />
          <Slider id="l5-etac" label="learning rate η" value={etaC} min={0.005} max={0.26} stepSize={0.001} onChange={setEtaC} display={fmt(etaC, 3)} />
          {kind !== "gd" && <Slider id="l5-betac" label="momentum β" value={betaC} min={0} max={0.99} stepSize={0.005} onChange={setBetaC} display={fmt(betaC, 3)} />}
          <Goal met={okC}>settled within 0.1 by step 30 {sC >= 0 ? "(step " + sC + ")" : ""}</Goal>
          <Hint
            hints={[
              "Watch the distance trace. A slow straight slide means too little momentum. A long ringing tail means too much.",
              "Keep η at about half the gradient-descent limit (0.08 to 0.12) and try β between 0.7 and 0.8. Nesterov gives a wider window.",
            ]}
          />
        </Panel>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="Let the zig-zags cancel" question={question} stage={stage} side={side} />;
}
