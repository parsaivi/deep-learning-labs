import { useState, useMemo } from "react";
import { THEME, MONO } from "./common";

// Lesson 7 - The learnable scale and shift gamma, beta (Batch Normalization, slides 16-20).
// Aha: normalizing forces every unit into tanh's straight middle, which takes away the network's
// ability to saturate or to sit off-centre. Two learnable numbers per unit, z = gamma * u + beta, give
// that choice back, and gamma = sigma_B, beta = mu_B undoes the normalization exactly.

// @sim-start
const B7 = 32;
const BATCH7 = (function () {
  const r = rng(71);
  const z = [];
  for (let i = 0; i < B7; i++) z.push(1.7 + 0.6 * gauss(r));
  return z;
})();
const BN7 = bnForward(BATCH7);
function apply7(gamma, beta) {
  const zh = BN7.u.map((u) => gamma * u + beta);
  const a = zh.map((v) => Math.tanh(v));
  const hard = a.filter((v) => Math.abs(v) > 0.9).length / a.length;
  const late = a.filter((v) => v > 0).length / a.length;
  let miss = 0;
  for (let i = 0; i < B7; i++) miss += Math.abs(zh[i] - BATCH7[i]) / B7;
  return { zh, a, hard, late, miss };
}
const GOAL7 = 0.03;
// @sim-end

const AX7 = [-4, 4];
const ROW_STYLE = {
  z: { label: "z  (raw)", shape: "square" },
  u: { label: "u  (normalized)", shape: "circle" },
  zh: { label: "γu + β", shape: "triangle" },
};

function GammaStage({ res, showRaw, ghost }) {
  const X = (v) => PX0 + ((clamp(v, AX7[0], AX7[1]) - AX7[0]) / (AX7[1] - AX7[0])) * (PX1 - PX0);
  const rows = [
    ["u", BN7.u, THEME.accent],
    ["zh", res.zh, THEME.highlight],
  ];
  if (showRaw) rows.unshift(["z", BATCH7, THEME.muted]);
  const top = 30;
  const rowH = 36;
  const curveTop = top + rows.length * rowH + 14;
  const curveBot = curveTop + 120;
  const YT = (v) => curveBot - ((v + 1.1) / 2.2) * (curveBot - curveTop);
  const tanhPts = [];
  for (let i = 0; i <= 160; i++) {
    const z = AX7[0] + ((AX7[1] - AX7[0]) * i) / 160;
    tanhPts.push(X(z).toFixed(1) + "," + YT(Math.tanh(z)).toFixed(1));
  }
  const H = curveBot + 30;
  return (
    <svg viewBox={"0 0 620 " + H} style={{ width: "100%", display: "block" }} role="img" aria-label="Batch values before and after normalization, scale and shift, and the tanh they feed">
      <rect x={PX0} y={top - 22} width={PX1 - PX0} height={rows.length * rowH + 6} fill={THEME.bg} stroke={THEME.grid} />
      <line x1={X(0)} x2={X(0)} y1={top - 12} y2={curveBot} stroke={THEME.grid} />
      {rows.map((row, k) => {
        const y = top + k * rowH + 4;
        const st = ROW_STYLE[row[0]];
        return (
          <g key={row[0]}>
            <text x={PX0 + 4} y={y - 11} fontFamily={MONO} fontSize={10} fill={row[2]}>
              {st.label}
            </text>
            {row[0] === "zh" && ghost &&
              BATCH7.map((v, i) => <rect key={"g" + i} x={X(v) - 2.5} y={y + 1} width={5} height={5} fill="none" stroke={THEME.muted} strokeWidth={1} />)}
            {row[1].map((v, i) => (
              <Marker key={i} x={X(v)} y={y + ((i * 5) % 9) - 4} shape={st.shape} color={row[2]} r={2.8} />
            ))}
          </g>
        );
      })}
      <rect x={PX0} y={curveTop - 6} width={PX1 - PX0} height={curveBot - curveTop + 12} fill={THEME.bg} stroke={THEME.grid} />
      <rect x={PX0} y={YT(1.1)} width={PX1 - PX0} height={YT(0.9) - YT(1.1)} fill={rgba(THEME.warn, 0.12)} />
      <rect x={PX0} y={YT(-0.9)} width={PX1 - PX0} height={YT(-1.1) - YT(-0.9)} fill={rgba(THEME.warn, 0.12)} />
      <polyline points={tanhPts.join(" ")} fill="none" stroke={THEME.accent} strokeWidth={2} />
      {res.zh.map((v, i) => (
        <Marker key={"t" + i} x={X(v)} y={YT(Math.tanh(v))} shape="triangle" color={THEME.highlight} r={3} />
      ))}
      <text x={PX0 + 4} y={curveTop + 8} fontFamily={MONO} fontSize={10} fill={THEME.muted}>
        tanh(γu + β) · shaded rows = |output| &gt; 0.9 (a hard yes / no)
      </text>
      {[-4, -2, 0, 2, 4].map((t) => (
        <text key={t} x={X(t)} y={H - 12} textAnchor="middle" fontFamily={MONO} fontSize={10} fill={THEME.muted}>
          {t}
        </text>
      ))}
      {ghost && (
        <text x={PX1} y={H - 1} textAnchor="end" fontFamily={MONO} fontSize={10} fill={THEME.muted}>
          □ = the raw z you are trying to rebuild
        </text>
      )}
    </svg>
  );
}

export default function GammaLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [gamma, setGamma] = useState(1);
  const [beta, setBeta] = useState(0);
  const [gC, setGC] = useState(1);
  const [bC, setBC] = useState(0);
  const g = screen === 0 ? 1 : gamma;
  const bb = screen === 0 ? 0 : beta;
  const res = useMemo(() => apply7(g, bb), [g, bb]);
  const resC = useMemo(() => apply7(gC, bC), [gC, bC]);
  const okC = resC.miss < GOAL7;
  const rows = (r) => [
    ["outputs that are a hard yes/no (|a| > 0.9)", fmt(r.hard * 100, 0) + "%"],
    ["orders the unit calls late (a > 0)", fmt(r.late * 100, 0) + "%"],
  ];

  let question, stage, side;
  if (screen === 0) {
    question = (
      <>
        This unit is meant to be a hard switch, "will this order be late?", feeding a tanh. Before batch norm, the raw z of its 32 orders (■) sat mostly to the right. After normalization (●), where do they land on the tanh? How many orders get a hard yes or no? Could the network ever learn a hard switch, or a unit that says "late" for most orders, if every batch is forced to mean 0 and std 1?
      </>
    );
    stage = <GammaStage res={res} showRaw />;
    side = (
      <>
        <Panel>
          <Readout rows={rows(res)} />
        </Panel>
        <Note>Here γ = 1 and β = 0, so the third row equals the normalized row.</Note>
      </>
    );
  } else if (screen === 1) {
    question = (
      <>
        Add two numbers per unit: a scale γ and a shift β, applied after normalizing. Can you make the unit a hard switch (most outputs beyond ±0.9)? Can you make it call 80% of orders late? Which number does which job?
      </>
    );
    stage = <GammaStage res={res} showRaw />;
    side = (
      <>
        <Panel>
          <Slider id="l7-g" label="scale γ" value={gamma} min={0} max={5} stepSize={0.01} onChange={setGamma} />
          <Slider id="l7-b" label="shift β" value={beta} min={-3} max={3} stepSize={0.01} onChange={setBeta} />
          <Readout rows={rows(res)} />
        </Panel>
        <Note>γ and β are ordinary weights. Training sets them by gradient descent, like any other weight.</Note>
      </>
    );
  } else if (screen === 2) {
    question = (
      <>
        Name it. Batch norm's full output is ẑ = γ·u + β, with γ and β learned for each unit. Normalizing gives every layer a standard starting place; γ and β let the network move away from it when that helps, including all the way back: γ = √(σ_B² + ε), β = μ_B gives ẑ = z.
      </>
    );
    stage = <GammaStage res={res} showRaw />;
    side = (
      <>
        <Panel>
          <Eq>
            u<sup>(i)</sup> = (z<sup>(i)</sup> − μ_B) / √(σ_B² + ε)
            <br />
            ẑ<sup>(i)</sup> = γ · u<sup>(i)</sup> + β &nbsp; (γ, β learned per unit)
            <br />
            γ = √(σ_B² + ε), β = μ_B &nbsp;⇒&nbsp; ẑ = z
          </Eq>
          <Slider id="l7-g2" label="scale γ" value={gamma} min={0} max={5} stepSize={0.01} onChange={setGamma} />
          <Slider id="l7-b2" label="shift β" value={beta} min={-3} max={3} stepSize={0.01} onChange={setBeta} />
          <Readout rows={rows(res)} />
        </Panel>
        <Note>In a layer: z = Wx (the bias is dropped because β replaces it), then BN, then the activation. Gradients for the two new weights: ∂L/∂β = Σ ∂L/∂ẑ and ∂L/∂γ = Σ u · ∂L/∂ẑ.</Note>
      </>
    );
  } else {
    question = (
      <>
        Suppose the best thing for this unit is to change nothing. Without reading the batch's mean and std, set γ and β so batch norm gives back the raw values: each △ should land on its □. Get the average miss below {GOAL7}.
      </>
    );
    stage = <GammaStage res={resC} ghost />;
    side = (
      <>
        <Panel edge={okC ? THEME.ok : THEME.grid}>
          <Slider id="l7-gc" label="scale γ" value={gC} min={0} max={3} stepSize={0.005} onChange={setGC} display={fmt(gC, 3)} />
          <Slider id="l7-bc" label="shift β" value={bC} min={-3} max={3} stepSize={0.005} onChange={setBC} display={fmt(bC, 3)} />
          <Readout rows={[["average miss |ẑ − z|", fmt(resC.miss, 3)]]} />
          <Goal met={okC}>average miss below {GOAL7}</Goal>
          <Hint
            hints={[
              "First move only β until the centre of the △ cloud sits on the centre of the □ cloud. Then use γ.",
              "γ stretches the cloud around β. Make the △ cloud exactly as wide as the □ cloud: γ is then the batch's std, and β its mean.",
            ]}
          />
        </Panel>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="γ and β give the choice back" question={question} stage={stage} side={side} />;
}
