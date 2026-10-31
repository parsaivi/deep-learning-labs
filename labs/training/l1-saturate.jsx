import { useState, useMemo, useCallback } from "react";
import { THEME, MONO } from "./common";

// Lesson 1 - Activation functions and saturation (Practical Issues, slides 5-23).
// Aha: the gradient a neuron sends back is the slope of its activation at the point where each
// order's input lands. Sigmoid and tanh are flat at both ends, so orders that land there send back
// nothing; sigmoid's slope is never above 0.25, so a stack of them shrinks the gradient every layer.
// ReLU is flat only on one side, and a ReLU whose inputs all land there is dead.

// @sim-start
const ORD1 = makeOrders(41, 140, 2.2, 8, 3);
const KM1 = Array.from(ORD1.km);
function neuron(act, w, b, metres, al) {
  const zs = KM1.map((km) => w * (metres ? km * 1000 : km) + b);
  const f = zs.map((z) => actF(act, z, al));
  const d = zs.map((z) => actD(act, z, al));
  const silent = d.filter((v) => v < 0.01).length / d.length;
  return { zs, f, d, slope: meanOf(d), silent, spread: stdOf(f) };
}
const DEPTH1 = 6;
const GOAL1G = 0.1;
const GOAL1S = 0.35;
// @sim-end

const ZR = [-10, 10];
const ACT_OPTS = [
  ["sigmoid", "sigmoid"],
  ["tanh", "tanh"],
  ["relu", "ReLU"],
  ["leaky", "leaky ReLU"],
  ["elu", "ELU"],
];
const UNIT_OPTS = [
  ["m", "distance in metres"],
  ["km", "distance in km"],
];

// two panels sharing the z axis: f(z) on top, f'(z) below; each order is a dot on both
function ActStage({ act, al, n }) {
  const W0 = PX0;
  const W1 = PX1;
  const X = (z) => W0 + ((clamp(z, ZR[0], ZR[1]) - ZR[0]) / (ZR[1] - ZR[0])) * (W1 - W0);
  const panel = (top, bot, lo, hi) => (v) => bot - ((clamp(v, lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo)) - lo) / (hi - lo)) * (bot - top);
  const fLo = act === "sigmoid" || act === "relu" ? -0.1 : act === "leaky" ? -1.1 : -1.1;
  const fHi = act === "relu" || act === "leaky" || act === "elu" ? 3 : 1.1;
  const YF = panel(22, 150, fLo, fHi);
  const YD = panel(190, 290, 0, 1.05);
  const curve = (fn, Y) => {
    const a = [];
    for (let i = 0; i <= 240; i++) {
      const z = ZR[0] + ((ZR[1] - ZR[0]) * i) / 240;
      a.push(X(z).toFixed(1) + "," + Y(fn(z)).toFixed(1));
    }
    return a.join(" ");
  };
  const left = n.zs.filter((z) => z < ZR[0]).length;
  const right = n.zs.filter((z) => z > ZR[1]).length;
  const ticks = [-10, -5, 0, 5, 10];
  return (
    <svg viewBox="0 0 620 318" style={{ width: "100%", display: "block" }} role="img" aria-label="Activation curve and its slope, with one dot per order at its input z">
      <rect x={W0} y={16} width={W1 - W0} height={140} fill={THEME.bg} stroke={THEME.grid} />
      <rect x={W0} y={184} width={W1 - W0} height={112} fill={THEME.bg} stroke={THEME.grid} />
      <line x1={X(0)} x2={X(0)} y1={16} y2={156} stroke={THEME.grid} />
      <line x1={X(0)} x2={X(0)} y1={184} y2={296} stroke={THEME.grid} />
      <line x1={W0} x2={W1} y1={YD(0.25)} y2={YD(0.25)} stroke={THEME.grid} strokeDasharray="2 3" />
      <text x={W0 - 4} y={YD(0.25) + 3} textAnchor="end" fontFamily={MONO} fontSize={9.5} fill={THEME.muted}>
        0.25
      </text>
      <text x={W0 - 4} y={YD(1) + 3} textAnchor="end" fontFamily={MONO} fontSize={9.5} fill={THEME.muted}>
        1
      </text>
      <text x={W0 - 4} y={YD(0) + 3} textAnchor="end" fontFamily={MONO} fontSize={9.5} fill={THEME.muted}>
        0
      </text>
      <polyline points={curve((z) => actF(act, z, al), YF)} fill="none" stroke={THEME.accent} strokeWidth={2.2} />
      <polyline points={curve((z) => actD(act, z, al), YD)} fill="none" stroke={THEME.highlight} strokeWidth={2} strokeDasharray="6 3" />
      {n.zs.map((z, i) => (
        <g key={i}>
          <circle cx={X(z)} cy={YF(n.f[i])} r={2.6} fill={THEME.text} opacity={0.7} />
          <circle cx={X(z)} cy={YD(n.d[i])} r={2.6} fill={n.d[i] < 0.01 ? THEME.warn : THEME.text} opacity={0.75} />
        </g>
      ))}
      <g fontFamily={MONO} fontSize={10.5} stroke={THEME.bg} strokeWidth={3} paintOrder="stroke">
        <text x={W0 + 6} y={30} fill={THEME.accent}>
          ── output f(z) = {ACT_NAMES[act]}
        </text>
        <text x={W0 + 6} y={198} fill={THEME.highlight}>
          - - slope f′(z): the share of the gradient each order sends back
        </text>
        {left > 0 && (
          <text x={W0 + 6} y={146} fill={THEME.warn}>
            ◀ {left} orders beyond z = −10
          </text>
        )}
        {right > 0 && (
          <text x={W1 - 6} y={146} textAnchor="end" fill={THEME.warn}>
            {right} orders beyond z = +10 ▶
          </text>
        )}
      </g>
      {ticks.map((t) => (
        <text key={t} x={X(t)} y={310} textAnchor="middle" fontFamily={MONO} fontSize={10} fill={THEME.muted}>
          {t}
        </text>
      ))}
      <text x={W1} y={318} textAnchor="end" fontFamily={MONO} fontSize={10} fill={THEME.muted}>
        z = w · distance + b →
      </text>
    </svg>
  );
}

export default function SaturateLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [act, setAct] = useState("sigmoid");
  const [unit, setUnit] = useState("m");
  const [w, setW] = useState(0.003);
  const [b, setB] = useState(0);
  const [al, setAl] = useState(0.1);
  const [wC, setWC] = useState(1.2);
  const [bC, setBC] = useState(0);
  const metres = unit === "m";
  const actNow = screen === 0 ? "sigmoid" : act;
  const metresNow = screen === 0 ? true : metres;
  const n = useMemo(() => neuron(actNow, w, b, metresNow, al), [actNow, w, b, metresNow, al]);
  const nC = useMemo(() => neuron("tanh", wC, bC, false, al), [wC, bC, al]);
  const deepC = Math.pow(nC.slope, DEPTH1);
  const pickUnit = useCallback(
    (u) => {
      setUnit(u);
      setW(u === "m" ? 0.003 : 1);
    },
    []
  );

  const rows = (nn) => [
    ["average slope f′ over the orders", fmt(nn.slope, 3)],
    ["orders sending back < 1% (● silent)", fmt(nn.silent * 100, 0) + "%", nn.silent > 0.5 ? THEME.warn : undefined],
    ["spread of outputs (std of f)", fmt(nn.spread, 3)],
  ];
  const wSlider = (id, lo, hi) => <LogSlider id={id} label={"weight w (per " + (metresNow ? "metre" : "km") + ")"} value={w} lo={lo} hi={hi} onChange={setW} digits={metresNow ? 5 : 3} />;

  let question, stage, side;
  if (screen === 0) {
    question = (
      <>
        One neuron of the ETA network reads each order's distance, in metres as the app stores it, and passes w · distance + b through a sigmoid. Training has stalled: w has not moved for 1000 steps. Slide w and b. Can you find a setting where most of the 140 orders send a gradient back? What do the orders that do send one have in common?
      </>
    );
    stage = <ActStage act="sigmoid" al={al} n={n} />;
    side = (
      <>
        <Panel>
          {wSlider("l1-w0", -4.5, -1)}
          <Slider id="l1-b0" label="bias b" value={b} min={-12} max={12} stepSize={0.01} onChange={setB} />
          <Readout rows={rows(n)} />
        </Panel>
        <Note>Each dot is one order. A dot drawn as a red ● in the lower panel sends back less than 1% of the gradient that reaches this neuron.</Note>
      </>
    );
  } else if (screen === 1) {
    question = (
      <>
        Now switch the activation and the units. What is the largest slope a sigmoid ever has? Which activation sends back the full gradient on a whole half of the line? Set b to −10 with ReLU: what happens, and could training ever undo it?
      </>
    );
    stage = <ActStage act={act} al={al} n={n} />;
    side = (
      <>
        <Panel>
          <Label>activation</Label>
          <Pills options={ACT_OPTS} value={act} onPick={setAct} />
          <Pills options={UNIT_OPTS} value={unit} onPick={pickUnit} />
          {wSlider("l1-w1", metres ? -4.5 : -2, metres ? -1 : 1)}
          <Slider id="l1-b1" label="bias b" value={b} min={-12} max={12} stepSize={0.01} onChange={setB} />
          {act === "leaky" && <Slider id="l1-al" label="leak α (slope for z < 0)" value={al} min={0.01} max={0.5} stepSize={0.01} onChange={setAl} />}
          <Readout rows={rows(n)} />
        </Panel>
        <Note>Watch the lower panel: training can only change w and b through orders whose dot is off the floor.</Note>
      </>
    );
  } else if (screen === 2) {
    question = (
      <>
        Name it. Back-propagation multiplies the incoming gradient by f′(z) of each order. Where the curve is flat, f′ ≈ 0 and the neuron is <i>saturated</i>: those orders teach w nothing. A neuron whose orders all land in a flat part has stopped learning.
      </>
    );
    stage = <ActStage act={act} al={al} n={n} />;
    side = (
      <>
        <Panel>
          <Eq>
            ∂L/∂w = ∂L/∂a · f′(z) · x
            <br />
            sigmoid: σ′ = σ(1 − σ) ≤ 0.25
            <br />
            tanh: 1 − tanh² ≤ 1, flat at both ends
            <br />
            ReLU: f′ = 1 if z &gt; 0, else 0
            <br />
            leaky ReLU: f′ = 1 or α, never 0
          </Eq>
          <Pills options={ACT_OPTS} value={act} onPick={setAct} />
          <Readout rows={rows(n)} />
        </Panel>
        <Note>
          Three more problems from the slides. Sigmoid's outputs are always positive (lesson 2 shows why that hurts the next layer). In a deep stack the slopes multiply: six sigmoid layers give at most 0.25⁶ ≈ 0.0002 of the gradient. A ReLU whose z is negative for every order is <i>dead</i>: its gradient is exactly 0, so no update can bring it back. Bad initialization or a too-large learning rate can push it there. In practice: use ReLU, try leaky ReLU / ELU / maxout for small gains, and avoid sigmoid in hidden layers.
        </Note>
      </>
    );
  } else {
    question = (
      <>
        A new layer of the ETA network uses tanh and you cannot change that. Distance comes in km. The layer sits {DEPTH1} tanh layers deep, so the gradient that reaches the bottom is roughly (average slope)<sup>{DEPTH1}</sup>. Pick w and b so that at least {GOAL1G} of the gradient gets through, while the outputs still tell near orders from far ones (spread ≥ {GOAL1S}).
      </>
    );
    stage = <ActStage act="tanh" al={al} n={nC} />;
    side = (
      <>
        <Panel edge={deepC >= GOAL1G && nC.spread >= GOAL1S ? THEME.ok : THEME.grid}>
          <LogSlider id="l1-wc" label="weight w (per km)" value={wC} lo={-2} hi={1} onChange={setWC} digits={3} />
          <Slider id="l1-bc" label="bias b" value={bC} min={-6} max={6} stepSize={0.01} onChange={setBC} />
          <Readout
            rows={[
              ["average slope f′", fmt(nC.slope, 3)],
              ["(average slope)^" + DEPTH1, fmt(deepC, 3)],
              ["spread of outputs", fmt(nC.spread, 3)],
            ]}
          />
          <Goal met={deepC >= GOAL1G}>
            gradient through {DEPTH1} layers ≥ {GOAL1G}
          </Goal>
          <Goal met={nC.spread >= GOAL1S}>output spread ≥ {GOAL1S}</Goal>
          <Hint
            hints={[
              "Notice where the cloud of dots sits on the slope curve. Is it centred on the peak at z = 0?",
              "Orders run from 0.5 to 8 km, so the middle is about 4 km. Choose b ≈ −4·w to centre the cloud, then make the cloud just wide enough for the spread goal without reaching the flat ends.",
            ]}
          />
        </Panel>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="Flat ends kill the signal" question={question} stage={stage} side={side} />;
}
