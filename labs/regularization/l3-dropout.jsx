import { useState, useMemo, useCallback } from "react";
import { THEME, MONO } from "./common";

// Lesson 3 - Dropout: no single input to lean on (Regularization deck, slides 11-24).
// Aha: if any input can vanish during training, the unit cannot put all its trust in one of them.
// It spreads the weight over every input that carries the signal, so it survives when one goes dark.
// For a linear unit the expected dropout loss is exactly the squared error plus an L2-like penalty
// ((1 - a) / a) * sum_k w_k^2 E[s_k^2]: the same "keep the weights small and spread" effect as weight decay.

// @sim-start
const FEEDS3 = ["courier GPS", "map traffic", "restaurant"];
const SIG3 = [0.25, 0.7, 1.0];
function makeSens3(seed, n) {
  const r = rng(seed);
  const X = [];
  const y = [];
  for (let i = 0; i < n; i++) {
    const t = 2.2 * gauss(r);
    X.push(SIG3.map((s) => t + s * gauss(r)));
    y.push(2 * t + 0.4 * gauss(r));
  }
  return { X, y, n };
}
const TR3 = makeSens3(31, 200);
const TE3 = makeSens3(32, 400);
// expected squared error under inverted dropout with keep probability a (closed form for a linear unit)
function dropFit3(a) {
  const A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  const b = [0, 0, 0];
  for (let s = 0; s < TR3.n; s++)
    for (let i = 0; i < 3; i++) {
      b[i] += (TR3.X[s][i] * TR3.y[s]) / TR3.n;
      for (let j = 0; j < 3; j++) A[i][j] += (TR3.X[s][i] * TR3.X[s][j]) / TR3.n;
    }
  const r = (1 - a) / a;
  for (let i = 0; i < 3; i++) A[i][i] *= 1 + r;
  return solveLin(A, b);
}
function miss3(w, dead) {
  let s = 0;
  for (let i = 0; i < TE3.n; i++) {
    let p = 0;
    for (let k = 0; k < 3; k++) if (k !== dead) p += w[k] * TE3.X[i][k];
    s += Math.pow(p - TE3.y[i], 2);
  }
  return Math.sqrt(s / TE3.n);
}
function eval3(a) {
  const w = dropFit3(a);
  const out = [0, 1, 2].map((k) => miss3(w, k));
  return { w, ok: miss3(w, -1), out, worst: Math.max.apply(null, out) };
}
const GOAL3 = { ok: 0.9, worst: 2.35 };
// @sim-end

function SensorNet({ w, dead, mask }) {
  const ys = [50, 115, 180];
  const ox = 470;
  const oy = 115;
  return (
    <svg viewBox="0 0 620 230" style={{ width: "100%", display: "block" }} role="img" aria-label="Three traffic feeds feeding the extra-delay unit; line thickness is the weight">
      <rect x={0} y={0} width={620} height={230} fill={THEME.bg} stroke={THEME.grid} />
      {ys.map((y, k) => {
        const off = dead === k || (mask && !mask[k]);
        const wd = Math.min(14, 0.6 + Math.abs(w[k]) * 6);
        return (
          <g key={k}>
            <line x1={170} y1={y} x2={ox - 26} y2={oy} stroke={off ? THEME.grid : THEME.accent} strokeWidth={off ? 1.5 : wd} strokeDasharray={off ? "4 4" : undefined} opacity={0.9} />
            <rect x={36} y={y - 18} width={134} height={36} rx={3} fill={THEME.surface} stroke={off ? THEME.warn : THEME.accent} strokeDasharray={off ? "4 3" : undefined} />
            <text x={103} y={y - 2} textAnchor="middle" fontFamily={MONO} fontSize={11.5} fill={off ? THEME.muted : THEME.text}>
              {FEEDS3[k]}
            </text>
            <text x={103} y={y + 12} textAnchor="middle" fontFamily={MONO} fontSize={10} fill={off ? THEME.warn : THEME.muted}>
              {off ? (dead === k ? "✕ feed down" : "✕ dropped") : "noise ±" + SIG3[k]}
            </text>
            <text x={(170 + ox - 26) / 2 + 10} y={(y + oy) / 2 - 8} textAnchor="middle" fontFamily={MONO} fontSize={11} fill={THEME.highlight} stroke={THEME.bg} strokeWidth={3} paintOrder="stroke">
              w{k + 1} = {fmt(w[k], 2)}
            </text>
          </g>
        );
      })}
      <circle cx={ox} cy={oy} r={26} fill={THEME.surface} stroke={THEME.highlight} strokeWidth={2} />
      <text x={ox} y={oy + 4} textAnchor="middle" fontFamily={MONO} fontSize={12} fill={THEME.text}>
        Σ
      </text>
      <text x={ox + 34} y={oy - 4} fontFamily={MONO} fontSize={11} fill={THEME.muted}>
        extra
      </text>
      <text x={ox + 34} y={oy + 10} fontFamily={MONO} fontSize={11} fill={THEME.muted}>
        delay
      </text>
      <text x={10} y={222} fontFamily={MONO} fontSize={10} fill={THEME.muted}>
        line thickness = |weight| · dashed = no signal
      </text>
    </svg>
  );
}

function OutageBars({ res, goal }) {
  const rows = [["all feeds up", res.ok, "circle"]].concat(FEEDS3.map((f, k) => [f + " down", res.out[k], "triangle"]));
  const X = (v) => 190 + (Math.min(v, 5) / 5) * 400;
  return (
    <svg viewBox="0 0 620 150" style={{ width: "100%", display: "block" }} role="img" aria-label="Typical miss on new orders, with every feed up and with each feed down">
      <rect x={0} y={0} width={620} height={150} fill={THEME.bg} stroke={THEME.grid} />
      <text x={10} y={16} fontFamily={MONO} fontSize={10.5} fill={THEME.muted}>
        miss on 400 new orders (minutes)
      </text>
      {goal && (
        <g>
          <line x1={X(goal.worst)} x2={X(goal.worst)} y1={24} y2={140} stroke={THEME.ok} strokeDasharray="4 3" />
          <text x={X(goal.worst) + 3} y={146} fontFamily={MONO} fontSize={9.5} fill={THEME.ok}>
            {goal.worst}
          </text>
          <line x1={X(goal.ok)} x2={X(goal.ok)} y1={24} y2={50} stroke={THEME.ok} strokeDasharray="2 2" />
        </g>
      )}
      {rows.map((r, i) => {
        const y = 30 + i * 28;
        return (
          <g key={r[0]}>
            <text x={182} y={y + 11} textAnchor="end" fontFamily={MONO} fontSize={11} fill={i === 0 ? THEME.text : THEME.muted}>
              {r[0]}
            </text>
            <rect x={190} y={y} width={X(r[1]) - 190} height={16} fill={i === 0 ? THEME.accent : THEME.highlight} opacity={0.8} />
            <Marker x={X(r[1])} y={y + 8} shape={r[2]} color={i === 0 ? THEME.accent : THEME.highlight} r={4} />
            <text x={X(r[1]) + 8} y={y + 12} fontFamily={MONO} fontSize={11} fill={THEME.text}>
              {fmt(r[1], 2)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export default function DropoutLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [dead, setDead] = useState(-1);
  const [keep, setKeep] = useState(1);
  const [keepC, setKeepC] = useState(1);
  const [mask, setMask] = useState(null);
  const [seed, setSeed] = useState(1);
  const a = screen === 0 ? 1 : keep;
  const res = useMemo(() => eval3(a), [a]);
  const resC = useMemo(() => eval3(keepC), [keepC]);
  const okC = resC.ok < GOAL3.ok && resC.worst < GOAL3.worst;
  const pickDead = useCallback((v) => setDead(Number(v)), []);
  const sample = useCallback(() => {
    const r = rng(seed * 7919 + Math.round(keep * 1000));
    setMask([0, 1, 2].map(() => (r() < keep ? 1 : 0)));
    setSeed((s) => s + 1);
  }, [seed, keep]);
  const clearMask = useCallback(() => setMask(null), []);
  const deadOpts = [["-1", "none"]].concat(FEEDS3.map((f, k) => [String(k), f]));

  let question, stage, side;
  if (screen === 0) {
    question = (
      <>
        The extra-delay unit reads three traffic feeds that all measure the same thing with different noise. Trained on 200 orders, it puts almost all its weight on the courier GPS, the cleanest feed. Rush hour, and the GPS feed goes down. How bad does the ETA get? Is any one feed safe to lose?
      </>
    );
    stage = (
      <>
        <SensorNet w={res.w} dead={dead} />
        <OutageBars res={res} />
      </>
    );
    side = (
      <Panel>
        <Label>feed down at test time</Label>
        <Pills options={deadOpts} value={String(dead)} onPick={pickDead} />
        <Readout rows={[["miss with this outage", fmt(dead < 0 ? res.ok : res.out[dead], 2) + " min", THEME.highlight]]} />
        <Note>Feeds are made-up: each is the true traffic delay plus its own noise. The target is the order's extra minutes.</Note>
      </Panel>
    );
  } else if (screen === 1) {
    question = (
      <>
        Change the training: on every pass, switch each feed off with probability 1 − a (and divide the survivors by a, so the sum keeps its size). Lower the keep probability a. What happens to the three weights? To the miss when all feeds are up, and when one is down?
      </>
    );
    stage = (
      <>
        <SensorNet w={res.w} mask={mask} />
        <OutageBars res={res} />
      </>
    );
    side = (
      <Panel>
        <Slider id="r3-a" label="keep probability a (training only)" value={keep} min={0.3} max={1} stepSize={0.005} onChange={setKeep} display={fmt(keep, 3)} />
        <Btn onClick={sample}>sample one training pass</Btn> <Btn onClick={clearMask}>clear</Btn>
        <Note>Each pass sees a different random subset of feeds, so the unit can never count on the GPS being there. The weights shown are what training settles on for this a.</Note>
      </Panel>
    );
  } else if (screen === 2) {
    question = (
      <>
        Name it. <b>Dropout</b>: during training, multiply each unit's output by a random mask m ~ Bernoulli(a), and only the "on" part of the network gets gradient that pass. A unit cannot rely on any one input, so it spreads its weights: for this linear unit the expected loss is exactly the squared error plus an L2-style penalty. Each mask is a different thinned network; with M units there are 2<sup>M</sup> of them, all sharing weights.
      </>
    );
    stage = (
      <>
        <SensorNet w={res.w} mask={mask} />
        <OutageBars res={res} />
      </>
    );
    side = (
      <Panel>
        <Eq size={13}>
          m<sub>k</sub> ~ Bernoulli(a), &nbsp;ŝ<sub>k</sub> = m<sub>k</sub> s<sub>k</sub> / a
          <br />
          E[(y − Σ w<sub>k</sub>ŝ<sub>k</sub>)²] = (y − Σ w<sub>k</sub>s<sub>k</sub>)²
          <br />
          &nbsp;&nbsp;+ ((1 − a)/a) Σ w<sub>k</sub>² s<sub>k</sub>²
        </Eq>
        <Slider id="r3-a2" label="keep probability a" value={keep} min={0.3} max={1} stepSize={0.005} onChange={setKeep} display={fmt(keep, 3)} />
        <Readout
          rows={[
            ["penalty factor (1 − a)/a", fmt((1 - keep) / keep, 3), THEME.highlight],
            ["sum of weights", fmt(res.w[0] + res.w[1] + res.w[2], 2)],
          ]}
        />
        <Note>Too little keep (a → 0.3) and the penalty swamps the fit: every weight shrinks and even the all-feeds-up ETA gets worse. 0.5 is common for big hidden layers; inputs usually keep more, about 0.8.</Note>
      </Panel>
    );
  } else {
    question = (
      <>
        Operations wants both: an ETA that stays good when all three feeds are up (miss under {GOAL3.ok} min) and one that survives any single feed going down (worst miss under {GOAL3.worst} min). Pick the keep probability.
      </>
    );
    stage = (
      <>
        <SensorNet w={resC.w} />
        <OutageBars res={resC} goal={GOAL3} />
      </>
    );
    side = (
      <Panel edge={okC ? THEME.ok : THEME.grid}>
        <Slider id="r3-ac" label="keep probability a" value={keepC} min={0.3} max={1} stepSize={0.005} onChange={setKeepC} display={fmt(keepC, 3)} />
        <Readout
          rows={[
            ["miss, all feeds up", fmt(resC.ok, 2) + " min"],
            ["worst miss, one feed down", fmt(resC.worst, 2) + " min", THEME.highlight],
          ]}
        />
        <Goal met={resC.ok < GOAL3.ok}>all feeds up: below {GOAL3.ok} min</Goal>
        <Goal met={resC.worst < GOAL3.worst}>any one feed down: below {GOAL3.worst} min</Goal>
        <Hint
          hints={[
            "The two goals pull in opposite directions. Find the a where the worst outage stops improving, then check the other number.",
            "a = 1 trusts the GPS alone; a = 0.5 shrinks everything. The answer sits between: dropout of roughly one feed in five.",
          ]}
        />
      </Panel>
    );
  }
  return <LessonFrame screen={screen} onScreen={setScreen} title="No single feed to lean on" question={question} stage={stage} side={side} />;
}
