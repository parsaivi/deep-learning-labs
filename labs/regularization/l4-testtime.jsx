import { useState, useMemo, useCallback } from "react";
import { THEME, MONO } from "./common";

// Lesson 4 - Dropout at test time (Regularization deck, slides 25-34).
// Aha: with dropout left on, the same order gets a different ETA on every refresh. What we want is the
// average over all masks, and for the layer right after the dropout that average is simply the full
// network with each dropped unit scaled by its keep probability a. Inverted dropout moves the 1/a into
// training so the test-time network needs no change at all.

// @sim-start
const H4 = 6;
const NET4 = (function () {
  const r = rng(41);
  const W = [];
  const c = [];
  const v = [];
  for (let j = 0; j < H4; j++) {
    W.push([0.9 + 0.5 * r(), 0.3 + 0.9 * r(), (r() - 0.3) * 1.2]);
    c.push(j % 2 ? 1.2 : -0.8 + 0.4 * r());
    v.push(1.3 + 1.4 * r());
  }
  return { W, c, v, b: 9 };
})();
// orders: [distance / 4 km, traffic level, rain]
const ORD4 = [
  { name: "A · 3 km, light traffic", x: [0.75, 0.4, 0] },
  { name: "B · 6 km, rush hour, rain", x: [1.5, 1.3, 1] },
  { name: "C · 2 km, jammed", x: [0.5, 1.7, 0.4] },
];
function hidden4(x) {
  return NET4.W.map((w, j) => Math.max(0, NET4.c[j] + w[0] * x[0] + w[1] * x[1] + w[2] * x[2]));
}
// one pass: mask m (0/1 per hidden unit), scale s on every hidden unit (1/a for inverted dropout)
function out4(h, m, s) {
  let o = NET4.b;
  for (let j = 0; j < H4; j++) o += NET4.v[j] * h[j] * (m ? m[j] : 1) * s;
  return o;
}
// exact average over all 2^6 masks
function mean4(h, a, s) {
  let o = 0;
  for (let B = 0; B < 1 << H4; B++) {
    let p = 1;
    const m = [];
    for (let j = 0; j < H4; j++) {
      const on = (B >> j) & 1;
      m.push(on);
      p *= on ? a : 1 - a;
    }
    o += p * out4(h, m, s);
  }
  return o;
}
function samples4(h, a, s, n, seed) {
  const r = rng(seed);
  const out = [];
  for (let i = 0; i < n; i++) {
    const m = [];
    for (let j = 0; j < H4; j++) m.push(r() < a ? 1 : 0);
    out.push(out4(h, m, s));
  }
  return out;
}
const KEEPC4 = 0.65;
const TOLC4 = 0.5;
// @sim-end

const HR4 = [0, 90];
function EtaHist({ vals, marks, label, height }) {
  const H = height || 150;
  const top = 18;
  const bot = H - 24;
  const bins = 45;
  const counts = new Array(bins).fill(0);
  for (const v of vals) counts[clamp(Math.floor(((v - HR4[0]) / (HR4[1] - HR4[0])) * bins), 0, bins - 1)]++;
  const mx = Math.max(1, Math.max.apply(null, counts));
  const X = (v) => PX0 + ((clamp(v, HR4[0], HR4[1]) - HR4[0]) / (HR4[1] - HR4[0])) * (PX1 - PX0);
  const bw = (PX1 - PX0) / bins;
  return (
    <svg viewBox={"0 0 620 " + H} style={{ width: "100%", display: "block" }} role="img" aria-label={label || "Histogram of ETAs over random dropout masks"}>
      <rect x={PX0} y={top - 4} width={PX1 - PX0} height={bot - top + 8} fill={THEME.bg} stroke={THEME.grid} />
      <text x={PX0 + 6} y={top + 8} fontFamily={MONO} fontSize={10.5} fill={THEME.muted}>
        {label}
      </text>
      {counts.map((c, i) => (c ? <rect key={i} x={PX0 + i * bw + 0.5} y={bot - (c / mx) * (bot - top - 16)} width={bw - 1} height={(c / mx) * (bot - top - 16)} fill={THEME.accent} opacity={0.55} /> : null))}
      {[0, 15, 30, 45, 60, 75, 90].map((t) => (
        <text key={t} x={X(t)} y={H - 8} textAnchor="middle" fontFamily={MONO} fontSize={10} fill={THEME.muted}>
          {t}′
        </text>
      ))}
      {(marks || []).map((m, i) => (
        <g key={i}>
          <line x1={X(m.x)} x2={X(m.x)} y1={top + 12} y2={bot} stroke={m.color} strokeWidth={1.6} strokeDasharray={m.dash} />
          <Marker x={X(m.x)} y={top + 14 + i * 12} shape={m.shape} color={m.color} r={4.5} />
          <text x={X(m.x) + (m.x > 70 ? -8 : 8)} y={top + 18 + i * 12} textAnchor={m.x > 70 ? "end" : "start"} fontFamily={MONO} fontSize={10.5} fill={m.color} stroke={THEME.bg} strokeWidth={3} paintOrder="stroke">
            {m.label}
          </text>
        </g>
      ))}
    </svg>
  );
}

export default function TestTimeLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [order, setOrder] = useState("1");
  const [keep, setKeep] = useState(0.5);
  const [nPass, setNPass] = useState(20);
  const [refresh, setRefresh] = useState([]);
  const [inverted, setInverted] = useState(false);
  const [sC, setSC] = useState(1);
  const o = ORD4[Number(order)];
  const h = useMemo(() => hidden4(o.x), [o]);
  const a = screen === 0 ? 0.5 : keep;
  const sTrain = inverted ? 1 / a : 1;
  const exact = mean4(h, a, sTrain);
  const allOn = out4(h, null, 1);
  const scaled = out4(h, null, inverted ? 1 : a);
  const n = Math.round(nPass);
  const smp = useMemo(() => samples4(h, a, sTrain, screen === 0 ? 400 : n, 77), [h, a, sTrain, n, screen]);
  const run = smp.reduce((p, v) => p + v, 0) / smp.length;
  const press = useCallback(() => {
    setRefresh((r) => {
      const v = samples4(h, 0.5, 1, 1, 1000 + r.length * 31)[0];
      return r.concat([v]).slice(-8);
    });
  }, [h]);
  const pickOrder = useCallback((v) => {
    setOrder(v);
    setRefresh([]);
  }, []);
  const orderPills = <Pills options={ORD4.map((q, i) => [String(i), q.name])} value={order} onPick={pickOrder} />;

  // challenge data: three orders, 300 training-time passes each, keep probability not shown
  const chal = useMemo(
    () =>
      ORD4.map((q, i) => {
        const hh = hidden4(q.x);
        return { name: q.name, h: hh, smp: samples4(hh, KEEPC4, 1, 300, 500 + i), mean: mean4(hh, KEEPC4, 1) };
      }),
    []
  );
  const chalRes = chal.map((c) => ({ test: out4(c.h, null, sC), mean: c.mean }));
  const okC = chalRes.every((c) => Math.abs(c.test - c.mean) < TOLC4);

  let question, stage, side;
  if (screen === 0) {
    question = (
      <>
        The ETA network was trained with dropout (each hidden unit kept with probability a = 0.5), and someone shipped it with dropout still switched on. Press refresh a few times for the same order. Which of these ETAs should the customer see?
      </>
    );
    stage = (
      <EtaHist
        vals={smp}
        label={"order " + o.name + " · 400 refreshes"}
        marks={refresh.length ? [{ x: refresh[refresh.length - 1], color: THEME.highlight, shape: "diamond", label: "latest refresh ◆" }] : []}
      />
    );
    side = (
      <Panel>
        {orderPills}
        <Btn onClick={press} strong>
          refresh ETA
        </Btn>
        <div style={{ fontFamily: MONO, fontSize: 13, margin: "10px 0 4px", color: THEME.text, lineHeight: 1.7 }}>
          {refresh.length ? refresh.map((v) => fmt(v, 1) + "′").join("  ·  ") : "no refreshes yet"}
        </div>
        <Note>The network and orders are made-up: 3 inputs, 6 ReLU hidden units, dropout on the hidden layer.</Note>
      </Panel>
    );
  } else if (screen === 1) {
    question = (
      <>
        The fair answer is the average over every mask. Average more and more random passes: where does the average settle? Compare it with two cheap single passes: all units on, and all units on with each multiplied by a. Which one lands on the average, for every order and every a?
      </>
    );
    stage = (
      <EtaHist
        vals={smp}
        label={"order " + o.name + " · " + n + " random passes"}
        marks={[
          { x: run, color: THEME.highlight, shape: "diamond", label: "average of passes " + fmt(run, 1) },
          { x: allOn, color: THEME.warn, shape: "square", dash: "4 3", label: "all on " + fmt(allOn, 1) },
          { x: scaled, color: THEME.ok, shape: "circle", dash: "2 3", label: "all on × a " + fmt(scaled, 1) },
        ]}
      />
    );
    side = (
      <Panel>
        {orderPills}
        <Slider id="r4-n" label="random passes averaged" value={nPass} min={1} max={2000} stepSize={1} onChange={setNPass} display={String(n)} />
        <Slider id="r4-a" label="keep probability a" value={keep} min={0.2} max={1} stepSize={0.005} onChange={setKeep} display={fmt(keep, 3)} />
        <Readout
          rows={[
            ["exact average (all 64 masks)", fmt(exact, 2) + " min", THEME.highlight],
            ["all on", fmt(allOn, 2) + " min"],
            ["all on × a", fmt(scaled, 2) + " min"],
          ]}
        />
      </Panel>
    );
  } else if (screen === 2) {
    question = (
      <>
        Name it. A dropped unit outputs D·σ(z) with D ~ Bernoulli(a), so on average it outputs a·σ(z). At test time keep every unit on and scale its output by a: output at test = expected output at training. <b>Inverted dropout</b> does the scaling at training time instead (divide by a), so the test network is left untouched. Switch it on.
      </>
    );
    stage = (
      <EtaHist
        vals={smp}
        label={"order " + o.name + " · " + n + " random passes" + (inverted ? " · inverted dropout" : "")}
        marks={[
          { x: exact, color: THEME.highlight, shape: "diamond", label: "exact average " + fmt(exact, 1) },
          { x: allOn, color: THEME.warn, shape: "square", dash: "4 3", label: "test pass, untouched " + fmt(allOn, 1) },
        ]}
      />
    );
    side = (
      <Panel>
        <Eq size={13}>
          train: a<sub>i</sub> = D · σ(Σ<sub>j</sub> w<sub>ji</sub>a<sub>j</sub> + b<sub>i</sub>), D ~ Bernoulli(a)
          <br />
          test: a<sub>i</sub> = a · σ(Σ<sub>j</sub> w<sub>ji</sub>a<sub>j</sub> + b<sub>i</sub>)
          <br />
          inverted: train with D/a, test with no change
        </Eq>
        <Toggle label="inverted dropout (÷a while training)" value={inverted} onChange={setInverted} />
        <Slider id="r4-a2" label="keep probability a" value={keep} min={0.2} max={1} stepSize={0.005} onChange={setKeep} display={fmt(keep, 3)} />
        <Note>Here the scaling is exact because the layer after dropout is linear. After a nonlinearity it is a close approximation, which is what everyone uses. Backprop only flows through the units that were on.</Note>
      </Panel>
    );
  } else {
    question = (
      <>
        Another team trained this network with plain (not inverted) dropout and lost the log that says the keep probability. You have 300 training-time passes for three orders. Set the test-time scale s on the hidden units so one deterministic pass hits the training-time average (◆) of every order within {TOLC4} min.
      </>
    );
    stage = (
      <>
        {chal.map((c, i) => (
          <EtaHist
            key={i}
            vals={c.smp}
            height={120}
            label={c.name + " · 300 training passes"}
            marks={[
              { x: c.mean, color: THEME.highlight, shape: "diamond", label: "average " + fmt(c.mean, 1) },
              { x: chalRes[i].test, color: THEME.ok, shape: "circle", dash: "2 3", label: "your test pass " + fmt(chalRes[i].test, 1) },
            ]}
          />
        ))}
      </>
    );
    side = (
      <Panel edge={okC ? THEME.ok : THEME.grid}>
        <Slider id="r4-sc" label="test-time scale s on hidden units" value={sC} min={0.2} max={1.2} stepSize={0.001} onChange={setSC} display={fmt(sC, 3)} />
        <Readout rows={chalRes.map((c, i) => [ORD4[i].name.split(" ·")[0] + ": test − average", sgn(c.test - c.mean, 2) + " min", Math.abs(c.test - c.mean) < TOLC4 ? THEME.ok : THEME.muted])} />
        <Goal met={okC}>all three within {TOLC4} min</Goal>
        <Hint
          hints={[
            "Can one s fix all three orders at once? Get order B right first; it has the biggest gap.",
            "s is not a fudge factor: it is the keep probability they trained with. Find it to about two decimals.",
          ]}
        />
      </Panel>
    );
  }
  return <LessonFrame screen={screen} onScreen={setScreen} title="Test time: average the masks" question={question} stage={stage} side={side} />;
}
