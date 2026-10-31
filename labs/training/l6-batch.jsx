import { useState, useMemo, useCallback } from "react";
import { THEME, MONO } from "./common";

// Lesson 6 - Covariate shift and normalizing each mini-batch (Batch Normalization, slides 3-15, 19, 42).
// Aha: if every mini-batch arrives somewhere else on the axis, the next layer keeps chasing it.
// Batch norm moves each batch to mean 0, std 1 using that batch's own statistics. The price: an
// order's normalized value now depends on its batch-mates, so small batches make it jitter, and
// unusual orders jitter most. (That jitter is also the "funny" regularization the slides mention.)

// @sim-start
const DAYS6 = ["Sat", "Sun", "Mon", "Tue", "Wed", "Thu"];
const DAYMEAN6 = [0.5, 1.2, 2.0, 2.4, 3.1, 3.6];
const PER_DAY6 = 300;
// z of one ETA unit for every order, by day; `shift` scales how far apart the days are
function population6(shift) {
  const r = rng(61);
  const z = [];
  const day = [];
  for (let d = 0; d < DAYS6.length; d++)
    for (let i = 0; i < PER_DAY6; i++) {
      z.push(2.05 + (DAYMEAN6[d] - 2.05) * shift + 0.8 * gauss(r));
      day.push(d);
    }
  return { z, day };
}
const POP6 = population6(1);
const TRACK6 = { typical: 2.0, rush: 6.0 };
// spread (std) of one order's normalised value u over many random batches of size B
function jitter6(zOrder, B, trials, seed) {
  const r = rng(seed || 9);
  const us = [];
  const n = POP6.z.length;
  for (let t = 0; t < trials; t++) {
    const batch = [zOrder];
    for (let k = 1; k < B; k++) batch.push(POP6.z[Math.floor(r() * n)]);
    const bn = bnForward(batch);
    us.push(bn.u[0]);
  }
  return { us, sd: stdOf(us), mean: meanOf(us) };
}
const GPU6 = 96;
const GOAL6 = 0.2;
const STAND6 = 2.75;
// @sim-end

const AX6 = [-4, 8];
const ROWS6 = 6;
const B6_SHOWN = 16;
const TRACK_OPTS = [
  ["typical", "typical order (z = 2)"],
  ["rush", "rush-hour order (z = 6)"],
];
const BATCHING_OPTS = [
  ["day", "one batch per day"],
  ["shuffled", "shuffled batches"],
];

// rows of mini-batches on a common axis; with bn, each row is normalised by its own mean and std
function batchRows(pop, batching, B, bn, track, seed) {
  const r = rng(seed);
  const rows = [];
  for (let k = 0; k < ROWS6; k++) {
    let zs = [];
    if (batching === "day") {
      for (let i = 0; i < B6_SHOWN; i++) zs.push(pop.z[k * PER_DAY6 + Math.floor(r() * PER_DAY6)]);
    } else {
      zs = [track];
      for (let i = 1; i < B; i++) zs.push(pop.z[Math.floor(r() * pop.z.length)]);
    }
    const st = bnForward(zs);
    rows.push({ label: batching === "day" ? DAYS6[k] : "batch " + (k + 1), vals: bn ? st.u : zs, mu: st.mu, sd: st.sd, tracked: batching === "day" ? -1 : 0 });
  }
  return rows;
}
function BatchStage({ rows, bn }) {
  const X = (v) => PX0 + ((clamp(v, AX6[0], AX6[1]) - AX6[0]) / (AX6[1] - AX6[0])) * (PX1 - PX0);
  const rowH = 30;
  const top = 26;
  const H = top + ROWS6 * rowH + 30;
  const ticks = [-4, -2, 0, 2, 4, 6, 8];
  return (
    <svg viewBox={"0 0 620 " + H} style={{ width: "100%", display: "block" }} role="img" aria-label="Mini-batches drawn as rows of dots on the z axis">
      <rect x={PX0} y={top - 10} width={PX1 - PX0} height={ROWS6 * rowH + 6} fill={THEME.bg} stroke={THEME.grid} />
      <rect x={X(-1)} y={top - 10} width={X(1) - X(-1)} height={ROWS6 * rowH + 6} fill={rgba(THEME.ok, 0.08)} />
      <line x1={X(0)} x2={X(0)} y1={top - 10} y2={top + ROWS6 * rowH - 4} stroke={THEME.grid} />
      <text x={PX0} y={12} fontFamily={MONO} fontSize={10.5} fill={THEME.muted}>
        {bn ? "after batch norm: u = (z − μ_B) / σ_B" : "raw z of one ETA unit"} · shaded = −1…1 · | = batch mean
      </text>
      {rows.map((row, k) => {
        const y = top + k * rowH + 8;
        const vals = row.vals.length > 40 ? row.vals.slice(0, 40) : row.vals;
        const m = meanOf(row.vals);
        return (
          <g key={k}>
            <text x={PX0 - 5} y={y + 4} textAnchor="end" fontFamily={MONO} fontSize={10} fill={THEME.muted}>
              {row.label}
            </text>
            {vals.map((v, i) =>
              i === row.tracked ? null : <circle key={i} cx={X(v)} cy={y + ((i * 7) % 11) - 5} r={2.6} fill={THEME.accent} opacity={0.75} />
            )}
            <line x1={X(m)} x2={X(m)} y1={y - 10} y2={y + 10} stroke={THEME.text} strokeWidth={2} />
            {row.tracked >= 0 && <Marker x={X(row.vals[row.tracked])} y={y} shape="diamond" color={THEME.highlight} r={6} />}
          </g>
        );
      })}
      {ticks.map((t) => (
        <text key={t} x={X(t)} y={H - 12} textAnchor="middle" fontFamily={MONO} fontSize={10} fill={THEME.muted}>
          {t}
        </text>
      ))}
      <text x={PX1} y={H - 1} textAnchor="end" fontFamily={MONO} fontSize={10} fill={THEME.muted}>
        {bn ? "u →" : "z →"} {rows[0] && rows[0].tracked >= 0 ? "· ◆ the tracked order" : ""}
      </text>
    </svg>
  );
}
function JitterPlot({ jit, label }) {
  const h = histOf(jit.us, -2, 6, 40);
  const xs = [];
  const ys = [];
  for (let k = 0; k < 40; k++) {
    xs.push(-2 + (8 * (k + 0.5)) / 40);
    ys.push(h[k]);
  }
  return (
    <Plot
      series={[{ xs, ys, color: THEME.highlight, width: 2 }]}
      xr={[-2, 6]}
      yr={[0, 0.4]}
      xl="normalised value u of the tracked order →"
      label={label}
      height={150}
      vlines={[{ x: jit.mean, color: THEME.text, label: "spread " + fmt(jit.sd, 2) }]}
    />
  );
}

export default function BatchLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [bn, setBn] = useState(false);
  const [shift, setShift] = useState(1);
  const [batching, setBatching] = useState("day");
  const [B, setB] = useState(8);
  const [track, setTrack] = useState("typical");
  const [seed, setSeed] = useState(3);
  const [BC, setBC] = useState(16);
  const pop = useMemo(() => (shift === 1 ? POP6 : population6(shift)), [shift]);
  const batchingNow = screen === 0 ? "day" : batching;
  const bnNow = screen >= 1 && batchingNow === "shuffled" ? true : bn;
  const rows = useMemo(() => batchRows(pop, batchingNow, B, bnNow, TRACK6[track], seed), [pop, batchingNow, B, bnNow, track, seed]);
  const jit = useMemo(() => jitter6(TRACK6[track], B, 400, 9), [track, B]);
  const rowsC = useMemo(() => batchRows(POP6, "shuffled", BC, true, TRACK6.rush, seed), [BC, seed]);
  const jitC = useMemo(() => jitter6(TRACK6.rush, BC, 400, 9), [BC]);
  const reseed = useCallback(() => setSeed((s) => s + 1), []);
  const changeB = useCallback((v) => setB(Math.round(Math.pow(2, v))), []);
  const changeBC = useCallback((v) => setBC(Math.round(v)), []);
  const okC = jitC.sd <= GOAL6 && jitC.mean >= STAND6 && BC <= GPU6;
  const means = rows.map((r) => r.mu);
  const spreadMeans = stdOf(rows.map((r) => meanOf(r.vals)));

  let question, stage, side;
  if (screen === 0) {
    question = (
      <>
        Each row is a mini-batch of 16 orders, one batch per day, as one unit of the ETA network sees them (its z). The next layer was tuned on Saturday's batch. Raise the day-to-day difference: where does each batch land? Then normalize each batch by its own mean and std. What does the next layer see now?
      </>
    );
    stage = <BatchStage rows={rows} bn={bn} />;
    side = (
      <>
        <Panel>
          <Slider id="l6-shift" label="how different the days are" value={shift} min={0} max={2} stepSize={0.01} onChange={setShift} />
          <Toggle label="normalize each batch" value={bn} onChange={setBn} />
          <Readout
            rows={[
              ["spread of the batch means", fmt(spreadMeans, 2)],
              ["Thu batch mean (raw z)", fmt(means[5], 2)],
            ]}
          />
        </Panel>
        <Note>Inside a network the same thing happens with no change of days at all: every update of the earlier layers moves where this unit's z lands.</Note>
      </>
    );
  } else if (screen === 1) {
    question = (
      <>
        Now use shuffled batches and follow one order (◆). Its normalized value depends on which orders share its batch. Shrink the batch to 2: what values can u take? Compare a typical order with a rush-hour order far out in the tail. Which one jitters more, and how does the jitter change each time you double the batch?
      </>
    );
    stage = (
      <>
        <BatchStage rows={rows} bn={bnNow} />
        <JitterPlot jit={jit} label={"u of the ◆ order over 400 random batches of " + B} />
      </>
    );
    side = (
      <>
        <Panel>
          <Pills options={BATCHING_OPTS} value={batching} onPick={setBatching} />
          {batchingNow === "shuffled" && (
            <>
              <Pills options={TRACK_OPTS} value={track} onPick={setTrack} />
              <Slider id="l6-B" label="batch size B" value={Math.log2(B)} min={1} max={7} stepSize={0.01} onChange={changeB} display={String(B)} />
              <Btn onClick={reseed}>draw new batches</Btn>
              <div style={{ height: 8 }} />
              <Readout
                rows={[
                  ["spread of the ◆ order's u", fmt(jit.sd, 3)],
                  ["spread × √B", fmt(jit.sd * Math.sqrt(B), 2)],
                ]}
              />
            </>
          )}
        </Panel>
        <Note>With B = 2 there are two numbers, z₁ and z₂. Work out (z₁ − μ)/σ by hand.</Note>
      </>
    );
  } else if (screen === 2) {
    question = (
      <>
        Name it: <i>batch normalization</i> (Ioffe & Szegedy, 2015). For each unit, after the weighted sum and before the activation, subtract the mini-batch mean and divide by the mini-batch std. It removes the covariate shift between batches, and between training steps, so later layers stand on stable ground. The jitter shrinks like 1/√B.
      </>
    );
    stage = (
      <>
        <BatchStage rows={rows} bn={bnNow} />
        <JitterPlot jit={jit} label={"u of the ◆ order over 400 random batches of " + B} />
      </>
    );
    side = (
      <>
        <Panel>
          <Eq>
            μ_B = (1/B) Σ<sub>i</sub> z<sup>(i)</sup>
            <br />
            σ_B² = (1/B) Σ<sub>i</sub> (z<sup>(i)</sup> − μ_B)²
            <br />
            u<sup>(i)</sup> = (z<sup>(i)</sup> − μ_B) / √(σ_B² + ε)
          </Eq>
          <Pills options={TRACK_OPTS} value={track} onPick={setTrack} />
          <Slider id="l6-B2" label="batch size B" value={Math.log2(B)} min={1} max={7} stepSize={0.01} onChange={changeB} display={String(B)} />
          <Readout rows={[["spread of the ◆ order's u", fmt(jit.sd, 3)]]} />
        </Panel>
        <Note>Done independently for each unit. For large B the spread is about √((1 + u²/2) / B), so orders far from the mean (large |u|) jitter more. In small batches an unusual order also inflates its own batch's std, which squashes it toward the others. The slides also ask for well-shuffled data: batches made of a single day would each be normalized to a different reference.</Note>
      </>
    );
  } else {
    question = (
      <>
        The rush-hour order (z = 6) is exactly the kind the ETA network must get right. Measured against all orders it sits about 3 std above the mean, so after batch norm it should still stand out (average u at least {STAND6}) and should not jump around with its batch-mates (spread at most {GOAL6}). The GPU holds at most {GPU6} orders per batch. Pick a batch size that does all three.
      </>
    );
    stage = (
      <>
        <BatchStage rows={rowsC} bn />
        <JitterPlot jit={jitC} label={"u of the rush-hour order over 400 random batches of " + BC} />
      </>
    );
    side = (
      <>
        <Panel edge={okC ? THEME.ok : THEME.grid}>
          <Slider id="l6-BC" label="batch size B" value={BC} min={2} max={160} stepSize={1} onChange={changeBC} display={String(BC)} />
          <Readout
            rows={[
              ["average u of the rush-hour order", fmt(jitC.mean, 2)],
              ["spread of its u", fmt(jitC.sd, 3)],
              ["orders per batch", String(BC), BC > GPU6 ? THEME.warn : undefined],
            ]}
          />
          <Goal met={jitC.mean >= STAND6}>stands out: average u ≥ {STAND6}</Goal>
          <Goal met={jitC.sd <= GOAL6}>steady: spread ≤ {GOAL6}</Goal>
          <Goal met={BC <= GPU6}>fits on the GPU (B ≤ {GPU6})</Goal>
          <Hint
            hints={[
              "Start at B = 2: the rush order is always exactly +1, as unusual as anyone. A small batch squashes it, because it inflates its own batch's std. Watch its average u as B grows.",
              "Both the squashing and the jitter fade as B grows, but slowly (about 1/√B). Find where both goals hold, then check the GPU limit.",
            ]}
          />
        </Panel>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="Every batch to the same place" question={question} stage={stage} side={side} />;
}
