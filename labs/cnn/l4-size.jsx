import { useState, useMemo } from "react";
import { THEME, MONO } from "./common";

// Lesson 4 - Stride and zero padding set the output size (CNN deck, slides 63-88).
// Aha: the output size is just a count of where the filter fits. It starts at the left edge and jumps S
// cells at a time; it fits (N + 2P - F) / S + 1 times. If that isn't a whole number the last jump hangs off
// the edge. Padding with zeros adds room so the filter can be centred on edge pixels, which is how a layer
// keeps the size (P = (F - 1) / 2 at stride 1).

// @sim-start
function fits4(N, F, S, P) {
  const q = (N + 2 * P - F) / S + 1;
  return { q, n: Math.floor(q), exact: Number.isInteger(q) && q >= 1 };
}
const GOALC4 = { from: 32, to: 8 };
// @sim-end

function RowStage({ N, F, S, P, label }) {
  const total = N + 2 * P;
  const cell = Math.min(28, 560 / total);
  const x0 = 310 - (total * cell) / 2;
  const f = fits4(N, F, S, P);
  const starts = [];
  for (let s = 0; s + F <= total; s += S) starts.push({ s, ok: true });
  if (!f.exact && starts.length && starts[starts.length - 1].s + S < total) starts.push({ s: starts[starts.length - 1].s + S, ok: false });
  const shown = starts.slice(0, 14);
  const rowH = 15;
  const top = 64;
  const H = top + shown.length * rowH + 50;
  return (
    <svg viewBox={"0 0 620 " + H} style={{ width: "100%", display: "block" }} role="img" aria-label={"One row of the input with every filter position; " + f.n + " fit"}>
      <text x={10} y={16} fontFamily={MONO} fontSize={10.5} fill={THEME.muted}>
        {label || "one row of the input · dashed cells = zero padding · each bracket = one filter position"}
      </text>
      {Array.from({ length: total }).map((_, i) => {
        const pad = i < P || i >= P + N;
        return (
          <g key={i}>
            <rect x={x0 + i * cell} y={26} width={cell - 2} height={cell - 2} fill={pad ? THEME.bg : rgba(THEME.text, 0.25)} stroke={pad ? THEME.muted : THEME.grid} strokeDasharray={pad ? "3 2" : undefined} />
            {cell > 18 && (
              <text x={x0 + i * cell + cell / 2 - 1} y={26 + cell / 2 + 3} textAnchor="middle" fontFamily={MONO} fontSize={9.5} fill={pad ? THEME.muted : THEME.text}>
                {pad ? "0" : i - P + 1}
              </text>
            )}
          </g>
        );
      })}
      {shown.map((p, k) => {
        const y = top + k * rowH;
        const x1 = x0 + p.s * cell;
        const x2 = x0 + Math.min(p.s + F, total + F) * cell - 2;
        const col = p.ok ? THEME.accent : THEME.warn;
        return (
          <g key={k}>
            <path d={"M" + x1 + "," + (y - 4) + "v6h" + (x2 - x1) + "v-6"} fill="none" stroke={col} strokeWidth={2} strokeDasharray={p.ok ? undefined : "4 3"} />
            <text x={x2 + 6} y={y + 3} fontFamily={MONO} fontSize={10} fill={col}>
              {p.ok ? "out " + (k + 1) : "✕ hangs off the edge"}
            </text>
          </g>
        );
      })}
      {starts.length > shown.length && (
        <text x={10} y={H - 30} fontFamily={MONO} fontSize={10} fill={THEME.muted}>
          … {starts.length - shown.length} more positions
        </text>
      )}
      <text x={10} y={H - 10} fontFamily={MONO} fontSize={12} fill={f.exact ? THEME.ok : THEME.warn}>
        ({N} + 2·{P} − {F}) / {S} + 1 = {fmt(f.q, Number.isInteger(f.q) ? 0 : 2)} {f.exact ? "◆ fits exactly" : "✕ not a whole number"}
      </text>
    </svg>
  );
}

function LayerCtl({ id, n, F, S, P, setF, setS, setP }) {
  const f = fits4(n, F, S, P);
  return (
    <div>
      <Label>
        layer {id}: {n} → {f.exact ? f.n : fmt(f.q, 2)}
      </Label>
      <Slider id={"c4-f" + id} label="filter F" value={F} min={1} max={9} stepSize={1} onChange={setF} display={String(F)} />
      <Slider id={"c4-s" + id} label="stride S" value={S} min={1} max={4} stepSize={1} onChange={setS} display={String(S)} />
      <Slider id={"c4-p" + id} label="padding P" value={P} min={0} max={4} stepSize={1} onChange={setP} display={String(P)} />
    </div>
  );
}

export default function SizeLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [S0, setS0] = useState(1);
  const [N, setN] = useState(7);
  const [F, setF] = useState(3);
  const [S, setS] = useState(2);
  const [P, setP] = useState(0);
  const [f1, setF1] = useState(3);
  const [s1, setS1] = useState(1);
  const [p1, setP1] = useState(0);
  const [f2, setF2] = useState(3);
  const [s2, setS2] = useState(1);
  const [p2, setP2] = useState(0);
  const l1 = fits4(GOALC4.from, f1, s1, p1);
  const l2 = fits4(l1.exact ? l1.n : 0, f2, s2, p2);
  const okC = l1.exact && l2.exact && l2.n === GOALC4.to && f1 >= 3 && f2 >= 3;

  let question, stage, side;
  if (screen === 0) {
    question = (
      <>
        A 3-wide filter scans a 7-pixel row of the doorstep photo. With stride 1 it moves one pixel at a time. Raise the stride: how many outputs do you get with stride 2? With stride 3, where would the last jump land?
      </>
    );
    stage = <RowStage N={7} F={3} S={S0} P={0} />;
    side = (
      <Panel>
        <Slider id="c4-s0" label="stride" value={S0} min={1} max={3} stepSize={1} onChange={setS0} display={String(S0)} />
        <Readout rows={[["filter positions that fit", String(fits4(7, 3, S0, 0).n)]]} />
        <Note>The same count happens along rows and along columns, so a 7 × 7 input with stride 2 gives a 3 × 3 output.</Note>
      </Panel>
    );
  } else if (screen === 1) {
    question = (
      <>
        Add zero padding around the row. Which padding keeps the output as long as the input for a 3-wide filter at stride 1? For a 5-wide one? Find a setting where the filter does not fit a whole number of times, then fix it with padding alone.
      </>
    );
    stage = <RowStage N={N} F={F} S={S} P={P} />;
    side = (
      <Panel>
        <Slider id="c4-n" label="input size N" value={N} min={4} max={15} stepSize={1} onChange={setN} display={String(N)} />
        <Slider id="c4-f" label="filter size F" value={F} min={1} max={7} stepSize={1} onChange={setF} display={String(F)} />
        <Slider id="c4-s" label="stride S" value={S} min={1} max={4} stepSize={1} onChange={setS} display={String(S)} />
        <Slider id="c4-p" label="zero padding P" value={P} min={0} max={3} stepSize={1} onChange={setP} display={String(P)} />
        <Readout rows={[["output size", fits4(N, F, S, P).exact ? String(fits4(N, F, S, P).n) : "doesn't fit", THEME.highlight]]} />
      </Panel>
    );
  } else if (screen === 2) {
    question = (
      <>
        Name it. A conv layer's spatial output is (N + 2P − F)/S + 1 on each axis. <b>Stride</b> S &gt; 1 shrinks the map by about S and saves computation, at some risk of losing detail. <b>Zero padding</b> keeps the size: F = 3, P = 1 or F = 5, P = 2 at stride 1. Without padding a stack shrinks every layer (32 → 28 → 24 with 5 × 5 filters), which is usually too fast.
      </>
    );
    stage = <RowStage N={N} F={F} S={S} P={P} />;
    side = (
      <Panel>
        <Eq>
          W₂ = (W₁ − F + 2P)/S + 1
          <br />
          H₂ = (H₁ − F + 2P)/S + 1, &nbsp;D₂ = K
          <br />
          same size: S = 1, P = (F − 1)/2
        </Eq>
        <Slider id="c4-n2" label="input size N" value={N} min={4} max={15} stepSize={1} onChange={setN} display={String(N)} />
        <Slider id="c4-f2" label="filter size F" value={F} min={1} max={7} stepSize={1} onChange={setF} display={String(F)} />
        <Slider id="c4-s2" label="stride S" value={S} min={1} max={4} stepSize={1} onChange={setS} display={String(S)} />
        <Slider id="c4-p2" label="zero padding P" value={P} min={0} max={3} stepSize={1} onChange={setP} display={String(P)} />
        <Note>Common settings: K a power of 2; F = 3, S = 1, P = 1; F = 5, S = 1, P = 2; F = 1, S = 1, P = 0.</Note>
      </Panel>
    );
  } else {
    question = (
      <>
        The parcel checker takes 32 × 32 photos and its last stage expects an 8 × 8 map. Design two conv layers (each filter at least 3 wide) that take 32 down to exactly 8, with every filter position fitting exactly.
      </>
    );
    stage = (
      <>
        <RowStage N={GOALC4.from} F={f1} S={s1} P={p1} label={"layer 1 on one row of 32 pixels"} />
        {l1.exact ? <RowStage N={l1.n} F={f2} S={s2} P={p2} label={"layer 2 on one row of " + l1.n} /> : <Note>Layer 2 waits until layer 1 fits exactly.</Note>}
      </>
    );
    side = (
      <Panel edge={okC ? THEME.ok : THEME.grid}>
        <LayerCtl id={1} n={GOALC4.from} F={f1} S={s1} P={p1} setF={setF1} setS={setS1} setP={setP1} />
        <LayerCtl id={2} n={l1.exact ? l1.n : 0} F={f2} S={s2} P={p2} setF={setF2} setS={setS2} setP={setP2} />
        <Goal met={l1.exact && l2.exact}>both layers fit exactly</Goal>
        <Goal met={f1 >= 3 && f2 >= 3}>both filters at least 3 wide</Goal>
        <Goal met={l2.exact && l2.n === GOALC4.to}>final size 8</Goal>
        <Hint
          hints={[
            "32 → 8 is a factor of 4. Stride 1 barely changes the size, so at least one layer needs a stride above 1.",
            "Try halving twice, 32 → 16 → 8. With stride 2 on an even row, N + 2P − F must be even, so an odd F never fits exactly. With F = 4, which padding gives 16?",
          ]}
        />
      </Panel>
    );
  }
  return <LessonFrame screen={screen} onScreen={setScreen} title="Stride and padding set the size" question={question} stage={stage} side={side} />;
}
