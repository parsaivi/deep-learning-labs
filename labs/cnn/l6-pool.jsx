import { useState, useMemo, useCallback } from "react";
import { THEME, MONO } from "./common";

// Lesson 6 - Pooling and the whole ConvNet (CNN deck, slides 114-133).
// Aha: pooling keeps "how strongly did the detector fire in this block" and throws away "exactly where".
// A parcel that moves a pixel inside a 2x2 block gives the same pooled map, the map gets 4x smaller, and every
// layer above sees twice as far. Stacking [(CONV-RELU)*N - POOL]*M - FC trades space for channels until one
// small map summarises the whole photo.

// @sim-start
const SLIDE6 = [
  [1, 1, 2, 4],
  [5, 6, 7, 8],
  [3, 2, 1, 0],
  [1, 2, 3, 4],
];
const BOX6 = [
  [1, 1, 1, 1],
  [1, 0, 0, 1],
  [1, 0, 0, 1],
  [1, 1, 1, 1],
];
const F6 = BOX6.map((r) => r.map((v) => (v ? 1 : -1)));
function parcelMap6(dr, dc) {
  const img = stamp(zeros2(11, 11), BOX6, 2 + dr, 2 + dc);
  return relu2(conv2(img, F6, -6, 1, 0));
}
function sameMap(a, b) {
  for (let i = 0; i < a.length; i++) for (let j = 0; j < a[0].length; j++) if (Math.abs(a[i][j] - b[i][j]) > 1e-9) return false;
  return true;
}
function arch6(N, M, K) {
  let side = 32;
  let C = 3;
  let rf = 1;
  let jump = 1;
  let params = 0;
  const rows = [["input", "32 × 32 × 3", ""]];
  for (let b = 0; b < M; b++) {
    const Co = K * Math.pow(2, b);
    for (let n = 0; n < N; n++) {
      params += 9 * C * Co + Co;
      C = Co;
      rf += 2 * jump;
    }
    rows.push([N + " × (conv3-ReLU)", side + " × " + side + " × " + C, "rf " + rf]);
    rf += jump;
    jump *= 2;
    side = Math.floor(side / 2);
    rows.push(["max pool 2, s2", side + " × " + side + " × " + C, "rf " + rf]);
  }
  const fcIn = side * side * C;
  params += fcIn * 2 + 2;
  rows.push(["FC → parcel / none", fcIn + " → 2", ""]);
  return { side, C, rf, fcIn, params, rows };
}
const GOALC6 = { rf: 32, params: 100000, K: 16 };
// @sim-end

function PoolStage({ a, kind, p, title, vmax, ref0 }) {
  const out = pool2(a, 2, 2, kind, p);
  const n = a.length;
  const cell = Math.min(40, 260 / n);
  const boxes = [];
  for (let i = 0; i + 2 <= n; i += 2) for (let j = 0; j + 2 <= n; j += 2) boxes.push({ r: i, c: j, w: 2, h: 2, color: THEME.muted, width: 1.2, dash: "3 2" });
  const big = n <= 4;
  const changed = ref0 ? out.map((r, i) => r.map((v, j) => Math.abs(v - ref0[i][j]) > 1e-9)) : null;
  const ob = [];
  if (changed) changed.forEach((r, i) => r.forEach((c, j) => c && ob.push({ r: i, c: j, w: 1, h: 1, color: THEME.warn, width: 2.5 })));
  return (
    <svg viewBox="0 0 620 316" style={{ width: "100%", display: "block" }} role="img" aria-label={title + ": map and its 2 × 2 pooled version"}>
      <PixelGrid cells={heatCells(a, vmax)} x={14} y={30} cell={cell} values={big || cell > 26 ? a.map((r) => r.map((v) => (v ? v : null))) : null} valueFmt={(v) => fmt(v, 0)} boxes={boxes} label={title + " " + n + " × " + n} />
      <text x={335} y={30 + (n * cell) / 2} textAnchor="middle" fontFamily={MONO} fontSize={12} fill={THEME.muted}>
        {kind === "p" ? "p-norm, p = " + fmt(p, 1) : kind} 2 × 2, s 2 →
      </text>
      <PixelGrid cells={heatCells(out, vmax)} x={400} y={30} cell={Math.min(64, 180 / out.length)} values={out.map((r) => r.map((v) => (v ? v : null)))} valueFmt={(v) => fmt(v, Number.isInteger(v) ? 0 : 2)} boxes={ob} label={"pooled " + out.length + " × " + out.length} />
      {changed && (
        <text x={14} y={310} fontFamily={MONO} fontSize={10.5} fill={THEME.warn}>
          ▢ pink box = changed vs. the unshifted parcel
        </text>
      )}
    </svg>
  );
}

export default function PoolLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [dc, setDc] = useState(0);
  const [dr, setDr] = useState(0);
  const [kind, setKind] = useState("max");
  const [p, setP] = useState(5);
  const [src, setSrc] = useState("parcel");
  const [N, setN] = useState(1);
  const [M, setM] = useState(1);
  const [K, setK] = useState(8);
  const map = useMemo(() => parcelMap6(dr, dc), [dr, dc]);
  const ref0 = useMemo(() => pool2(parcelMap6(0, 0), 2, 2, screen === 0 ? "max" : kind, p), [kind, p, screen]);
  const pooled = pool2(map, 2, 2, screen === 0 ? "max" : kind, p);
  const same = sameMap(pooled, ref0);
  const rawSame = sameMap(map, parcelMap6(0, 0));
  const pickK = useCallback((v) => setK(Number(v)), []);
  const A = useMemo(() => arch6(N, M, K), [N, M, K]);
  const okC = A.rf >= GOALC6.rf && A.params <= GOALC6.params && K >= GOALC6.K;
  const kindPills = <Pills options={[["max", "max"], ["mean", "mean"], ["p", "p-norm"]]} value={kind} onPick={setKind} />;

  let question, stage, side;
  if (screen === 0) {
    question = (
      <>
        The parcel detector's map (after ReLU) lights up where the parcel is. Shrink it by keeping only the biggest value in each 2 × 2 block. Now nudge the parcel right or down by a pixel. The raw map changes. Does the pooled map? What about a 2-pixel nudge?
      </>
    );
    stage = <PoolStage a={map} kind="max" p={p} title="detector map" vmax={6} ref0={ref0} />;
    side = (
      <Panel>
        <Slider id="c6-dc" label="nudge parcel right (px)" value={dc} min={0} max={3} stepSize={1} onChange={setDc} display={String(dc)} />
        <Slider id="c6-dr" label="nudge parcel down (px)" value={dr} min={0} max={3} stepSize={1} onChange={setDr} display={String(dr)} />
        <Readout
          rows={[
            ["raw 8 × 8 map unchanged?", rawSame ? "yes" : "no"],
            ["pooled 4 × 4 map unchanged?", same ? "◆ yes" : "✕ no", same ? THEME.ok : THEME.warn],
          ]}
        />
      </Panel>
    );
  } else if (screen === 1) {
    question = (
      <>
        Compare max, mean and p-norm pooling on the slide example and on the parcel map. Which keeps the strongest response? Which is unchanged by small nudges? What happens to the p-norm as p grows?
      </>
    );
    stage = src === "slide" ? <PoolStage a={SLIDE6} kind={kind} p={p} title="slide example" vmax={8} /> : <PoolStage a={map} kind={kind} p={p} title="detector map" vmax={6} ref0={ref0} />;
    side = (
      <Panel>
        <Pills options={[["slide", "slide 4 × 4"], ["parcel", "parcel map"]]} value={src} onPick={setSrc} />
        {kindPills}
        {kind === "p" && <Slider id="c6-p" label="p" value={p} min={1} max={20} stepSize={0.1} onChange={setP} />}
        {src === "parcel" && <Slider id="c6-dc1" label="nudge parcel right (px)" value={dc} min={0} max={3} stepSize={1} onChange={setDc} display={String(dc)} />}
        {src === "parcel" && <Readout rows={[["pooled map unchanged?", same ? "◆ yes" : "✕ no", same ? THEME.ok : THEME.warn]]} />}
        <Note>p = 1 is mean pooling (for non-negative inputs); p → ∞ approaches max. The slide's answers: max 6 8 / 3 4, mean 3.25 5.25 / 2 2.</Note>
      </Panel>
    );
  } else if (screen === 2) {
    question = (
      <>
        Name it. A <b>pooling layer</b> downsamples each activation map on its own: an N × N map with P × P windows and stride S gives (N − P)/S + 1, usually F = 2, S = 2 and no padding. It has no weights. It cuts computation and parameters, gives tolerance to small shifts, and helps against overfitting. A conv with stride 2 can do the same job (an "all-convolutional" net). The usual ConvNet: [(CONV-ReLU)×N − POOL]×M − (FC-ReLU)×K − softmax.
      </>
    );
    stage = <PoolStage a={map} kind={kind} p={p} title="detector map" vmax={6} ref0={ref0} />;
    side = (
      <Panel>
        <Eq size={13}>
          max: y = max<sub>(i,j)∈block</sub> x<sub>ij</sub>
          <br />
          p-norm: y = ((1/P²) Σ x<sub>ij</sub><sup>p</sup>)<sup>1/p</sup>
          <br />
          size: (N − P)/S + 1, depth unchanged
        </Eq>
        {kindPills}
        {kind === "p" && <Slider id="c6-p2" label="p" value={p} min={1} max={20} stepSize={0.1} onChange={setP} />}
        <Slider id="c6-dc2" label="nudge parcel right (px)" value={dc} min={0} max={3} stepSize={1} onChange={setDc} display={String(dc)} />
        <Note>Common settings: F = 2, S = 2 or F = 3, S = 2 (overlapping). The trend in modern nets is fewer pooling and FC layers, more plain conv.</Note>
      </Panel>
    );
  } else {
    question = (
      <>
        Design the full parcel checker for 32 × 32 × 3 photos: M blocks, each N conv3-ReLU layers then a 2 × 2 max pool, filters doubling every block, then one FC layer to "parcel / none". It needs at least {GOALC6.K} filters in its first layer, a final receptive field of at least {GOALC6.rf} (the whole photo), and at most {(GOALC6.params / 1000).toFixed(0)}k weights.
      </>
    );
    stage = (
      <svg viewBox={"0 0 620 " + (A.rows.length * 26 + 30)} style={{ width: "100%", display: "block" }} role="img" aria-label="Layer table of the designed network">
        {A.rows.map((r, i) => {
          const y = 20 + i * 26;
          const w = r[1].split(" × ");
          const side = Number(w[0]) || 0;
          return (
            <g key={i}>
              <rect x={6} y={y - 14} width={608} height={22} fill={i % 2 ? THEME.bg : THEME.surface} />
              <text x={14} y={y + 1} fontFamily={MONO} fontSize={12} fill={THEME.text}>
                {r[0]}
              </text>
              <text x={250} y={y + 1} fontFamily={MONO} fontSize={12} fill={THEME.highlight}>
                {r[1]}
              </text>
              <text x={420} y={y + 1} fontFamily={MONO} fontSize={12} fill={THEME.muted}>
                {r[2]}
              </text>
              {side > 0 && i < A.rows.length - 1 && <rect x={500} y={y - 10} width={Math.max(2, side * 3)} height={14} fill={rgba(THEME.accent, 0.5)} />}
            </g>
          );
        })}
      </svg>
    );
    side = (
      <Panel edge={okC ? THEME.ok : THEME.grid}>
        <Slider id="c6-n" label="conv layers per block N" value={N} min={1} max={4} stepSize={1} onChange={setN} display={String(N)} />
        <Slider id="c6-m" label="blocks M (each ends in a pool)" value={M} min={1} max={5} stepSize={1} onChange={setM} display={String(M)} />
        <Pills options={[["8", "K = 8"], ["16", "K = 16"], ["32", "K = 32"]]} value={String(K)} onPick={pickK} />
        <Readout
          rows={[
            ["final map", A.side + " × " + A.side + " × " + A.C],
            ["receptive field", String(A.rf)],
            ["weights", A.params.toLocaleString("en-US"), THEME.highlight],
          ]}
        />
        <Goal met={K >= GOALC6.K}>first layer has ≥ {GOALC6.K} filters</Goal>
        <Goal met={A.rf >= GOALC6.rf}>receptive field ≥ {GOALC6.rf}</Goal>
        <Goal met={A.params <= GOALC6.params}>≤ {GOALC6.params.toLocaleString("en-US")} weights</Goal>
        <Hint
          hints={[
            "Each pool doubles how far every later conv sees, so pooling is the cheap way to grow the receptive field. Where do most of the weights sit: early or late blocks?",
            "Late blocks have the most channels, and a conv costs 9·C_in·C_out. Try more blocks with one conv each, or three blocks with two convs each.",
          ]}
        />
      </Panel>
    );
  }
  return <LessonFrame screen={screen} onScreen={setScreen} title="Pool: keep the vote, drop the where" question={question} stage={stage} side={side} />;
}
