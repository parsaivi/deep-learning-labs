import { useState, useMemo, useCallback } from "react";
import { THEME, MONO } from "./common";

// Lesson 1 - Shift invariance: scan one small detector everywhere (CNN deck, slides 5-15).
// Aha: a fully connected unit has one weight per pixel, so it learns "a parcel HERE". Move the parcel a
// pixel and it sees a different input. Use one small set of weights (local) and apply the same weights at
// every position (shared): the map of scores moves with the parcel, and its maximum doesn't care where it is.

// @sim-start
const N1 = 12;
const PARCEL1 = [
  [1, 1, 1, 1],
  [1, 0, 0, 1],
  [1, 0, 0, 1],
  [1, 1, 1, 1],
];
function filt1(u) {
  return PARCEL1.map((r) => r.map((v) => (v ? 1 : u)));
}
function photo1(r, c) {
  return stamp(zeros2(N1, N1), PARCEL1, r, c);
}
// the fully connected unit was trained on parcels at row 4, col 4: weights = the filter placed there, 0 elsewhere
const HOME1 = [4, 4];
function fcScore1(img, u) {
  const f = filt1(u);
  let s = 0;
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) s += f[i][j] * img[HOME1[0] + i][HOME1[1] + j];
  return s;
}
function scan1(img, u) {
  return conv2(img, filt1(u), 0, 1, 0);
}
// challenge photos
const CHAL1 = (function () {
  const L = [];
  L.push({ name: "parcel, porch", has: true, img: photo1(1, 2) });
  const p2 = photo1(6, 7);
  p2[8][8] = 1;
  L.push({ name: "parcel, leaf on top", has: true, img: p2 });
  const p3 = photo1(7, 1);
  p3[10][4] = 0;
  L.push({ name: "parcel, corner hidden", has: true, img: p3 });
  L.push({ name: "parcel, by the door", has: true, img: photo1(3, 6) });
  L.push({ name: "parcel, far corner", has: true, img: photo1(8, 8) });
  const mat = zeros2(N1, N1);
  stamp(mat, [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], 5, 3);
  L.push({ name: "doormat, no parcel", has: false, img: mat });
  const bench = zeros2(N1, N1);
  for (let j = 2; j < 10; j++) {
    bench[4][j] = 1;
    bench[7][j] = 1;
  }
  L.push({ name: "bench, no parcel", has: false, img: bench });
  const frame = zeros2(N1, N1);
  stamp(frame, [[1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1], [1, 1, 1, 1, 1, 1]], 3, 3);
  L.push({ name: "window frame, no parcel", has: false, img: frame });
  return L;
})();
function judge1(u, T) {
  return CHAL1.map((p) => {
    const s = max2(scan1(p.img, u)).m;
    return { s, yes: s > T, ok: s > T === p.has };
  });
}
// @sim-end

function Photo({ img, x, y, cell, onCell, boxes, label }) {
  return <PixelGrid cells={grayCells(img)} x={x} y={y} cell={cell} onCell={onCell} boxes={boxes} label={label} />;
}

function ShiftStage({ img, mode, u, onCell }) {
  const cell = 22;
  const boxes = [{ r: HOME1[0], c: HOME1[1], w: 4, h: 4, color: THEME.highlight, dash: "5 3" }];
  const map = scan1(img, u);
  const best = max2(map);
  if (mode === "scan") boxes.push({ r: best.at[0], c: best.at[1], w: 4, h: 4, color: THEME.ok, width: 2.5 });
  return (
    <svg viewBox="0 0 620 300" style={{ width: "100%", display: "block", touchAction: "none" }} role="img" aria-label="Doorstep photo; click to move the parcel">
      <Photo img={img} x={20} y={24} cell={cell} onCell={onCell} boxes={boxes} label="doorstep photo · click to move the parcel" />
      {mode === "scan" ? (
        <g>
          <PixelGrid cells={heatCells(map, 12)} x={330} y={24} cell={26} values={map.map((r) => r.map((v) => (v >= 6 ? v : null)))} valueFmt={(v) => fmt(v, 0)} boxes={[{ r: best.at[0], c: best.at[1], w: 1, h: 1, color: THEME.ok, width: 2.5 }]} label="score at every position (9 × 9 map)" />
        </g>
      ) : (
        <g>
          <text x={330} y={40} fontFamily={MONO} fontSize={11} fill={THEME.muted}>
            fully connected unit: one weight per pixel
          </text>
          <text x={330} y={58} fontFamily={MONO} fontSize={11} fill={THEME.muted}>
            nonzero only inside the dashed box ⬚
          </text>
          <text x={330} y={76} fontFamily={MONO} fontSize={11} fill={THEME.muted}>
            (where the training parcels sat)
          </text>
        </g>
      )}
      <text x={20} y={296} fontFamily={MONO} fontSize={10} fill={THEME.muted}>
        ⬚ dashed = where the training parcels were{mode === "scan" ? " · solid = best-scoring window" : ""}
      </text>
    </svg>
  );
}

export default function ShiftLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [pos, setPos] = useState([4, 4]);
  const [mode, setMode] = useState("fc");
  const [u, setU] = useState(0);
  const [T, setT] = useState(6);
  const img = useMemo(() => photo1(pos[0], pos[1]), [pos]);
  const place = useCallback((i, j) => setPos([clamp(i - 1, 0, N1 - 4), clamp(j - 1, 0, N1 - 4)]), []);
  const md = screen === 0 ? "fc" : mode;
  const fc = fcScore1(img, -1);
  const sc = max2(scan1(img, -1)).m;
  const res = useMemo(() => judge1(u, T), [u, T]);
  const okC = res.every((r) => r.ok);

  let question, stage, side;
  if (screen === 0) {
    question = (
      <>
        Couriers upload a photo at the door. A fully connected unit was trained to say "parcel" and scores 12 for the training photos, where the parcel always sat in the dashed box. Click to move the parcel one pixel, then anywhere. What does the unit say now? Would more neurons fix it?
      </>
    );
    stage = <ShiftStage img={img} mode="fc" u={-1} onCell={place} />;
    side = (
      <Panel>
        <Readout
          rows={[
            ["parcel at (row, col)", pos[0] + ", " + pos[1]],
            ["fully connected score", fmt(fc, 0) + " / 12", THEME.highlight],
          ]}
        />
        <Note>Weights: +1 on the parcel's edge, −1 on its middle, placed in the dashed box. Photos are 12 × 12 made-up pixels.</Note>
      </Panel>
    );
  } else if (screen === 1) {
    question = (
      <>
        Keep only the 16 weights of the dashed box, and slide them over every position of the photo, computing the same score at each. Move the parcel around. What happens to the map of scores? To its largest value?
      </>
    );
    stage = <ShiftStage img={img} mode={md} u={-1} onCell={place} />;
    side = (
      <Panel>
        <Pills options={[["fc", "fully connected (144 weights)"], ["scan", "scan 16 shared weights"]]} value={mode} onPick={setMode} />
        <Readout
          rows={[
            ["fully connected score", fmt(fc, 0) + " / 12"],
            ["scan: best score anywhere", fmt(sc, 0) + " / 12", THEME.highlight],
          ]}
        />
        <Note>The map moves with the parcel (it is equivariant); its maximum stays put (it is invariant). An MLP would need training photos with the parcel in every spot to learn the same.</Note>
      </Panel>
    );
  } else if (screen === 2) {
    question = (
      <>
        Name it. A <b>convolutional layer</b> makes two changes to a fully connected one. <b>Locality</b>: each output looks at a small patch only. <b>Parameter sharing</b>: every position uses the same weights, a filter. The filter is a perceptron that scans the image; a high score means the patch correlates with the weights. Taking the maximum of the map asks "is there a parcel anywhere?".
      </>
    );
    stage = <ShiftStage img={img} mode="scan" u={-1} onCell={place} />;
    side = (
      <Panel>
        <Eq size={13}>
          z(i, j) = Σ<sub>k,l</sub> w(k, l) · x(i + k, j + l) + b
          <br />
          parcel present ⇔ max<sub>i,j</sub> z(i, j) &gt; T
        </Eq>
        <Readout
          rows={[
            ["fully connected weights", "12 × 12 = 144"],
            ["shared filter weights", "4 × 4 = 16"],
            ["positions it is used at", "9 × 9 = 81"],
          ]}
        />
        <Note>The same idea finds the word "Welcome" anywhere in a recording: scan one small detector along time.</Note>
      </Panel>
    );
  } else {
    question = (
      <>
        Eight new photos. Scan the 4 × 4 filter (+1 on the edge, u in the middle) and flag a photo when its best score passes T. Set u and T so every parcel is flagged and nothing else is.
      </>
    );
    const cell = 12;
    stage = (
      <svg viewBox="0 0 620 420" style={{ width: "100%", display: "block" }} role="img" aria-label="Eight photos with their best scan score and verdict">
        {CHAL1.map((p, k) => {
          const x = 10 + (k % 4) * 152;
          const y = 26 + Math.floor(k / 4) * 204;
          const r = res[k];
          return (
            <g key={k}>
              <Photo img={p.img} x={x} y={y} cell={cell} label={p.name} />
              <text x={x} y={y + 158} fontFamily={MONO} fontSize={11} fill={THEME.text}>
                best {fmt(r.s, 2)} → {r.yes ? "■ parcel" : "□ none"}
              </text>
              <text x={x} y={y + 173} fontFamily={MONO} fontSize={11} fill={r.ok ? THEME.ok : THEME.warn}>
                {r.ok ? "◆ right" : "✕ wrong"}
              </text>
            </g>
          );
        })}
      </svg>
    );
    side = (
      <Panel edge={okC ? THEME.ok : THEME.grid}>
        <Slider id="c1-u" label="middle weight u" value={u} min={-1} max={1} stepSize={0.01} onChange={setU} />
        <Slider id="c1-t" label="threshold T" value={T} min={0} max={16} stepSize={0.05} onChange={setT} />
        <Goal met={okC}>all 8 photos right ({res.filter((r) => r.ok).length} / 8)</Goal>
        <Hint
          hints={[
            "With u = 0 the filter only looks at the edge. What does a solid doormat score then, compared with a parcel?",
            "A negative u charges for anything in the middle, which a parcel lacks and a doormat has. Then put T between the weakest parcel and the strongest non-parcel.",
          ]}
        />
      </Panel>
    );
  }
  return <LessonFrame screen={screen} onScreen={setScreen} title="Move the parcel, keep the answer" question={question} stage={stage} side={side} />;
}
