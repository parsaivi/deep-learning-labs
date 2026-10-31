import { useState, useMemo, useCallback } from "react";
import { THEME, MONO } from "./common";

// Lesson 2 - What a filter computes (CNN deck, slides 15-17, 23-28, 40-42).
// Aha: at each position the filter is a tiny perceptron: multiply the patch under it cell by cell with
// the weights, add them and the bias. The output map is a picture of "where does the image look like my
// weights?". Edit the weights and you choose what the map lights up on.

// @sim-start
const IMG2 = [
  [1, 1, 1, 0, 0],
  [0, 1, 1, 1, 0],
  [0, 0, 1, 1, 1],
  [0, 0, 1, 1, 0],
  [0, 1, 1, 0, 0],
];
const SLIDE2 = [
  [1, 0, 1],
  [0, 1, 0],
  [1, 0, 1],
];
const BOX2 = [
  [1, 1, 1, 1],
  [1, 0, 0, 1],
  [1, 0, 0, 1],
  [1, 1, 1, 1],
];
const PHOTO2 = (function () {
  const p = zeros2(10, 10);
  stamp(p, BOX2, 0, 1);
  stamp(p, BOX2, 5, 5);
  stamp(p, [[1, 1, 1], [1, 1, 1], [1, 1, 1]], 6, 0);
  for (let j = 6; j < 10; j++) p[1][j] = 1;
  return p;
})();
// targets of the challenge: windows whose top-left cell is a parcel's top-left corner
const TARGET2 = [
  [0, 1],
  [5, 5],
];
function check2(f, b) {
  const out = conv2(PHOTO2, f, b, 1, 0);
  let hits = 0;
  let extra = 0;
  for (let i = 0; i < out.length; i++)
    for (let j = 0; j < out[0].length; j++) {
      const t = TARGET2.some((q) => q[0] === i && q[1] === j);
      if (out[i][j] > 0 && t) hits++;
      if (out[i][j] > 0 && !t) extra++;
    }
  return { out, hits, extra };
}
// @sim-end

const PRESETS2 = {
  slide: SLIDE2,
  vert: [
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1],
  ],
  horiz: [
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1],
  ],
  blank: [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ],
};
function cycle(v) {
  return v === 0 ? 1 : v === 1 ? -1 : 0;
}

function FilterEditor({ f, onCell, x, y }) {
  return (
    <PixelGrid
      cells={heatCells(f, 1)}
      x={x}
      y={y}
      cell={34}
      values={f}
      valueFmt={(v) => (v > 0 ? "+" + v : String(v))}
      onCell={onCell}
      label="filter (click cells)"
    />
  );
}

export default function FilterLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [sel, setSel] = useState([0, 0]);
  const [f, setF] = useState(PRESETS2.vert);
  const [b, setB] = useState(0);
  const [relu, setRelu] = useState(false);
  const [fC, setFC] = useState(PRESETS2.blank);
  const [bC, setBC] = useState(0);
  const pickOut = useCallback((i, j) => setSel([i, j]), []);
  const edit = useCallback((i, j) => setF((g) => g.map((r, a) => r.map((v, c) => (a === i && c === j ? cycle(v) : v)))), []);
  const editC = useCallback((i, j) => setFC((g) => g.map((r, a) => r.map((v, c) => (a === i && c === j ? cycle(v) : v)))), []);
  const preset = useCallback((k) => setF(PRESETS2[k]), []);
  const out0 = useMemo(() => conv2(IMG2, SLIDE2, 0, 1, 0), []);
  const map = useMemo(() => {
    const m = conv2(PHOTO2, f, b, 1, 0);
    return relu ? relu2(m) : m;
  }, [f, b, relu]);
  const chk = useMemo(() => check2(fC, bC), [fC, bC]);
  const okC = chk.hits === TARGET2.length && chk.extra === 0;

  let question, stage, side;
  if (screen === 0) {
    const [si, sj] = sel;
    const prod = SLIDE2.map((r, k) => r.map((w, l) => w * IMG2[si + k][sj + l]));
    const sum = prod.flat().reduce((p, v) => p + v, 0);
    question = (
      <>
        A 5 × 5 black-and-white patch and a 3 × 3 grid of weights. Put the weights over one 3 × 3 window of the patch, multiply cell by cell, add up: that's one output number. Click each output cell. Which window scores highest, and what does that window have in common with the weights?
      </>
    );
    stage = (
      <svg viewBox="0 0 620 260" style={{ width: "100%", display: "block", touchAction: "none" }} role="img" aria-label="Input patch, filter, output map; click an output cell">
        <PixelGrid cells={grayCells(IMG2)} x={14} y={30} cell={36} values={IMG2} boxes={[{ r: si, c: sj, w: 3, h: 3, color: THEME.highlight, width: 3 }]} label="input patch" />
        <PixelGrid cells={heatCells(SLIDE2, 1)} x={226} y={30} cell={30} values={SLIDE2} label="filter" />
        <PixelGrid cells={heatCells(prod, 1)} x={226} y={150} cell={22} values={prod} label="products" />
        <PixelGrid cells={heatCells(out0, 5)} x={370} y={30} cell={50} values={out0} boxes={[{ r: si, c: sj, w: 1, h: 1, color: THEME.highlight, width: 3 }]} onCell={pickOut} label="output map (click a cell)" />
        <text x={370} y={208} fontFamily={MONO} fontSize={12} fill={THEME.highlight}>
          sum of products = {sum}
        </text>
      </svg>
    );
    side = (
      <Panel>
        <Readout
          rows={[
            ["window at (row, col)", si + ", " + sj],
            ["output", String(sum), THEME.highlight],
          ]}
        />
        <Note>This is the example from the slides: binary pixels, the "X" filter, bias 0.</Note>
      </Panel>
    );
  } else if (screen === 1 || screen === 2) {
    question =
      screen === 1 ? (
        <>
          Now a 10 × 10 doorstep photo: two parcels, a doormat and a railing. Try the presets, then edit the weights. Can you light up only the left edges of things? Only the bottom edges? What does the bias do to the map? What does ReLU (keep positives, zero the rest) change?
        </>
      ) : (
        <>
          Name it. A <b>convolution</b> (strictly a cross-correlation): slide the filter over every position, and at each one take the dot product of filter and patch plus a bias. The result is an <b>activation map</b>, usually passed through ReLU. The filter is a perceptron with a small <b>receptive field</b>; in a trained network the first-layer filters end up as edge and colour-blob detectors, like the ones you just built.
        </>
      );
    stage = (
      <svg viewBox="0 0 620 300" style={{ width: "100%", display: "block", touchAction: "none" }} role="img" aria-label="Photo, editable filter and the activation map">
        <PixelGrid cells={grayCells(PHOTO2)} x={10} y={24} cell={22} label="doorstep photo 10 × 10" />
        <FilterEditor f={f} onCell={edit} x={250} y={24} />
        <PixelGrid cells={heatCells(map, 4)} x={400} y={24} cell={26} values={map.map((r) => r.map((v) => (Math.abs(v) >= 0.5 ? v : null)))} valueFmt={(v) => fmt(v, v % 1 ? 1 : 0)} label={"activation map 8 × 8" + (relu ? " · ReLU" : "")} />
        <text x={10} y={290} fontFamily={MONO} fontSize={10} fill={THEME.muted}>
          map: blue = positive, pink = negative, stronger = darker
        </text>
      </svg>
    );
    side = (
      <Panel>
        {screen === 2 && (
          <Eq size={13}>
            z(i, j) = b + Σ<sub>k=1..F</sub> Σ<sub>l=1..F</sub> w(k, l) · x(i + k − 1, j + l − 1)
            <br />
            a(i, j) = ReLU(z(i, j)) · map side = N − F + 1
          </Eq>
        )}
        <Label>presets</Label>
        <Pills options={[["slide", "slide X"], ["vert", "left edge"], ["horiz", "bottom edge"], ["blank", "blank"]]} value={null} onPick={preset} />
        <Slider id={"c2-b" + screen} label="bias b" value={b} min={-6} max={6} stepSize={0.5} onChange={setB} />
        <Toggle label="ReLU" value={relu} onChange={setRelu} />
        <Note>The map is 8 × 8, not 10 × 10: a 3 × 3 filter fits in 10 − 3 + 1 = 8 places per row.</Note>
      </Panel>
    );
  } else {
    question = (
      <>
        Build a "parcel top-left corner" detector. After ReLU, the map should be positive at exactly the two cells marked ◇ (where a window's top-left is a parcel's top-left corner) and zero everywhere else, doormat and railing included.
      </>
    );
    const marks = TARGET2.map((q) => ({ r: q[0], c: q[1], w: 1, h: 1, color: THEME.highlight, width: 2.5, dash: "3 2" }));
    const shown = relu2(chk.out);
    stage = (
      <svg viewBox="0 0 620 300" style={{ width: "100%", display: "block", touchAction: "none" }} role="img" aria-label="Photo, your filter, and the ReLU map with the two target cells">
        <PixelGrid cells={grayCells(PHOTO2)} x={10} y={24} cell={22} boxes={TARGET2.map((q) => ({ r: q[0], c: q[1], w: 3, h: 3, color: THEME.highlight, dash: "3 2" }))} label="photo · dashed = target windows" />
        <FilterEditor f={fC} onCell={editC} x={250} y={24} />
        <PixelGrid cells={heatCells(shown, 4)} x={400} y={24} cell={26} values={shown.map((r) => r.map((v) => (v > 0 ? v : null)))} valueFmt={(v) => fmt(v, v % 1 ? 1 : 0)} boxes={marks} label="ReLU map · ◇ dashed = targets" />
      </svg>
    );
    side = (
      <Panel edge={okC ? THEME.ok : THEME.grid}>
        <Slider id="c2-bc" label="bias b" value={bC} min={-9} max={3} stepSize={0.5} onChange={setBC} />
        <Readout
          rows={[
            ["targets lit", chk.hits + " / 2"],
            ["other cells lit", String(chk.extra), chk.extra ? THEME.warn : THEME.muted],
          ]}
        />
        <Goal met={okC}>exactly the two corners</Goal>
        <Hint
          hints={[
            "Draw the corner itself: what does a parcel's top-left 3 × 3 window look like? Put +1 where it is ink, −1 where it is empty.",
            "With that filter a perfect corner scores 5. The doormat's windows also have lots of ink. Lower the bias until only scores of 5 survive the ReLU.",
          ]}
        />
      </Panel>
    );
  }
  return <LessonFrame screen={screen} onScreen={setScreen} title="A filter is a perceptron that scans" question={question} stage={stage} side={side} />;
}
