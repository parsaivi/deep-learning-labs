import { useState, useMemo } from "react";
import { THEME, MONO } from "./common";

// Lesson 3 - Filters span every input channel; one filter = one output map (CNN deck, slides 18-37, 43-62, 84-90).
// Aha: a colour photo is three stacked maps, and a filter always reaches through all of them: at every
// position it sums over rows, columns AND channels. In grey a cardboard parcel and the pavement look the same;
// a filter that weighs red, green and blue differently tells them apart. K filters give K maps, and the
// layer's weights are K x (F x F x C_in + 1), however big the image.

// @sim-start
const COL3 = {
  pave: [0.45, 0.45, 0.45],
  parcel: [0.7, 0.5, 0.2],
  cone: [0.95, 0.4, 0.05],
  mat: [0.4, 0.25, 0.2],
  grass: [0.3, 0.6, 0.25],
  bin: [0.2, 0.3, 0.7],
};
const NAMES3 = { pave: "pavement", parcel: "parcel", cone: "traffic cone", mat: "doormat", grass: "grass", bin: "blue bin" };
const SCENE3 = (function () {
  const s = [];
  for (let i = 0; i < 10; i++) s.push(new Array(10).fill("pave"));
  const put = (k, r, c, h, w) => {
    for (let i = r; i < r + h; i++) for (let j = c; j < c + w; j++) s[i][j] = k;
  };
  put("grass", 0, 0, 10, 2);
  put("mat", 6, 3, 3, 4);
  put("parcel", 2, 4, 3, 3);
  put("bin", 1, 8, 4, 2);
  put("cone", 7, 8, 2, 1);
  put("cone", 6, 8, 1, 1);
  return s;
})();
function map3(w, b) {
  return SCENE3.map((r) => r.map((k) => b + w[0] * COL3[k][0] + w[1] * COL3[k][1] + w[2] * COL3[k][2]));
}
function judge3(w, b) {
  const out = {};
  for (const k of Object.keys(COL3)) out[k] = b + w[0] * COL3[k][0] + w[1] * COL3[k][1] + w[2] * COL3[k][2];
  const ok = out.parcel > 0 && Object.keys(COL3).every((k) => k === "parcel" || out[k] <= 0);
  return { out, ok };
}
// @sim-end

function rgbCss(c) {
  return "rgb(" + c.map((v) => Math.round(v * 255)).join(",") + ")";
}
function ChannelStage({ w, b, relu, gray }) {
  const photo = SCENE3.map((r) => r.map((k) => (gray ? rgbCss([1, 1, 1].map(() => (COL3[k][0] + COL3[k][1] + COL3[k][2]) / 3)) : rgbCss(COL3[k]))));
  const chans = [0, 1, 2].map((c) => SCENE3.map((r) => r.map((k) => COL3[k][c])));
  const m = map3(w, b);
  const shown = relu ? relu2(m) : m;
  const vmax = Math.max(0.3, Math.max.apply(null, shown.flat().map(Math.abs)));
  const cname = ["R", "G", "B"];
  return (
    <svg viewBox="0 0 620 300" style={{ width: "100%", display: "block" }} role="img" aria-label="Colour photo, its three channels, and the filter's output map">
      <PixelGrid cells={photo} x={10} y={24} cell={18} label={gray ? "photo, in grey" : "photo 10 × 10 × 3"} />
      {chans.map((ch, c) => (
        <g key={c}>
          <PixelGrid cells={grayCells(ch)} x={210 + c * 66} y={24} cell={6} label={cname[c]} />
          <text x={210 + c * 66 + 30} y={100} textAnchor="middle" fontFamily={MONO} fontSize={11} fill={THEME.highlight}>
            × {fmt(w[c], 2)}
          </text>
        </g>
      ))}
      <text x={210} y={124} fontFamily={MONO} fontSize={10.5} fill={THEME.muted}>
        sum over channels + b ({fmt(b, 2)})
      </text>
      <text x={300} y={150} textAnchor="middle" fontFamily={MONO} fontSize={18} fill={THEME.muted}>
        ↓
      </text>
      <PixelGrid cells={heatCells(shown, vmax)} x={410} y={24} cell={18} label={"output map 10 × 10" + (relu ? " · ReLU" : "")} />
      <text x={10} y={232} fontFamily={MONO} fontSize={10.5} fill={THEME.muted}>
        map: blue = positive, pink = negative
      </text>
      {Object.keys(COL3).map((k, i) => (
        <g key={k}>
          <rect x={10 + (i % 3) * 200} y={244 + Math.floor(i / 3) * 24} width={14} height={14} fill={rgbCss(COL3[k])} stroke={THEME.grid} />
          <text x={30 + (i % 3) * 200} y={255 + Math.floor(i / 3) * 24} fontFamily={MONO} fontSize={11} fill={THEME.text}>
            {NAMES3[k]} {fmt(b + w[0] * COL3[k][0] + w[1] * COL3[k][1] + w[2] * COL3[k][2], 2)}
          </text>
        </g>
      ))}
    </svg>
  );
}

export default function ChannelsLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [w, setW] = useState([0.33, 0.33, 0.33]);
  const [b, setB] = useState(-0.4);
  const [relu, setRelu] = useState(true);
  const [K, setK] = useState(10);
  const [F, setF] = useState(5);
  const [Cin, setCin] = useState(3);
  const [N, setN] = useState(32);
  const [wc, setWc] = useState([0, 0, 0]);
  const [bc, setBc] = useState(0);
  const sR = useMemo(() => (v) => setW((a) => [v, a[1], a[2]]), []);
  const sG = useMemo(() => (v) => setW((a) => [a[0], v, a[2]]), []);
  const sB = useMemo(() => (v) => setW((a) => [a[0], a[1], v]), []);
  const cR = useMemo(() => (v) => setWc((a) => [v, a[1], a[2]]), []);
  const cG = useMemo(() => (v) => setWc((a) => [a[0], v, a[2]]), []);
  const cB = useMemo(() => (v) => setWc((a) => [a[0], a[1], v]), []);
  const jC = judge3(wc, bc);

  let question, stage, side;
  if (screen === 0) {
    question = (
      <>
        Grey photos first: the filter averages R, G and B equally. Move the bias to make the map light up on the parcel only. Can you? Why do the pavement and the cone keep coming along?
      </>
    );
    stage = <ChannelStage w={[1 / 3, 1 / 3, 1 / 3]} b={b} relu gray />;
    side = (
      <Panel>
        <Slider id="c3-b0" label="bias b" value={b} min={-1} max={0.2} stepSize={0.005} onChange={setB} />
        <Note>Each number under the swatches is that colour's output, w·(R, G, B) + b. Positive means the ReLU map lights up there. Colours are made-up.</Note>
      </Panel>
    );
  } else if (screen === 1) {
    question = (
      <>
        Now the filter has a separate weight for each colour channel. It still looks at one pixel position, but it reaches through all three channels there. Which weights make cardboard stand out from the pavement? From the orange cone? From the brown doormat?
      </>
    );
    stage = <ChannelStage w={w} b={b} relu={relu} />;
    side = (
      <Panel>
        <Slider id="c3-r" label="weight on R" value={w[0]} min={-3} max={3} stepSize={0.01} onChange={sR} />
        <Slider id="c3-g" label="weight on G" value={w[1]} min={-3} max={3} stepSize={0.01} onChange={sG} />
        <Slider id="c3-bl" label="weight on B" value={w[2]} min={-3} max={3} stepSize={0.01} onChange={sB} />
        <Slider id="c3-bb" label="bias b" value={b} min={-4} max={2} stepSize={0.01} onChange={setB} />
        <Toggle label="ReLU" value={relu} onChange={setRelu} />
      </Panel>
    );
  } else if (screen === 2) {
    const P = Math.floor((F - 1) / 2);
    const out = Math.floor((N - F + 2 * P) / 1) + 1;
    const params = K * (F * F * Cin + 1);
    question = (
      <>
        Name it. Filters always extend the full depth of the input: a filter on a 32 × 32 × 3 image is F × F × 3, and each output is one F·F·3-dimensional dot product plus a bias. One filter makes one activation map; K filters make K maps, stacked into an output of depth K that the next layer reads as its channels. Your colour filter was the special case F = 1: a <b>1 × 1 convolution</b>, a neuron over the depth at one pixel.
      </>
    );
    stage = <ChannelStage w={w} b={b} relu={relu} />;
    side = (
      <Panel>
        <Eq size={12.5}>
          z(s, i, j) = b(s) + Σ<sub>p</sub> Σ<sub>k</sub> Σ<sub>l</sub> w(s, p, k, l) · a(p, i + k, j + l)
          <br />
          params = K · (F · F · C<sub>in</sub> + 1)
        </Eq>
        <Slider id="c3-n" label="input side N (pixels)" value={N} min={8} max={64} stepSize={1} onChange={setN} display={String(N)} />
        <Slider id="c3-c" label="input channels C_in" value={Cin} min={1} max={64} stepSize={1} onChange={setCin} display={String(Cin)} />
        <Slider id="c3-f" label="filter size F (odd, padding (F−1)/2)" value={F} min={1} max={11} stepSize={2} onChange={setF} display={F + " × " + F} />
        <Slider id="c3-k" label="number of filters K" value={K} min={1} max={64} stepSize={1} onChange={setK} display={String(K)} />
        <Readout
          rows={[
            ["input volume", N + " × " + N + " × " + Cin],
            ["one filter", F + " × " + F + " × " + Cin + " + 1 bias"],
            ["output volume", out + " × " + out + " × " + K, THEME.highlight],
            ["learnable weights", params.toLocaleString("en-US"), THEME.highlight],
            ["a dense layer instead", (N * N * Cin * out * out * K).toExponential(1)],
          ]}
        />
        <Note>Slide example: 32 × 32 × 3, ten 5 × 5 filters, pad 2 → 32 × 32 × 10 and 10 · (75 + 1) = 760 weights. The count doesn't depend on N.</Note>
      </Panel>
    );
  } else {
    question = (
      <>
        Build the cardboard detector: a 1 × 1 × 3 filter plus bias whose ReLU map is positive on every parcel pixel and zero on everything else: pavement, cone, doormat, grass and bin.
      </>
    );
    stage = <ChannelStage w={wc} b={bc} relu />;
    side = (
      <Panel edge={jC.ok ? THEME.ok : THEME.grid}>
        <Slider id="c3-cr" label="weight on R" value={wc[0]} min={-3} max={3} stepSize={0.01} onChange={cR} />
        <Slider id="c3-cg" label="weight on G" value={wc[1]} min={-3} max={3} stepSize={0.01} onChange={cG} />
        <Slider id="c3-cb" label="weight on B" value={wc[2]} min={-3} max={3} stepSize={0.01} onChange={cB} />
        <Slider id="c3-cbb" label="bias b" value={bc} min={-4} max={2} stepSize={0.01} onChange={setBc} />
        <Goal met={jC.out.parcel > 0}>parcel output above 0</Goal>
        <Goal met={Object.keys(COL3).every((k) => k === "parcel" || jC.out[k] <= 0)}>every other colour at or below 0</Goal>
        <Hint
          hints={[
            "Compare the colour values. In grey, cardboard and the cone are the same. Cardboard has more green than the cone, more red than the pavement, more of everything than the doormat. One weight can't do all of that.",
            "Keep all three weights positive so dark things like the doormat score low, and make G about twice R and B: cardboard has more green than the cone and more red than the pavement. Then pull the bias down until only cardboard survives.",
          ]}
        />
      </Panel>
    );
  }
  return <LessonFrame screen={screen} onScreen={setScreen} title="Filters reach through every channel" question={question} stage={stage} side={side} />;
}
