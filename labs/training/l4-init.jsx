import { useState, useMemo } from "react";
import { THEME, MONO } from "./common";

// Lesson 4 - Weight initialization scale: Xavier and Kaiming (Practical Issues, slides 35-45).
// Aha: each layer multiplies the size of the signal by about sqrt(fan_in) * sigma (times a factor for
// the activation). Over ten layers that factor is raised to the tenth power, so it must be 1:
// sigma = 1/sqrt(fan_in) for tanh (Xavier), sqrt(2/fan_in) for ReLU, which zeroes half its inputs (Kaiming).

// @sim-start
const L4 = 10;
const BATCH4 = 48;
const WIDTHS4 = [50, 100, 200];
const BASES4 = {};
for (const w of WIDTHS4) BASES4[w] = makeDeep(Array(L4 + 1).fill(w), BATCH4, 11);
// challenge net: layer widths vary a lot
const CW4 = [64, 240, 240, 30, 30, 240, 30, 240, 240, 40, 40, 160, 160];
const CBASE4 = makeDeep(CW4, BATCH4, 5);
function healthy(o) {
  return o.every((q) => q.zrms >= 0.5 && q.zrms <= 2);
}
// @sim-end

const RULE_OPTS = [
  ["fixed", "fixed σ"],
  ["xavier", "Xavier 1/√fan_in"],
  ["kaiming", "Kaiming √(2/fan_in)"],
];
const ACT4_OPTS = [
  ["tanh", "tanh"],
  ["relu", "ReLU"],
];
const W_OPTS = WIDTHS4.map((w) => [w, w + " units"]);
const CRULE_OPTS = [
  ["same", "same σ for every layer"],
  ["fan", "σ = g / √fan_in"],
];

function histRange(act) {
  return act === "tanh" ? [-1, 1] : [0, 3];
}
function Stats({ out, act, title, widths }) {
  const [lo, hi] = histRange(act);
  const layers = out.map((q, i) => ({ h: histOf(q.a, lo, hi, 24), label: "L" + (i + 1) + (widths ? " ·" + widths[i + 1] : ""), value: fmt(q.astd, 2), bad: q.zrms < 0.5 || q.zrms > 2 }));
  const ks = out.map((q, i) => i + 1);
  return (
    <>
      <LayerHists layers={layers} lo={lo} hi={hi} cap={act === "relu" ? 0.6 : 0.3} height={150} label={title + " · histogram of each layer's outputs, std below"} />
      <Plot
        series={[{ xs: ks, ys: out.map((q) => q.zrms), color: THEME.highlight, width: 2.2 }]}
        marks={out.map((q, i) => ({ x: i + 1, y: q.zrms, shape: q.zrms < 0.5 || q.zrms > 2 ? "triangle" : "circle", color: q.zrms < 0.5 || q.zrms > 2 ? THEME.warn : THEME.highlight }))}
        hlines={[
          { y: 0.5, color: THEME.ok, label: "0.5" },
          { y: 2, color: THEME.ok, label: "2" },
        ]}
        xr={[0, out.length]}
        yr={[-3, 1]}
        logY
        xl="layer →"
        label="size of each layer's input z (rms, log) · ▲ outside 0.5–2"
        height={160}
      />
    </>
  );
}

export default function InitLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [act, setAct] = useState("tanh");
  const [rule, setRule] = useState("fixed");
  const [sigma, setSigma] = useState(0.01);
  const [width, setWidth] = useState(100);
  const [crule, setCrule] = useState("same");
  const [cs, setCs] = useState(0.1);
  const [cg, setCg] = useState(1);
  const actNow = screen === 0 ? "tanh" : act;
  const ruleNow = screen < 2 ? "fixed" : rule;
  const widthNow = screen === 0 ? 100 : width;
  const out = useMemo(() => {
    const f = ruleNow === "fixed" ? () => sigma : ruleNow === "xavier" ? (l, n) => 1 / Math.sqrt(n) : (l, n) => Math.sqrt(2 / n);
    return deepForward(BASES4[widthNow], actNow, f);
  }, [actNow, ruleNow, sigma, widthNow]);
  const outC = useMemo(() => deepForward(CBASE4, "relu", crule === "same" ? () => cs : (l, n) => cg / Math.sqrt(n)), [crule, cs, cg]);
  const okC = healthy(outC);
  const last = out[out.length - 1];

  const rows = [
    ["σ·√fan_in", fmt((ruleNow === "fixed" ? sigma : ruleNow === "xavier" ? 1 / Math.sqrt(widthNow) : Math.sqrt(2 / widthNow)) * Math.sqrt(widthNow), 3)],
    ["layer 1 output std", fmt(out[0].astd, 3)],
    ["layer 10 output std", fmt(last.astd, 4), last.zrms < 0.5 || last.zrms > 2 ? THEME.warn : undefined],
  ];
  const sigmaSlider = <LogSlider id="l4-sigma" label="weight std σ" value={sigma} lo={-3} hi={0} onChange={setSigma} digits={3} />;

  let question, stage, side;
  if (screen === 0) {
    question = (
      <>
        The ETA network grows to 10 tanh layers of 100 units. Following the first idea on the slides, its weights start as small random numbers: Gaussian, mean 0, std 0.01. Look at what reaches the last layer. Then find a σ for which layer 10 looks like layer 1. What goes wrong at σ = 0.3?
      </>
    );
    stage = <Stats out={out} act="tanh" title="10 tanh layers × 100" />;
    side = (
      <>
        <Panel>
          {sigmaSlider}
          <Readout rows={rows} />
        </Panel>
        <Note>When outputs are near 0, the gradient on each layer's weights (input × upstream gradient) is near 0 too. When they pile up at ±1, the tanh is saturated (lesson 1).</Note>
      </>
    );
  } else if (screen === 1) {
    question = (
      <>
        Change the width of the layers and find the good σ again for 50, 100 and 200 units. Which single number stays the same at the good setting? Then switch to ReLU: does the same rule still work?
      </>
    );
    stage = <Stats out={out} act={actNow} title={"10 " + ACT_NAMES[actNow] + " layers × " + widthNow} />;
    side = (
      <>
        <Panel>
          <Pills options={ACT4_OPTS} value={act} onPick={setAct} />
          <Pills options={W_OPTS} value={width} onPick={setWidth} />
          {sigmaSlider}
          <Readout rows={rows} />
        </Panel>
        <Note>Write down the good σ for each width and multiply it by √width.</Note>
      </>
    );
  } else if (screen === 2) {
    question = (
      <>
        Name it. A unit adds fan_in terms w·x, so Var(z) = fan_in · σ² · E[x²]. Keeping it fixed from layer to layer needs σ = 1/√fan_in (Xavier, Glorot & Bengio 2010). ReLU sets half its inputs to 0, which halves E[x²], so it needs twice the variance: σ = √(2/fan_in) (Kaiming / MSRA, He et al. 2015).
      </>
    );
    stage = <Stats out={out} act={actNow} title={"10 " + ACT_NAMES[actNow] + " layers × " + widthNow} />;
    side = (
      <>
        <Panel>
          <Eq>
            Var(z) = fan_in · σ² · E[x²]
            <br />
            Xavier (tanh): σ = 1 / √fan_in
            <br />
            Kaiming (ReLU): σ = √(2 / fan_in)
          </Eq>
          <Pills options={ACT4_OPTS} value={act} onPick={setAct} />
          <Pills options={W_OPTS} value={width} onPick={setWidth} />
          <Pills options={RULE_OPTS} value={rule} onPick={setRule} />
          {rule === "fixed" && sigmaSlider}
          <Readout rows={rows} />
        </Panel>
        <Note>Try Xavier with ReLU: the signal halves its variance every layer and collapses again, as on the slides. With tanh, Xavier keeps the signal far better than any std of 0.01, though it still shrinks slowly because tanh's slope is below 1 away from 0. For conv layers fan_in = filter_size² × input_channels.</Note>
      </>
    );
  } else {
    question = (
      <>
        The production ETA network: 12 ReLU layers whose widths jump around (64 inputs, then 240, 240, 30, 30, 240, 30, 240, 240, 40, 40, 160, 160). Keep every layer's input z between 0.5 and 2 in size. Choose a rule and its number.
      </>
    );
    stage = <Stats out={outC} act="relu" title="12 ReLU layers, mixed widths" widths={CW4} />;
    side = (
      <>
        <Panel edge={okC ? THEME.ok : THEME.grid}>
          <Pills options={CRULE_OPTS} value={crule} onPick={setCrule} />
          {crule === "same" ? (
            <LogSlider id="l4-cs" label="σ for every layer" value={cs} lo={-2.5} hi={0} onChange={setCs} digits={3} />
          ) : (
            <Slider id="l4-cg" label="gain g in σ = g / √fan_in" value={cg} min={0.5} max={3} stepSize={0.005} onChange={setCg} display={fmt(cg, 3)} />
          )}
          <Readout
            rows={[
              ["smallest layer input z", fmt(Math.min.apply(null, outC.map((q) => q.zrms)), 3)],
              ["largest layer input z", fmt(Math.max.apply(null, outC.map((q) => q.zrms)), 3)],
            ]}
          />
          <Goal met={okC}>all 12 layers between 0.5 and 2</Goal>
          <Hint
            hints={[
              "With one σ for all layers, look at what happens right after a wide layer feeds a narrow one. Can any single σ suit fan_in = 30 and fan_in = 240 together?",
              "Per layer the variance is multiplied by g²/2 for ReLU. Over 12 layers that factor is raised to the 12th power, so g²/2 must be very close to 1.",
            ]}
          />
        </Panel>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="Keep the signal the same size" question={question} stage={stage} side={side} />;
}
