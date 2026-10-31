import { useState, useMemo, useCallback } from "react";
import { THEME, MONO } from "./common";

// Lesson 5 - Data augmentation and the common pattern (Regularization deck, slides 39-54).
// Aha: you can make new labelled examples from old ones with any change that leaves the label true
// (an order 0.3 km longer takes about as long). Training on the randomly changed copies smooths the fit
// like a penalty would. Changing the label instead adds nothing. Dropout, batch norm and augmentation are
// all one pattern: add randomness while training, average it out at test time.

// @sim-start
const TR5 = makeETA(1, 10, 1.6);
const VA5 = makeETA(901, 60, 1.6);
const COPIES5 = 30;
function labelJitter(data, sigma, copies, seed) {
  const r = rng(seed);
  const km = [];
  const y = [];
  for (let i = 0; i < data.n; i++)
    for (let k = 0; k < copies; k++) {
      km.push(data.km[i]);
      y.push(data.y[i] + sigma * gauss(r));
    }
  return { km, y, n: km.length };
}
function fit5(tr, va, sigma, mode) {
  const aug = mode === "y" ? labelJitter(tr, sigma * 6, COPIES5, 5) : jitterCopies(tr, sigma, COPIES5, 5);
  const w = ridgeFit(aug, 9, 1e-7);
  return { aug, w, tr: rmseOf(w, tr), va: rmseOf(w, va) };
}
const SIGS5 = [];
for (let s = 0; s <= 1.5001; s += 0.025) SIGS5.push(s);
const SWEEP5 = SIGS5.map((s) => fit5(TR5, VA5, s, "x").va);
const TRC5 = makeETA(6, 10, 1.6);
const VAC5 = makeETA(906, 60, 1.6);
const GOALC5 = 1.75;
// @sim-end

const PATTERN5 = [
  ["drop", "dropout", "which units are on (mask)", "all units on, scaled by a"],
  ["bn", "batch norm", "which orders share the mini-batch", "fixed running mean and std"],
  ["aug", "augmentation", "flip, crop, scale, color jitter, nudge", "the clean input (or average a few crops)"],
  ["dc", "DropConnect", "which single weights are zeroed", "all weights, scaled"],
  ["sd", "stochastic depth", "which whole layers are skipped", "every layer, scaled"],
  ["cut", "cutout", "a random square of the image blanked", "the full image"],
  ["mix", "mixup", "blend two examples and their labels", "one clean example"],
];

function SigmaPlot({ sigma, sweep, goal }) {
  return (
    <Plot
      series={[{ xs: SIGS5, ys: sweep.map((v) => Math.min(v, 11)), color: THEME.highlight, dash: "6 3", label: "new orders ▲" }]}
      xr={[0, 1.5]}
      yr={[0, 12]}
      xl="nudge size σ (km) →"
      label="miss on new orders (minutes) · clipped at 11"
      height={160}
      hlines={goal ? [{ y: goal, color: THEME.ok, label: "goal" }] : []}
      vlines={[{ x: sigma, color: THEME.muted }]}
    />
  );
}

export default function AugmentLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [on, setOn] = useState(false);
  const [sigma, setSigma] = useState(0.3);
  const [mode, setMode] = useState("x");
  const [tech, setTech] = useState("aug");
  const [sigC, setSigC] = useState(0);
  const sig = screen === 0 ? (on ? 0.3 : 0) : sigma;
  const md = screen === 1 ? mode : "x";
  const cur = useMemo(() => fit5(TR5, VA5, sig, md), [sig, md]);
  const curC = useMemo(() => fit5(TRC5, VAC5, sigC, "x"), [sigC]);
  const okC = curC.va < GOALC5;
  const sweepC = useMemo(() => SIGS5.map((s) => fit5(TRC5, VAC5, s, "x").va), []);

  const plot = (fit, tr, va, lbl) => (
    <EtaPlot
      sets={[
        { data: va, shape: "triangle", color: THEME.highlight, faint: true },
        ...(sig > 0 || screen === 3 ? [{ data: fit.aug, shape: "circle", color: THEME.muted, r: 1.4 }] : []),
        { data: tr, shape: "circle", color: THEME.accent },
      ]}
      curves={[{ w: fit.w, color: THEME.accent, label: lbl }]}
      label="● 10 real orders · · made-up copies · ▲ new orders"
      height={230}
    />
  );
  const rows = [
    ["training examples", sig > 0 ? String(TR5.n * COPIES5) + " (10 real)" : "10"],
    ["miss on new orders", fmt(cur.va, 2) + " min", THEME.highlight],
  ];

  let question, stage, side;
  if (screen === 0) {
    question = (
      <>
        Still the same 10 orders and the wild degree-9 fit. You can't buy more orders, but you know something for free: an order a few hundred metres longer takes about the same time. Make 30 nudged copies of every order, keeping its ETA. Does adding made-up orders poison the fit or help it?
      </>
    );
    stage = plot(cur, TR5, VA5, on ? "fit on copies" : "fit on 10 orders");
    side = (
      <Panel>
        <Toggle label="add 30 copies per order, nudged ±0.3 km" value={on} onChange={setOn} />
        <Readout rows={rows} />
        <Note>No penalty here (λ ≈ 0): the only change is the extra, randomly nudged training data.</Note>
      </Panel>
    );
  } else if (screen === 1) {
    question = (
      <>
        Vary the nudge size σ. When is it too small to help, and when does it start to blur the real shape of the city? Then nudge the ETA (the label) instead of the distance. Why does that do nothing to the wiggles?
      </>
    );
    stage = (
      <>
        {plot(cur, TR5, VA5, "fit")}
        {mode === "x" && <SigmaPlot sigma={sigma} sweep={SWEEP5} />}
      </>
    );
    side = (
      <Panel>
        <Label>what gets nudged</Label>
        <Pills options={[["x", "distance (input)"], ["y", "ETA (label)"]]} value={mode} onPick={setMode} />
        <Slider id="r5-s" label={mode === "x" ? "nudge size σ (km)" : "nudge size (×6 minutes)"} value={sigma} min={0} max={1.5} stepSize={0.005} onChange={setSigma} display={fmt(sigma, 3)} />
        <Readout rows={rows} />
        <Note>Nudging the input says "nearby inputs share this answer", which forbids sharp wiggles between orders. Noisy labels only average back to the same 10 targets.</Note>
      </Panel>
    );
  } else if (screen === 2) {
    const t = PATTERN5.find((p) => p[0] === tech);
    question = (
      <>
        Name it. <b>Data augmentation</b>: distort training examples in ways that keep the label true (for images: flips, random crops and scales, color jitter, rotation). It is one case of a common pattern behind most regularizers: add randomness z while training, then average it out at test time.
      </>
    );
    stage = (
      <>
        {plot(cur, TR5, VA5, "fit")}
        <SigmaPlot sigma={sigma} sweep={SWEEP5} />
      </>
    );
    side = (
      <Panel>
        <Eq size={13}>
          train: y = f(x, z; W), z random
          <br />
          test: y = E<sub>z</sub>[f(x, z; W)] = ∫ p(z) f(x, z; W) dz
        </Eq>
        <Label>pick a regularizer</Label>
        <Pills options={PATTERN5.map((p) => [p[0], p[1]])} value={tech} onPick={setTech} />
        <Readout
          rows={[
            ["random z in training", t[2]],
            ["at test time", t[3]],
          ]}
        />
        <Slider id="r5-s2" label="nudge size σ (km)" value={sigma} min={0} max={1.5} stepSize={0.005} onChange={setSigma} display={fmt(sigma, 3)} />
        <Note>In practice: batch norm and augmentation almost always; dropout for big fully connected layers; cutout and mixup for small image datasets.</Note>
      </Panel>
    );
  } else {
    question = (
      <>
        The second city again, with its ten orders and degree 9, no penalty. Use only augmentation: choose the nudge size so the miss on new orders drops below {GOALC5} minutes.
      </>
    );
    stage = (
      <>
        <EtaPlot
          sets={[
            { data: VAC5, shape: "triangle", color: THEME.highlight, faint: true },
            ...(sigC > 0 ? [{ data: curC.aug, shape: "circle", color: THEME.muted, r: 1.4 }] : []),
            { data: TRC5, shape: "circle", color: THEME.accent },
          ]}
          curves={[{ w: curC.w, color: THEME.accent, label: "your fit" }]}
          label="second city · ● real orders · · copies · ▲ new orders"
          height={230}
        />
        <SigmaPlot sigma={sigC} sweep={sweepC} goal={GOALC5} />
      </>
    );
    side = (
      <Panel edge={okC ? THEME.ok : THEME.grid}>
        <Slider id="r5-sc" label="nudge size σ (km)" value={sigC} min={0} max={1.5} stepSize={0.005} onChange={setSigC} display={fmt(sigC, 3)} />
        <Readout rows={[["miss on new orders", fmt(curC.va, 2) + " min", THEME.highlight]]} />
        <Goal met={okC}>below {GOALC5} min</Goal>
        <Hint
          hints={[
            "Too small and the copies sit on top of the originals: nothing changes. Too big and an order at 2 km pretends to be at 4 km.",
            "The orders here are about 0.75 km apart. A nudge around half that spacing fills the gaps without crossing into the neighbour's territory.",
          ]}
        />
      </Panel>
    );
  }
  return <LessonFrame screen={screen} onScreen={setScreen} title="Made-up orders that tell the truth" question={question} stage={stage} side={side} />;
}
