import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import { THEME, MONO } from "./common";

// Lesson 4 - Choosing the learning rate from the loss curve (Optimization II, slides 52-58).
// Aha: the shape of the loss curve tells you which way to turn eta (flat: up; drops then sits high and
// noisy: a bit down; explodes to NaN: well down), and the size of each update relative to the weights
// (~1e-3) tells you the right scale before the curve does, whatever the network.
// The "network" is a quadratic with 100 directions whose curvatures span 1000x, plus mini-batch noise.

// @sim-start
const D4 = 100;
const K4 = 300;
const NOISE4 = 0.12;
function makeNet(lmax, wScale) {
  const lam = [];
  const x0 = [];
  for (let i = 0; i < D4; i++) lam.push(lmax * Math.pow(10, (-3 * i) / (D4 - 1)));
  for (let i = 0; i < D4; i++) x0.push(Math.sqrt((2 * (2.0 / D4)) / lam[i]));
  let g = 0;
  for (let i = 0; i < D4; i++) g += Math.pow(lam[i] * x0[i], 2);
  return { lam, x0, wnorm: Math.sqrt(g) * wScale, lmax };
}
const NET_A = makeNet(1000, 1);
const NET_B = makeNet(40, 30);
function trainNet(net, eta, seed) {
  const r = rng(seed || 2);
  const x = net.x0.slice();
  const loss = [];
  const ratio = [];
  for (let k = 0; k <= K4; k++) {
    let L = 0.3;
    for (let i = 0; i < D4; i++) L += 0.5 * net.lam[i] * x[i] * x[i];
    if (!isFinite(L) || L > 50) {
      loss.push(60);
      return { loss, ratio, nanAt: k };
    }
    loss.push(L);
    if (k === K4) break;
    let g2 = 0;
    for (let i = 0; i < D4; i++) {
      const g = net.lam[i] * x[i] + NOISE4 * Math.sqrt(net.lam[i]) * gauss(r);
      g2 += g * g;
      x[i] -= eta * g;
    }
    ratio.push((eta * Math.sqrt(g2)) / net.wnorm);
  }
  return { loss, ratio, nanAt: -1 };
}
function finalLoss(t) {
  if (t.nanAt >= 0) return Infinity;
  return t.loss.slice(-30).reduce((a, c) => a + c, 0) / 30;
}
function firstRatio(t) {
  const a = t.ratio.slice(0, 10);
  return a.length ? a.reduce((s, v) => s + v, 0) / a.length : NaN;
}
const TARGET4C = 0.55;
// @sim-end

const KS = Array.from({ length: K4 + 1 }, (_, k) => k);
const PIN_STYLE = [
  [THEME.muted, "2 3"],
  [THEME.text, "6 3"],
  [THEME.warn, "1 3"],
  [THEME.ok, "8 3 2 3"],
  [THEME.accent, "4 4"],
];
function etaText(e) {
  return e < 1e-3 ? e.toExponential(1) : fmt(e, 4);
}
function lossSeries(t, color, dash, label, width) {
  return { xs: KS.slice(0, t.loss.length), ys: t.loss, color, dash, label, width };
}

function LossPanel({ t, pins, height, eta }) {
  return (
    <Plot
      series={pins.map((p, i) => lossSeries(p.t, PIN_STYLE[i][0], PIN_STYLE[i][1], "", 1.4)).concat([lossSeries(t, THEME.highlight, undefined, "", 2.4)])}
      legend={[{ color: THEME.highlight, width: 2.4, text: "now: η " + etaText(eta) + (t.nanAt >= 0 ? " NaN" : "") }].concat(pins.map((p, i) => ({ color: PIN_STYLE[i][0], dash: PIN_STYLE[i][1], text: "η " + etaText(p.eta) + (p.t.nanAt >= 0 ? " NaN" : "") })))}
      xr={[0, K4]}
      yr={[-0.6, 1.2]}
      logY
      xl="iteration →"
      label="training loss (log scale) · NaN runs shoot off the top"
      height={height || 220}
    />
  );
}
function RatioPanel({ t, pins }) {
  const rs = (tt, color, dash, label, width) => ({ xs: KS.slice(0, tt.ratio.length), ys: tt.ratio, color, dash, label, width });
  return (
    <Plot
      series={pins.map((p, i) => rs(p.t, PIN_STYLE[i][0], PIN_STYLE[i][1], "", 1.2)).concat([rs(t, THEME.highlight, undefined, "now", 2.2)])}
      xr={[0, K4]}
      yr={[-7, -1]}
      logY
      xl="iteration →"
      label="size of update ÷ size of weights (log scale)"
      height={150}
      hlines={[{ y: 1e-3, color: THEME.ok, label: "≈ 1e-3" }]}
    />
  );
}

export default function CurveLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [eta, setEta] = useState(1e-5);
  const [pins, setPins] = useState([]);
  const [etaC, setEtaC] = useState(1e-3);
  const t = useMemo(() => trainNet(NET_A, eta), [eta]);
  const tC = useMemo(() => trainNet(NET_B, etaC), [etaC]);
  const keep = useCallback(() => setPins((ps) => ps.concat([{ eta, t }]).slice(-5)), [eta, t]);
  const clear = useCallback(() => setPins([]), []);
  const fl = finalLoss(t);
  const flC = finalLoss(tC);
  const okC = flC < TARGET4C;

  const lossRows = (tt) => [
    ["loss after 300 steps (avg of last 30)", tt.nanAt >= 0 ? "NaN at step " + tt.nanAt : fmt(finalLoss(tt), 3), tt.nanAt >= 0 ? THEME.warn : undefined],
    ["update ÷ weight, first 10 steps", tt.ratio.length ? firstRatio(tt).toExponential(1) : "—"],
  ];
  const pinBox = (
    <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
      <Btn onClick={keep} strong>
        keep this curve
      </Btn>
      <Btn onClick={clear} disabled={!pins.length}>
        clear kept
      </Btn>
    </div>
  );

  let question, stage, side;
  if (screen === 0) {
    question = (
      <>
        <b>Where we are:</b> we now know <i>how</i> to update: shuffled mini-batches. What's left is the deck's practical question (slides 52–58): how do you choose η for a real network, where you can't compute the curvature?
        <br />
        <b>The puzzle:</b> a network with millions of weights (here: a stand-in with the same kind of loss), trained for 300 mini-batch steps. Its loss starts at 2.3. Search η over five orders of magnitude and keep the curves you get. You should find three different ways it fails, and one η that gives the lowest loss.
      </>
    );
    stage = <LossPanel t={t} eta={eta} pins={pins} />;
    side = (
      <>
        <Panel>
          <LogSlider id="l4-eta" label="learning rate η" value={eta} lo={-6.5} hi={-2.3} onChange={setEta} />
          <Readout rows={lossRows(t).slice(0, 1)} />
          {pinBox}
        </Panel>
        <Note>Kept curves are drawn with their own dash pattern and labelled with their η. Give each failure a one-word name.</Note>
      </>
    );
  } else if (screen === 1) {
    question = (
      <>
        Add a second gauge: how big each update is compared with the weights themselves, ‖Δw‖ / ‖w‖. This is available after a single step, long before the loss curve has a shape. What value does it have for your best η? And for the runs that fail?
      </>
    );
    stage = (
      <>
        <LossPanel t={t} eta={eta} pins={pins} height={190} />
        <RatioPanel t={t} pins={pins} />
      </>
    );
    side = (
      <>
        <Panel>
          <LogSlider id="l4-eta2" label="learning rate η" value={eta} lo={-6.5} hi={-2.3} onChange={setEta} />
          <Readout rows={lossRows(t)} />
          {pinBox}
        </Panel>
        <Note>Why would the ratio work as a gauge on any network, while a good η of 1e-3 here might be useless on another one?</Note>
      </>
    );
  } else if (screen === 2) {
    question = (
      <>
        Name it. The learning rate is the first hyperparameter to tune (with small regularization), and the loss curve says which way to move it. A good update is about a thousandth of the weight it changes (slide 58). Next: η doesn't have to stay fixed. A schedule changes it during training.
      </>
    );
    stage = (
      <>
        <LossPanel t={t} eta={eta} pins={pins} height={190} />
        <RatioPanel t={t} pins={pins} />
      </>
    );
    side = (
      <>
        <Panel>
          <Readout
            rows={[
              ["loss barely moves", "η too low"],
              ["fast drop, then stalls high and jittery", "η a bit too high"],
              ["explodes, cost becomes NaN", "η too high"],
              ["steady drop, low end", "good η"],
              ["‖Δw‖ / ‖w‖", "aim for ≈ 1e-3"],
            ]}
          />
          <LogSlider id="l4-eta3" label="learning rate η" value={eta} lo={-6.5} hi={-2.3} onChange={setEta} />
          {pinBox}
        </Panel>
        <Note>
          Why the stall: the sharpest direction of the loss can only take steps below 2/λ<sub>max</sub> (= {fmt(2 / NET_A.lmax, 4)} here). Close to that limit, the mini-batch noise in that direction is barely damped. For this network a rough search range is 1e-5 to 1e-3, as on the slides. Anything from 3e-3 up gives NaN.
        </Note>
      </>
    );
  } else {
    question = (
      <>
        A different network: a different architecture, much larger weights, and a different curvature. Your best η from before is useless here. Get its loss (average of the last 30 steps) below {TARGET4C}, without making the run explode.
      </>
    );
    stage = (
      <>
        <LossPanel t={tC} eta={etaC} pins={[]} height={190} />
        <RatioPanel t={tC} pins={[]} />
      </>
    );
    side = (
      <>
        <Panel edge={okC ? THEME.ok : THEME.grid}>
          <LogSlider id="l4-etac" label="learning rate η" value={etaC} lo={-6.5} hi={-0.5} onChange={setEtaC} />
          <Readout rows={lossRows(tC)} />
          <Goal met={tC.nanAt < 0}>no NaN</Goal>
          <Goal met={okC}>final loss below {TARGET4C}</Goal>
          <Hint
            hints={[
              "Read the second gauge right after you move the slider. Where is the ratio, compared with where it was for your best η on the first network?",
              "Move η until ‖Δw‖/‖w‖ in the first steps is near 1e-3, then nudge it up or down while watching the end of the loss curve.",
            ]}
          />
        </Panel>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="Read the loss curve" question={question} stage={stage} side={side} />;
}
