import { useState, useMemo } from "react";
import { THEME, MONO } from "./common";

// Lesson 6 - Early stopping (Regularization deck, slide 55).
// Aha: gradient descent from small weights fits the big, smooth shape first and the noise last.
// So training time is itself a flexibility dial: stop when the validation error turns up and you keep
// the smooth fit. Stopping early behaves much like an L2 penalty, with lambda roughly 1 / (alpha * t).

// @sim-start
const TR6 = makeETA(1, 10, 1.6);
const VA6 = makeETA(901, 60, 1.6);
const LR6 = 0.3;
const EXP6 = [];
for (let e = 0; e <= 5.0001; e += 0.05) EXP6.push(e);
const SNAPS6 = Array.from(new Set(EXP6.map((e) => Math.round(Math.pow(10, e)))));
const PATH6 = gdPath(TR6, 9, LR6, SNAPS6);
const ERR6 = PATH6.map((w) => ({ tr: rmseOf(w, TR6), va: rmseOf(w, VA6) }));
const BEST6 = ERR6.reduce((b, e, i) => (e.va < ERR6[b].va ? i : b), 0);
// challenge: the ridge fit with this lambda; how close is the early-stopped curve?
const LAMC6 = 0.01;
const RIDGE6 = ridgeFit(TR6, 9, LAMC6);
function curveGap6(w) {
  let s = 0;
  const n = 60;
  for (let i = 0; i <= n; i++) {
    const km = KM0 + ((KM1 - KM0) * i) / n;
    s += Math.pow(polyAt(w, km) - polyAt(RIDGE6, km), 2);
  }
  return Math.sqrt(s / (n + 1));
}
const GAP6 = PATH6.map(curveGap6);
const GOALC6 = 0.62;
// @sim-end

function EpochPlot({ k, showVal, extra }) {
  const xs = SNAPS6.map((t) => Math.max(t, 1));
  const series = [{ xs, ys: ERR6.map((e) => Math.min(e.tr, 11)), color: THEME.accent, label: "training ●", ldy: 14 }];
  if (showVal) series.push({ xs, ys: ERR6.map((e) => Math.min(e.va, 11)), color: THEME.highlight, dash: "6 3", label: "new orders ▲" });
  if (extra) series.push(extra);
  const marks = [{ x: xs[k], y: Math.min(ERR6[k].tr, 11), shape: "circle", color: THEME.accent }];
  if (showVal) marks.push({ x: xs[k], y: Math.min(ERR6[k].va, 11), shape: "triangle", color: THEME.highlight });
  return <Plot series={series} xr={[0, 5]} logX yr={[0, 12]} xl="epochs of gradient descent (log scale) →" label="typical miss (minutes) · clipped at 11" height={180} marks={marks} vlines={[{ x: xs[k], color: THEME.muted }]} />;
}

export default function EarlyLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [k, setK] = useState(SNAPS6.length - 1);
  const [patience, setPatience] = useState(10);
  const [kC, setKC] = useState(0);
  const t = SNAPS6[k];
  const w = PATH6[k];
  const e = ERR6[k];
  const okC = GAP6[kC] < GOALC6;

  // patience rule: walk forward, stop once validation has not improved for `patience` checks
  const stopAt = useMemo(() => {
    let best = 0;
    for (let i = 1; i < ERR6.length; i++) {
      if (ERR6[i].va < ERR6[best].va) best = i;
      else if (i - best >= patience) return { stop: i, best };
    }
    return { stop: ERR6.length - 1, best };
  }, [patience]);

  const epochLabel = (n) => (n >= 1000 ? fmt(n / 1000, n >= 10000 ? 0 : 1) + "k" : String(n));
  const fitPlot = (
    <EtaPlot
      sets={[
        ...(screen >= 1 ? [{ data: VA6, shape: "triangle", color: THEME.highlight, faint: true }] : []),
        { data: TR6, shape: "circle", color: THEME.accent },
      ]}
      curves={[{ w, color: THEME.accent, label: "after " + epochLabel(t) + " epochs" }]}
      label={"degree 9, gradient descent from all-zero weights, α = " + LR6}
      height={220}
    />
  );
  const slider = <Slider id={"r6-k" + screen} label="stop after (epochs)" value={k} min={0} max={SNAPS6.length - 1} stepSize={1} onChange={setK} display={epochLabel(t)} />;

  let question, stage, side;
  if (screen === 0) {
    question = (
      <>
        Same ten orders, degree 9, no penalty, but now trained the way networks are: gradient descent from all-zero weights, overnight, 100k epochs. Drag back through training. What does the curve learn first, the big shape or the wiggles? At what epoch would you have liked to stop it?
      </>
    );
    stage = (
      <>
        {fitPlot}
        <EpochPlot k={k} showVal={false} />
      </>
    );
    side = (
      <Panel>
        {slider}
        <Readout rows={[["miss on training orders", fmt(e.tr, 2) + " min"]]} />
        <Note>Only the training curve is shown. It never goes up, so it can't tell you when to stop.</Note>
      </Panel>
    );
  } else if (screen === 1) {
    question = (
      <>
        Now hold out 60 orders the descent never trains on and track them. Where is the validation miss lowest? Is that where the training miss stops falling?
      </>
    );
    stage = (
      <>
        {fitPlot}
        <EpochPlot k={k} showVal />
      </>
    );
    side = (
      <Panel>
        {slider}
        <Readout
          rows={[
            ["miss on training orders", fmt(e.tr, 2) + " min"],
            ["miss on new orders", fmt(e.va, 2) + " min", THEME.highlight],
            ["best new-order miss so far", fmt(ERR6[BEST6].va, 2) + " at " + epochLabel(SNAPS6[BEST6])],
          ]}
        />
      </Panel>
    );
  } else if (screen === 2) {
    question = (
      <>
        Name it. <b>Early stopping</b>: track a held-out validation set during training and stop when it degrades, keeping the best weights seen. The usual criterion is patience: stop after the validation error has not improved for p checks. Starting small and stopping early limits how far the weights travel, much like an L2 penalty with λ ≈ 1/(2αt).
      </>
    );
    stage = (
      <>
        {fitPlot}
        <EpochPlot k={k} showVal />
      </>
    );
    side = (
      <Panel>
        <Eq size={13}>
          stop at the first t where val(t) ≥ min<sub>s&lt;t</sub> val(s) for p checks
          <br />
          keep W at argmin val
        </Eq>
        <Slider id="r6-p" label="patience p (checks, here 1 check ≈ ×1.12 epochs)" value={patience} min={1} max={40} stepSize={1} onChange={setPatience} display={String(patience)} />
        <Readout
          rows={[
            ["training stops at", epochLabel(SNAPS6[stopAt.stop]) + " epochs"],
            ["weights kept from", epochLabel(SNAPS6[stopAt.best]) + " epochs"],
            ["their new-order miss", fmt(ERR6[stopAt.best].va, 2) + " min", THEME.highlight],
          ]}
        />
        {slider}
        <Note>Patience guards against stopping at a small bump. Here the validation curve is smooth, so any patience finds the minimum; with noisy mini-batch training it matters more.</Note>
      </Panel>
    );
  } else {
    question = (
      <>
        Lesson 2 found that λ = {fmt(LAMC6, 4)} gives a good curve for these orders (dashed). Without any penalty, reach the same curve by stopping gradient descent at the right time: get the typical gap between the two curves below {GOALC6} min.
      </>
    );
    const wC = PATH6[kC];
    stage = (
      <>
        <EtaPlot
          sets={[{ data: TR6, shape: "circle", color: THEME.accent }]}
          curves={[
            { w: RIDGE6, color: THEME.ok, dash: "7 4", label: "L2, λ = " + fmt(LAMC6, 4), ldy: 14 },
            { w: wC, color: THEME.accent, label: "stopped at " + epochLabel(SNAPS6[kC]) },
          ]}
          label="● training orders · dashed = the L2 fit to match"
          height={230}
        />
        <Plot
          series={[{ xs: SNAPS6.map((s) => Math.max(s, 1)), ys: GAP6.map((g) => Math.min(g, 11)), color: THEME.ok, label: "gap to L2 curve" }]}
          xr={[0, 5]}
          logX
          yr={[0, 6]}
          xl="epochs (log scale) →"
          label="typical gap between the two curves (minutes)"
          height={150}
          hlines={[{ y: GOALC6, color: THEME.ok, label: "goal" }]}
          vlines={[{ x: Math.max(SNAPS6[kC], 1), color: THEME.muted }]}
        />
      </>
    );
    side = (
      <Panel edge={okC ? THEME.ok : THEME.grid}>
        <Slider id="r6-kc" label="stop after (epochs)" value={kC} min={0} max={SNAPS6.length - 1} stepSize={1} onChange={setKC} display={epochLabel(SNAPS6[kC])} />
        <Readout
          rows={[
            ["gap to the L2 curve", fmt(GAP6[kC], 2) + " min", THEME.highlight],
            ["1/(2αt) at this stop", fmt(1 / (2 * LR6 * Math.max(SNAPS6[kC], 1)), 4)],
          ]}
        />
        <Goal met={okC}>gap below {GOALC6} min</Goal>
        <Hint
          hints={[
            "Early in training both curves are too flat; late in training yours wiggles. The match is in between.",
            "Use the rule of thumb λ ≈ 1/(2αt) with α = " + LR6 + " to guess t, then fine-tune around it.",
          ]}
        />
      </Panel>
    );
  }
  return <LessonFrame screen={screen} onScreen={setScreen} title="Stop before it memorises" question={question} stage={stage} side={side} />;
}
