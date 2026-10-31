import { useState, useMemo } from "react";
import { THEME, MONO } from "./common";

// Lesson 1 - Overfitting (Regularization deck, slides 3-6).
// Aha: a curve that passes through every training order has learned the orders, not the city.
// Training error only goes down as the model gets more flexible; the error on orders it never saw
// goes down, then shoots up. You can only see overfitting on data the fit did not use.

// @sim-start
const TR1 = makeETA(1, 10, 1.6);
const VA1 = makeETA(901, 60, 1.6);
const DEG1 = [1, 2, 3, 4, 5, 6, 7, 8, 9];
const CURVE1 = DEG1.map((d) => {
  const w = ridgeFit(TR1, d, 0);
  return { d, w, tr: rmseOf(w, TR1), va: rmseOf(w, VA1) };
});
// challenge: another city, 14 orders, noisier
const TRC1 = makeETA(5, 14, 2.4);
const VAC1 = makeETA(905, 60, 2.4);
const CURVEC1 = DEG1.concat([10, 11, 12, 13]).map((d) => {
  const w = ridgeFit(TRC1, d, 0);
  return { d, w, tr: rmseOf(w, TRC1), va: rmseOf(w, VAC1) };
});
const BESTC1 = Math.min.apply(null, CURVEC1.map((c) => c.va));
const TOLC1 = 1.07;
// @sim-end

function ErrPlot({ curve, deg, showVal, maxD }) {
  const series = [{ xs: curve.map((c) => c.d), ys: curve.map((c) => c.tr), color: THEME.accent, label: "training orders ●" }];
  if (showVal) series.push({ xs: curve.map((c) => c.d), ys: curve.map((c) => Math.min(c.va, 11)), color: THEME.highlight, dash: "6 3", label: "new orders ▲" });
  const cur = curve.find((c) => c.d === deg);
  const marks = [{ x: deg, y: cur.tr, shape: "circle", color: THEME.accent }];
  if (showVal) marks.push({ x: deg, y: Math.min(cur.va, 11), shape: "triangle", color: THEME.highlight });
  return (
    <Plot
      series={series}
      xr={[0, maxD + 1]}
      yr={[0, 12]}
      xl="polynomial degree (flexibility) →"
      label="typical miss (RMSE, minutes) · clipped at 12"
      height={190}
      marks={marks}
      vlines={[{ x: deg, color: THEME.muted }]}
    />
  );
}

export default function OverfitLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [deg, setDeg] = useState(9);
  const [showVal, setShowVal] = useState(false);
  const [degC, setDegC] = useState(1);
  const cur = CURVE1[deg - 1];
  const curC = CURVEC1.find((c) => c.d === degC);
  const okC = curC.va <= BESTC1 * TOLC1;
  const val = screen >= 1 || showVal;

  const sets = [{ data: TR1, shape: "circle", color: THEME.accent, label: "training" }];
  if (val) sets.unshift({ data: VA1, shape: "triangle", color: THEME.highlight, faint: true });
  const stage = (
    <>
      <EtaPlot sets={sets} curves={[{ w: cur.w, color: THEME.accent, label: "degree " + deg + " fit" }]} label={val ? "● 10 orders the fit saw · ▲ 60 orders it never saw" : "● the 10 orders the fit saw"} />
      {screen >= 1 && <ErrPlot curve={CURVE1} deg={deg} showVal maxD={9} />}
    </>
  );
  const rows = [["miss on its 10 orders", fmt(cur.tr, 2) + " min"]];
  if (val) rows.push(["miss on 60 new orders", fmt(cur.va, 1) + " min", THEME.highlight]);

  let question, side;
  if (screen === 0) {
    question = (
      <>
        Ten delivered orders, each a distance and a delivery time (●). A degree-9 polynomial threads every one of them: its miss is 0.00 minutes. Would you put it in the app to quote ETAs for tomorrow's orders? Look at what it says for a 7.5 km order, then reveal the orders it never saw.
      </>
    );
    side = (
      <Panel>
        <Readout rows={rows} />
        <Toggle label="show 60 new orders (▲)" value={showVal} onChange={setShowVal} />
        <Note>The ETAs here are made-up teaching data: a smooth city curve plus about ±1.6 minutes of noise per order.</Note>
      </Panel>
    );
  } else if (screen === 1) {
    question = (
      <>
        Turn the flexibility down and up. Which number tells you how good the fit is for the app: the miss on its own 10 orders, or on the 60 new ones? Why can the first only go down as the degree rises?
      </>
    );
    side = (
      <Panel>
        <Slider id="r1-d" label="polynomial degree" value={deg} min={1} max={9} stepSize={1} onChange={setDeg} display={String(deg)} />
        <Readout rows={rows} />
        <Note>Each degree adds one more weight. With 10 weights and 10 orders the curve can hit every point exactly, noise included.</Note>
      </Panel>
    );
  } else if (screen === 2) {
    question = (
      <>
        Name it. The network was asked to learn a whole function from a few snapshots of it. Past some flexibility it fits the noise in those snapshots: <b>overfitting</b>. The training error keeps falling; the error on held-out data, the <b>validation error</b>, turns up. The gap between them is what regularization tries to close.
      </>
    );
    side = (
      <Panel>
        <Eq>
          training error = (1/N) Σ<sub>n∈train</sub> (f(x<sub>n</sub>) − y<sub>n</sub>)²
          <br />
          validation error = same sum over orders the fit never saw
        </Eq>
        <Slider id="r1-d2" label="polynomial degree" value={deg} min={1} max={9} stepSize={1} onChange={setDeg} display={String(deg)} />
        <Readout rows={rows} />
        <Note>Underfit (degree 1): both errors high. Overfit (degree 8–9): training error near 0, validation error huge. The rest of this lab is tools that let you keep a flexible model without the blow-up.</Note>
      </Panel>
    );
  } else {
    question = (
      <>
        A second city: 14 orders and noisier traffic. You may go up to degree 13. Pick the degree you would ship, judged on 60 new orders. Get within {Math.round((TOLC1 - 1) * 100)}% of the best validation miss any degree can reach.
      </>
    );
  }

  if (screen === 3) {
    const setsC = [
      { data: VAC1, shape: "triangle", color: THEME.highlight, faint: true },
      { data: TRC1, shape: "circle", color: THEME.accent },
    ];
    const stageC = (
      <>
        <EtaPlot sets={setsC} curves={[{ w: curC.w, color: THEME.accent, label: "degree " + degC }]} label="second city · ● 14 training orders · ▲ 60 new orders" />
        <ErrPlot curve={CURVEC1} deg={degC} showVal maxD={14} />
      </>
    );
    side = (
      <Panel edge={okC ? THEME.ok : THEME.grid}>
        <Slider id="r1-dc" label="polynomial degree" value={degC} min={1} max={13} stepSize={1} onChange={setDegC} display={String(degC)} />
        <Readout
          rows={[
            ["miss on 14 training orders", fmt(curC.tr, 2) + " min"],
            ["miss on 60 new orders", fmt(curC.va, 2) + " min", THEME.highlight],
          ]}
        />
        <Goal met={okC}>new-order miss within {Math.round((TOLC1 - 1) * 100)}% of the best degree</Goal>
        <Hint
          hints={[
            "Ignore the solid line in the lower chart. Only the dashed (▲) line says how the app will do.",
            "Find where the dashed line bottoms out before it climbs. More orders allow a bit more flexibility than 10 did, but not 13 weights for 14 noisy orders.",
          ]}
        />
      </Panel>
    );
    return <LessonFrame screen={screen} onScreen={setScreen} title="Fits every order, predicts nothing" question={question} stage={stageC} side={side} />;
  }
  return <LessonFrame screen={screen} onScreen={setScreen} title="Fits every order, predicts nothing" question={question} stage={stage} side={side} />;
}
