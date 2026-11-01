import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import { THEME, MONO } from "./common";

// Lesson 1 - Batch vs stochastic updates, and the order of presentation (Optimization II, slides 3-19).
// Aha: a batch update waits to hear from every sample before it moves once; SGD moves after each sample,
// so for the same work it gets far further. But a fixed, structured order makes the pulls repeat every
// epoch, and the weights repeat with them: a cycle that never closes in on the best fit.

// @sim-start
const O1 = makeOrders(11, 40, 0.8, 2.0, 0.5);
const C1 = makeOrders(31, 60, 1.1, 1.5, 0.3);
const START = [2.2, 1.8];
const START_C = [0.2, 8];
const ETA_BATCH = 0.35;
const ETA_SGD = 0.04;
const EPOCHS_C = 3;
const TOL_C = 0.2;
const CHECK_KM = [1, 8];
function feeAt(w, km) {
  return w[1] + w[0] * (km - 4);
}
function fairness(o, w) {
  return Math.max(...CHECK_KM.map((km) => Math.abs(feeAt(w, km) - feeAt(o.best, km))));
}
// @sim-end

const SURF = { f: O1.loss, xr: [-0.6, 2.6], yr: [1, 7.6], xl: "slope w₁", yl: "w₂ = prediction at x = 4" };
const SURF_C = { f: C1.loss, xr: [-0.6, 2.6], yr: [1, 9.4], xl: "slope w₁", yl: "w₂ = prediction at x = 4" };
const ORDERS = [
  ["sorted", "dataset order: sorted by x"],
  ["fixed", "one random order, reused"],
  ["shuffle", "reshuffle every epoch"],
];
const METHODS = [
  ["batch", "batch: 1 update per pass"],
  ["sgd", "SGD: 1 update per sample"],
];

function epochDiamonds(run, upto) {
  return function (P) {
    return (
      <g>
        {run.epochEnd
          .filter((i) => i <= upto)
          .map((i) => {
            const q = P(run.pts[i][0], run.pts[i][1]);
            return <Marker key={i} x={q[0]} y={q[1]} shape="diamond" color={THEME.text} r={4} />;
          })}
      </g>
    );
  };
}

export default function SGDLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [reads, setReads] = useState(40);
  const [start, setStart] = useState(START);
  const [order, setOrder] = useState("sorted");
  const [eta, setEta] = useState(0.08);
  const [upto, setUpto] = useState(400);
  const [methodC, setMethodC] = useState("batch");
  const [orderC, setOrderC] = useState("sorted");
  const [etaC, setEtaC] = useState(0.1);
  const [seedC, setSeedC] = useState(5);
  const onHandle = useCallback((x, y) => setStart([x, y]), []);
  const reshuffle = useCallback(() => setSeedC((s) => s + 1), []);

  const batchRun = useMemo(() => runSGD(O1, start, { b: 40, eta: ETA_BATCH, steps: Math.floor(reads / 40) }), [start, reads]);
  const sgdRun = useMemo(() => runSGD(O1, start, { b: 1, eta: ETA_SGD, steps: reads, order: "shuffle", seed: 3 }), [start, reads]);
  const exRun = useMemo(() => runSGD(O1, START, { b: 1, eta, steps: 400, order, seed: 3 }), [eta, order]);
  const chRun = useMemo(
    () => (methodC === "batch" ? runSGD(C1, START_C, { b: 60, eta: etaC, steps: EPOCHS_C }) : runSGD(C1, START_C, { b: 1, eta: etaC, steps: 60 * EPOCHS_C, order: orderC, seed: seedC })),
    [methodC, orderC, etaC, seedC]
  );

  const k = Math.min(upto, exRun.pts.length - 1);
  const wK = exRun.pts[k];
  const ends = exRun.epochEnd.filter((i) => i <= k).slice(-3);
  const chFair = chRun.diverged ? Infinity : fairness(C1, chRun.w);
  const chOk = chFair <= TOL_C;

  let question, stage, side;
  if (screen === 0) {
    const dB = dist(batchRun.w, O1.best);
    const dS = dist(sgdRun.w, O1.best);
    question = (
      <>
        <b>Where we are:</b> Optimization I studied how to step on a given loss. But the loss is an average over N training samples, and computing its full gradient means reading all N samples before every single step (slides 2–5). Optimization II asks: do we need all of them before we move?
        <br />
        <b>The puzzle:</b> 40 training samples (x, y), fitted with a line ŷ = w₁(x − 4) + w₂. Computing the gradient on one sample costs one unit of work. Batch gradient descent reads all 40, then takes one step. The alternative takes a step after every sample. With the same work budget, which ends closer to the best fit (+)?
      </>
    );
    stage = (
      <LossMap
        surf={SURF}
        target={O1.best}
        handle={start}
        onHandle={onHandle}
        paths={[
          { key: "b", pts: batchRun.pts, color: THEME.highlight, shape: "square", dash: "6 4", label: "batch" },
          { key: "s", pts: sgdRun.pts, color: THEME.accent, shape: "circle", label: "one at a time", width: 1.4 },
        ]}
        ariaLabel="Loss over the two weights with the batch path and the per-sample path"
      />
    );
    side = (
      <>
        <Panel>
          <Slider id="l1-reads" label="work budget (sample gradients)" value={reads} min={40} max={400} stepSize={1} onChange={setReads} display={reads + " = " + fmt(reads / 40, 1) + " passes"} />
          <Readout
            rows={[
              ["■ batch: updates made", Math.floor(reads / 40)],
              ["■ batch: distance to +", fmt(dB, 3)],
              ["● one-at-a-time: updates made", reads],
              ["● one-at-a-time: distance to +", fmt(dS, 3)],
            ]}
          />
        </Panel>
        <FeePlot
          orders={O1}
          height={190}
          lines={[
            { w: O1.best, color: THEME.text, dash: "2 3", label: "best", ldy: 14 },
            { w: batchRun.w, color: THEME.highlight, dash: "6 4", label: "batch" },
            { w: sgdRun.w, color: THEME.accent, label: "one at a time" },
          ]}
        />
        <Note>Drag the start anywhere. Does the winner change? At what budget does batch catch up?</Note>
      </>
    );
  } else if (screen === 1 || screen === 2) {
    const seqText = order === "sorted" ? "small x first, large x last, every pass" : order === "fixed" ? "the same shuffled list, every pass" : "a fresh shuffle each pass";
    question =
      screen === 1 ? (
        <>
          <b>Claim to test:</b> per-sample updates only work if the samples come in random order (slides 10–14). Here is per-sample updating for 10 passes over the data (epochs), with the dataset sorted by x. Scrub through the updates: the ring on the right is the current sample. Where does each pass end (◆)? Then change the visiting order and compare.
        </>
      ) : (
        <>
          Name it. Updating after every sample is <b>stochastic gradient descent</b>; one pass over all N samples is an <b>epoch</b>, and gives N updates. Each update pulls the line toward one sample. With a fixed order, the same pulls come in the same sequence every epoch, so the weights trace the same loop: a <b>cycle</b>. The ◆ at each epoch end sits at the same spot instead of closing in. Reshuffling every epoch breaks the cycle (slides 10–15). Next: even with shuffling, a fixed η never fully settles.
        </>
      );
    stage = (
      <LossMap
        surf={SURF}
        target={O1.best}
        paths={[{ key: "e", pts: exRun.pts.slice(0, k + 1), color: THEME.accent, shape: "circle", width: 1.3, label: "update " + k }]}
        overlay={epochDiamonds(exRun, k)}
        ariaLabel="Loss over the two weights with the SGD path; diamonds mark the end of each epoch"
      />
    );
    side = (
      <>
        <Panel>
          <Pills options={ORDERS} value={order} onPick={setOrder} />
          <Slider id={"l1-eta" + screen} label="learning rate η" value={eta} min={0.005} max={0.12} stepSize={0.001} onChange={setEta} display={fmt(eta, 3)} />
          <Slider id={"l1-k" + screen} label="show updates up to" value={upto} min={0} max={400} stepSize={1} onChange={setUpto} display={k + " (epoch " + Math.min(10, Math.ceil(k / 40)) + ")"} />
          <Readout
            rows={[["visiting order", seqText]].concat(ends.map((i, j) => ["◆ distance at end of epoch " + Math.round(exRun.ks[i] / 40), fmt(dist(exRun.pts[i], O1.best), 3)]))}
          />
        </Panel>
        <FeePlot
          orders={O1}
          height={180}
          hi={k > 0 ? exRun.idx[k - 1] : null}
          lines={[
            { w: O1.best, color: THEME.text, dash: "2 3", label: "best", ldy: 14 },
            { w: wK, color: THEME.accent, label: "after update " + k },
          ]}
        />
        {screen === 1 ? (
          <Note>Scrub through one epoch of the sorted data. Which way does the line tilt in the first half, and in the second?</Note>
        ) : (
          <Eq size={12.5}>
            for each epoch:
            <br />
            &nbsp;&nbsp;shuffle the N samples
            <br />
            &nbsp;&nbsp;for n = 1..N: W ← W − η ∇loss(x⁽ⁿ⁾, y⁽ⁿ⁾)
          </Eq>
        )}
      </>
    );
  } else {
    const ticks = CHECK_KM.map((km) => {
      const b = feeAt(C1.best, km);
      return { km, lo: b - TOL_C, hi: b + TOL_C, ok: !chRun.diverged && Math.abs(feeAt(chRun.w, km) - b) <= TOL_C };
    });
    question = (
      <>
        A new dataset: 60 samples, again stored sorted by x. You can afford {EPOCHS_C} passes over it (180 sample gradients). The fitted line must be accurate at both ends: its prediction at x = 1 and at x = 8 within {fmt(TOL_C, 2)} of the best fit's. Pick the method, the visiting order and η.
      </>
    );
    stage = (
      <LossMap
        surf={SURF_C}
        target={C1.best}
        paths={[{ key: "c", pts: chRun.pts, color: chOk ? THEME.ok : THEME.accent, shape: methodC === "batch" ? "square" : "circle", width: 1.4, dash: methodC === "batch" ? "6 4" : undefined, label: methodC === "batch" ? "batch" : "SGD" }]}
        overlay={methodC === "batch" ? undefined : epochDiamonds(chRun, chRun.pts.length)}
        ariaLabel="Loss over the two weights for the new dataset with your run"
      />
    );
    side = (
      <>
        <Panel edge={chOk ? THEME.ok : THEME.grid}>
          <Pills options={METHODS} value={methodC} onPick={setMethodC} />
          {methodC === "sgd" && <Pills options={ORDERS} value={orderC} onPick={setOrderC} />}
          <LogSlider id="l1-etac" label="learning rate η" value={etaC} lo={-2.3} hi={-0.3} onChange={setEtaC} digits={3} />
          <Readout
            rows={[
              ["updates made", methodC === "batch" ? EPOCHS_C : 60 * EPOCHS_C],
              ["worst prediction error at x = 1 / x = 8", chRun.diverged ? "diverged" : fmt(chFair, 3)],
            ]}
          />
          <Goal met={chOk}>both predictions within {fmt(TOL_C, 2)} of the best fit</Goal>
          {methodC === "sgd" && orderC === "shuffle" && <Btn onClick={reshuffle}>new shuffle (seed {seedC})</Btn>}
          <Hint
            hints={[
              "Batch makes only 3 updates in 3 passes. Is that enough, whatever η is?",
              "With sorted data, the last samples of each pass all have large x, so the pass always ends tilted toward them. Change the visiting order (any well-mixed order removes that structure), then tune η: big enough to arrive in 180 updates, small enough that the last few single-sample kicks do not throw the line off. Around 0.02 is a good region.",
            ]}
          />
        </Panel>
        <FeePlot orders={C1} height={180} ticks={ticks} lines={[{ w: C1.best, color: THEME.text, dash: "2 3", label: "best", ldy: 14 }].concat(chRun.diverged ? [] : [{ w: chRun.w, color: chOk ? THEME.ok : THEME.accent, label: "yours" }])} />
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="One sample at a time" question={question} stage={stage} side={side} />;
}
