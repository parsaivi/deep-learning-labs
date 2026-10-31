import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import { THEME, MONO } from "./common";

// Lesson 6 - Adaptive per-weight step sizes: Adagrad, RMSprop, Adam (Optimization I, slides 76-92).
// Aha: divide each weight's gradient by its own typical gradient size. Weights with big, jumpy gradients
// get small steps, weights with tiny gradients get big ones, so all weights move at a similar pace.

// w1 = per-km rate (every delivery has a distance: big gradients), w2 = snow surcharge (1 delivery in 30 has snow)
const W_STAR = [1.5, 3.0];
const A = [
  [6, 0],
  [0, 0.2],
];
const VALLEY = { f: quad(A, W_STAR), xr: [-1.5, 4.5], yr: [0, 6], xl: "w₁ €/km", yl: "w₂ snow surcharge €" };
const START = [-1.0, 5.5];
const TILTED = { f: quad(rotA(6, 0.2, 40), W_STAR), xr: [-1.5, 4.5], yr: [0, 6], xl: "w₁", yl: "w₂" };
function plateau(w) {
  return 2 * w[0] * w[0] + 0.5 * (Math.pow(w[1], 4) / 4 - (w[1] * w[1]) / 2);
}
const PLATEAU = { f: plateau, xr: [-1.8, 1.8], yr: [-1.8, 1.8], xl: "w₁", yl: "w₂", floor: 0.01 };
const P_START = [1.5, 0.002];
const SURFS = { valley: [VALLEY, START, W_STAR], tilted: [TILTED, START, W_STAR], plateau: [PLATEAU, P_START, [0, 1]] };
const OPTS = {
  gd: { kind: "gd", label: "GD", color: THEME.text, dash: "4 3", shape: "square" },
  momentum: { kind: "momentum", beta: 0.9, label: "momentum", color: THEME.warn, dash: "2 3", shape: "triangle" },
  rmsprop: { kind: "rmsprop", beta: 0.9, label: "RMSprop", color: THEME.highlight, shape: "circle" },
  adam: { kind: "adam", beta: 0.9, beta2: 0.999, bias: true, label: "Adam", color: THEME.ok, dash: "6 3", shape: "diamond" },
};

function perCoord(f, w0, e1, e2, n) {
  let w = w0.slice();
  const pts = [w.slice()];
  const g1 = [];
  const g2 = [];
  for (let k = 0; k < n; k++) {
    const g = grad2(f, w);
    g1.push(g[0]);
    g2.push(g[1]);
    w = [w[0] - e1 * g[0], w[1] - e2 * g[1]];
    if (Math.abs(w[0]) > 60 || Math.abs(w[1]) > 60) return { pts, g1, g2, diverged: true };
    pts.push(w.slice());
  }
  return { pts, g1, g2, diverged: false };
}
function rms(a) {
  if (!a.length) return 0;
  let s = 0;
  for (const v of a) s += v * v;
  return Math.sqrt(s / a.length);
}
function plateauHit(pts) {
  return pts.findIndex((p) => Math.min(Math.hypot(p[0], p[1] - 1), Math.hypot(p[0], p[1] + 1)) < 0.15);
}
function EmaPlot({ gamma, t }) {
  // running average of a constant signal 1 started from zero, with and without bias correction
  const raw = [];
  const fix = [];
  let v = 0;
  for (let k = 1; k <= 40; k++) {
    v = gamma * v + (1 - gamma) * 1;
    raw.push(v);
    fix.push(v / (1 - Math.pow(gamma, k)));
  }
  return (
    <Trace
      series={[
        { vals: [0].concat(raw), color: THEME.warn, dash: "3 3", label: "v_t (starts at 0)" },
        { vals: [0].concat(fix), color: THEME.highlight, label: "v_t / (1 − γᵗ)" },
      ]}
      n={40}
      yr={[0, 1.2]}
      label="running average of a gradient that is always 1"
      height={140}
    />
  );
}

export default function AdaptiveLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [e1, setE1] = useState(0.1);
  const [e2, setE2] = useState(0.1);
  const [surfKey, setSurfKey] = useState("valley");
  const [etaG, setEtaG] = useState(0.15);
  const [etaA, setEtaA] = useState(0.3);
  const [on, setOn] = useState({ gd: true, momentum: false, rmsprop: true, adam: false });
  const [gamma, setGamma] = useState(0.9);
  const [kindC, setKindC] = useState("gd");
  const [etaC, setEtaC] = useState(0.2);

  const pc = useMemo(() => perCoord(VALLEY.f, START, e1, e2, 10), [e1, e2]);
  const pcEnd = pc.pts[pc.pts.length - 1];
  const pcOk = !pc.diverged && Math.hypot(pcEnd[0] - W_STAR[0], pcEnd[1] - W_STAR[1]) < 0.1;

  const [surf, st, tgt] = SURFS[surfKey];
  const runs = useMemo(() => {
    const out = {};
    for (const k of Object.keys(OPTS)) {
      const o = OPTS[k];
      out[k] = runOpt(surf.f, st, Object.assign({}, o, { eta: k === "gd" || k === "momentum" ? (k === "momentum" ? etaG / 4 : etaG) : etaA }), 40, 60);
    }
    return out;
  }, [surf, st, etaG, etaA]);
  const togglers = useMemo(() => {
    const t = {};
    for (const k of Object.keys(OPTS)) t[k] = (v) => setOn((p) => Object.assign({}, p, { [k]: v }));
    return t;
  }, []);
  const pickSurf = useCallback((k) => setSurfKey(k), []);
  const pickKindC = useCallback((k) => setKindC(k), []);
  const pathsFor = (keys) =>
    keys
      .filter((k) => on[k])
      .map((k) => ({ key: k, pts: runs[k].pts, color: OPTS[k].color, dash: OPTS[k].dash, shape: OPTS[k].shape, label: OPTS[k].label + (runs[k].diverged ? " ✕" : "") }));
  const rmsScale = runs.rmsprop.scale;
  const adagrad = useMemo(() => runOpt(surf.f, st, { kind: "adagrad", eta: etaA }, 40, 60), [surf, st, etaA]);

  const cRun = useMemo(() => runOpt(plateau, P_START, Object.assign({}, OPTS[kindC] || { kind: "adagrad" }, { kind: kindC, eta: etaC }), 30, 60), [kindC, etaC]);
  const hit = cRun.diverged ? -1 : plateauHit(cRun.pts);
  const okC = hit >= 0 && hit <= 15;

  let question, stage, side;
  if (screen === 0) {
    question = (
      <>
        Two weights again: the per-km rate, which every delivery uses, and a snow surcharge, which only snowy deliveries use. This time each weight gets its own learning rate. Find a pair that lands within 0.1 of the best in 10 steps. Then compare each weight's typical gradient size with the rate you gave it.
      </>
    );
    stage = (
      <>
        <LossMap surf={VALLEY} paths={[{ pts: pc.pts, color: THEME.highlight, label: pc.diverged ? "diverged" : "after 10 steps" }]} target={W_STAR} ariaLabel="Per-weight learning rates on the fee valley" />
      </>
    );
    side = (
      <>
        <Panel edge={pcOk ? THEME.ok : THEME.grid}>
          <Slider id="l6-e1" label="η for the per-km rate" value={e1} min={0.01} max={0.4} stepSize={0.001} onChange={setE1} display={fmt(e1, 3)} />
          <Slider id="l6-e2" label="η for the snow surcharge" value={e2} min={0.05} max={10} stepSize={0.01} onChange={setE2} display={fmt(e2, 2)} />
          <Goal met={pcOk}>within 0.1 of the best after 10 steps</Goal>
          <Readout
            rows={[
              ["typical |gradient|, per-km rate", fmt(rms(pc.g1), 3)],
              ["typical |gradient|, surcharge", fmt(rms(pc.g2), 3)],
              ["their ratio", fmt(rms(pc.g1) / (rms(pc.g2) || 1), 1) + " : 1"],
              ["ratio of your η's (surcharge : km)", fmt(e2 / e1, 1) + " : 1"],
            ]}
          />
        </Panel>
        <Note>Could the optimizer choose these rates itself, using only the gradients it has seen so far?</Note>
      </>
    );
  } else if (screen === 1) {
    question = (
      <>
        RMSprop does it automatically. It keeps a running average of each weight's squared gradient and divides that weight's step by its root. Race it against plain gradient descent. Which weight gets the bigger effective learning rate, and why? Then try the tilted bowl and the flat saddle. Where does per-weight scaling help least?
      </>
    );
    stage = (
      <>
        <LossMap surf={surf} paths={pathsFor(["gd", "momentum", "rmsprop", "adam"])} target={tgt} ariaLabel="Optimizer race" />
        <Trace
          series={[
            { vals: rmsScale.map((s) => s[0]), color: THEME.highlight, label: "RMSprop η on w₁" },
            { vals: rmsScale.map((s) => s[1]), color: THEME.highlight, dash: "5 3", label: "RMSprop η on w₂" },
          ]}
          n={40}
          yr={[-2, 2]}
          log
          label="effective learning rate η/√(v + ε) per weight (log scale)"
          height={130}
        />
      </>
    );
    side = (
      <>
        <Panel>
          <Pills
            options={[
              ["valley", "fee valley"],
              ["tilted", "tilted valley"],
              ["plateau", "flat saddle"],
            ]}
            value={surfKey}
            onPick={pickSurf}
          />
          {Object.keys(OPTS).map((k) => (
            <Toggle key={k} label={OPTS[k].label} value={on[k]} onChange={togglers[k]} />
          ))}
          <Slider id="l6-etag" label="η for GD (momentum uses η/4, β = 0.9)" value={etaG} min={0.01} max={0.34} stepSize={0.001} onChange={setEtaG} display={fmt(etaG, 3)} />
          <Slider id="l6-etaa" label="η for RMSprop / Adam" value={etaA} min={0.01} max={1} stepSize={0.001} onChange={setEtaA} display={fmt(etaA, 3)} />
        </Panel>
        <Note>RMSprop scales each axis of the coordinate system on its own. A tilted bowl's axes are not the coordinate axes.</Note>
      </>
    );
  } else if (screen === 2) {
    question = (
      <>
        Name it. All three methods divide by a root mean square of past gradients, one value per weight. Adagrad sums every squared gradient ever seen, so its steps keep shrinking. RMSprop and Adam use a fading average instead. A fading average that starts at 0 begins far too small. Drag γ and see why Adam divides by (1 − γᵗ).
      </>
    );
    stage = (
      <>
        <EmaPlot gamma={gamma} />
        <Trace
          series={[
            { vals: adagrad.scale.map((s) => s[1]), color: THEME.warn, dash: "3 3", label: "Adagrad η on w₂" },
            { vals: rmsScale.map((s) => s[1]), color: THEME.highlight, label: "RMSprop η on w₂" },
          ]}
          n={40}
          yr={[-2, 2]}
          log
          label={"effective learning rate on w₂ over 40 steps, " + (surfKey === "plateau" ? "flat saddle" : surfKey === "tilted" ? "tilted valley" : "fee valley")}
          height={130}
        />
      </>
    );
    side = (
      <>
        <Panel>
          <Slider id="l6-gamma" label="averaging factor γ" value={gamma} min={0.5} max={0.999} stepSize={0.001} onChange={setGamma} display={fmt(gamma, 3)} />
          <Eq size={12}>
            Adagrad: s_k = s_k−1 + g², w −= η·g/√(s_k + ε)
            <br />
            RMSprop: v_k = βv_k−1 + (1−β)g², w −= η·g/√(v_k + ε)
            <br />
            Adam: m_k = β₁m_k−1 + (1−β₁)g, v_k = β₂v_k−1 + (1−β₂)g²
            <br />
            m̂ = m_k/(1−β₁ᵏ), v̂ = v_k/(1−β₂ᵏ), w −= η·m̂/(√v̂ + ε)
            <br />
            defaults: η = 0.001, β₁ = 0.9, β₂ = 0.999, ε = 1e−8
          </Eq>
          <Note>Adam is RMSprop plus momentum: m̂ is the momentum average from lesson 5, √v̂ is the per-weight scale. With β₂ = 0.999 the uncorrected v_k starts about 1000 times too small, so the first steps would be far too large (slides 82-86).</Note>
        </Panel>
      </>
    );
  } else {
    question = (
      <>
        A new weight starts on a flat saddle at w₂ = 0.002, where its gradient is almost zero. Two minima wait at w₂ = ±1. Reach one of them (within 0.15) in at most 15 steps. Pick the method and its η.
      </>
    );
    stage = (
      <>
        <LossMap surf={PLATEAU} paths={[{ pts: cRun.pts.slice(0, 16), color: THEME.highlight, label: cRun.diverged ? "diverged" : "first 15 steps" }]} target={[0, 1]} ariaLabel="Flat saddle challenge" />
        <Trace series={[{ vals: cRun.pts.map((p) => Math.abs(p[1])), color: THEME.highlight, label: "|w₂|" }]} n={15} yr={[-3, 0.3]} log label="how far w₂ has moved off the saddle (log scale)" height={120} />
      </>
    );
    side = (
      <>
        <Panel edge={okC ? THEME.ok : THEME.grid}>
          <Pills
            options={[
              ["gd", "GD"],
              ["momentum", "momentum β=.9"],
              ["adagrad", "Adagrad"],
              ["rmsprop", "RMSprop β=.9"],
              ["adam", "Adam"],
            ]}
            value={kindC}
            onPick={pickKindC}
          />
          <Slider id="l6-etac" label="learning rate η" value={etaC} min={0.01} max={1} stepSize={0.001} onChange={setEtaC} display={fmt(etaC, 3)} />
          <Goal met={okC}>reached a minimum within 15 steps {hit >= 0 ? "(step " + hit + ")" : ""}</Goal>
          <Hint
            hints={[
              "With plain gradient descent, the step on w₂ is η times a gradient near 0.001. What would make that step independent of how small the gradient is?",
              "A method that divides by √(average g²) moves each weight by about η per step, whatever the gradient's size. Try RMSprop with η between 0.1 and 0.4.",
            ]}
          />
        </Panel>
        <Note>One caution from the last slide: adaptive methods often reach a different solution than SGD, and on some tasks it generalizes worse (Wilson et al., 2017).</Note>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="A step size per weight" question={question} stage={stage} side={side} />;
}
