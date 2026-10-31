import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import { THEME, MONO } from "./common";

// Lesson 6 - Back-propagation on a computational graph (back-prop deck, slides 28-64).
// Aha: nobody differentiates the whole formula. Each node knows only its own little derivative and
// multiplies it by the gradient arriving from above: downstream = local × upstream. Repeat to the leaves.

function GraphView({ nodes, edges, W, H, showGrad, focus, ariaLabel }) {
  const by = {};
  nodes.forEach((n) => (by[n.id] = n));
  return (
    <div style={{ overflowX: "auto" }}>
      <svg viewBox={"0 0 " + W + " " + H} style={{ width: "100%", minWidth: 520, maxWidth: W * 1.25, display: "block", margin: "0 auto" }} role="img" aria-label={ariaLabel}>
        <rect x="0" y="0" width={W} height={H} fill={THEME.bg} stroke={THEME.grid} />
        {edges.map(([a, b], i) => {
          const A = by[a];
          const B = by[b];
          const mx = (A.x + B.x) / 2;
          const my = (A.y + B.y) / 2;
          const g = showGrad[a];
          return (
            <g key={i}>
              <line x1={A.x} y1={A.y} x2={B.x} y2={B.y} stroke={focus === a || focus === b ? THEME.text : THEME.grid} strokeWidth={1.4} />
              <text x={mx} y={my - 6} textAnchor="middle" fontFamily={MONO} fontSize={12} fill={THEME.accent} stroke={THEME.bg} strokeWidth={3} paintOrder="stroke">
                {fmt(A.val)}
              </text>
              {g !== undefined && (
                <text x={mx} y={my + 15} textAnchor="middle" fontFamily={MONO} fontSize={12} fill={THEME.highlight} stroke={THEME.bg} strokeWidth={3} paintOrder="stroke">
                  {fmt(g)}
                </text>
              )}
            </g>
          );
        })}
        {nodes.map((n) => {
          const leaf = !n.op;
          const g = showGrad[n.id];
          const isF = focus === n.id;
          return (
            <g key={n.id}>
              {leaf ? (
                <rect x={n.x - 24} y={n.y - 14} width={48} height={28} rx={2} fill={THEME.surface} stroke={isF ? THEME.highlight : THEME.grid} strokeWidth={isF ? 2 : 1.2} />
              ) : (
                <circle cx={n.x} cy={n.y} r={20} fill={THEME.surface} stroke={isF ? THEME.highlight : THEME.accent} strokeWidth={isF ? 2.4 : 1.4} />
              )}
              <text x={n.x} y={n.y + 4.5} textAnchor="middle" fontFamily={MONO} fontSize={leaf ? 12 : 13} fill={THEME.text}>
                {leaf ? n.name : n.op}
              </text>
              {leaf && (
                <text x={n.x} y={n.y - 20} textAnchor="middle" fontFamily={MONO} fontSize={11} fill={THEME.accent}>
                  {fmt(n.val)}
                </text>
              )}
              {leaf && g !== undefined && (
                <text x={n.x - 30} y={n.y + 4} textAnchor="end" fontFamily={MONO} fontSize={11.5} fill={THEME.highlight}>
                  ∂ {fmt(g)}
                </text>
              )}
              {!leaf && n.out && g !== undefined && (
                <text x={n.x + 26} y={n.y + 4} fontFamily={MONO} fontSize={12} fill={THEME.highlight}>
                  {fmt(g)}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      <div style={{ display: "flex", gap: 16, flexWrap: "wrap", fontFamily: MONO, fontSize: 12, marginTop: 4 }}>
        <span style={{ color: THEME.accent }}>above an edge: forward value</span>
        <span style={{ color: THEME.highlight }}>below an edge, or ∂ under a leaf: gradient of the output</span>
      </div>
    </div>
  );
}

// ---- screen 1: group order total f = (x + y) * z ----
function orderGraph(x, y, z) {
  const q = x + y;
  const f = q * z;
  return {
    f,
    nodes: [
      { id: "x", name: "x food", x: 60, y: 60, val: x },
      { id: "y", name: "y fee", x: 60, y: 150, val: y },
      { id: "z", name: "z ppl", x: 60, y: 240, val: z },
      { id: "q", op: "+", x: 220, y: 105, val: q },
      { id: "f", op: "×", x: 380, y: 170, val: f, out: true },
    ],
    edges: [
      ["x", "q"],
      ["y", "q"],
      ["q", "f"],
      ["z", "f"],
    ],
    grads: { f: 1, q: z, z: q, x: z, y: z },
  };
}

// ---- screen 2: the slides' sigmoid neuron f = 1 / (1 + exp(-(w0 x0 + w1 x1 + w2))) ----
function neuronGraph(v) {
  const m0 = v.w0 * v.x0;
  const m1 = v.w1 * v.x1;
  const s1 = m0 + m1;
  const s2 = s1 + v.w2;
  const n = -s2;
  const e = Math.exp(n);
  const a = e + 1;
  const f = 1 / a;
  const gf = 1;
  const ga = (-1 / (a * a)) * gf;
  const ge = 1 * ga;
  const gn = e * ge;
  const gs2 = -1 * gn;
  const gs1 = gs2;
  const gw2 = gs2;
  const gm0 = gs1;
  const gm1 = gs1;
  const grads = { f: gf, a: ga, e: ge, n: gn, s2: gs2, s1: gs1, w2: gw2, m0: gm0, m1: gm1, w0: v.x0 * gm0, x0: v.w0 * gm0, w1: v.x1 * gm1, x1: v.w1 * gm1 };
  const nodes = [
    { id: "w0", name: "w0", x: 100, y: 40, val: v.w0 },
    { id: "x0", name: "x0", x: 100, y: 105, val: v.x0 },
    { id: "w1", name: "w1", x: 100, y: 185, val: v.w1 },
    { id: "x1", name: "x1", x: 100, y: 250, val: v.x1 },
    { id: "w2", name: "w2", x: 310, y: 290, val: v.w2 },
    { id: "m0", op: "×", x: 200, y: 72, val: m0 },
    { id: "m1", op: "×", x: 200, y: 218, val: m1 },
    { id: "s1", op: "+", x: 300, y: 145, val: s1 },
    { id: "s2", op: "+", x: 400, y: 190, val: s2 },
    { id: "n", op: "×−1", x: 500, y: 190, val: n },
    { id: "e", op: "exp", x: 595, y: 190, val: e },
    { id: "a", op: "+1", x: 690, y: 190, val: a },
    { id: "f", op: "1/x", x: 785, y: 190, val: f, out: true },
  ];
  const edges = [
    ["w0", "m0"],
    ["x0", "m0"],
    ["w1", "m1"],
    ["x1", "m1"],
    ["m0", "s1"],
    ["m1", "s1"],
    ["s1", "s2"],
    ["w2", "s2"],
    ["s2", "n"],
    ["n", "e"],
    ["e", "a"],
    ["a", "f"],
  ];
  return { nodes, edges, grads, vals: { m0, m1, s1, s2, n, e, a, f } };
}
const START_V = { w0: 2, x0: -1, w1: -3, x1: -2, w2: -3 };
// each backward step: which values get their gradient, and the local rule used
const STEPS = [
  { ids: ["f"], rule: (G) => "start: ∂f/∂f = 1" },
  { ids: ["a"], rule: (G) => "1/x gate: local = −1/a² = " + fmt(-1 / (G.vals.a * G.vals.a)) + ", upstream 1.00 → " + fmt(G.grads.a) },
  { ids: ["e"], rule: (G) => "+1 gate: local = 1, upstream " + fmt(G.grads.a) + " → " + fmt(G.grads.e) },
  { ids: ["n"], rule: (G) => "exp gate: local = eⁿ = " + fmt(G.vals.e) + ", upstream " + fmt(G.grads.e) + " → " + fmt(G.grads.n) },
  { ids: ["s2"], rule: (G) => "×−1 gate: local = −1, upstream " + fmt(G.grads.n) + " → " + fmt(G.grads.s2) },
  { ids: ["s1", "w2"], rule: (G) => "+ gate: local = 1 for both inputs, so it hands " + fmt(G.grads.s2) + " to each" },
  { ids: ["m0", "m1"], rule: (G) => "+ gate again: both inputs get " + fmt(G.grads.s1) },
  { ids: ["w0", "x0", "w1", "x1"], rule: (G) => "× gates: each input gets upstream × the OTHER input: ∂w0 = x0·" + fmt(G.grads.m0) + " = " + fmt(G.grads.w0) + ", ∂x0 = w0·" + fmt(G.grads.m0) + " = " + fmt(G.grads.x0) },
];

// ---- screen 4: courier pay P = max(d, m) * r + d * k ----
function payGraph(v) {
  const mx = Math.max(v.d, v.m);
  const p1 = mx * v.r;
  const p2 = v.d * v.k;
  const P = p1 + p2;
  return {
    P,
    nodes: [
      { id: "d", name: "d km", x: 60, y: 70, val: v.d },
      { id: "m", name: "m min", x: 60, y: 150, val: v.m },
      { id: "r", name: "r €/km", x: 220, y: 230, val: v.r },
      { id: "k", name: "k fuel", x: 220, y: 300, val: v.k },
      { id: "mx", op: "max", x: 200, y: 110, val: mx },
      { id: "p1", op: "×", x: 340, y: 160, val: p1 },
      { id: "p2", op: "×", x: 340, y: 270, val: p2 },
      { id: "P", op: "+", x: 470, y: 210, val: P, out: true },
    ],
    edges: [
      ["d", "mx"],
      ["m", "mx"],
      ["mx", "p1"],
      ["r", "p1"],
      ["d", "p2"],
      ["k", "p2"],
      ["p1", "P"],
      ["p2", "P"],
    ],
  };
}
const PAY = { d: 3, m: 4, r: 1.5, k: 0.2 };
const PAY_KEYS = [
  ["d", "distance d"],
  ["m", "minimum distance m"],
  ["r", "rate r"],
  ["k", "fuel allowance k"],
];
function slopeOf(fn, v, key) {
  const h = 1e-4;
  const up = Object.assign({}, v, { [key]: v[key] + h });
  const dn = Object.assign({}, v, { [key]: v[key] - h });
  return (fn(up) - fn(dn)) / (2 * h);
}

function GuessRow({ k, label, guess, onGuess, tested, onTest, truth }) {
  const change = useCallback((e) => onGuess(k, e.target.value), [onGuess, k]);
  const test = useCallback(() => onTest(k), [onTest, k]);
  const g = parseFloat(guess);
  const ok = tested && isFinite(g) && Math.abs(g - truth) < 0.011;
  const X = (v) => 100 + clamp(v, -4, 4) * 22;
  return (
    <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr) 74px auto", gap: 6, alignItems: "center", margin: "6px 0" }}>
      <label htmlFor={"g-" + k} style={{ fontSize: 13.5, color: THEME.muted }}>
        ∂P/∂{k} <span style={{ fontSize: 12 }}>({label})</span>
      </label>
      <input id={"g-" + k} value={guess} onChange={change} inputMode="decimal" placeholder="?" style={{ fontFamily: MONO, fontSize: 13, background: THEME.bg, color: THEME.text, border: "1px solid " + THEME.grid, borderRadius: 3, padding: "4px 6px", width: "100%" }} />
      <Btn onClick={test}>nudge {k}</Btn>
      <svg viewBox="0 0 200 22" preserveAspectRatio="none" style={{ gridColumn: "1 / -1", width: "100%", height: 22 }} aria-hidden="true">
        <line x1="12" x2="188" y1="11" y2="11" stroke={THEME.grid} />
        <line x1="100" x2="100" y1="3" y2="19" stroke={THEME.muted} />
        {isFinite(g) && <polygon points={X(g) + ",3 " + (X(g) + 5) + ",11 " + X(g) + ",19 " + (X(g) - 5) + ",11"} fill="none" stroke={THEME.text} strokeWidth={1.5} />}
        {tested && <circle cx={X(truth)} cy={11} r={4.5} fill={ok ? THEME.accent : THEME.highlight} />}
        {tested && (
          <text x={X(truth) + (truth > 2 ? -8 : 8)} y={9} textAnchor={truth > 2 ? "end" : "start"} fontFamily={MONO} fontSize={9} fill={THEME.muted}>
            measured
          </text>
        )}
      </svg>
    </div>
  );
}

export default function GraphLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [order, setOrder] = useState({ x: 12, y: 3, z: 4 });
  const [measured, setMeasured] = useState({});
  const [stepN, setStepN] = useState(1);
  const [gate, setGate] = useState("add");
  const [ga, setGa] = useState(3);
  const [gb, setGb] = useState(-1);
  const [up, setUp] = useState(2);
  const [fx, setFx] = useState(2);
  const [fy, setFy] = useState(3);
  const [guess, setGuess] = useState({ d: "", m: "", r: "", k: "" });
  const [tested, setTested] = useState({});

  // screen 1
  const og = orderGraph(order.x, order.y, order.z);
  const nudge = useCallback((key) => {
    setOrder((o) => {
      const before = orderGraph(o.x, o.y, o.z).f;
      const n = Object.assign({}, o, { [key]: Math.round((o[key] + 0.1) * 100) / 100 });
      const after = orderGraph(n.x, n.y, n.z).f;
      setMeasured((m) => Object.assign({}, m, { [key]: (after - before) / 0.1 }));
      return n;
    });
  }, []);
  const resetOrder = useCallback(() => {
    setOrder({ x: 12, y: 3, z: 4 });
    setMeasured({});
  }, []);

  // screen 2
  const G = useMemo(() => neuronGraph(START_V), []);
  const shown = {};
  STEPS.slice(0, stepN).forEach((s) => s.ids.forEach((id) => (shown[id] = G.grads[id])));
  const backOne = useCallback(() => setStepN((n) => Math.min(STEPS.length, n + 1)), []);
  const resetBack = useCallback(() => setStepN(1), []);
  const focus = STEPS[stepN - 1].ids[0];
  const gateOut = gate === "add" ? [up, up] : gate === "mul" ? [up * gb, up * ga] : ga >= gb ? [up, 0] : [0, up];

  // screen 3: fan-out f = x*y + x
  const pathA = fy;
  const pathB = 1;

  // screen 4
  const pay = payGraph(PAY);
  const payFn = useCallback((v) => payGraph(v).P, []);
  const onGuess = useCallback((k, val) => setGuess((g) => Object.assign({}, g, { [k]: val })), []);
  const onTest = useCallback((k) => setTested((t) => Object.assign({}, t, { [k]: true })), []);
  const truths = {};
  PAY_KEYS.forEach(([k]) => (truths[k] = slopeOf(payFn, PAY, k)));
  const right = PAY_KEYS.filter(([k]) => tested[k] && Math.abs(parseFloat(guess[k]) - truths[k]) < 0.011).length;

  let question, stage, side;
  if (screen === 0) {
    question = (
      <>
        An office group order: everyone pays the food price x plus the delivery fee y, and there are z people. The total is f = (x + y)·z. Which of the three numbers pulls the total hardest? Guess first, then nudge each one by €0.10 or by 0.1 person and measure.
      </>
    );
    stage = <GraphView nodes={og.nodes} edges={og.edges} W={460} H={300} showGrad={{}} ariaLabel="Graph of the group order total" />;
    side = (
      <>
        <Panel>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
            <Btn onClick={() => nudge("x")}>nudge x +0.1</Btn>
            <Btn onClick={() => nudge("y")}>nudge y +0.1</Btn>
            <Btn onClick={() => nudge("z")}>nudge z +0.1</Btn>
            <Btn onClick={resetOrder}>reset</Btn>
          </div>
          <div style={{ fontFamily: MONO, fontSize: 14, lineHeight: 1.8, marginTop: 8 }}>
            <div>total f = €{fmt(og.f)}</div>
            {["x", "y", "z"].map((k) => (
              <div key={k}>
                Δf / Δ{k} = <b style={{ color: THEME.highlight }}>{measured[k] === undefined ? "?" : fmt(measured[k])}</b>
              </div>
            ))}
          </div>
        </Panel>
        <Note>Compare each measured ratio with the numbers on the graph. Which value on the graph equals Δf/Δz? Which one equals Δf/Δx?</Note>
      </>
    );
  } else if (screen === 1) {
    question = (
      <>
        Here is one neuron from the delivery-zone net, broken into tiny gates: f = σ(w0·x0 + w1·x1 + w2), with the numbers from the slides. Press "send back one gate" repeatedly. At each gate, which two numbers get multiplied?
      </>
    );
    stage = (
      <>
        <GraphView nodes={G.nodes} edges={G.edges} W={840} H={330} showGrad={shown} focus={focus} ariaLabel="Computational graph of a sigmoid neuron" />
        <Panel style={{ marginTop: 8 }}>
          <div style={{ fontFamily: MONO, fontSize: 13.5, lineHeight: 1.6 }}>
            step {stepN} / {STEPS.length}: {STEPS[stepN - 1].rule(G)}
          </div>
          <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
            <Btn onClick={backOne} disabled={stepN === STEPS.length} strong>
              send back one gate
            </Btn>
            <Btn onClick={resetBack}>restart</Btn>
          </div>
        </Panel>
      </>
    );
    side = (
      <>
        <Panel>
          <Label>Three gate patterns (slide 40)</Label>
          <div style={{ display: "flex", gap: 6, marginBottom: 6 }}>
            <Pill value="add" active={gate === "add"} onPick={setGate}>
              add
            </Pill>
            <Pill value="mul" active={gate === "mul"} onPick={setGate}>
              multiply
            </Pill>
            <Pill value="max" active={gate === "max"} onPick={setGate}>
              max
            </Pill>
          </div>
          <Slider id="l6-a" label="input a" value={ga} min={-4} max={4} stepSize={0.05} onChange={setGa} />
          <Slider id="l6-b" label="input b" value={gb} min={-4} max={4} stepSize={0.05} onChange={setGb} />
          <Slider id="l6-u" label="upstream gradient" value={up} min={-3} max={3} stepSize={0.05} onChange={setUp} />
          <div style={{ fontFamily: MONO, fontSize: 14, lineHeight: 1.8 }}>
            <div>
              ∂/∂a = <b style={{ color: THEME.highlight }}>{fmt(gateOut[0])}</b>
            </div>
            <div>
              ∂/∂b = <b style={{ color: THEME.highlight }}>{fmt(gateOut[1])}</b>
            </div>
          </div>
        </Panel>
        <Note>Which gate copies the upstream gradient to both inputs? Which one swaps the inputs? Which one picks a single input and gives the other nothing?</Note>
      </>
    );
  } else if (screen === 2) {
    question = (
      <>
        Name it. Every gate did the same thing: downstream = local × upstream. That is the chain rule applied one gate at a time. What happens when a value feeds two gates? Change x and y below and compare the two paths with the total.
      </>
    );
    stage = (
      <>
        <Panel>
          <Label>chain rule, one gate at a time (slides 28, 38)</Label>
          <Eq size={14}>
            z = f(g(x)) &nbsp;⇒&nbsp; ∂L/∂x = ∂L/∂z · ∂z/∂x
            <br />
            downstream gradient = upstream gradient × local gradient
          </Eq>
          <Label>
            <span style={{ display: "block", marginTop: 6 }}>a value used twice: the paths add (slide 29)</span>
          </Label>
          <Slider id="l6-fx" label="x" value={fx} min={-3} max={3} stepSize={0.05} onChange={setFx} />
          <Slider id="l6-fy" label="y" value={fy} min={-3} max={3} stepSize={0.05} onChange={setFy} />
          <Eq size={13.5}>
            f = x·y + x = {fmt(fx * fy + fx)}
            <br />
            path through ×: y = {fmt(pathA)} &nbsp;&nbsp;path through +: 1
            <br />
            ∂f/∂x = {fmt(pathA)} + {fmt(pathB)} = <span style={{ color: THEME.highlight }}>{fmt(pathA + pathB)}</span> &nbsp;(nudge check: {fmt(slopeOf((v) => v.x * v.y + v.x, { x: fx, y: fy }, "x"))})
          </Eq>
        </Panel>
      </>
    );
    side = (
      <>
        <Panel>
          <Label>sigmoid gate shortcut (slides 57-59)</Label>
          <Eq size={13.5}>
            σ′(z) = σ(z)·(1 − σ(z))
            <br />= {fmt(G.vals.f)} × {fmt(1 - G.vals.f)} = {fmt(G.vals.f * (1 - G.vals.f))}
          </Eq>
          <Note>That equals the gradient you reached at s2 by passing through four small gates. A graph can be drawn coarse or fine. Pick gates whose local gradients are easy to write down.</Note>
        </Panel>
        <Panel>
          <Label>forward / backward API (slides 60-64)</Label>
          <pre style={{ fontFamily: MONO, fontSize: 12.5, margin: 0, color: THEME.text, overflowX: "auto", lineHeight: 1.5 }}>
            {"class Multiply:\n  def forward(self, a, b):\n    self.a, self.b = a, b   # cache\n    return a * b\n  def backward(self, up):\n    return up * self.b, up * self.a"}
          </pre>
          <Note>
            <span style={{ display: "block", marginTop: 6 }}>PyTorch's autograd is a library of these gates. Forward caches the inputs, backward multiplies the upstream gradient by the local one.</span>
          </Note>
        </Panel>
      </>
    );
  } else {
    question = (
      <>
        Courier pay for one trip: P = max(d, m)·r + d·k. The trip is 3 km, but couriers are paid for at least m = 4 km at r = €1.50/km, plus a fuel allowance k = €0.20 for each real km. Work out all four gradients by passing back through the graph. Type each one, then nudge that input to see if you were right.
      </>
    );
    stage = (
      <>
        <GraphView nodes={pay.nodes} edges={pay.edges} W={560} H={340} showGrad={{}} ariaLabel="Graph of courier pay" />
      </>
    );
    side = (
      <>
        <Panel edge={right === 4 ? THEME.accent : THEME.grid}>
          {PAY_KEYS.map(([k, label]) => (
            <GuessRow key={k} k={k} label={label} guess={guess[k]} onGuess={onGuess} tested={!!tested[k]} onTest={onTest} truth={truths[k]} />
          ))}
          <div style={{ fontFamily: MONO, fontSize: 12.5, color: THEME.muted, marginTop: 4 }}>◇ your guess · ● measured slope (cyan when they agree)</div>
          <Hint
            hints={[
              "Start at P with gradient 1. A + gate copies it to both branches. Then each × gate swaps: the branch to r gets the other input, max(d, m).",
              "d reaches P along two paths, and their contributions add. One path goes through max, and max routes the whole gradient to the bigger input. Which input is bigger here?",
            ]}
          />
        </Panel>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="Pass the blame backward" question={question} stage={stage} side={side} wide={screen === 1} />;
}
