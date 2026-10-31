import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import { THEME, MONO, ZoneMap } from "./common";

// Lesson 7 - Back-prop in vector form (back-prop deck, slides 65-100).
// Aha: for a layer z = W a + b the gradient of every weight is (error at its output neuron) x
// (activity at its input): dL/dW = delta a^T, one outer product, always the same shape as W.
// A dead ReLU zeroes a whole row, a zero input zeroes a whole column.

// inputs are centred map coordinates: u = 4(x - 0.5), v = 4(y - 0.5)
function inputs(x, y) {
  return [4 * (x - 0.5), 4 * (y - 0.5)];
}
function forward(net, x, y) {
  const a0 = inputs(x, y);
  const H = net.W1.length;
  const z1 = [];
  const a1 = [];
  for (let i = 0; i < H; i++) {
    const z = net.W1[i][0] * a0[0] + net.W1[i][1] * a0[1] + net.b1[i];
    z1.push(z);
    a1.push(Math.max(0, z));
  }
  let z2 = net.b2;
  for (let i = 0; i < H; i++) z2 += net.W2[i] * a1[i];
  return { a0, z1, a1, z2, o: sig(z2) };
}
function backward(net, f, y) {
  const d2 = f.o - y; // sigmoid + cross-entropy
  const gW2 = f.a1.map((a) => d2 * a);
  const d1 = net.W2.map((w, i) => w * d2 * (f.z1[i] > 0 ? 1 : 0));
  const gW1 = d1.map((d) => [d * f.a0[0], d * f.a0[1]]);
  return { d2, gW2, gb2: d2, d1, gW1, gb1: d1 };
}
function lossOf(o, y) {
  const p = clamp(o, 1e-7, 1 - 1e-7);
  return -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
}

const HAND = {
  W1: [
    [1.5, -0.8],
    [-1.0, 1.2],
    [0.6, 0.9],
  ],
  b1: [-0.3, 0.2, 0.6],
  W2: [1.4, -1.1, 0.8],
  b2: -1.9,
};
const ADDR = { x: 0.62, y: 0.35, s: 1 };

function Cell({ v, max, shown, onPick, id, label, hot }) {
  const click = useCallback(() => onPick && onPick(id), [onPick, id]);
  const t = shown ? clamp(Math.abs(v) / max, 0, 1) : 0;
  const pos = v >= 0;
  return (
    <button
      onClick={click}
      aria-label={label}
      style={{
        fontFamily: MONO,
        fontSize: 13,
        minWidth: 0,
        padding: "8px 4px",
        cursor: onPick ? "pointer" : "default",
        borderRadius: 2,
        border: "1px " + (shown && !pos ? "dashed " : "solid ") + (hot ? THEME.highlight : shown ? (pos ? THEME.accent : THEME.warn) : THEME.grid),
        background: shown ? rgba(pos ? THEME.accent : THEME.warn, 0.08 + 0.45 * t) : THEME.bg,
        color: THEME.text,
      }}
    >
      {shown ? sgn(v) : "?"}
    </button>
  );
}
function Matrix({ rows, max, shown, onPick, prefix, rowHead, colHead, hotRow }) {
  const cols = rows[0].length;
  return (
    <div style={{ display: "grid", gridTemplateColumns: (rowHead ? "auto " : "") + "repeat(" + cols + ", minmax(56px, 1fr))", gap: 4, alignItems: "center" }}>
      {colHead && (
        <>
          {rowHead && <span />}
          {colHead.map((c, j) => (
            <span key={j} style={{ fontFamily: MONO, fontSize: 11.5, color: THEME.muted, textAlign: "center" }}>
              {c}
            </span>
          ))}
        </>
      )}
      {rows.map((r, i) => (
        <React.Fragment key={i}>
          {rowHead && <span style={{ fontFamily: MONO, fontSize: 11.5, color: THEME.muted, paddingRight: 4 }}>{rowHead[i]}</span>}
          {r.map((v, j) => (
            <Cell key={j} v={v} max={max} shown={shown ? shown(i, j) : true} onPick={onPick} id={prefix + i + "-" + j} label={"row " + (i + 1) + " column " + (j + 1)} hot={hotRow === i} />
          ))}
        </React.Fragment>
      ))}
    </div>
  );
}

// ---------- training data for the challenge: the downtown disc ----------
const ZONE = (function () {
  const r = rng(29);
  const out = [];
  while (out.length < 44) {
    const x = 0.06 + 0.88 * r();
    const y = 0.06 + 0.88 * r();
    const d = Math.hypot(x - 0.5, y - 0.52);
    if (d < 0.2 && out.filter((p) => p.s).length < 18) out.push({ x, y, s: 1 });
    else if (d > 0.29 && out.filter((p) => !p.s).length < 26) out.push({ x, y, s: 0 });
  }
  return out;
})();
function initNet(H, seed) {
  const r = rng(seed);
  const gauss = () => Math.sqrt(-2 * Math.log(r() + 1e-9)) * Math.cos(2 * Math.PI * r());
  const W1 = [];
  const b1 = [];
  const W2 = [];
  for (let i = 0; i < H; i++) {
    W1.push([gauss(), gauss()]);
    b1.push(0.5 + 0.3 * gauss());
    W2.push(gauss() * Math.sqrt(2 / H));
  }
  return { W1, b1, W2, b2: 0 };
}
function epoch(net, lr) {
  const H = net.W1.length;
  const gW1 = net.W1.map(() => [0, 0]);
  const gb1 = new Array(H).fill(0);
  const gW2 = new Array(H).fill(0);
  let gb2 = 0;
  let L = 0;
  for (const p of ZONE) {
    const f = forward(net, p.x, p.y);
    const g = backward(net, f, p.s);
    L += lossOf(f.o, p.s);
    for (let i = 0; i < H; i++) {
      gW1[i][0] += g.gW1[i][0];
      gW1[i][1] += g.gW1[i][1];
      gb1[i] += g.gb1[i];
      gW2[i] += g.gW2[i];
    }
    gb2 += g.gb2;
  }
  const n = ZONE.length;
  return {
    net: {
      W1: net.W1.map((w, i) => [w[0] - (lr * gW1[i][0]) / n, w[1] - (lr * gW1[i][1]) / n]),
      b1: net.b1.map((b, i) => b - (lr * gb1[i]) / n),
      W2: net.W2.map((w, i) => w - (lr * gW2[i]) / n),
      b2: net.b2 - (lr * gb2) / n,
    },
    loss: L / n,
  };
}
function Spark({ vals }) {
  const W = 300;
  const H = 60;
  if (vals.length < 2) return <div style={{ height: H }} />;
  const mx = Math.max(0.8, ...vals);
  const pts = vals.map((v, i) => (i / (vals.length - 1)) * W + "," + (H - 4 - (v / mx) * (H - 10))).join(" ");
  return (
    <svg viewBox={"0 0 " + W + " " + H} style={{ width: "100%", height: H, display: "block" }} role="img" aria-label="Training loss over time">
      <rect x="0" y="0" width={W} height={H} fill={THEME.bg} stroke={THEME.grid} />
      <polygon points={"0," + H + " " + pts + " " + W + "," + H} fill={rgba(THEME.accent, 0.15)} />
      <polyline points={pts} fill="none" stroke={THEME.accent} strokeWidth={1.6} />
      <circle cx={W} cy={H - 4 - (vals[vals.length - 1] / mx) * (H - 10)} r={3} fill={THEME.highlight} />
      <text x={4} y={12} fontFamily={MONO} fontSize={10} fill={THEME.muted}>
        training loss
      </text>
    </svg>
  );
}

export default function LayerLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [measured, setMeasured] = useState({});
  const [addr, setAddr] = useState([ADDR.x, ADDR.y]);
  const [label, setLabel] = useState(1);
  const [H, setH] = useState(2);
  const [seed, setSeed] = useState(1);
  const [lr, setLr] = useState(0.6);
  const [net, setNet] = useState(() => initNet(2, 1));
  const [hist, setHist] = useState([]);
  const [running, setRunning] = useState(false);
  const [epochs, setEpochs] = useState(0);
  const netRef = useRef(net);
  netRef.current = net;

  // screen 1: measure one weight at a time by nudging it
  const f0 = forward(HAND, ADDR.x, ADDR.y);
  const L0 = lossOf(f0.o, ADDR.s);
  const g0 = backward(HAND, f0, ADDR.s);
  const measure = useCallback((id) => {
    const [i, j] = id.slice(1).split("-").map(Number);
    const h = 1e-4;
    const W1 = HAND.W1.map((r) => r.slice());
    W1[i][j] += h;
    const up = lossOf(forward(Object.assign({}, HAND, { W1 }), ADDR.x, ADDR.y).o, ADDR.s);
    W1[i][j] -= 2 * h;
    const dn = lossOf(forward(Object.assign({}, HAND, { W1 }), ADDR.x, ADDR.y).o, ADDR.s);
    setMeasured((m) => Object.assign({}, m, { [i + "-" + j]: (up - dn) / (2 * h) }));
  }, []);
  const nMeasured = Object.keys(measured).length;

  // screens 2 and 3: live address
  const onMove = useCallback((id, x, y) => setAddr([x, y]), []);
  const f = forward(HAND, addr[0], addr[1]);
  const g = backward(HAND, f, label);
  const handField = useCallback((x, y) => forward(HAND, x, y).o, []);
  const maxG = Math.max(0.05, ...g.gW1.map((r) => Math.max(Math.abs(r[0]), Math.abs(r[1]))));

  // screen 4: training
  const reinit = useCallback(
    (h, s) => {
      setRunning(false);
      setNet(initNet(h, s));
      setHist([]);
      setEpochs(0);
    },
    []
  );
  const setWidth = useCallback(
    (v) => {
      setH(v);
      reinit(v, seed);
    },
    [reinit, seed]
  );
  const newSeed = useCallback(() => {
    const s = seed + 1;
    setSeed(s);
    reinit(H, s);
  }, [seed, H, reinit]);
  const toggleRun = useCallback(() => setRunning((r) => !r), []);
  useEffect(() => {
    if (!running) return;
    let raf;
    let alive = true;
    const tick = () => {
      if (!alive) return;
      let n = netRef.current;
      let L = 0;
      for (let k = 0; k < (REDUCED ? 200 : 25); k++) {
        const r = epoch(n, lr);
        n = r.net;
        L = r.loss;
      }
      setNet(n);
      setEpochs((e) => e + (REDUCED ? 200 : 25));
      setHist((h) => (h.length > 160 ? h.slice(1) : h).concat([L]));
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => {
      alive = false;
      cancelAnimationFrame(raf);
    };
  }, [running, lr]);
  useEffect(() => {
    if (running && epochs >= 6000) setRunning(false);
  }, [epochs, running]);
  const trainField = useCallback((x, y) => forward(net, x, y).o, [net]);
  const trainWrong = countWrong(ZONE, trainField);
  const deadUnits = useMemo(() => {
    let dead = 0;
    for (let i = 0; i < net.W1.length; i++) {
      let any = false;
      for (const p of ZONE) if (forward(net, p.x, p.y).z1[i] > 0) any = true;
      if (!any) dead++;
    }
    return dead;
  }, [net]);

  const hiddenHead = ["h₁", "h₂", "h₃"];
  let question, stage, side;
  if (screen === 0) {
    question = (
      <>
        A small zone network (2 inputs → 3 ReLU hidden → 1 sigmoid output) gives a served address only o = {fmt(f0.o, 2)}. The six first-layer weights W₁ are hidden below. Click a cell to nudge that weight and measure how the loss responds. Which weights matter most? After a few, can you predict the rest?
      </>
    );
    stage = (
      <>
        <ZoneMap points={[ADDR]} field={handField} pred={handField} res={80} ariaLabel="Zone map of the hand-set network" maxWidth={420} />
        <Legend />
      </>
    );
    side = (
      <>
        <Panel>
          <Label>∂L/∂W₁, measured by nudging (3 × 2, same shape as W₁)</Label>
          <Matrix rows={HAND.W1.map((r, i) => r.map((_, j) => (measured[i + "-" + j] === undefined ? 0 : measured[i + "-" + j])))} max={Math.max(0.05, ...Object.values(measured).map(Math.abs))} shown={(i, j) => measured[i + "-" + j] !== undefined} onPick={measure} prefix="m" rowHead={hiddenHead} colHead={["from u (east)", "from v (north)"]} />
          <div style={{ fontFamily: MONO, fontSize: 13, color: THEME.muted, marginTop: 8, lineHeight: 1.6 }}>
            loss = {fmt(L0, 3)} · input (u, v) = ({fmt(f0.a0[0])}, {fmt(f0.a0[1])})
            <br />
            hidden a₁ = [{f0.a1.map((a) => fmt(a)).join(", ")}]
          </div>
        </Panel>
        {nMeasured >= 3 && (
          <Note>
            Compare the two columns. Is the second column a fixed multiple of the first? Now look at the hidden values a₁. Which row is zero, and why?
          </Note>
        )}
      </>
    );
  } else if (screen === 1) {
    question = (
      <>
        The gradient grid is built from just two vectors: δ₁, the error signal at each hidden neuron (left), and the input (u, v) (top). Each cell is the product of its row and column. Drag the address around and flip its label. Can you make a whole row vanish? A whole column?
      </>
    );
    stage = (
      <>
        <ZoneMap points={[{ x: addr[0], y: addr[1], s: label }]} field={handField} handles={[{ id: "addr", x: addr[0], y: addr[1], color: THEME.highlight, shape: "square" }]} onMove={onMove} res={80} ariaLabel="Drag the address" maxWidth={420} />
        <div style={{ display: "flex", gap: 6, marginTop: 8, alignItems: "center", flexWrap: "wrap" }}>
          <span style={{ fontSize: 13, color: THEME.muted }}>this address is</span>
          <Pill value={1} active={label === 1} onPick={setLabel}>
            ● served (y = 1)
          </Pill>
          <Pill value={0} active={label === 0} onPick={setLabel}>
            □ not served (y = 0)
          </Pill>
        </div>
      </>
    );
    side = (
      <>
        <Panel>
          <div style={{ display: "grid", gridTemplateColumns: "auto minmax(0, 1fr)", gap: 8, alignItems: "end" }}>
            <span />
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 4, fontFamily: MONO, fontSize: 12.5, textAlign: "center" }}>
              <span style={{ color: THEME.muted }}>
                u = <b style={{ color: THEME.text }}>{fmt(f.a0[0])}</b>
              </span>
              <span style={{ color: THEME.muted }}>
                v = <b style={{ color: THEME.text }}>{fmt(f.a0[1])}</b>
              </span>
            </div>
            <div style={{ display: "grid", gap: 4, fontFamily: MONO, fontSize: 12.5 }}>
              {g.d1.map((d, i) => (
                <div key={i} style={{ padding: "9px 0", color: f.z1[i] > 0 ? THEME.text : THEME.muted }}>
                  δ{i + 1} = {sgn(d)}
                  {f.z1[i] > 0 ? "" : " (off)"}
                </div>
              ))}
            </div>
            <Matrix rows={g.gW1} max={maxG} />
          </div>
          <div style={{ fontFamily: MONO, fontSize: 13, color: THEME.muted, marginTop: 8 }}>
            o = {fmt(f.o, 3)}, y = {label}, output error δ₂ = o − y = {sgn(g.d2, 3)}
          </div>
        </Panel>
        <Note>A hidden neuron that is off (ReLU input below 0) passes no gradient back, so its whole row is 0. The address at the map's centre line (u = 0) gives a zero first column.</Note>
      </>
    );
  } else if (screen === 2) {
    question = (
      <>
        Name it. These are the four lines of back-prop for this network, with the live numbers for the address you placed. Check each shape: does every gradient have exactly the shape of the thing it belongs to?
      </>
    );
    stage = (
      <Panel>
        <Label>forward</Label>
        <Eq size={13}>
          z₁ = W₁a₀ + b₁ &nbsp;&nbsp;(3×2)(2×1) → 3×1 = [{f.z1.map((v) => fmt(v)).join(", ")}]
          <br />
          a₁ = ReLU(z₁) = [{f.a1.map((v) => fmt(v)).join(", ")}]
          <br />o = σ(W₂a₁ + b₂) &nbsp;&nbsp;(1×3)(3×1) → 1×1 = {fmt(f.o, 3)}
        </Eq>
        <Label>
          <span style={{ display: "block", marginTop: 6 }}>backward</span>
        </Label>
        <Eq size={13}>
          δ₂ = o − y = {sgn(g.d2, 3)} &nbsp;&nbsp;(1×1)
          <br />
          ∂L/∂W₂ = δ₂ a₁ᵀ = [{g.gW2.map((v) => fmt(v)).join(", ")}] &nbsp;&nbsp;(1×3, like W₂)
          <br />
          δ₁ = (W₂ᵀ δ₂) ⊙ 1[z₁ &gt; 0] = [{g.d1.map((v) => fmt(v)).join(", ")}] &nbsp;&nbsp;(3×1)
          <br />
          ∂L/∂W₁ = δ₁ a₀ᵀ = [{g.gW1.map((r) => "[" + fmt(r[0]) + ", " + fmt(r[1]) + "]").join(", ")}] &nbsp;&nbsp;(3×2, like W₁)
          <br />
          ∂L/∂b₁ = δ₁, ∂L/∂b₂ = δ₂
        </Eq>
      </Panel>
    );
    side = (
      <>
        <Panel>
          <Label>why those factors (slides 67-82)</Label>
          <Note>
            ∂z/∂a = W for an affine layer, so the error flows back through Wᵀ. A scalar activation such as ReLU or sigmoid has a diagonal Jacobian, so its backward step is an elementwise product (⊙) with f′(z). For ReLU, f′ is 1 where the neuron is on and 0 where it is off.
          </Note>
          <Note>Shape rule (slide 74): the gradient of a scalar loss with respect to W has W's shape. If your product comes out 2×3 when W is 3×2, you transposed the wrong thing.</Note>
        </Panel>
        <Panel>
          <Label>the same thing from screen 1</Label>
          <Note>Every number you measured by nudging is one cell of δ₁a₀ᵀ. Back-prop gets all of them with one backward pass instead of one nudge per weight. A real network has millions of weights.</Note>
        </Panel>
      </>
    );
  } else {
    question = (
      <>
        Train a real network on the downtown zone, with every step computed by the formulas you just named. Choose the hidden width, press Train, and watch the boundary form. What is the narrowest hidden layer that reaches zero wrong addresses? Lesson 2 tells you what to expect.
      </>
    );
    stage = (
      <>
        <ZoneMap points={ZONE} field={trainField} pred={trainField} res={70} ariaLabel="Training map" />
        <Legend />
      </>
    );
    side = (
      <>
        <Panel edge={trainWrong === 0 ? THEME.accent : THEME.grid}>
          <Slider id="l7-h" label="hidden neurons (ReLU)" value={H} min={1} max={10} stepSize={1} onChange={setWidth} display={String(H)} />
          <Slider id="l7-lr" label="learning rate η" value={lr} min={0.05} max={3} stepSize={0.05} onChange={setLr} display={fmt(lr, 2)} />
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
            <Btn onClick={toggleRun} strong>
              {running ? "Pause" : epochs > 0 ? "Continue" : "Train"}
            </Btn>
            <Btn onClick={newSeed}>New random start</Btn>
          </div>
          <div style={{ marginTop: 8 }}>
            <Spark vals={hist} />
          </div>
          <div style={{ fontFamily: MONO, fontSize: 13.5, lineHeight: 1.7, marginTop: 6 }}>
            <div>
              wrong addresses: <b style={{ color: trainWrong ? THEME.warn : THEME.accent }}>{trainWrong}</b> / {ZONE.length}
            </div>
            <div>
              epochs: {epochs}
              {epochs >= 6000 ? " (stopped at 6000)" : ""} · loss {hist.length ? fmt(hist[hist.length - 1], 3) : "–"}
            </div>
            <div style={{ color: deadUnits ? THEME.warn : THEME.muted }}>dead ReLUs (never on): {deadUnits}</div>
          </div>
          <Hint
            hints={[
              "Each ReLU contributes one fold along a straight street. How many streets does it take to surround a district? Recall the polygon from lesson 2.",
              "Try 3 hidden neurons. If a run stalls with a dead ReLU (its row of gradients is always zero, as on screen 2), press New random start. The narrowest width that works is the smallest number of streets that can surround a point.",
            ]}
          />
        </Panel>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="Backprop for a whole layer" question={question} stage={stage} side={side} />;
}
