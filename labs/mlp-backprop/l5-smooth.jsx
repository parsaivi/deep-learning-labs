import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import { THEME, MONO } from "./common";

// Lesson 5 - Why training needs a smooth activation, and which loss to use (back-prop deck, slides 5-20).
// Aha: with a threshold neuron the error is a staircase: flat almost everywhere, so a small nudge to a
// weight tells you nothing. A sigmoid turns it into a slope you can follow downhill. With cross-entropy
// the slope for each address is simply (o - y).

// "express zone": is an address within the 30-minute promise? label depends on distance, with overlap
function makeData(seed, centre, k) {
  const r = rng(seed);
  const out = [];
  for (let i = 0; i < 24; i++) {
    const d = 0.3 + (9.4 * (i + r() * 0.8)) / 24;
    const p = sig(k * (d - centre));
    out.push({ d, y: r() < p ? 1 : 0 });
  }
  return out;
}
const DATA = makeData(3, 4.2, -1.6);
const LONG = makeData(17, 6, 1.5);

function zeroOne(data, b) {
  let e = 0;
  for (const p of data) if ((p.d < b ? 1 : 0) !== p.y) e++;
  return e / data.length;
}
function ce(data, b, s) {
  let L = 0;
  for (const p of data) {
    const o = clamp(sig(s * (b - p.d)), 1e-7, 1 - 1e-7);
    L += -(p.y * Math.log(o) + (1 - p.y) * Math.log(1 - o));
  }
  return L / data.length;
}
function ceGrad(data, b, s) {
  // dL/db = mean (o - y) * s
  let g = 0;
  for (const p of data) g += (sig(s * (b - p.d)) - p.y) * s;
  return g / data.length;
}
// challenge model: o = sigma(w (d - 5) + b)
function ce2(data, w, b) {
  let L = 0;
  for (const p of data) {
    const o = clamp(sig(w * (p.d - 5) + b), 1e-7, 1 - 1e-7);
    L += -(p.y * Math.log(o) + (1 - p.y) * Math.log(1 - o));
  }
  return L / data.length;
}
function grad2(data, w, b) {
  let gw = 0;
  let gb = 0;
  for (const p of data) {
    const e = sig(w * (p.d - 5) + b) - p.y;
    gw += e * (p.d - 5);
    gb += e;
  }
  return [gw / data.length, gb / data.length];
}

const X0 = 46;
const X1 = 590;
function xs(d) {
  return X0 + (d / 10) * (X1 - X0);
}
function DataStrip({ data, curve, sel, threshold }) {
  // addresses along the distance axis: y = 1 at the top row, y = 0 at the bottom row
  const H = 130;
  const Y = (v) => 108 - v * 84;
  const pts = [];
  if (curve)
    for (let i = 0; i <= 200; i++) {
      const d = (10 * i) / 200;
      pts.push(xs(d) + "," + Y(curve(d)));
    }
  return (
    <svg viewBox={"0 0 620 " + H} style={{ width: "100%", display: "block" }} role="img" aria-label="Addresses by distance and the neuron output">
      <rect x={X0} y={18} width={X1 - X0} height={96} fill={THEME.bg} stroke={THEME.grid} />
      <line x1={X0} x2={X1} y1={Y(0.5)} y2={Y(0.5)} stroke={THEME.grid} strokeDasharray="3 3" />
      <text x={X0 - 6} y={Y(1) + 4} textAnchor="end" fontFamily={MONO} fontSize={10.5} fill={THEME.muted}>
        1
      </text>
      <text x={X0 - 6} y={Y(0) + 4} textAnchor="end" fontFamily={MONO} fontSize={10.5} fill={THEME.muted}>
        0
      </text>
      {[0, 2, 4, 6, 8, 10].map((d) => (
        <text key={d} x={xs(d)} y={H - 2} textAnchor="middle" fontFamily={MONO} fontSize={10.5} fill={THEME.muted}>
          {d} km
        </text>
      ))}
      {threshold !== undefined && <line x1={xs(threshold)} x2={xs(threshold)} y1={18} y2={114} stroke={THEME.highlight} strokeWidth={1.5} />}
      {curve && <polyline points={pts.join(" ")} fill="none" stroke={THEME.highlight} strokeWidth={2.2} />}
      {data.map((p, i) => {
        const x = xs(p.d);
        const y = Y(p.y);
        const isSel = sel === i;
        return (
          <g key={i}>
            {isSel && curve && <line x1={x} x2={x} y1={y} y2={Y(curve(p.d))} stroke={THEME.warn} strokeWidth={2} />}
            {p.y ? <circle cx={x} cy={y} r={5} fill={THEME.highlight} stroke={THEME.bg} /> : <rect x={x - 4.5} y={y - 4.5} width={9} height={9} fill={THEME.bg} stroke={THEME.accent} strokeWidth={1.6} />}
            {isSel && <circle cx={x} cy={y} r={9} fill="none" stroke={THEME.warn} strokeWidth={1.6} />}
          </g>
        );
      })}
      <text x={X0 + 4} y={12} fontFamily={MONO} fontSize={10.5} fill={THEME.muted}>
        top row: inside the 30-min promise (y = 1) · bottom row: outside (y = 0)
      </text>
    </svg>
  );
}
function LossCurve({ fn, ghost, b, ymax, slope, label }) {
  const H = 190;
  const Y = (v) => 168 - (clamp(v, 0, ymax) / ymax) * 140;
  const pts = (f) => {
    const a = [];
    for (let i = 0; i <= 300; i++) {
      const bb = (10 * i) / 300;
      a.push(xs(bb) + "," + Y(f(bb)));
    }
    return a.join(" ");
  };
  const yb = fn(b);
  const dx = 0.9;
  return (
    <svg viewBox={"0 0 620 " + H} style={{ width: "100%", display: "block" }} role="img" aria-label="Loss as a function of the threshold">
      <rect x={X0} y={20} width={X1 - X0} height={150} fill={THEME.bg} stroke={THEME.grid} />
      {[0, ymax / 2, ymax].map((v) => (
        <text key={v} x={X0 - 6} y={Y(v) + 4} textAnchor="end" fontFamily={MONO} fontSize={10} fill={THEME.muted}>
          {fmt(v, 2)}
        </text>
      ))}
      {[0, 2, 4, 6, 8, 10].map((d) => (
        <text key={d} x={xs(d)} y={H - 4} textAnchor="middle" fontFamily={MONO} fontSize={10} fill={THEME.muted}>
          b = {d}
        </text>
      ))}
      {ghost && <polyline points={pts(ghost)} fill="none" stroke={THEME.muted} strokeWidth={1.3} strokeDasharray="4 3" />}
      <polyline points={pts(fn)} fill="none" stroke={THEME.accent} strokeWidth={2.2} />
      <line x1={xs(b - dx)} x2={xs(b + dx)} y1={Y(yb - slope * dx)} y2={Y(yb + slope * dx)} stroke={THEME.warn} strokeWidth={2} />
      <circle cx={xs(b)} cy={Y(yb)} r={7} fill={THEME.highlight} stroke={THEME.bg} strokeWidth={1.5} />
      <text x={X0 + 6} y={34} fontFamily={MONO} fontSize={11} fill={THEME.muted}>
        {label}
      </text>
    </svg>
  );
}

function Landscape({ data, path }) {
  // loss over (w, b), w and b in [-4, 4]; canvas underneath, path on top
  const ref = useRef(null);
  const R = 90;
  const field = useMemo(() => {
    const v = new Float32Array(R * R);
    for (let j = 0; j < R; j++)
      for (let i = 0; i < R; i++) {
        const w = -4 + (8 * (i + 0.5)) / R;
        const b = 4 - (8 * (j + 0.5)) / R;
        v[j * R + i] = ce2(data, w, b);
      }
    return v;
  }, [data]);
  useEffect(() => {
    const ctx = ref.current.getContext("2d");
    const img = ctx.createImageData(R, R);
    const [ar, ag, ab] = hexRGB(THEME.accent);
    const [br, bg, bb] = hexRGB(THEME.bg);
    for (let k = 0; k < R * R; k++) {
      const t = clamp(1 - Math.log(field[k] / 0.25) / Math.log(16), 0, 1); // brighter = lower loss
      const band = Math.floor(Math.log(field[k]) * 6) % 2 === 0 ? 1 : 0.85;
      img.data[4 * k] = Math.round((br + (ar - br) * t * 0.75) * band);
      img.data[4 * k + 1] = Math.round((bg + (ag - bg) * t * 0.75) * band);
      img.data[4 * k + 2] = Math.round((bb + (ab - bb) * t * 0.75) * band);
      img.data[4 * k + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [field]);
  const P = (w, b) => [((w + 4) / 8) * 300, ((4 - b) / 8) * 300];
  const last = path[path.length - 1];
  return (
    <div style={{ position: "relative", width: "100%", maxWidth: 360, aspectRatio: "1 / 1", margin: "0 auto" }}>
      <canvas ref={ref} width={R} height={R} style={{ position: "absolute", inset: 0, width: "100%", height: "100%", imageRendering: "pixelated" }} />
      <svg viewBox="0 0 300 300" style={{ position: "absolute", inset: 0, width: "100%", height: "100%", border: "1px solid " + THEME.grid }} role="img" aria-label="Loss landscape over w and b">
        <line x1={150} x2={150} y1={0} y2={300} stroke={THEME.grid} />
        <line x1={0} x2={300} y1={150} y2={150} stroke={THEME.grid} />
        <polyline points={path.map((q) => P(q[0], q[1]).join(",")).join(" ")} fill="none" stroke={THEME.highlight} strokeWidth={1.8} />
        {path.map((q, i) => {
          const [x, y] = P(q[0], q[1]);
          return <circle key={i} cx={x} cy={y} r={i === path.length - 1 ? 6 : 2.5} fill={THEME.highlight} stroke={THEME.bg} />;
        })}
        <text x={296} y={146} textAnchor="end" fontFamily={MONO} fontSize={10} fill={THEME.text}>
          w →
        </text>
        <text x={154} y={12} fontFamily={MONO} fontSize={10} fill={THEME.text}>
          b ↑
        </text>
        <text x={4} y={294} fontFamily={MONO} fontSize={10} fill={THEME.text}>
          brighter = lower loss
        </text>
        <text x={P(last[0], last[1])[0] + 8} y={P(last[0], last[1])[1] - 8} fontFamily={MONO} fontSize={10.5} fill={THEME.highlight} stroke={THEME.bg} strokeWidth={3} paintOrder="stroke">
          you
        </text>
      </svg>
    </div>
  );
}

const RESTAURANTS = ["Pizza Port", "Noodle Bar", "Green Bowl"];
function Bars({ vals, color, max, labels, signed }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "auto 1fr auto", gap: "4px 8px", alignItems: "center", fontFamily: MONO, fontSize: 12.5 }}>
      {vals.map((v, i) => (
        <React.Fragment key={i}>
          <span style={{ color: THEME.muted }}>{labels[i]}</span>
          <svg viewBox="0 0 200 12" style={{ width: "100%", height: 12 }} aria-hidden="true">
            <rect x="0" y="1" width="200" height="10" fill={THEME.bg} stroke={THEME.grid} />
            {signed && <line x1="100" x2="100" y1="0" y2="12" stroke={THEME.muted} />}
            {signed ? (
              <rect x={v >= 0 ? 100 : 100 + (v / max) * 100} y="2" width={Math.abs(v / max) * 100} height="8" fill={v >= 0 ? color : THEME.warn} />
            ) : (
              <rect x="0" y="2" width={clamp(v / max, 0, 1) * 200} height="8" fill={color} />
            )}
          </svg>
          <span>{signed ? sgn(v) : fmt(v)}</span>
        </React.Fragment>
      ))}
    </div>
  );
}

export default function SmoothLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [b, setB] = useState(7.2);
  const [s, setS] = useState(1.2);
  const [lr, setLr] = useState(2);
  const [sel, setSel] = useState(() => Math.max(0, DATA.findIndex((p) => p.d > 3.6)));
  const [z, setZ] = useState([1.2, 2.0, -0.5]);
  const [cls, setCls] = useState(0);
  const [eta, setEta] = useState(0.5);
  const [path, setPath] = useState([[-2.4, 1.6]]);

  const err01 = zeroOne(DATA, b);
  const L = ce(DATA, b, s);
  const g = ceGrad(DATA, b, s);
  const stepDown = useCallback(() => setB((bb) => clamp(bb - lr * ceGrad(DATA, bb, s), 0, 10)), [lr, s]);
  const setZ0 = useCallback((v) => setZ((p) => [v, p[1], p[2]]), []);
  const setZ1 = useCallback((v) => setZ((p) => [p[0], v, p[2]]), []);
  const setZ2 = useCallback((v) => setZ((p) => [p[0], p[1], v]), []);
  const ex = z.map((v) => Math.exp(v));
  const sum = ex[0] + ex[1] + ex[2];
  const soft = ex.map((v) => v / sum);
  const softLoss = -Math.log(soft[cls]);
  const softGrad = soft.map((o, i) => o - (i === cls ? 1 : 0));

  const cur = path[path.length - 1];
  const L2 = ce2(LONG, cur[0], cur[1]);
  const steps = path.length - 1;
  const takeStep = useCallback(
    (n) =>
      setPath((p) => {
        const out = p.slice();
        for (let k = 0; k < n; k++) {
          const q = out[out.length - 1];
          const gr = grad2(LONG, q[0], q[1]);
          out.push([clamp(q[0] - eta * gr[0], -4, 4), clamp(q[1] - eta * gr[1], -4, 4)]);
        }
        return out;
      }),
    [eta]
  );
  const one = useCallback(() => takeStep(1), [takeStep]);
  const five = useCallback(() => takeStep(5), [takeStep]);
  const restart = useCallback(() => setPath([[-2.4, 1.6]]), []);
  const p = DATA[sel];
  const oSel = sig(s * (b - p.d));

  let question, stage, side;
  if (screen === 0) {
    question = (
      <>
        One threshold neuron decides "inside the 30-minute promise" for every address closer than b km. You are training it blind. You can't see the data, only feel how the error changes when you nudge b. Start at 7.2 km. Which way should b move, and what does the slope under the ball tell you?
      </>
    );
    stage = (
      <>
        <LossCurve fn={(bb) => zeroOne(DATA, bb)} b={b} ymax={0.6} slope={0} label="error rate of the threshold neuron, as a function of b" />
        <DataStrip data={DATA} threshold={b} />
      </>
    );
    side = (
      <>
        <Panel>
          <Slider id="l5-b" label="threshold b (km)" value={b} min={0} max={10} stepSize={0.01} onChange={setB} display={fmt(b, 2) + " km"} />
          <div style={{ fontFamily: MONO, fontSize: 14, lineHeight: 1.8 }}>
            <div>error rate: {fmt(err01 * 100, 0)}%</div>
            <div>
              slope under the ball: <b style={{ color: THEME.warn }}>0.00</b>
            </div>
          </div>
        </Panel>
        <Note>Move b by a few tenths of a kilometre in either direction. How often does the error change at all? Gradient descent only ever sees the pink slope line.</Note>
      </>
    );
  } else if (screen === 1) {
    question = (
      <>
        Swap the step for a sigmoid: o = σ(s·(b − d)), read as "probability this address is inside the promise", and score it with cross-entropy. Press "step downhill" a few times. Then drag the steepness s up toward 15. What happens to the curve, and to how fast you get downhill?
      </>
    );
    stage = (
      <>
        <LossCurve fn={(bb) => ce(DATA, bb, s)} ghost={(bb) => zeroOne(DATA, bb) * 2} b={b} ymax={2} slope={g} label="cross-entropy loss (solid) · threshold error ×2 (dashed)" />
        <DataStrip data={DATA} curve={(d) => sig(s * (b - d))} />
      </>
    );
    side = (
      <>
        <Panel>
          <Slider id="l5-b2" label="b (km)" value={b} min={0} max={10} stepSize={0.01} onChange={setB} display={fmt(b, 2) + " km"} />
          <Slider id="l5-s" label="steepness s" value={s} min={0.2} max={15} stepSize={0.05} onChange={setS} display={fmt(s, 2)} />
          <Slider id="l5-lr" label="learning rate η" value={lr} min={0.1} max={6} stepSize={0.05} onChange={setLr} display={fmt(lr, 2)} />
          <Btn onClick={stepDown} strong>
            step downhill: b ← b − η·dL/db
          </Btn>
          <div style={{ fontFamily: MONO, fontSize: 14, lineHeight: 1.8, marginTop: 8 }}>
            <div>loss: {fmt(L, 3)}</div>
            <div>
              slope dL/db: <b style={{ color: THEME.warn }}>{sgn(g, 3)}</b>
            </div>
          </div>
        </Panel>
        <Note>As s grows the sigmoid approaches the step from screen 1. The slope stops being smooth: it is steep near the jumps and almost flat between them. That is why a threshold network can't be trained by gradient descent (slides 7-9).</Note>
      </>
    );
  } else if (screen === 2) {
    question = (
      <>
        Name it. Pick one address and look at its pink bar, the gap between the prediction o and the label y. That gap is all back-prop needs from the loss. On the right, the same idea for three classes: which restaurant will accept the order?
      </>
    );
    stage = (
      <>
        <DataStrip data={DATA} curve={(d) => sig(s * (b - d))} sel={sel} />
        <Panel>
          <Slider id="l5-sel" label="address" value={sel} min={0} max={DATA.length - 1} stepSize={1} onChange={setSel} display={"#" + (sel + 1) + " at " + fmt(p.d, 1) + " km"} />
          <Eq size={13.5}>
            o = σ(z) = {fmt(oSel, 3)} &nbsp;&nbsp;y = {p.y}
            <br />L = −y·log o − (1−y)·log(1−o) = {fmt(-(p.y * Math.log(Math.max(oSel, 1e-7)) + (1 - p.y) * Math.log(Math.max(1 - oSel, 1e-7))), 3)}
            <br />
            dL/do = {p.y ? "−1/o" : "1/(1−o)"} = {fmt(p.y ? -1 / oSel : 1 / (1 - oSel), 3)}
            <br />
            dL/dz = (dL/do)·σ′(z) = o − y = <span style={{ color: THEME.warn }}>{sgn(oSel - p.y, 3)}</span>
          </Eq>
          <Note>σ′(z) = σ(1 − σ) cancels the 1/o, so a sigmoid output with cross-entropy hands back just (o − y). For a real-valued target such as a fee, use a plain output with squared error ½(y − o)², whose slope is (o − y) as well (slides 11 and 20).</Note>
        </Panel>
      </>
    );
    side = (
      <>
        <Panel>
          <Label>Softmax over three restaurants (slides 16-19)</Label>
          <Slider id="l5-z0" label={"score z for " + RESTAURANTS[0]} value={z[0]} min={-3} max={3} stepSize={0.05} onChange={setZ0} />
          <Slider id="l5-z1" label={"score z for " + RESTAURANTS[1]} value={z[1]} min={-3} max={3} stepSize={0.05} onChange={setZ1} />
          <Slider id="l5-z2" label={"score z for " + RESTAURANTS[2]} value={z[2]} min={-3} max={3} stepSize={0.05} onChange={setZ2} />
          <Label>restaurant that actually took the order</Label>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 8 }}>
            {RESTAURANTS.map((r, i) => (
              <Pill key={r} value={i} active={cls === i} onPick={setCls}>
                {r}
              </Pill>
            ))}
          </div>
          <Label>oᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)</Label>
          <Bars vals={soft} color={THEME.highlight} max={1} labels={RESTAURANTS} />
          <Label>
            <span style={{ display: "block", marginTop: 8 }}>gradient dL/dzᵢ = oᵢ − yᵢ</span>
          </Label>
          <Bars vals={softGrad} color={THEME.accent} max={1} labels={RESTAURANTS} signed />
          <div style={{ fontFamily: MONO, fontSize: 13, marginTop: 6 }}>loss −log o_c = {fmt(softLoss, 3)}</div>
        </Panel>
      </>
    );
  } else {
    question = (
      <>
        New service: long-distance express, where far addresses are the served ones. The model o = σ(w·(d − 5) + b) starts with the wrong sign of w. Using only gradient steps, get the loss under 0.10 in at most 15 steps. You set the learning rate.
      </>
    );
    stage = (
      <>
        <DataStrip data={LONG} curve={(d) => sig(cur[0] * (d - 5) + cur[1])} />
        <Landscape data={LONG} path={path} />
      </>
    );
    side = (
      <>
        <Panel edge={L2 < 0.1 && steps <= 15 ? THEME.accent : THEME.grid}>
          <Slider id="l5-eta" label="learning rate η" value={eta} min={0.05} max={12} stepSize={0.05} onChange={setEta} display={fmt(eta, 2)} />
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
            <Btn onClick={one} strong>
              1 step
            </Btn>
            <Btn onClick={five}>5 steps</Btn>
            <Btn onClick={restart}>restart</Btn>
          </div>
          <Meter label="loss" value={L2} max={2} good={0.1} />
          <div style={{ fontFamily: MONO, fontSize: 13.5, lineHeight: 1.7 }}>
            <div style={{ color: steps > 15 ? THEME.warn : THEME.text }}>steps used: {steps} / 15</div>
            <div>
              w = {fmt(cur[0])}, b = {fmt(cur[1])}
            </div>
          </div>
          <Hint
            hints={[
              "Watch the yellow path on the landscape. Tiny steps crawl and huge steps bounce across the valley. Restart and try a different η each time.",
              "The slope here is a mean of (o − y)·(d − 5) terms, which are small. With η around 1 the path is still crawling after 15 steps. Something in the 3 to 10 range moves w across zero quickly without bouncing out of the valley.",
            ]}
          />
        </Panel>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="Why the staircase can't learn" question={question} stage={stage} side={side} />;
}
