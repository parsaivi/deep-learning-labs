import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import { THEME, MONO, ZoneMap } from "./common";

// Lesson 1 - The perceptron is one straight street; XOR needs a hidden layer.
// Aha: a hidden layer moves every address to new coordinates (h1, h2). A layout that no
// single street can split on the map becomes splittable by one street in hidden space.

const HOODS = {
  sw: { name: "Old Town", c: [0.25, 0.25] },
  nw: { name: "Harbor", c: [0.25, 0.75] },
  se: { name: "Market", c: [0.75, 0.25] },
  ne: { name: "Hills", c: [0.75, 0.75] },
};
const PATTERNS = {
  and: { label: "Hills only", served: ["ne"], story: "The Hills kitchen delivers only inside Hills." },
  or: { label: "All but Old Town", served: ["nw", "se", "ne"], story: "Old Town has its own couriers; we cover the other three." },
  xor: { label: "Harbor + Market", served: ["nw", "se"], story: "Two partner kitchens, one in Harbor and one in Market. Each delivers in its own quarter." },
};
const CH_HOODS = {
  st: { name: "Station", c: [0.5, 0.42], s: 1 },
  w: { name: "West End", c: [0.14, 0.42], s: 0 },
  e: { name: "East End", c: [0.86, 0.42], s: 0 },
  n: { name: "Uptown", c: [0.5, 0.84], s: 0 },
};

function scatter(centres, seed, per, rad) {
  const r = rng(seed);
  const out = [];
  for (const key of Object.keys(centres)) {
    const c = centres[key].c;
    for (let k = 0; k < per; k++) {
      const ang = r() * Math.PI * 2;
      const d = rad * Math.sqrt(0.15 + 0.85 * r());
      out.push({ x: c[0] + d * Math.cos(ang), y: c[1] + d * Math.sin(ang), hood: key });
    }
  }
  return out;
}
const BASE_PTS = scatter(HOODS, 7, 5, 0.11);
const CH_PTS = scatter(CH_HOODS, 21, 5, 0.075).map((p) => Object.assign({}, p, { s: CH_HOODS[p.hood].s }));

// hidden space is drawn in a unit square with the corners (0/1, 0/1) placed at 0.25 / 0.75
function u(h) {
  return 0.25 + 0.5 * h;
}
const START = {
  s: { a: [0.15, 0.6], b: [0.85, 0.35], flip: false },
  h1: { a: [0.5, 0.06], b: [0.5, 0.94], flip: false },
  h2: { a: [0.06, 0.5], b: [0.94, 0.5], flip: true },
  o: { a: [0.06, 0.5], b: [0.94, 0.5], flip: false },
  c1: { a: [0.3, 0.06], b: [0.3, 0.94], flip: false },
  c2: { a: [0.7, 0.06], b: [0.7, 0.94], flip: true },
  co: { a: [0.06, 0.62], b: [0.94, 0.62], flip: false },
};
const SLIDE_NET = {
  h1: { a: [0.05, 0.7], b: [0.7, 0.05], flip: false },
  h2: { a: [0.3, 0.95], b: [0.95, 0.3], flip: false },
  o: { a: [0.27, 0.02], b: [0.98, 0.73], flip: true },
};

function twoLayer(L1, L2, LO) {
  const n1 = lineNeuron(L1.a, L1.b, L1.flip);
  const n2 = lineNeuron(L2.a, L2.b, L2.flip);
  const no = lineNeuron(LO.a, LO.b, LO.flip);
  const hid = (x, y) => [step(zOf(n1, x, y)), step(zOf(n2, x, y))];
  const out = (x, y) => {
    const h = hid(x, y);
    return step(zOf(no, u(h[0]), u(h[1])));
  };
  return { n1, n2, no, hid, out };
}

// addresses re-drawn at their hidden coordinates, packed around each corner
function hiddenPoints(points, hid) {
  const groups = {};
  return points.map((p) => {
    const h = hid(p.x, p.y);
    const key = h[0] + "" + h[1];
    const k = (groups[key] = (groups[key] || 0) + 1) - 1;
    const ring = Math.floor(k / 8);
    const ang = (k % 8) * (Math.PI / 4) + ring * 0.4;
    const r = k === 0 ? 0 : 0.035 + 0.035 * ring;
    return { x: u(h[0]) + r * Math.cos(ang), y: u(h[1]) + r * Math.sin(ang), s: p.s };
  });
}

function HiddenAxes() {
  const c0 = toSvg(u(0), u(0));
  const c1 = toSvg(u(1), u(1));
  return (
    <g fontFamily={MONO} fontSize={12} fill={THEME.muted}>
      <rect x={c0[0]} y={c1[1]} width={c1[0] - c0[0]} height={c0[1] - c1[1]} fill="none" stroke={THEME.grid} strokeDasharray="4 4" />
      <text x={c0[0]} y={c0[1] + 30} textAnchor="middle">h1=0</text>
      <text x={c1[0]} y={c0[1] + 30} textAnchor="middle">h1=1</text>
      <text x={c0[0] - 28} y={c0[1] + 4} textAnchor="middle">h2=0</text>
      <text x={c0[0] - 28} y={c1[1] + 4} textAnchor="middle">h2=1</text>
      <text x={200} y={392} textAnchor="middle" fill={THEME.text}>
        hidden space: where each address lands
      </text>
    </g>
  );
}
function HoodLabels({ hoods }) {
  return (
    <g fontFamily={MONO} fontSize={11} fill={THEME.muted}>
      {Object.keys(hoods).map((k) => {
        const [sx, sy] = toSvg(hoods[k].c[0], hoods[k].c[1]);
        return (
          <text key={k} x={sx} y={sy - 40} textAnchor="middle" stroke={THEME.bg} strokeWidth={3} paintOrder="stroke">
            {hoods[k].name}
          </text>
        );
      })}
    </g>
  );
}

function WBox({ title, vals, z }) {
  return (
    <div style={{ border: "1px solid " + THEME.grid, borderRadius: 3, padding: "5px 7px", background: THEME.bg }}>
      <div style={{ color: THEME.muted, fontSize: 11 }}>{title}</div>
      <div>[{fmt(vals[0])}, {fmt(vals[1])}] {sgn(vals[2])}</div>
      <div style={{ color: z >= 0 ? THEME.highlight : THEME.muted }}>z = {fmt(z)} → {z >= 0 ? "fires" : "silent"}</div>
    </div>
  );
}
function NetDiagram({ net, probe }) {
  const h = net.hid(probe[0], probe[1]);
  const z1 = zOf(net.n1, probe[0], probe[1]);
  const z2 = zOf(net.n2, probe[0], probe[1]);
  // output weights on h (undo the 0.25 / 0.5 drawing offset)
  const v1 = 0.5 * net.no.w1;
  const v2 = 0.5 * net.no.w2;
  const c = net.no.b + 0.25 * (net.no.w1 + net.no.w2);
  const zo = v1 * h[0] + v2 * h[1] + c;
  const y = step(zo);
  const N = {
    x1: [60, 60, "x₁ east", fmt(probe[0])],
    x2: [60, 140, "x₂ north", fmt(probe[1])],
    one: [60, 220, "1", "1"],
    h1: [270, 80, "h₁", String(h[0])],
    h2: [270, 190, "h₂", String(h[1])],
    one2: [270, 260, "1", "1"],
    y: [460, 140, "y", String(y)],
  };
  const E = [
    ["x1", "h1", net.n1.w1],
    ["x2", "h1", net.n1.w2],
    ["one", "h1", net.n1.b],
    ["x1", "h2", net.n2.w1],
    ["x2", "h2", net.n2.w2],
    ["one", "h2", net.n2.b],
    ["h1", "y", v1],
    ["h2", "y", v2],
    ["one2", "y", c],
  ];
  return (
    <>
    <svg viewBox="0 0 520 290" style={{ width: "100%", display: "block" }} role="img" aria-label="Network diagram with live weights">
      {E.map(([a, b, w], i) => {
        const A = N[a];
        const B = N[b];
        const t = b === "y" ? 0.42 : a === "one" ? 0.3 : a === "x2" ? 0.55 : 0.4;
        const mx = A[0] + (B[0] - A[0]) * t;
        const my = A[1] + (B[1] - A[1]) * t;
        return <line key={i} x1={A[0]} y1={A[1]} x2={B[0]} y2={B[1]} stroke={w >= 0 ? THEME.accent : THEME.warn} strokeWidth={0.8 + Math.min(3, Math.abs(w) * 2.2)} strokeDasharray={w >= 0 ? undefined : "5 4"} opacity={0.75} />;
      })}
      {Object.keys(N).map((k) => {
        const [x, yy, name, val] = N[k];
        const isStep = k === "h1" || k === "h2" || k === "y";
        const on = isStep && val === "1";
        return (
          <g key={k}>
            <circle cx={x} cy={yy} r={isStep ? 24 : 20} fill={on ? rgba(THEME.highlight, 0.3) : THEME.bg} stroke={isStep ? THEME.highlight : THEME.grid} strokeWidth={1.6} />
            <text x={x} y={yy + 5} textAnchor="middle" fontFamily={MONO} fontSize={14} fill={THEME.text}>
              {val}
            </text>
            <text x={x} y={yy - (isStep ? 30 : 26)} textAnchor="middle" fontFamily={MONO} fontSize={11.5} fill={THEME.muted}>
              {name}
            </text>
          </g>
        );
      })}
    </svg>
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))", gap: 6, fontFamily: MONO, fontSize: 12.5, marginTop: 4 }}>
      <WBox title="h₁ row of W, b" vals={[net.n1.w1, net.n1.w2, net.n1.b]} z={z1} />
      <WBox title="h₂ row of W, b" vals={[net.n2.w1, net.n2.w2, net.n2.b]} z={z2} />
      <WBox title="output v, c" vals={[v1, v2, c]} z={zo} />
    </div>
    <div style={{ fontSize: 12, color: THEME.muted, marginTop: 6 }}>solid cyan edge = positive weight, dashed pink = negative, thickness = size. Map units: 1 = 10 km.</div>
    </>
  );
}

export default function XorLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [pattern, setPattern] = useState("xor");
  const [L, setL] = useState(START);
  const [probe, setProbe] = useState([0.25, 0.75]);
  const [best, setBest] = useState({ and: 99, or: 99, xor: 99 });

  const onMove = useCallback((id, x, y) => {
    if (id === "probe") {
      setProbe([x, y]);
      return;
    }
    const [name, end] = id.split(":");
    setL((prev) => Object.assign({}, prev, { [name]: Object.assign({}, prev[name], { [end]: [x, y] }) }));
  }, []);
  const flip = useCallback((name) => setL((prev) => Object.assign({}, prev, { [name]: Object.assign({}, prev[name], { flip: !prev[name].flip }) })), []);
  const loadSlide = useCallback(() => setL((prev) => Object.assign({}, prev, SLIDE_NET)), []);
  const resetHidden = useCallback(() => setL((prev) => Object.assign({}, prev, { h1: START.h1, h2: START.h2, o: START.o })), []);

  const pts = useMemo(() => {
    const served = PATTERNS[pattern].served;
    return BASE_PTS.map((p) => Object.assign({}, p, { s: served.indexOf(p.hood) >= 0 ? 1 : 0 }));
  }, [pattern]);
  const xorPts = useMemo(() => BASE_PTS.map((p) => Object.assign({}, p, { s: PATTERNS.xor.served.indexOf(p.hood) >= 0 ? 1 : 0 })), []);

  // screen 1: one neuron
  const single = useMemo(() => lineNeuron(L.s.a, L.s.b, L.s.flip), [L.s]);
  const singleOut = useCallback((x, y) => step(zOf(single, x, y)), [single]);
  const singleWrong = countWrong(pts, singleOut);
  useEffect(() => {
    if (screen === 0) setBest((b) => (singleWrong < b[pattern] ? Object.assign({}, b, { [pattern]: singleWrong }) : b));
  }, [singleWrong, pattern, screen]);

  // screens 2 and 3: two hidden neurons
  const net = useMemo(() => twoLayer(L.h1, L.h2, L.o), [L.h1, L.h2, L.o]);
  const netWrong = countWrong(xorPts, net.out);
  const hidPts = useMemo(() => hiddenPoints(xorPts, net.hid), [xorPts, net]);
  const hiddenField = useCallback((x, y) => step(zOf(net.no, x, y)), [net]);

  // screen 4: challenge
  const cnet = useMemo(() => twoLayer(L.c1, L.c2, L.co), [L.c1, L.c2, L.co]);
  const cWrong = countWrong(CH_PTS, cnet.out);
  const cHidPts = useMemo(() => hiddenPoints(CH_PTS, cnet.hid), [cnet]);
  const cHiddenField = useCallback((x, y) => step(zOf(cnet.no, x, y)), [cnet]);

  const lineObj = (name, label, color, dash) => Object.assign({ id: name, label, color, dash }, L[name]);
  const ends = (name, color) => [
    { id: name + ":a", x: L[name].a[0], y: L[name].a[1], color },
    { id: name + ":b", x: L[name].b[0], y: L[name].b[1], color },
  ];

  let question, stage, side;
  if (screen === 0) {
    question = (
      <>
        A single neuron draws one straight street and says "deliver" on one side of it. Drag the street's two ends (and flip its side) to cover the served addresses. Which delivery contracts can one street handle?
      </>
    );
    stage = (
      <>
        <ZoneMap points={pts} field={singleOut} pred={singleOut} lines={[lineObj("s", "deliver", THEME.accent)]} handles={ends("s")} onMove={onMove} overlay={<HoodLabels hoods={HOODS} />} ariaLabel="Map with one draggable street" />
        <Legend />
      </>
    );
    side = (
      <>
        <Panel>
          <Label>Delivery contract</Label>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
            {Object.keys(PATTERNS).map((k) => (
              <Pill key={k} value={k} active={pattern === k} onPick={setPattern}>
                {PATTERNS[k].label}
              </Pill>
            ))}
          </div>
          <Note>
            <span style={{ display: "block", marginTop: 8 }}>{PATTERNS[pattern].story}</span>
          </Note>
          <Btn onClick={() => flip("s")}>Flip which side delivers</Btn>
        </Panel>
        <Panel edge={singleWrong === 0 ? THEME.accent : THEME.grid}>
          <div style={{ fontFamily: MONO, fontSize: 14, lineHeight: 1.8 }}>
            <div>wrong addresses now: <b style={{ color: singleWrong ? THEME.warn : THEME.accent }}>{singleWrong}</b> / 20</div>
            {Object.keys(PATTERNS).map((k) => (
              <div key={k} style={{ color: THEME.muted }}>
                best for "{PATTERNS[k].label}": {best[k] === 99 ? "not tried" : best[k]}
              </div>
            ))}
          </div>
        </Panel>
        <Note>Try all three contracts. Then spend a minute on "Harbor + Market" and see how low you can get it.</Note>
      </>
    );
  } else if (screen === 1) {
    question = (
      <>
        Give the network two streets, h₁ and h₂. Each address now gets two answers, one per street, so it lands on a corner of the square on the right. A third street (the output neuron) works there. Can you place h₁ and h₂ so that one straight line on the right splits Harbor + Market from the rest?
      </>
    );
    stage = (
      <>
        <ZoneMap
          points={xorPts}
          field={net.out}
          pred={net.out}
          lines={[lineObj("h1", "h₁", THEME.accent), lineObj("h2", "h₂", THEME.text, true)]}
          handles={ends("h1").concat(ends("h2", THEME.text))}
          onMove={onMove}
          overlay={<HoodLabels hoods={HOODS} />}
          ariaLabel="Map with two hidden-neuron streets"
        />
        <Legend />
      </>
    );
    side = (
      <>
        <Panel>
          <ZoneMap plain points={hidPts} field={hiddenField} pred={hiddenField} lines={[lineObj("o", "y", THEME.highlight)]} handles={ends("o", THEME.highlight)} onMove={onMove} overlay={<HiddenAxes />} maxWidth={330} ariaLabel="Hidden space with the output street" />
        </Panel>
        <Panel edge={netWrong === 0 ? THEME.accent : THEME.grid}>
          <div style={{ fontFamily: MONO, fontSize: 14 }}>
            wrong addresses: <b style={{ color: netWrong ? THEME.warn : THEME.accent }}>{netWrong}</b> / 20
          </div>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginTop: 8 }}>
            <Btn onClick={() => flip("h1")}>flip h₁</Btn>
            <Btn onClick={() => flip("h2")}>flip h₂</Btn>
            <Btn onClick={() => flip("o")}>flip y</Btn>
            <Btn onClick={resetHidden}>reset</Btn>
          </div>
        </Panel>
        <Note>
          The streets start in the obvious place: one north-south, one east-west. Look at the square on the right first. Why can't the yellow street split it?
        </Note>
      </>
    );
  } else if (screen === 2) {
    question = (
      <>
        This is the network you just built, with its real weights. Drag the square probe around the map. Which numbers change when the probe crosses h₁? What does the output neuron actually see?
      </>
    );
    stage = (
      <>
        <ZoneMap
          points={xorPts}
          field={net.out}
          pred={net.out}
          lines={[lineObj("h1", "h₁", THEME.accent), lineObj("h2", "h₂", THEME.text, true)]}
          handles={[{ id: "probe", x: probe[0], y: probe[1], color: THEME.highlight, shape: "square" }]}
          onMove={onMove}
          overlay={<HoodLabels hoods={HOODS} />}
          ariaLabel="Map with a draggable probe"
        />
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginTop: 8 }}>
          <Btn onClick={loadSlide}>Load the slides' XOR net (OR and AND streets)</Btn>
        </div>
      </>
    );
    side = (
      <>
        <Panel>
          <NetDiagram net={net} probe={probe} />
        </Panel>
        <Panel>
          <Label>What you built, in symbols</Label>
          <Eq>
            hᵢ = step(wᵢ₁x₁ + wᵢ₂x₂ + bᵢ)
            <br />y = step(v₁h₁ + v₂h₂ + c)
            <br />
            y = step(v · step(W x + b) + c)
          </Eq>
          <Note>
            Each row of W is one street: (wᵢ₁, wᵢ₂) points across the street, bᵢ slides it. The output neuron never sees the map. It only sees the pair (h₁, h₂), and in those coordinates the served addresses can be cut off by one straight line. One perceptron draws one line (slide 4). XOR can't be done with one (slide 19). With a hidden layer it takes three perceptrons (slide 20).
          </Note>
        </Panel>
      </>
    );
  } else {
    question = (
      <>
        New contract. A kitchen at the train Station delivers only around the Station, and West End, East End and Uptown are not served. You have two hidden streets and one output street again. Make the wrong count reach zero.
      </>
    );
    stage = (
      <>
        <ZoneMap
          points={CH_PTS}
          field={cnet.out}
          pred={cnet.out}
          lines={[lineObj("c1", "h₁", THEME.accent), lineObj("c2", "h₂", THEME.text, true)]}
          handles={ends("c1").concat(ends("c2", THEME.text))}
          onMove={onMove}
          overlay={<HoodLabels hoods={CH_HOODS} />}
          ariaLabel="Challenge map"
        />
        <Legend />
      </>
    );
    side = (
      <>
        <Panel>
          <ZoneMap plain points={cHidPts} field={cHiddenField} pred={cHiddenField} lines={[lineObj("co", "y", THEME.highlight)]} handles={ends("co", THEME.highlight)} onMove={onMove} overlay={<HiddenAxes />} maxWidth={330} ariaLabel="Challenge hidden space" />
        </Panel>
        <Panel edge={cWrong === 0 ? THEME.accent : THEME.grid}>
          <div style={{ fontFamily: MONO, fontSize: 14 }}>
            wrong addresses: <b style={{ color: cWrong ? THEME.warn : THEME.accent }}>{cWrong}</b> / 20
          </div>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginTop: 8 }}>
            <Btn onClick={() => flip("c1")}>flip h₁</Btn>
            <Btn onClick={() => flip("c2")}>flip h₂</Btn>
            <Btn onClick={() => flip("co")}>flip y</Btn>
          </div>
          <Hint
            hints={[
              "Look at the square on the right. Station has to end up alone on one corner, or share a corner only with other served addresses.",
              "Tilt both streets so they cross above Station, like a roof. Station is then on the inner side of both, and every other neighbourhood is outside at least one.",
            ]}
          />
        </Panel>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="One street is not enough" question={question} stage={stage} side={side} />;
}
