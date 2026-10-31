import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import { THEME, MONO, ZoneMap } from "./common";

// Lesson 2 - MLPs as universal classifiers (slides 45-67).
// Aha: each hidden neuron casts one vote ("I'm on the inner side of my street"). Requiring all N
// votes carves a convex polygon; with many streets the vote count becomes a cylinder (N inside,
// about N/2 far away). Subtract N/2 and cylinders can be added, so one hidden layer can fence any zone.

const TAU = Math.PI * 2;
function makeRing(n) {
  const c = new Float64Array(n);
  const s = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    c[i] = Math.cos((TAU * i) / n + 0.3);
    s[i] = Math.sin((TAU * i) / n + 0.3);
  }
  return { n, c, s };
}
function votes(ring, cx, cy, r, x, y) {
  let v = 0;
  const dx = x - cx;
  const dy = y - cy;
  for (let i = 0; i < ring.n; i++) if (r - (dx * ring.c[i] + dy * ring.s[i]) >= 0) v++;
  return v;
}

// downtown: served inside a rounded zone around (0.5, 0.52)
const DT = [0.5, 0.52];
const DOWNTOWN = (function () {
  const r = rng(11);
  const out = [];
  while (out.length < 26) {
    const x = 0.05 + 0.9 * r();
    const y = 0.05 + 0.9 * r();
    const d = Math.hypot(x - DT[0], y - DT[1]);
    if (d < 0.2) out.push({ x, y, s: 1 });
    else if (d > 0.3 && out.filter((p) => !p.s).length < 16) out.push({ x, y, s: 0 });
  }
  return out;
})();

// challenge: an L-shaped zone along the river bend
function distToL(x, y) {
  const rect = (x0, x1, y0, y1) => Math.hypot(Math.max(x0 - x, 0, x - x1), Math.max(y0 - y, 0, y - y1));
  return Math.min(rect(0.18, 0.42, 0.18, 0.78), rect(0.18, 0.82, 0.18, 0.4));
}
const LZONE = (function () {
  const r = rng(5);
  const out = [];
  for (let k = 0; k < 7; k++) out.push({ x: 0.3 + (r() - 0.5) * 0.1, y: 0.32 + k * 0.07 + (r() - 0.5) * 0.04, s: 1 });
  for (let k = 1; k < 7; k++) out.push({ x: 0.3 + k * 0.075 + (r() - 0.5) * 0.04, y: 0.29 + (r() - 0.5) * 0.1, s: 1 });
  while (out.length < 33) {
    const x = 0.04 + 0.92 * r();
    const y = 0.04 + 0.92 * r();
    if (distToL(x, y) > 0.1) out.push({ x, y, s: 0 });
  }
  return out;
})();

const CH_N = 48;
const CH_RING = makeRing(CH_N);
// fire threshold for summed discs: a disc gives N/2 inside, so 0.4N keeps its edge close to r
const TH = 0.4 * CH_N;

function VoteNumbers({ ring, c, r }) {
  const out = [];
  for (let i = 0; i < 10; i++)
    for (let j = 0; j < 10; j++) {
      const x = 0.05 + 0.1 * i;
      const y = 0.05 + 0.1 * j;
      const v = votes(ring, c[0], c[1], r, x, y);
      out.push(
        <text key={i + "-" + j} x={x * 400} y={(1 - y) * 400 + 4} textAnchor="middle" fontFamily={MONO} fontSize={10.5} fill={v === ring.n ? THEME.text : THEME.muted} opacity={0.85}>
          {v}
        </text>
      );
    }
  return <g pointerEvents="none">{out}</g>;
}
function Streets({ ring, c, r }) {
  if (ring.n > 16) return null;
  const out = [];
  for (let i = 0; i < ring.n; i++) {
    const px = c[0] + r * ring.c[i];
    const py = c[1] + r * ring.s[i];
    const a = [px - ring.s[i], py + ring.c[i]];
    const b = [px + ring.s[i], py - ring.c[i]];
    out.push({ id: "st" + i, a, b, flip: true, color: THEME.accent, thin: true, noArrow: ring.n > 8 });
  }
  return out;
}
// votes along the east-west street through y0
function CrossSection({ fn, n, y0, threshold, labelTop, labelMid }) {
  const W = 400;
  const H = 120;
  const top = Math.max(1, n);
  const pts = [];
  for (let i = 0; i <= 200; i++) {
    const x = i / 200;
    pts.push(x * W + "," + (H - 14 - (fn(x) / top) * (H - 30)));
  }
  const yOf = (v) => H - 14 - (v / top) * (H - 30);
  return (
    <svg viewBox={"0 0 " + W + " " + H} style={{ width: "100%", display: "block" }} role="img" aria-label="Votes along one street">
      <rect x="0" y="0" width={W} height={H} fill={THEME.bg} stroke={THEME.grid} />
      <line x1="0" x2={W} y1={yOf(n)} y2={yOf(n)} stroke={THEME.grid} strokeDasharray="3 3" />
      <line x1="0" x2={W} y1={yOf(n / 2)} y2={yOf(n / 2)} stroke={THEME.grid} strokeDasharray="3 3" />
      {threshold !== undefined && <line x1="0" x2={W} y1={yOf(threshold)} y2={yOf(threshold)} stroke={THEME.highlight} strokeWidth={1.4} />}
      <polyline points={pts.join(" ")} fill="none" stroke={THEME.accent} strokeWidth={2} />
      <text x={W - 4} y={yOf(n) - 3} textAnchor="end" fontFamily={MONO} fontSize={10.5} fill={THEME.muted}>
        {labelTop || "N = " + n}
      </text>
      <text x={W - 4} y={yOf(n / 2) - 3} textAnchor="end" fontFamily={MONO} fontSize={10.5} fill={THEME.muted}>
        {labelMid || "N/2 = " + n / 2}
      </text>
      {threshold !== undefined && (
        <text x={4} y={yOf(threshold) - 3} fontFamily={MONO} fontSize={10.5} fill={THEME.highlight}>
          fire above this
        </text>
      )}
      <text x={4} y={H - 3} fontFamily={MONO} fontSize={10} fill={THEME.muted}>
        west
      </text>
      <text x={W - 4} y={H - 3} textAnchor="end" fontFamily={MONO} fontSize={10} fill={THEME.muted}>
        east · votes along y = {fmt(y0 * 10, 1)} km
      </text>
    </svg>
  );
}

const CIRCLE_START = [
  { c: [0.3, 0.62], r: 0.14 },
  { c: [0.55, 0.3], r: 0.14 },
  { c: [0.75, 0.75], r: 0.12 },
  { c: [0.75, 0.55], r: 0.12 },
  { c: [0.85, 0.15], r: 0.1 },
];

export default function ZonesLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [n, setN] = useState(4);
  const [k, setK] = useState(2);
  const [r, setR] = useState(0.17);
  const [c, setC] = useState([0.46, 0.5]);
  const [two, setTwo] = useState([
    [0.3, 0.62],
    [0.66, 0.38],
  ]);
  const [circles, setCircles] = useState(CIRCLE_START);
  const [used, setUsed] = useState(2);
  const [sel, setSel] = useState(0);
  const [hover, setHover] = useState(null);

  const big = screen >= 1 ? Math.max(n, 3) : Math.min(n, 10);
  const ring = useMemo(() => makeRing(big), [big]);
  const kk = Math.min(k, big);
  const ring48 = CH_RING;

  const onMove = useCallback((id, x, y) => {
    if (id === "c") setC([x, y]);
    else if (id === "t0" || id === "t1") {
      const i = id === "t0" ? 0 : 1;
      setTwo((prev) => prev.map((p, j) => (j === i ? [x, y] : p)));
    } else if (id.startsWith("q")) {
      const i = parseInt(id.slice(1), 10);
      setSel(i);
      setCircles((prev) => prev.map((q, j) => (j === i ? { c: [x, y], r: q.r } : q)));
    }
  }, []);
  const onHover = useCallback((x, y) => setHover(x === null ? null : [x, y]), []);
  const setNk = useCallback((v) => {
    setN(v);
    setK((kv) => Math.min(kv, v));
  }, []);
  const setSelR = useCallback((v) => setCircles((prev) => prev.map((q, j) => (j === sel ? { c: q.c, r: v } : q))), [sel]);

  // screens 1 and 2: one polygon net
  const V = useCallback((x, y) => votes(ring, c[0], c[1], r, x, y), [ring, c, r]);
  const polyField = useCallback(
    (x, y) => {
      const v = V(x, y);
      return v >= kk ? 0.55 + 0.45 * (v / big) : 0.45 * (v / big);
    },
    [V, kk, big]
  );
  const polyWrong = countWrong(DOWNTOWN, polyField);

  // screen 3: two cylinders added
  const S2 = useCallback(
    (x, y) => votes(ring48, two[0][0], two[0][1], 0.15, x, y) - CH_N / 2 + votes(ring48, two[1][0], two[1][1], 0.15, x, y) - CH_N / 2,
    [ring48, two]
  );
  const twoField = useCallback((x, y) => (S2(x, y) >= TH ? 1 : 0.2), [S2]);

  // screen 4: up to five cylinders
  const active = circles.slice(0, used);
  const SC = useCallback(
    (x, y) => {
      let s = 0;
      for (const q of active) s += votes(ring48, q.c[0], q.c[1], q.r, x, y) - CH_N / 2;
      return s;
    },
    [active, ring48]
  );
  const chField = useCallback((x, y) => (SC(x, y) >= TH ? 1 : 0.15), [SC]);
  const chWrong = countWrong(LZONE, chField);

  const hoverText =
    hover === null
      ? "point at the map to read the votes there"
      : screen <= 1
      ? "votes here: " + V(hover[0], hover[1]) + " of " + big + (V(hover[0], hover[1]) >= kk ? " → fires" : " → silent")
      : screen === 2
      ? "sum here: " + fmt(S2(hover[0], hover[1]), 0) + (S2(hover[0], hover[1]) >= TH ? " → fires" : " → silent")
      : "sum here: " + fmt(SC(hover[0], hover[1]), 0) + (SC(hover[0], hover[1]) >= TH ? " → fires" : " → silent");

  let question, stage, side;
  if (screen === 0) {
    const sl = Streets({ ring, c, r });
    question = (
      <>
        The pedestrian district downtown is the delivery zone. You have N hidden neurons, each a street with an arrow pointing inward. The output neuron counts how many streets say "inside" and fires if the count is at least K. Which K fences in exactly the district?
      </>
    );
    stage = (
      <>
        <ZoneMap points={DOWNTOWN} field={polyField} pred={polyField} lines={sl} handles={[{ id: "c", x: c[0], y: c[1], color: THEME.highlight, shape: "square" }]} onMove={onMove} onHover={onHover} overlay={<VoteNumbers ring={ring} c={c} r={r} />} ariaLabel="Map with a ring of streets and vote counts" />
        <Legend />
      </>
    );
    side = (
      <>
        <Panel>
          <Slider id="l2-n" label="N, number of hidden neurons (streets)" value={n} min={3} max={10} stepSize={1} onChange={setNk} display={String(Math.min(n, 10))} />
          <Slider id="l2-k" label="K, votes needed to fire" value={kk} min={1} max={big} stepSize={1} onChange={setK} display={String(kk)} />
          <Slider id="l2-r" label="zone size (distance of each street from the centre)" value={r} min={0.08} max={0.3} stepSize={0.005} onChange={setR} display={fmt(r * 10, 1) + " km"} />
          <Note>Drag the yellow square to move the whole ring. Small grey numbers are the vote count at that spot.</Note>
        </Panel>
        <Panel edge={polyWrong === 0 ? THEME.accent : THEME.grid}>
          <div style={{ fontFamily: MONO, fontSize: 14, lineHeight: 1.7 }}>
            <div>
              wrong addresses: <b style={{ color: polyWrong ? THEME.warn : THEME.accent }}>{polyWrong}</b> / {DOWNTOWN.length}
            </div>
            <div style={{ color: THEME.muted, fontSize: 13 }}>{hoverText}</div>
          </div>
        </Panel>
      </>
    );
  } else if (screen === 1) {
    question = (
      <>
        Now raise N well past 10. Watch the vote count along the east-west street through the zone (the plot under the map). Inside the zone every street votes yes. How many vote yes far away from it, and does that number depend on how far away you go?
      </>
    );
    stage = (
      <>
        <ZoneMap points={DOWNTOWN} field={polyField} pred={polyField} lines={Streets({ ring, c, r }) || []} handles={[{ id: "c", x: c[0], y: c[1], color: THEME.highlight, shape: "square" }]} onMove={onMove} onHover={onHover} res={90} ariaLabel="Map with many streets" />
        <div style={{ marginTop: 8 }}>
          <CrossSection fn={(x) => V(x, c[1])} n={big} y0={c[1]} threshold={kk} />
        </div>
      </>
    );
    side = (
      <>
        <Panel>
          <Slider id="l2-n2" label="N, number of hidden neurons" value={n} min={3} max={64} stepSize={1} onChange={setNk} display={String(big)} />
          <Slider id="l2-k2" label="K, votes needed to fire" value={kk} min={1} max={big} stepSize={1} onChange={setK} display={String(kk)} />
          <Slider id="l2-r2" label="zone size" value={r} min={0.08} max={0.3} stepSize={0.005} onChange={setR} display={fmt(r * 10, 1) + " km"} />
        </Panel>
        <Panel edge={polyWrong === 0 ? THEME.accent : THEME.grid}>
          <div style={{ fontFamily: MONO, fontSize: 14, lineHeight: 1.7 }}>
            <div>
              wrong addresses: <b style={{ color: polyWrong ? THEME.warn : THEME.accent }}>{polyWrong}</b> / {DOWNTOWN.length}
            </div>
            <div style={{ color: THEME.muted, fontSize: 13 }}>{hoverText}</div>
            <div style={{ color: THEME.muted, fontSize: 13 }}>votes at the map corner (0, 0): {V(0.01, 0.01)} of {big}</div>
          </div>
        </Panel>
        <Note>Try K a little above N/2 with N = 64. Then try K = N. The polygon becomes a disc either way. Which K still works if two districts overlap?</Note>
      </>
    );
  } else if (screen === 2) {
    question = (
      <>
        Each disc above is a "cylinder": N votes inside, about N/2 outside. Subtract N/2 and a disc gives N/2 inside and roughly 0 outside. That makes discs addable. Drag the two disc centres. Where does the summed signal cross the threshold?
      </>
    );
    stage = (
      <>
        <ZoneMap
          points={[]}
          field={twoField}
          handles={[
            { id: "t0", x: two[0][0], y: two[0][1], color: THEME.highlight, shape: "square" },
            { id: "t1", x: two[1][0], y: two[1][1], color: THEME.highlight, shape: "square" },
          ]}
          onMove={onMove}
          onHover={onHover}
          res={80}
          ariaLabel="Map with two added cylinders"
        />
        <div style={{ marginTop: 8 }}>
          <CrossSection fn={(x) => S2(x, two[0][1]) + CH_N / 2} n={CH_N} y0={two[0][1]} threshold={TH + CH_N / 2} labelTop="N/2 (one disc)" labelMid="0 (outside both)" />
        </div>
        <Note>{hoverText}</Note>
      </>
    );
    side = (
      <>
        <Panel>
          <Label>What the votes are doing</Label>
          <Eq size={13.5}>
            convex polygon: &nbsp;y = [ Σᵢ zᵢ ≥ N ]
            <br />
            one cylinder: &nbsp;&nbsp;&nbsp;Σᵢ zᵢ − N/2 ≈ N/2 inside, ≈ 0 outside
            <br />
            any shape: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;y = [ Σ_discs (Σᵢ zᵢ − N/2) &gt; θ ]
          </Eq>
          <Note>Every zᵢ is a hidden step neuron, and all the discs share the same single hidden layer. The output neuron is one weighted sum and one threshold. Smaller and more numerous discs fit any outline as closely as you like (slides 63-67).</Note>
        </Panel>
        <Panel>
          <Label>Slide 55, the classic table (step activations)</Label>
          <div style={{ display: "grid", gridTemplateColumns: "auto 1fr", gap: "4px 12px", fontSize: 13.5 }}>
            <span style={{ fontFamily: MONO, color: THEME.muted }}>no hidden layer</span>
            <span>a half-plane (one street)</span>
            <span style={{ fontFamily: MONO, color: THEME.muted }}>1 hidden layer</span>
            <span>a convex polygon (AND of streets)</span>
            <span style={{ fontFamily: MONO, color: THEME.muted }}>2 hidden layers</span>
            <span>any union of polygons (OR of ANDs)</span>
          </div>
          <Note>
            <span style={{ display: "block", marginTop: 8 }}>The disc trick beats the table: one hidden layer already gets any region, if you can afford a huge number of neurons. Lesson 4 is about that cost.</span>
          </Note>
        </Panel>
      </>
    );
  } else {
    const handles = active.map((q, i) => ({ id: "q" + i, x: q.c[0], y: q.c[1], color: i === sel ? THEME.highlight : THEME.text, shape: "square" }));
    question = (
      <>
        The river-bend contract: deliver along an L-shaped strip on the river bank, and nowhere else. Each disc costs 48 hidden neurons. Fence in the strip with zero wrong addresses. What is the cheapest network you can find?
      </>
    );
    stage = (
      <>
        <ZoneMap points={LZONE} field={chField} pred={chField} handles={handles} onMove={onMove} onHover={onHover} res={80} ariaLabel="Challenge map with draggable discs" />
        <Legend />
      </>
    );
    side = (
      <>
        <Panel>
          <Label>Discs in the network</Label>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
            {[1, 2, 3, 4, 5].map((m) => (
              <Pill key={m} value={m} active={used === m} onPick={setUsed}>
                {m}
              </Pill>
            ))}
          </div>
          <Label>
            <span style={{ display: "block", marginTop: 10 }}>Selected disc (drag one to select it)</span>
          </Label>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
            {active.map((q, i) => (
              <Pill key={i} value={i} active={sel === i} onPick={setSel}>
                disc {i + 1}
              </Pill>
            ))}
          </div>
          <Slider id="l2-cr" label={"radius of disc " + (Math.min(sel, used - 1) + 1)} value={(circles[Math.min(sel, used - 1)] || circles[0]).r} min={0.05} max={0.3} stepSize={0.005} onChange={setSelR} display={fmt((circles[Math.min(sel, used - 1)] || circles[0]).r * 10, 1) + " km"} />
        </Panel>
        <Panel edge={chWrong === 0 ? THEME.accent : THEME.grid}>
          <div style={{ fontFamily: MONO, fontSize: 14, lineHeight: 1.7 }}>
            <div>
              wrong addresses: <b style={{ color: chWrong ? THEME.warn : THEME.accent }}>{chWrong}</b> / {LZONE.length}
            </div>
            <div>hidden neurons: {used * CH_N} · output: 1</div>
            <div style={{ color: THEME.muted, fontSize: 13 }}>{hoverText}</div>
          </div>
          <Hint
            hints={[
              "Point at the gap between two discs. If the sum there is below the threshold, the strip has a hole. Overlap the discs a little.",
              "Start with three or four discs on the middle line of each arm, then try to drop one. It can be done with two: a small disc on the upper arm and a big one over the corner and the lower arm.",
            ]}
          />
        </Panel>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="Fence any zone" question={question} stage={stage} side={side} />;
}
