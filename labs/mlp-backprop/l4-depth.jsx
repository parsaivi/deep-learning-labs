import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import { THEME, MONO } from "./common";

// Lesson 4 - Depth vs width, using parity (slides 23-41).
// Aha: one hidden layer can compute any Boolean function, but for parity every "odd" row needs
// its own detector (no two of them can be merged), so width grows as 2^(D-1). Stacking XOR blocks
// re-uses earlier answers, and the deep net grows only as 3(D-1).

// The city is cut by a winding river. A route crosses some of D bridges; every crossing flips the bank.
// y = 1 if the courier ends on the far bank, i.e. the route crosses an odd number of bridges.

function bitsOf(row, D) {
  const b = [];
  for (let i = D - 1; i >= 0; i--) b.push((row >> i) & 1);
  return b;
}
function parity(bits) {
  return bits.reduce((a, b) => a ^ b, 0);
}
function matches(pattern, bits) {
  for (let i = 0; i < bits.length; i++) if (pattern[i] !== "*" && Number(pattern[i]) !== bits[i]) return false;
  return true;
}

function BridgeRow({ bits, onToggle, small }) {
  const D = bits.length;
  const W = D * 46 + 20;
  const H = small ? 70 : 96;
  const crossings = bits.reduce((a, b) => a + b, 0);
  const far = crossings % 2 === 1;
  return (
    <div>
    <svg viewBox={"0 0 " + W + " " + H} style={{ width: "100%", maxWidth: W * 1.3, display: "block" }} role="group" aria-label="Bridges on the route">
      <rect x="0" y="0" width={W} height={H} fill={THEME.bg} stroke={THEME.grid} />
      {bits.map((b, i) => {
        const x = 30 + i * 46;
        return (
          <g key={i} onClick={onToggle ? () => onToggle(i) : undefined} style={{ cursor: onToggle ? "pointer" : "default" }}>
            <path d={"M" + (x + 4) + " 6 C " + (x + 22) + " " + H * 0.35 + ", " + (x - 6) + " " + H * 0.65 + ", " + (x + 12) + " " + (H - 6)} fill="none" stroke={rgba(THEME.accent, 0.45)} strokeWidth={9} />
            <rect x={x - 10} y={H / 2 - 9} width={34} height={18} rx={2} fill={b ? rgba(THEME.highlight, 0.3) : THEME.surface} stroke={b ? THEME.highlight : THEME.grid} strokeWidth={1.6} />
            <text x={x + 7} y={H / 2 + 4.5} textAnchor="middle" fontFamily={MONO} fontSize={12} fill={THEME.text}>
              {b ? "✓" : "·"}
            </text>
            {!small && (
              <text x={x + 7} y={H - 8} textAnchor="middle" fontFamily={MONO} fontSize={10.5} fill={THEME.muted}>
                x{i + 1}
              </text>
            )}
          </g>
        );
      })}
    </svg>
    <div style={{ fontFamily: MONO, fontSize: 13, marginTop: 4, color: far ? THEME.highlight : THEME.muted }}>
      {crossings} crossings → {far ? "■ far bank (y = 1)" : "□ home bank (y = 0)"} · click a bridge to toggle it
    </div>
    </div>
  );
}

function DetectorChip({ pattern, idx, onCycle, onRemove }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 4, fontFamily: MONO, fontSize: 13, border: "1px solid " + THEME.grid, borderRadius: 3, padding: "3px 5px", background: THEME.bg }}>
      <span style={{ color: THEME.muted, fontSize: 11 }}>n{idx + 1}</span>
      {pattern.map((c, i) => (
        <button
          key={i}
          onClick={() => onCycle(idx, i)}
          title="click to cycle 1 → 0 → don't care"
          style={{ fontFamily: MONO, fontSize: 13, width: 22, padding: "1px 0", cursor: "pointer", background: c === "*" ? rgba(THEME.accent, 0.15) : "transparent", color: c === "*" ? THEME.accent : THEME.text, border: "1px solid " + THEME.grid, borderRadius: 2 }}
        >
          {c === "*" ? "–" : c}
        </button>
      ))}
      <button onClick={() => onRemove(idx)} aria-label="remove detector" style={{ fontFamily: MONO, background: "transparent", color: THEME.warn, border: "none", cursor: "pointer", fontSize: 14 }}>
        ×
      </button>
    </div>
  );
}

function TruthTable({ D, rowOk, rowOut, onRow, compact }) {
  const rows = [];
  for (let r = 0; r < 1 << D; r++) {
    const b = bitsOf(r, D);
    const y = parity(b);
    const out = rowOut(b);
    const ok = out === y;
    rows.push(
      <tr key={r} onClick={onRow ? () => onRow(b) : undefined} style={{ cursor: onRow ? "pointer" : "default", background: ok ? "transparent" : rgba(THEME.warn, 0.14) }}>
        {b.map((v, i) => (
          <td key={i} style={{ padding: compact ? "0 5px" : "1px 6px", color: v ? THEME.text : THEME.muted, textAlign: "center" }}>
            {v}
          </td>
        ))}
        <td style={{ padding: "0 8px", color: y ? THEME.highlight : THEME.muted, textAlign: "center", borderLeft: "1px solid " + THEME.grid }}>{y}</td>
        <td style={{ padding: "0 8px", textAlign: "center", color: ok ? THEME.accent : THEME.warn }}>
          {out} {ok ? "✓" : "✗"}
        </td>
      </tr>
    );
  }
  return (
    <div style={{ overflowX: "auto", maxHeight: 420, overflowY: "auto", border: "1px solid " + THEME.grid, borderRadius: 3 }}>
      <table style={{ borderCollapse: "collapse", fontFamily: MONO, fontSize: compact ? 12 : 13, width: "100%" }}>
        <thead>
          <tr style={{ color: THEME.muted, position: "sticky", top: 0, background: THEME.surface }}>
            {Array.from({ length: D }, (_, i) => (
              <th key={i} style={{ padding: "2px 6px", fontWeight: 500 }}>
                x{i + 1}
              </th>
            ))}
            <th style={{ padding: "2px 8px", fontWeight: 500, borderLeft: "1px solid " + THEME.grid }}>far?</th>
            <th style={{ padding: "2px 8px", fontWeight: 500 }}>net</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
  );
}

function GrowthChart({ D }) {
  const W = 560;
  const H = 230;
  const Dmax = 16;
  const ymax = Math.log10(40000);
  const X = (d) => 50 + ((d - 2) / (Dmax - 2)) * (W - 70);
  const Y = (v) => H - 28 - (Math.log10(Math.max(1, v)) / ymax) * (H - 48);
  const shallow = (d) => Math.pow(2, d - 1) + 1;
  const deep = (d) => 3 * (d - 1);
  const pts = (fn) => {
    const a = [];
    for (let d = 2; d <= Dmax; d++) a.push(X(d) + "," + Y(fn(d)));
    return a.join(" ");
  };
  return (
    <svg viewBox={"0 0 " + W + " " + H} style={{ width: "100%", display: "block" }} role="img" aria-label="Neurons needed, shallow vs deep, log scale">
      <rect x="0" y="0" width={W} height={H} fill={THEME.bg} stroke={THEME.grid} />
      {[1, 10, 100, 1000, 10000].map((v) => (
        <g key={v}>
          <line x1={50} x2={W - 20} y1={Y(v)} y2={Y(v)} stroke={THEME.grid} strokeWidth={0.6} />
          <text x={44} y={Y(v) + 4} textAnchor="end" fontFamily={MONO} fontSize={10} fill={THEME.muted}>
            {v}
          </text>
        </g>
      ))}
      {[2, 4, 6, 8, 10, 12, 14, 16].map((d) => (
        <text key={d} x={X(d)} y={H - 10} textAnchor="middle" fontFamily={MONO} fontSize={10} fill={THEME.muted}>
          {d}
        </text>
      ))}
      <text x={W - 20} y={H - 10} textAnchor="end" fontFamily={MONO} fontSize={10} fill={THEME.muted}>
        bridges D
      </text>
      <line x1={X(D)} x2={X(D)} y1={14} y2={H - 28} stroke={THEME.text} strokeDasharray="3 3" />
      <polyline points={pts(shallow)} fill="none" stroke={THEME.warn} strokeWidth={2.2} strokeDasharray="6 4" />
      <polyline points={pts(deep)} fill="none" stroke={THEME.accent} strokeWidth={2.2} />
      <circle cx={X(D)} cy={Y(shallow(D))} r={4.5} fill={THEME.warn} />
      <rect x={X(D) - 4} y={Y(deep(D)) - 4} width={8} height={8} fill={THEME.accent} />
      <text x={X(12) - 8} y={Y(shallow(12)) - 4} textAnchor="end" fontFamily={MONO} fontSize={11} fill={THEME.warn}>
        one hidden layer: 2^(D−1)+1
      </text>
      <text x={X(10)} y={Y(deep(10)) + 18} fontFamily={MONO} fontSize={11} fill={THEME.accent}>
        chain of XOR blocks: 3(D−1)
      </text>
      <text x={56} y={24} fontFamily={MONO} fontSize={10.5} fill={THEME.muted}>
        neurons (log scale)
      </text>
    </svg>
  );
}

// XOR block = 3 threshold neurons: OR, AND, and "OR but not AND"
function xorBlock(a, b) {
  const or = a + b - 0.5 >= 0 ? 1 : 0;
  const and = a + b - 1.5 >= 0 ? 1 : 0;
  return { or, and, y: or - 2 * and - 0.5 >= 0 ? 1 : 0 };
}
function ChainDiagram({ bits }) {
  const D = bits.length;
  const W = Math.max(420, 70 + (D - 1) * 66);
  const H = 150;
  let acc = bits[0];
  const blocks = [];
  for (let i = 1; i < D; i++) {
    const r = xorBlock(acc, bits[i]);
    blocks.push({ i, inA: acc, inB: bits[i], out: r.y });
    acc = r.y;
  }
  const bx = (i) => 40 + i * 66;
  return (
    <div style={{ overflowX: "auto" }}>
      <svg viewBox={"0 0 " + W + " " + H} style={{ width: "100%", minWidth: Math.min(W, 560), display: "block" }} role="img" aria-label="Chain of XOR blocks">
        <rect x="0" y="0" width={W} height={H} fill={THEME.bg} stroke={THEME.grid} />
        {bits.map((b, i) => (
          <g key={i}>
            <circle cx={bx(i)} cy={H - 24} r={11} fill={b ? rgba(THEME.highlight, 0.35) : THEME.surface} stroke={THEME.grid} />
            <text x={bx(i)} y={H - 20} textAnchor="middle" fontFamily={MONO} fontSize={11} fill={THEME.text}>
              {b}
            </text>
          </g>
        ))}
        {blocks.map((B) => {
          const x = bx(B.i);
          const y = 60;
          return (
            <g key={B.i}>
              <line x1={bx(B.i)} y1={H - 35} x2={x} y2={y + 16} stroke={THEME.grid} />
              <line x1={B.i === 1 ? bx(0) : bx(B.i - 1)} y1={B.i === 1 ? H - 35 : y} x2={x - 18} y2={y + 4} stroke={B.inA ? THEME.highlight : THEME.grid} strokeWidth={B.inA ? 1.8 : 1} />
              <rect x={x - 20} y={y - 16} width={40} height={32} rx={3} fill={B.out ? rgba(THEME.highlight, 0.28) : THEME.surface} stroke={THEME.accent} />
              <text x={x} y={y - 2} textAnchor="middle" fontFamily={MONO} fontSize={10} fill={THEME.muted}>
                XOR
              </text>
              <text x={x} y={y + 11} textAnchor="middle" fontFamily={MONO} fontSize={11.5} fill={THEME.text}>
                {B.out}
              </text>
            </g>
          );
        })}
        <text x={8} y={16} fontFamily={MONO} fontSize={10.5} fill={THEME.muted}>
          each box = 3 neurons · {D - 1} boxes · {3 * (D - 1)} neurons · {2 * (D - 1)} layers deep
        </text>
      </svg>
    </div>
  );
}

const OPTS4 = ["x1", "x2", "x3", "x4", "B1", "B2"];
function BlockSelect({ id, value, options, onPick }) {
  const change = useCallback((e) => onPick(e.target.value), [onPick]);
  return (
    <select id={id} value={value} onChange={change} style={{ fontFamily: MONO, fontSize: 13, background: THEME.bg, color: THEME.text, border: "1px solid " + THEME.grid, borderRadius: 3, padding: "3px 4px" }}>
      {options.map((o) => (
        <option key={o} value={o}>
          {o}
        </option>
      ))}
    </select>
  );
}
function evalWiring(w, bits) {
  const v = { x1: bits[0], x2: bits[1], x3: bits[2], x4: bits[3] };
  for (let i = 0; i < 3; i++) v["B" + (i + 1)] = xorBlock(v[w[i][0]], v[w[i][1]]).y;
  return v[w.out];
}
function wiringDepth(w) {
  const d = { x1: 0, x2: 0, x3: 0, x4: 0 };
  for (let i = 0; i < 3; i++) d["B" + (i + 1)] = 1 + Math.max(d[w[i][0]], d[w[i][1]]);
  return d[w.out];
}

export default function DepthLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [D, setD] = useState(3);
  const [bits, setBits] = useState([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1]);
  const [dets, setDets] = useState([]);
  const [DD, setDD] = useState(8);
  const [wiring, setWiring] = useState({ 0: ["x1", "x2"], 1: ["x1", "x3"], 2: ["B1", "B2"], out: "B3" });

  const toggle = useCallback((i) => setBits((prev) => prev.map((b, j) => (j === i ? 1 - b : b))), []);
  const setDim = useCallback((v) => {
    setD(v);
    setDets([]);
  }, []);
  const addRow = useCallback((b) => setDets((prev) => (prev.some((p) => p.join("") === b.join("")) ? prev : prev.concat([b.map(String)]))), []);
  const cycle = useCallback((idx, i) => setDets((prev) => prev.map((p, j) => (j !== idx ? p : p.map((c, k) => (k !== i ? c : c === "1" ? "0" : c === "0" ? "*" : "1"))))), []);
  const remove = useCallback((idx) => setDets((prev) => prev.filter((_, j) => j !== idx)), []);
  const netOut = useCallback((b) => (dets.some((p) => matches(p, b)) ? 1 : 0), [dets]);
  const wrongRows = useMemo(() => {
    let w = 0;
    for (let r = 0; r < 1 << D; r++) {
      const b = bitsOf(r, D);
      if (netOut(b) !== parity(b)) w++;
    }
    return w;
  }, [D, netOut]);

  const pick = useCallback((key, slot, val) => setWiring((prev) => {
    const next = Object.assign({}, prev);
    if (key === "out") next.out = val;
    else next[key] = slot === 0 ? [val, prev[key][1]] : [prev[key][0], val];
    return next;
  }), []);
  const p00 = useCallback((v) => pick(0, 0, v), [pick]);
  const p01 = useCallback((v) => pick(0, 1, v), [pick]);
  const p10 = useCallback((v) => pick(1, 0, v), [pick]);
  const p11 = useCallback((v) => pick(1, 1, v), [pick]);
  const p20 = useCallback((v) => pick(2, 0, v), [pick]);
  const p21 = useCallback((v) => pick(2, 1, v), [pick]);
  const pOut = useCallback((v) => pick("out", 0, v), [pick]);
  const wOut = useCallback((b) => evalWiring(wiring, b), [wiring]);
  const wWrong = useMemo(() => {
    let w = 0;
    for (let r = 0; r < 16; r++) {
      const b = bitsOf(r, 4);
      if (evalWiring(wiring, b) !== parity(b)) w++;
    }
    return w;
  }, [wiring]);
  const depth = wiringDepth(wiring);

  let question, stage, side;
  if (screen === 0) {
    question = (
      <>
        A winding river cuts the city, and every bridge a route crosses flips which bank the courier is on. The dispatcher needs y = 1 when a route ends on the far bank. With one hidden layer, each hidden neuron can detect one pattern of crossed bridges. Click truth-table rows to add detectors. What is the fewest you can get away with?
      </>
    );
    stage = (
      <>
        <BridgeRow bits={bits.slice(0, D)} onToggle={toggle} />
        <div style={{ marginTop: 8 }}>
          <TruthTable D={D} rowOut={netOut} onRow={addRow} />
        </div>
      </>
    );
    side = (
      <>
        <Panel>
          <Slider id="l4-d" label="bridges on the route, D" value={D} min={2} max={5} stepSize={1} onChange={setDim} display={String(D)} />
          <Label>hidden neurons (pattern detectors) → output = OR</Label>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
            {dets.length === 0 && <span style={{ color: THEME.muted, fontSize: 13 }}>none yet: click a row on the left</span>}
            {dets.map((p, i) => (
              <DetectorChip key={i} idx={i} pattern={p} onCycle={cycle} onRemove={remove} />
            ))}
          </div>
          <Note>
            <span style={{ display: "block", marginTop: 8 }}>A bit set to "–" means "don't care", so one detector covers two rows. Try it to save a neuron and watch what happens to the table.</span>
          </Note>
        </Panel>
        <Panel edge={wrongRows === 0 ? THEME.accent : THEME.grid}>
          <div style={{ fontFamily: MONO, fontSize: 14, lineHeight: 1.7 }}>
            <div>
              wrong rows: <b style={{ color: wrongRows ? THEME.warn : THEME.accent }}>{wrongRows}</b> / {1 << D}
            </div>
            <div>hidden neurons used: {dets.length}</div>
          </div>
        </Panel>
      </>
    );
  } else if (screen === 1) {
    const b = bits.slice(0, DD);
    question = (
      <>
        A deep network can do this differently. It answers "far bank after bridges 1 and 2?", then feeds that answer forward with bridge 3, and so on, one XOR block (3 neurons) per bridge. Slide D up. Where do the two curves cross, and how many neurons does each design need for 16 bridges?
      </>
    );
    stage = (
      <>
        <GrowthChart D={DD} />
        <div style={{ marginTop: 8 }}>
          <BridgeRow bits={b} onToggle={toggle} small />
        </div>
        <div style={{ marginTop: 6 }}>
          <ChainDiagram bits={b} />
        </div>
      </>
    );
    side = (
      <>
        <Panel>
          <Slider id="l4-dd" label="bridges D" value={DD} min={2} max={16} stepSize={1} onChange={setDD} display={String(DD)} />
          <div style={{ display: "grid", gridTemplateColumns: "1fr auto auto", gap: "4px 12px", fontFamily: MONO, fontSize: 13 }}>
            <span />
            <span style={{ color: THEME.muted }}>neurons</span>
            <span style={{ color: THEME.muted }}>weights+biases</span>
            <span style={{ color: THEME.warn }}>● one hidden layer</span>
            <span>{Math.pow(2, DD - 1) + 1}</span>
            <span>{(DD + 2) * Math.pow(2, DD - 1) + 1}</span>
            <span style={{ color: THEME.accent }}>■ XOR chain</span>
            <span>{3 * (DD - 1)}</span>
            <span>{9 * (DD - 1)}</span>
          </div>
        </Panel>
        <Note>Toggle bridges in the small row. The XOR chain passes one bit forward: "odd so far?". The wide network can't pass anything forward, so it has to list every odd pattern separately.</Note>
      </>
    );
  } else if (screen === 2) {
    question = (
      <>
        Name it. You found that no detector can merge two odd rows: flipping any one bridge flips the answer, so odd rows are never neighbours. That is the worst case of slide 33. Which number on the right is the price of having one layer?
      </>
    );
    stage = (
      <>
        <Panel>
          <Label>Parity on D inputs (slides 38-40)</Label>
          <Eq size={13.5}>
            one hidden layer (DNF): &nbsp;2^(D−1) hidden neurons, (D+2)·2^(D−1)+1 weights
            <br />
            chain of XORs: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3(D−1) neurons, 9(D−1) weights, 2(D−1) layers
            <br />
            balanced tree of XORs: 3(D−1) neurons, only 2⌈log₂D⌉ layers
          </Eq>
          <Label>
            <span style={{ display: "block", marginTop: 8 }}>One XOR block, using the threshold neurons from lesson 1</span>
          </Label>
          <Eq size={13.5}>
            h₁ = step(a + b − 0.5) &nbsp;&nbsp;(OR)
            <br />
            h₂ = step(a + b − 1.5) &nbsp;&nbsp;(AND)
            <br />y = step(h₁ − 2h₂ − 0.5) &nbsp;(OR but not AND)
          </Eq>
        </Panel>
      </>
    );
    side = (
      <>
        <Panel>
          <Slider id="l4-dd2" label="bridges D" value={DD} min={2} max={16} stepSize={1} onChange={setDD} display={String(DD)} />
          <div style={{ fontFamily: MONO, fontSize: 13.5, lineHeight: 1.8 }}>
            <div style={{ color: THEME.warn }}>● shallow: {Math.pow(2, DD - 1)} hidden</div>
            <div style={{ color: THEME.accent }}>■ deep chain: {3 * (DD - 1)} neurons</div>
            <div>ratio: {fmt(Math.pow(2, DD - 1) / (3 * (DD - 1)), 1)}×</div>
          </div>
        </Panel>
        <Note>
          The general lesson (slide 41): one hidden layer is a universal Boolean machine, but it may need exponentially many neurons. Deeper networks can need exponentially fewer. Caveat (slide 42, Shannon): for most Boolean functions of D inputs, about 2^D/D gates are needed whatever the depth. Depth helps functions with structure, like parity, that can reuse partial answers.
        </Note>
      </>
    );
  } else {
    question = (
      <>
        Four bridges and three XOR blocks (9 neurons). Wire the blocks so the network gets all 16 routes right. Then rewire for the fewest layers. How deep must it be?
      </>
    );
    stage = (
      <>
        <TruthTable D={4} rowOut={wOut} compact />
      </>
    );
    side = (
      <>
        <Panel edge={wWrong === 0 ? THEME.accent : THEME.grid}>
          <Label>wiring</Label>
          <div style={{ display: "grid", gridTemplateColumns: "auto auto auto auto", gap: "6px 8px", alignItems: "center", fontFamily: MONO, fontSize: 13 }}>
            <span>B1 = XOR(</span>
            <BlockSelect id="w00" value={wiring[0][0]} options={OPTS4.slice(0, 4)} onPick={p00} />
            <BlockSelect id="w01" value={wiring[0][1]} options={OPTS4.slice(0, 4)} onPick={p01} />
            <span>)</span>
            <span>B2 = XOR(</span>
            <BlockSelect id="w10" value={wiring[1][0]} options={OPTS4.slice(0, 5)} onPick={p10} />
            <BlockSelect id="w11" value={wiring[1][1]} options={OPTS4.slice(0, 5)} onPick={p11} />
            <span>)</span>
            <span>B3 = XOR(</span>
            <BlockSelect id="w20" value={wiring[2][0]} options={OPTS4} onPick={p20} />
            <BlockSelect id="w21" value={wiring[2][1]} options={OPTS4} onPick={p21} />
            <span>)</span>
            <span>output =</span>
            <BlockSelect id="wout" value={wiring.out} options={["B1", "B2", "B3"]} onPick={pOut} />
            <span />
            <span />
          </div>
          <div style={{ fontFamily: MONO, fontSize: 14, lineHeight: 1.7, marginTop: 10 }}>
            <div>
              wrong routes: <b style={{ color: wWrong ? THEME.warn : THEME.accent }}>{wWrong}</b> / 16
            </div>
            <div>
              depth: {depth} block{depth === 1 ? "" : "s"} = {2 * depth} neuron layers
            </div>
          </div>
          <Hint
            hints={[
              "Every bridge has to reach the output exactly once. Check which x's feed into B3, directly or through B1 and B2.",
              "Pair the bridges up: (x1, x2) in one block, (x3, x4) in another, then combine the two answers. That is the tree from slide 40.",
            ]}
          />
        </Panel>
        <Note>A one-hidden-layer network needs 8 detectors plus the output here: 9 neurons, a tie with yours. Look back at the chart: by 16 bridges it is 32 769 against 45.</Note>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="Wide or deep?" question={question} stage={stage} side={side} />;
}
