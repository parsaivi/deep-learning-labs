import { useState, useMemo, useCallback } from "react";
import { THEME, MONO } from "./common";

// Lesson 5 - Receptive field: stacking small filters (CNN deck, slides 91-113).
// Aha: a unit in layer 2 only sees 3 units of layer 1, but each of those saw 3 pixels, so together it sees
// 5 pixels: two 3x3 layers look at the same 5x5 block as one 5x5 filter, with fewer weights (18 C^2 vs 25 C^2)
// and an extra nonlinearity. Lower layers find the parts (tape, edges), higher layers arrange them (a parcel).

// @sim-start
const IN5 = 27;
function stack5(L, F, s1) {
  const layers = [];
  let n = IN5;
  for (let l = 0; l < L; l++) {
    const S = l === 0 && s1 ? 2 : 1;
    n = Math.floor((n - F) / S) + 1;
    layers.push({ F, S, n });
  }
  let rf = 1;
  let jump = 1;
  for (const ly of layers) {
    rf += (ly.F - 1) * jump;
    jump *= ly.S;
  }
  return { layers, rf, params: L * F * F };
}
const GOALC5 = { rf: 11, params: 50 };
// @sim-end

function RFStage({ st, pick, onPick }) {
  const L = st.layers.length;
  const ok = st.layers.every((l) => l.n >= 1);
  const cell = 21;
  const x0 = 310 - (IN5 * cell) / 2;
  const rowGap = 34;
  const H = 40 + (L + 1) * rowGap + 20;
  const yOf = (l) => H - 40 - l * rowGap;
  // receptive interval in every layer, top unit = pick (clamped to the top layer)
  const iv = [];
  if (ok) {
    const top = st.layers[L - 1];
    let a = clamp(pick, 0, top.n - 1);
    let b = a;
    iv[L] = [a, b];
    for (let l = L - 1; l >= 0; l--) {
      const ly = st.layers[l];
      a = a * ly.S;
      b = b * ly.S + ly.F - 1;
      iv[l] = [a, b];
    }
  }
  // x position of unit i in layer l (layer 0 = input): centre of its receptive field in input coordinates
  const pos = [];
  let start = 0;
  let span = 1;
  let jump = 1;
  pos.push((i) => x0 + (i + 0.5) * cell);
  for (let l = 0; l < L; l++) {
    const ly = st.layers[l];
    span = span + (ly.F - 1) * jump;
    const j = jump * ly.S;
    const s0 = start;
    const jj = jump;
    const sp = span;
    pos.push((i) => x0 + (s0 + i * jj * ly.S + sp / 2) * cell);
    start = s0;
    jump = j;
  }
  const click = useCallback(
    (e) => {
      if (!ok) return;
      const svg = e.currentTarget.ownerSVGElement;
      const p = svgPoint(svg, e);
      const top = st.layers[L - 1];
      let best = 0;
      let bd = Infinity;
      for (let i = 0; i < top.n; i++) {
        const d = Math.abs(pos[L](i) - p.x);
        if (d < bd) {
          bd = d;
          best = i;
        }
      }
      onPick(best);
    },
    [ok, st, L, onPick]
  );
  return (
    <svg viewBox={"0 0 620 " + H} style={{ width: "100%", display: "block", touchAction: "none" }} role="img" aria-label={"Stack of " + L + " conv layers over one row of pixels; the highlighted cells are the receptive field"}>
      <text x={10} y={16} fontFamily={MONO} fontSize={10.5} fill={THEME.muted}>
        one row of the photo (bottom) and the layers above it · click a top unit
      </text>
      {Array.from({ length: IN5 }).map((_, i) => {
        const inRF = ok && i >= iv[0][0] && i <= iv[0][1];
        return <rect key={"i" + i} x={x0 + i * cell + 1} y={yOf(0) - 8} width={cell - 2} height={16} fill={inRF ? rgba(THEME.highlight, 0.85) : rgba(THEME.text, 0.12)} stroke={THEME.grid} />;
      })}
      <text x={6} y={yOf(0) + 4} fontFamily={MONO} fontSize={10} fill={THEME.muted}>
        input
      </text>
      {ok &&
        st.layers.map((ly, l) => (
          <g key={l}>
            <text x={6} y={yOf(l + 1) + 4} fontFamily={MONO} fontSize={10} fill={THEME.muted}>
              L{l + 1}
              {ly.S > 1 ? " s2" : ""}
            </text>
            {Array.from({ length: ly.n }).map((_, i) => {
              const inRF = i >= iv[l + 1][0] && i <= iv[l + 1][1];
              const isTop = l === L - 1 && inRF;
              return (
                <g key={i}>
                  {inRF &&
                    Array.from({ length: ly.F }).map((__, k) => (
                      <line key={k} x1={pos[l + 1](i)} y1={yOf(l + 1) + 6} x2={pos[l](i * ly.S + k)} y2={yOf(l) - 7} stroke={THEME.highlight} strokeWidth={0.8} opacity={0.6} />
                    ))}
                  <circle cx={pos[l + 1](i)} cy={yOf(l + 1)} r={isTop ? 7 : 5} fill={inRF ? THEME.highlight : THEME.surface} stroke={inRF ? THEME.highlight : THEME.accent} strokeWidth={1.2} />
                </g>
              );
            })}
          </g>
        ))}
      {ok && <rect x={0} y={yOf(L) - 14} width={620} height={28} fill="transparent" style={{ cursor: "pointer" }} onPointerDown={click} />}
      {!ok && (
        <text x={310} y={H / 2} textAnchor="middle" fontFamily={MONO} fontSize={12} fill={THEME.warn}>
          too deep for this {IN5}-pixel strip to draw (the numbers on the right still hold)
        </text>
      )}
      <text x={10} y={H - 6} fontFamily={MONO} fontSize={11} fill={THEME.highlight}>
        ■ receptive field of the chosen unit: {st.rf} pixels wide ({st.rf} × {st.rf} in 2-D)
      </text>
    </svg>
  );
}

export default function ReceptiveLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [arch, setArch] = useState("a");
  const [L, setL] = useState(2);
  const [F, setF] = useState(3);
  const [s1, setS1] = useState(false);
  const [pick, setPick] = useState(100);
  const [LC, setLC] = useState(1);
  const [FC, setFC] = useState(3);
  const [sC, setSC] = useState(false);
  const st0 = useMemo(() => (arch === "a" ? stack5(1, 5, false) : stack5(2, 3, false)), [arch]);
  const st = useMemo(() => stack5(L, F, s1), [L, F, s1]);
  const stC = useMemo(() => stack5(LC, FC, sC), [LC, FC, sC]);
  const top = (s) => Math.floor(s.layers[s.layers.length - 1].n / 2);
  const pk = (s) => (pick >= 100 ? top(s) : pick);
  const okC = stC.rf >= GOALC5.rf && stC.params <= GOALC5.params;

  let question, stage, side;
  if (screen === 0) {
    question = (
      <>
        Two ways to look at a 5-pixel-wide stretch of the doorstep: A, one layer of 5-wide filters, or B, two layers of 3-wide filters stacked. Click a unit at the top of each. How much of the photo does it see? Which design needs fewer weights?
      </>
    );
    stage = <RFStage st={st0} pick={pk(st0)} onPick={setPick} />;
    side = (
      <Panel>
        <Pills options={[["a", "A: one 5 × 5 layer"], ["b", "B: two 3 × 3 layers"]]} value={arch} onPick={setArch} />
        <Readout
          rows={[
            ["receptive field", st0.rf + " × " + st0.rf],
            ["weights (C channels in and out)", st0.params + " C²", THEME.highlight],
            ["ReLUs on the way", String(st0.layers.length)],
          ]}
        />
      </Panel>
    );
  } else if (screen === 1) {
    question = (
      <>
        Stack more layers, change the filter size, try a stride of 2 in the first layer. How fast does the receptive field grow with depth? What does the stride do to every layer above it?
      </>
    );
    stage = <RFStage st={st} pick={pk(st)} onPick={setPick} />;
    side = (
      <Panel>
        <Slider id="c5-l" label="layers" value={L} min={1} max={6} stepSize={1} onChange={setL} display={String(L)} />
        <Slider id="c5-f" label="filter size F" value={F} min={3} max={7} stepSize={2} onChange={setF} display={F + " × " + F} />
        <Toggle label="stride 2 in layer 1" value={s1} onChange={setS1} />
        <Readout
          rows={[
            ["receptive field", st.rf + " × " + st.rf, THEME.highlight],
            ["weights", st.params + " C²"],
          ]}
        />
        <Note>A parcel is about 11 pixels wide in these photos. Layer 1 can see a strip of tape; only a deeper unit can see the whole box.</Note>
      </Panel>
    );
  } else if (screen === 2) {
    question = (
      <>
        Name it. A unit's <b>receptive field</b> is the patch of the input it depends on. Each layer adds (F − 1) times the product of the strides below it. Two 3 × 3 layers see a 5 × 5 block, three see 7 × 7, with 27C² weights instead of 49C² and three nonlinearities instead of one. So the trend is small filters, deep stacks: low layers detect parts (petals, tape), higher layers learn how parts are arranged (a flower, a parcel).
      </>
    );
    stage = <RFStage st={st} pick={pk(st)} onPick={setPick} />;
    side = (
      <Panel>
        <Eq size={13}>
          r<sub>L</sub> = 1 + Σ<sub>l=1..L</sub> (F<sub>l</sub> − 1) · Π<sub>k&lt;l</sub> S<sub>k</sub>
          <br />
          weights = Σ<sub>l</sub> F<sub>l</sub>² · C<sub>in</sub> · C<sub>out</sub>
        </Eq>
        <Slider id="c5-l2" label="layers" value={L} min={1} max={6} stepSize={1} onChange={setL} display={String(L)} />
        <Slider id="c5-f2" label="filter size F" value={F} min={3} max={7} stepSize={2} onChange={setF} display={F + " × " + F} />
        <Toggle label="stride 2 in layer 1" value={s1} onChange={setS1} />
        <Readout rows={[["receptive field", st.rf + " × " + st.rf, THEME.highlight], ["weights", st.params + " C²"]]} />
        <Note>A 1 × 1 filter adds nothing to the receptive field (F − 1 = 0): it only mixes channels at one pixel.</Note>
      </Panel>
    );
  } else {
    question = (
      <>
        The top unit must see a whole parcel: a receptive field of at least {GOALC5.rf} × {GOALC5.rf}. The phone budget allows at most {GOALC5.params} C² weights. Choose depth, filter size and whether layer 1 strides by 2.
      </>
    );
    stage = <RFStage st={stC} pick={pk(stC)} onPick={setPick} />;
    side = (
      <Panel edge={okC ? THEME.ok : THEME.grid}>
        <Slider id="c5-lc" label="layers" value={LC} min={1} max={6} stepSize={1} onChange={setLC} display={String(LC)} />
        <Slider id="c5-fc" label="filter size F" value={FC} min={3} max={11} stepSize={2} onChange={setFC} display={FC + " × " + FC} />
        <Toggle label="stride 2 in layer 1" value={sC} onChange={setSC} />
        <Readout rows={[["receptive field", stC.rf + " × " + stC.rf], ["weights", stC.params + " C²", THEME.highlight]]} />
        <Goal met={stC.rf >= GOALC5.rf}>receptive field ≥ {GOALC5.rf}</Goal>
        <Goal met={stC.params <= GOALC5.params}>weights ≤ {GOALC5.params} C²</Goal>
        <Hint
          hints={[
            "One 11 × 11 layer sees enough but costs 121 C². Every 3 × 3 layer costs only 9 C². How many do you need?",
            "Five 3 × 3 layers reach 11 for 45 C². A stride of 2 at the bottom doubles what every later layer adds: three layers might be enough.",
          ]}
        />
      </Panel>
    );
  }
  return <LessonFrame screen={screen} onScreen={setScreen} title="Small filters, stacked, see far" question={question} stage={stage} side={side} />;
}
