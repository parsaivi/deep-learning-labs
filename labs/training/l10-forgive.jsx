import { useState, useMemo, useCallback } from "react";
import { THEME, MONO } from "./common";

// Lesson 10 - What batch norm buys: independence from the weight scale (Batch Normalization, slides 14, 40-42).
// Aha: BN(a * W x) = BN(W x) for any a > 0, so a normalized layer's output does not care how big
// its weights are. A badly scaled initialization can no longer blow up or kill the signal, which is
// why BN makes deep nets far less sensitive to init and lets them use larger learning rates.
// BN can go on only some layers; then the error compounds only between two BN layers.

// @sim-start
const CW10 = [64, 240, 240, 30, 30, 240, 30, 240, 240, 40, 40, 160, 160];
const BASE10 = makeDeep(CW10, 48, 5);
const L10 = CW10.length - 1;
const INTERN10 = 2.2; // sigma = 2.2 / sqrt(fan_in): 1.56x Kaiming
const MAXBN10 = 6;
function healthy10(o) {
  return o.every((q) => q.zrms >= 0.5 && q.zrms <= 2);
}
// @sim-end

function Stats10({ out, title, bn }) {
  const layers = out.map((q, i) => ({ h: histOf(q.a, 0, 3, 24), label: (bn && bn[i] ? "■" : "") + "L" + (i + 1), value: fmt(q.zrms, 2), bad: q.zrms < 0.5 || q.zrms > 2 }));
  const ks = out.map((q, i) => i + 1);
  return (
    <>
      <LayerHists layers={layers} lo={0} hi={3} cap={0.6} height={150} label={title + " · ReLU outputs per layer, size of z below · ■ = batch norm"} />
      <Plot
        series={[{ xs: ks, ys: out.map((q) => q.zrms), color: THEME.highlight, width: 2.2 }]}
        marks={out.map((q, i) => ({ x: i + 1, y: q.zrms, shape: bn && bn[i] ? "square" : q.zrms < 0.5 || q.zrms > 2 ? "triangle" : "circle", color: q.zrms < 0.5 || q.zrms > 2 ? THEME.warn : THEME.highlight }))}
        hlines={[
          { y: 0.5, color: THEME.ok, label: "0.5" },
          { y: 2, color: THEME.ok, label: "2" },
        ]}
        xr={[0, out.length]}
        yr={[-3, 3]}
        logY
        xl="layer →"
        label="size of each layer's z (rms, log) · ■ BN layer · ▲ outside 0.5–2"
        height={160}
      />
    </>
  );
}
function BnSwitch({ i, on, onFlip }) {
  const flip = useCallback(() => onFlip(i), [onFlip, i]);
  return (
    <button
      onClick={flip}
      aria-pressed={on}
      style={{ fontFamily: MONO, fontSize: 12, padding: "4px 0", width: 44, borderRadius: 3, cursor: "pointer", border: "1px solid " + (on ? THEME.highlight : THEME.grid), background: on ? rgba(THEME.highlight, 0.16) : THEME.bg, color: THEME.text }}
    >
      {on ? "■" : "□"}
      {i + 1}
    </button>
  );
}

const NO_BN = Array(L10).fill(false);
const ALL_BN = Array(L10).fill(true);

export default function ForgiveLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [bnAll, setBnAll] = useState(false);
  const [sigma, setSigma] = useState(0.2);
  const [mask, setMask] = useState(NO_BN);
  const sigNow = screen === 0 ? null : sigma;
  const out = useMemo(() => deepForward(BASE10, "relu", sigNow === null ? (l, n) => INTERN10 / Math.sqrt(n) : () => sigNow, bnAll ? ALL_BN : NO_BN), [sigNow, bnAll]);
  const outC = useMemo(() => deepForward(BASE10, "relu", (l, n) => INTERN10 / Math.sqrt(n), mask), [mask]);
  const flip = useCallback((i) => setMask((m) => m.map((v, k) => (k === i ? !v : v))), []);
  const nBn = mask.filter(Boolean).length;
  const okC = healthy10(outC) && nBn <= MAXBN10;
  const last = out[out.length - 1];
  const rows = [
    ["layer 12 z size", fmt(last.zrms, 3), last.zrms < 0.5 || last.zrms > 2 ? THEME.warn : undefined],
    ["layers outside 0.5–2", String(out.filter((q) => q.zrms < 0.5 || q.zrms > 2).length)],
  ];

  let question, stage, side;
  if (screen === 0) {
    question = (
      <>
        Back to the 12-layer ReLU ETA network of lesson 4. An intern initialized it with σ = 2.2/√fan_in, about 1.6 × the Kaiming value. Look at layer 12. Now put batch norm after the weighted sum of every layer (before the ReLU). What happens, and why does the intern's mistake no longer matter?
      </>
    );
    stage = <Stats10 out={out} title="intern's init" bn={bnAll ? ALL_BN : NO_BN} />;
    side = (
      <>
        <Panel>
          <Toggle label="batch norm on every layer" value={bnAll} onChange={setBnAll} />
          <Readout rows={rows} />
        </Panel>
      </>
    );
  } else if (screen === 1) {
    question = (
      <>
        Use one σ for every layer, which lesson 4 showed can never work for these mixed widths. Sweep it over three powers of ten, with and without batch norm. With batch norm on, does σ change anything at all? What would happen to a weight's gradient if you doubled every weight of a BN layer?
      </>
    );
    stage = <Stats10 out={out} title={"same σ = " + fmt(sigma, 3) + " for every layer"} bn={bnAll ? ALL_BN : NO_BN} />;
    side = (
      <>
        <Panel>
          <LogSlider id="l10-s" label="σ for every layer" value={sigma} lo={-2.5} hi={0.5} onChange={setSigma} digits={3} />
          <Toggle label="batch norm on every layer" value={bnAll} onChange={setBnAll} />
          <Readout rows={rows} />
        </Panel>
        <Note>Multiplying W by a multiplies z, μ_B and σ_B all by a; the normalized u does not change.</Note>
      </>
    );
  } else if (screen === 2) {
    question = (
      <>
        Name it. BN(aWx) = BN(Wx): a normalized layer ignores the scale of its weights. So the signal can neither explode nor vanish through it, whatever the init, gradients flow better, and much higher learning rates are safe. If the weights grow, their gradients shrink as 1/a, which steadies training on its own.
      </>
    );
    stage = <Stats10 out={out} title={"same σ = " + fmt(sigma, 3) + " for every layer"} bn={bnAll ? ALL_BN : NO_BN} />;
    side = (
      <>
        <Panel>
          <Eq>
            BN(a·Wx) = BN(Wx), a &gt; 0
            <br />
            ∂L/∂(aW) = (1/a) · ∂L/∂W
          </Eq>
          <LogSlider id="l10-s2" label="σ for every layer" value={sigma} lo={-2.5} hi={0.5} onChange={setSigma} digits={3} />
          <Toggle label="batch norm on every layer" value={bnAll} onChange={setBnAll} />
        </Panel>
        <Note>From the slides' summary: BN improves gradient flow, allows higher learning rates and faster decay, reduces the dependence on initialization, and acts as a form of regularization (lesson 6's jitter); some reports find it removes the need for dropout. It needs well-shuffled data, and it may be applied to only some layers, or even some units.</Note>
      </>
    );
  } else {
    question = (
      <>
        Batch norm costs compute, so the team allows at most {MAXBN10} BN layers in the intern's network. Choose which layers get one (click to toggle) so that every layer's z stays between 0.5 and 2.
      </>
    );
    stage = <Stats10 out={outC} title={"intern's init, " + nBn + " BN layers"} bn={mask} />;
    side = (
      <>
        <Panel edge={okC ? THEME.ok : THEME.grid}>
          <Label>batch norm after layer…</Label>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 5, marginBottom: 8 }}>
            {mask.map((on, i) => (
              <BnSwitch key={i} i={i} on={on} onFlip={flip} />
            ))}
          </div>
          <Readout
            rows={[
              ["BN layers used", nBn + " of " + MAXBN10, nBn > MAXBN10 ? THEME.warn : undefined],
              ["layers outside 0.5–2", String(outC.filter((q) => q.zrms < 0.5 || q.zrms > 2).length)],
            ]}
          />
          <Goal met={healthy10(outC)}>every layer between 0.5 and 2</Goal>
          <Goal met={nBn <= MAXBN10}>at most {MAXBN10} BN layers</Goal>
          <Hint
            hints={[
              "Turn on one BN layer and watch the layers after it: by how much does z grow per layer once it has been reset to 1?",
              "The intern's σ multiplies z by about 1.56 per layer, so one layer after a reset is still fine (1.56) but two are not (2.4). Layer 1 already starts at 2.2.",
            ]}
          />
        </Panel>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="BN forgives a bad init" question={question} stage={stage} side={side} />;
}
