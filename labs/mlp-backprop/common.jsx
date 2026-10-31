import { useState, useMemo, useCallback, useRef, useEffect } from "react";

// Shared by every lesson (the build prepends this file to each one).
// Theme: Blueprint. Two channels everywhere: served addresses are filled amber dots,
// unserved addresses are hollow cyan squares, wrong predictions get a pink ring and a cross.

const THEME = {
  bg: "#071528",
  surface: "#0b1f3a",
  text: "#e6f0ff",
  muted: "#9db3d1",
  accent: "#4cc9f0",
  highlight: "#ffc857",
  grid: "#24497a",
  warn: "#ff8fa3",
};
const MONO = "'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
const SANS = "'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif";
const DISPLAY = "'IBM Plex Sans Condensed', 'Arial Narrow', system-ui, sans-serif";
const SCREENS = ["Puzzle", "Explore", "Name", "Challenge"];
const REDUCED =
  typeof window !== "undefined" && !!window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function clamp(v, a, b) {
  return v < a ? a : v > b ? b : v;
}
function sig(z) {
  return 1 / (1 + Math.exp(-z));
}
function step(z) {
  return z >= 0 ? 1 : 0;
}
function fmt(v, d) {
  if (d === undefined) d = 2;
  if (!isFinite(v)) return "∞";
  const s = v.toFixed(d);
  return s === "-0.00" || s === "-0.0" || s === "-0" ? s.slice(1) : s;
}
function sgn(v, d) {
  return (v >= 0 ? "+" : "−") + fmt(Math.abs(v), d);
}
function rgba(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return "rgba(" + ((n >> 16) & 255) + "," + ((n >> 8) & 255) + "," + (n & 255) + "," + a + ")";
}
function hexRGB(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
function svgPoint(svg, e) {
  const pt = svg.createSVGPoint();
  pt.x = e.clientX;
  pt.y = e.clientY;
  return pt.matrixTransform(svg.getScreenCTM().inverse());
}
// small deterministic RNG so the "city" is the same for every learner
function rng(seed) {
  let s = seed >>> 0;
  return function () {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

// ---------- a neuron as a street: z = w·x + b, fires on the side the arrow points to ----------
function lineNeuron(a, b, flip) {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const len = Math.hypot(dx, dy) || 1e-6;
  let w1 = -dy / len;
  let w2 = dx / len;
  if (flip) {
    w1 = -w1;
    w2 = -w2;
  }
  return { w1, w2, b: -(w1 * a[0] + w2 * a[1]) };
}
function zOf(n, x, y) {
  return n.w1 * x + n.w2 * y + n.b;
}
// the part of the infinite line through a,b that lies in the unit square
function clipLine(a, b) {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const ts = [];
  if (Math.abs(dx) > 1e-9) {
    ts.push((0 - a[0]) / dx, (1 - a[0]) / dx);
  }
  if (Math.abs(dy) > 1e-9) {
    ts.push((0 - a[1]) / dy, (1 - a[1]) / dy);
  }
  const pts = ts
    .map((t) => [a[0] + t * dx, a[1] + t * dy])
    .filter((p) => p[0] >= -1e-6 && p[0] <= 1 + 1e-6 && p[1] >= -1e-6 && p[1] <= 1 + 1e-6);
  if (pts.length < 2) return null;
  pts.sort((p, q) => p[0] - q[0] || p[1] - q[1]);
  return [pts[0], pts[pts.length - 1]];
}

// ---------- small UI pieces ----------
function Panel({ children, edge, style }) {
  return (
    <div style={Object.assign({ background: THEME.surface, border: "1px solid " + (edge || THEME.grid), borderRadius: 4, padding: "10px 12px" }, style || {})}>
      {children}
    </div>
  );
}
function Note({ children }) {
  return <div style={{ color: THEME.muted, fontSize: 14, lineHeight: 1.5, margin: "0 0 8px" }}>{children}</div>;
}
function Label({ children }) {
  return (
    <div style={{ fontFamily: MONO, fontSize: 11, letterSpacing: 1, textTransform: "uppercase", color: THEME.muted, margin: "2px 0 6px" }}>{children}</div>
  );
}
function Pill({ value, active, onPick, children, title }) {
  const click = useCallback(() => onPick(value), [onPick, value]);
  return (
    <button
      onClick={click}
      title={title}
      aria-pressed={!!active}
      style={{
        fontFamily: MONO,
        fontSize: 12.5,
        padding: "5px 10px",
        borderRadius: 3,
        cursor: "pointer",
        border: "1px solid " + (active ? THEME.highlight : THEME.grid),
        background: active ? rgba(THEME.highlight, 0.16) : THEME.bg,
        color: THEME.text,
      }}
    >
      {children}
    </button>
  );
}
function Btn({ onClick, children, disabled, strong }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        fontFamily: MONO,
        fontSize: 12.5,
        padding: "6px 11px",
        borderRadius: 3,
        cursor: disabled ? "default" : "pointer",
        border: "1px solid " + (disabled ? THEME.grid : THEME.accent),
        background: strong && !disabled ? rgba(THEME.accent, 0.18) : "transparent",
        color: disabled ? THEME.muted : THEME.accent,
      }}
    >
      {children}
    </button>
  );
}
function Slider({ id, label, value, min, max, stepSize, onChange, display, width }) {
  const change = useCallback((e) => onChange(parseFloat(e.target.value)), [onChange]);
  return (
    <label htmlFor={id} style={{ display: "block", margin: "4px 0 8px", minWidth: 0, width: width || "auto" }}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 8, fontSize: 13, color: THEME.muted }}>
        <span>{label}</span>
        <span style={{ fontFamily: MONO, color: THEME.text, fontVariantNumeric: "tabular-nums" }}>{display !== undefined ? display : fmt(value)}</span>
      </div>
      <input id={id} type="range" min={min} max={max} step={stepSize} value={value} onChange={change} style={{ width: "100%" }} />
    </label>
  );
}
function Hint({ hints }) {
  const [n, setN] = useState(0);
  const more = useCallback(() => setN((v) => Math.min(v + 1, hints.length)), [hints.length]);
  return (
    <div style={{ marginTop: 8 }}>
      {hints.slice(0, n).map((h, i) => (
        <div key={i} style={{ color: THEME.text, fontSize: 13.5, margin: "4px 0 6px", paddingLeft: 10, borderLeft: "2px dashed " + THEME.accent, lineHeight: 1.45 }}>
          <span style={{ fontFamily: MONO, color: THEME.accent }}>hint {i + 1} </span>
          {h}
        </div>
      ))}
      {n < hints.length && (
        <button
          onClick={more}
          style={{ fontFamily: MONO, fontSize: 12.5, background: "transparent", color: THEME.accent, border: "1px dashed " + THEME.accent, borderRadius: 3, padding: "4px 10px", cursor: "pointer" }}
        >
          {n === 0 ? "Stuck?" : "Another hint"}
        </button>
      )}
    </div>
  );
}
function Meter({ label, value, max, good, unit, invert }) {
  // horizontal bar with a target tick; "good" is the target value
  const frac = clamp(value / max, 0, 1);
  const ok = invert ? value >= good : value <= good;
  const col = ok ? THEME.accent : THEME.highlight;
  return (
    <div style={{ margin: "6px 0" }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, color: THEME.muted }}>
        <span>{label}</span>
        <span style={{ fontFamily: MONO, color: col }}>
          {ok ? "◆ " : "◇ "}
          {fmt(value, unit === "" ? 0 : 2)}
          {unit || ""}
        </span>
      </div>
      <svg viewBox="0 0 200 10" style={{ width: "100%", height: 10, display: "block" }} aria-hidden="true">
        <rect x="0" y="2" width="200" height="6" fill={THEME.bg} stroke={THEME.grid} />
        <rect x="0" y="2" width={200 * frac} height="6" fill={col} opacity="0.8" />
        <line x1={200 * clamp(good / max, 0, 1)} x2={200 * clamp(good / max, 0, 1)} y1="0" y2="10" stroke={THEME.text} strokeWidth="1.5" />
      </svg>
    </div>
  );
}
function Eq({ children, size }) {
  return (
    <div
      style={{
        fontFamily: MONO,
        fontSize: size || 14.5,
        color: THEME.text,
        background: THEME.bg,
        border: "1px solid " + THEME.grid,
        borderRadius: 3,
        padding: "8px 10px",
        margin: "6px 0",
        lineHeight: 1.7,
        overflowX: "auto",
        whiteSpace: "nowrap",
      }}
    >
      {children}
    </div>
  );
}
function Legend() {
  return (
    <div style={{ display: "flex", gap: 14, flexWrap: "wrap", fontSize: 12.5, color: THEME.muted, margin: "6px 0 0" }}>
      <span>
        <svg width="12" height="12" aria-hidden="true">
          <circle cx="6" cy="6" r="4.5" fill={THEME.highlight} />
        </svg>{" "}
        served address
      </span>
      <span>
        <svg width="12" height="12" aria-hidden="true">
          <rect x="1.5" y="1.5" width="9" height="9" fill="none" stroke={THEME.accent} strokeWidth="1.6" />
        </svg>{" "}
        not served
      </span>
      <span>
        <svg width="14" height="14" aria-hidden="true">
          <circle cx="7" cy="7" r="5.5" fill="none" stroke={THEME.warn} strokeWidth="1.6" />
        </svg>{" "}
        network gets it wrong
      </span>
      <span>
        <svg width="14" height="12" aria-hidden="true">
          <rect x="1" y="1" width="12" height="10" fill={rgba(THEME.highlight, 0.3)} />
        </svg>{" "}
        network says "deliver"
      </span>
    </div>
  );
}

// ---------- lesson frame: the four screens ----------
function ScreenTab({ i, active, done, onPick }) {
  const click = useCallback(() => onPick(i), [onPick, i]);
  return (
    <button
      onClick={click}
      aria-current={active ? "step" : undefined}
      style={{
        fontFamily: MONO,
        fontSize: 12.5,
        padding: "5px 10px",
        cursor: "pointer",
        borderRadius: 3,
        border: "1px solid " + (active ? THEME.accent : THEME.grid),
        background: active ? rgba(THEME.accent, 0.15) : "transparent",
        color: active ? THEME.text : done ? THEME.text : THEME.muted,
      }}
    >
      {i + 1} {SCREENS[i]}
    </button>
  );
}
function LessonFrame({ screen, onScreen, title, question, stage, side, below, wide }) {
  const back = useCallback(() => onScreen(Math.max(0, screen - 1)), [onScreen, screen]);
  const next = useCallback(() => onScreen(Math.min(3, screen + 1)), [onScreen, screen]);
  return (
    <div style={{ fontFamily: SANS, color: THEME.text, background: rgba(THEME.surface, 0.55), border: "1px solid " + THEME.grid, borderRadius: 4, padding: "12px 14px 14px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 10, flexWrap: "wrap", marginBottom: 8 }}>
        <div style={{ fontFamily: DISPLAY, fontSize: 22, fontWeight: 700, letterSpacing: 0.2 }}>{title}</div>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {SCREENS.map((s, i) => (
            <ScreenTab key={s} i={i} active={i === screen} done={i < screen} onPick={onScreen} />
          ))}
        </div>
      </div>
      <div style={{ fontSize: 16, lineHeight: 1.45, margin: "4px 0 12px", paddingLeft: 10, borderLeft: "3px solid " + THEME.highlight, maxWidth: 900 }}>
        <span style={{ fontFamily: MONO, fontSize: 12, color: THEME.highlight, marginRight: 8 }}>{SCREENS[screen].toUpperCase()}</span>
        {question}
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 16, alignItems: "flex-start" }}>
        <div style={{ flex: wide ? "1 1 100%" : "1 1 460px", minWidth: 0, maxWidth: wide ? "none" : 680 }}>{stage}</div>
        <div style={{ flex: "1 1 300px", minWidth: 0, display: "flex", flexDirection: "column", gap: 10 }}>{side}</div>
      </div>
      {below}
      <div style={{ display: "flex", justifyContent: "space-between", marginTop: 14 }}>
        <Btn onClick={back} disabled={screen === 0}>
          ← Back
        </Btn>
        <Btn onClick={next} disabled={screen === 3} strong>
          {screen === 3 ? "Last screen" : "Next: " + SCREENS[screen + 1] + " →"}
        </Btn>
      </div>
    </div>
  );
}

// ---------- the city map ----------
// world coordinates: x east 0..1, y north 0..1 (1 unit = 10 km). SVG viewBox is 400 x 400.
const MAP = 400;
function toSvg(x, y) {
  return [x * MAP, (1 - y) * MAP];
}
function MapPoint({ p, wrong, dim }) {
  const [sx, sy] = toSvg(p.x, p.y);
  return (
    <g opacity={dim ? 0.35 : 1}>
      {p.s ? (
        <circle cx={sx} cy={sy} r={5.2} fill={THEME.highlight} stroke={THEME.bg} strokeWidth={1.2} />
      ) : (
        <rect x={sx - 4.6} y={sy - 4.6} width={9.2} height={9.2} fill={THEME.bg} stroke={THEME.accent} strokeWidth={1.7} />
      )}
      {wrong && (
        <g>
          <circle cx={sx} cy={sy} r={10} fill="none" stroke={THEME.warn} strokeWidth={1.8} />
          <line x1={sx + 7} y1={sy - 13} x2={sx + 13} y2={sy - 7} stroke={THEME.warn} strokeWidth={1.8} />
          <line x1={sx + 13} y1={sy - 13} x2={sx + 7} y2={sy - 7} stroke={THEME.warn} strokeWidth={1.8} />
        </g>
      )}
    </g>
  );
}
function MapLine({ L }) {
  const seg = clipLine(L.a, L.b);
  if (!seg) return null;
  const [x1, y1] = toSvg(seg[0][0], seg[0][1]);
  const [x2, y2] = toSvg(seg[1][0], seg[1][1]);
  const n = lineNeuron(L.a, L.b, L.flip);
  const mx = (L.a[0] + L.b[0]) / 2;
  const my = (L.a[1] + L.b[1]) / 2;
  const [ax, ay] = toSvg(mx, my);
  const [tx, ty] = toSvg(mx + n.w1 * 0.06, my + n.w2 * 0.06);
  const col = L.color || THEME.accent;
  return (
    <g opacity={L.dim ? 0.35 : 1}>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={col} strokeWidth={L.thin ? 1.2 : 2.2} strokeDasharray={L.dash ? "6 5" : undefined} />
      {!L.noArrow && (
        <g>
          <line x1={ax} y1={ay} x2={tx} y2={ty} stroke={col} strokeWidth={1.6} />
          <circle cx={tx} cy={ty} r={2.6} fill={col} />
        </g>
      )}
      {L.label && (
        <text x={tx + (tx - ax) * 0.5} y={ty + (ty - ay) * 0.5 + 4} textAnchor="middle" fontFamily={MONO} fontSize={12} fill={col} stroke={THEME.bg} strokeWidth={3} paintOrder="stroke">
          {L.label}
        </text>
      )}
    </g>
  );
}
function MapHandle({ h }) {
  const [sx, sy] = toSvg(h.x, h.y);
  const col = h.color || THEME.accent;
  return (
    <g data-hid={h.id} style={{ cursor: "grab" }}>
      <circle data-hid={h.id} cx={sx} cy={sy} r={14} fill="transparent" />
      {h.shape === "square" ? (
        <rect data-hid={h.id} x={sx - 6} y={sy - 6} width={12} height={12} fill={THEME.bg} stroke={col} strokeWidth={2} />
      ) : (
        <circle data-hid={h.id} cx={sx} cy={sy} r={6.5} fill={THEME.bg} stroke={col} strokeWidth={2} />
      )}
    </g>
  );
}
// field(x, y) -> value in [0, 1]; values >= 0.5 mean "deliver". redrawn every render (res x res cells).
function ZoneMap({ points, field, pred, lines, handles, onMove, onHover, overlay, res, maxWidth, ariaLabel, plain }) {
  const canvasRef = useRef(null);
  const svgRef = useRef(null);
  const dragRef = useRef(null);
  const R = res || 100;
  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv || !field) return;
    const ctx = cv.getContext("2d");
    const img = ctx.createImageData(R, R);
    const [hr, hg, hb] = hexRGB(THEME.highlight);
    const vals = new Float32Array(R * R);
    for (let j = 0; j < R; j++) {
      const y = 1 - (j + 0.5) / R;
      for (let i = 0; i < R; i++) vals[j * R + i] = field((i + 0.5) / R, y);
    }
    for (let j = 0; j < R; j++) {
      for (let i = 0; i < R; i++) {
        const v = vals[j * R + i];
        const on = v >= 0.5;
        const edge = (i + 1 < R && on !== vals[j * R + i + 1] >= 0.5) || (j + 1 < R && on !== vals[(j + 1) * R + i] >= 0.5);
        const k = 4 * (j * R + i);
        img.data[k] = hr;
        img.data[k + 1] = hg;
        img.data[k + 2] = hb;
        img.data[k + 3] = edge ? 200 : Math.round(255 * (0.04 + 0.3 * clamp(v, 0, 1)));
      }
    }
    ctx.putImageData(img, 0, 0);
  });
  const down = useCallback((e) => {
    const id = e.target.getAttribute && e.target.getAttribute("data-hid");
    if (!id) return;
    dragRef.current = id;
    svgRef.current.setPointerCapture(e.pointerId);
    e.preventDefault();
  }, []);
  const move = useCallback(
    (e) => {
      const p = svgPoint(svgRef.current, e);
      const x = clamp(p.x / MAP, 0, 1);
      const y = clamp(1 - p.y / MAP, 0, 1);
      if (dragRef.current && onMove) onMove(dragRef.current, clamp(x, 0.02, 0.98), clamp(y, 0.02, 0.98));
      else if (onHover) onHover(x, y);
    },
    [onMove, onHover]
  );
  const up = useCallback(() => {
    dragRef.current = null;
  }, []);
  const leave = useCallback(() => {
    if (!dragRef.current && onHover) onHover(null, null);
  }, [onHover]);
  const gridLines = [];
  for (let k = 1; k < 10; k++) {
    gridLines.push(<line key={"v" + k} x1={k * 40} y1={0} x2={k * 40} y2={MAP} stroke={THEME.grid} strokeWidth={k === 5 ? 1 : 0.5} opacity={0.6} />);
    gridLines.push(<line key={"h" + k} x1={0} y1={k * 40} x2={MAP} y2={k * 40} stroke={THEME.grid} strokeWidth={k === 5 ? 1 : 0.5} opacity={0.6} />);
  }
  return (
    <div style={{ position: "relative", width: "100%", maxWidth: maxWidth || 520, aspectRatio: "1 / 1", margin: "0 auto" }}>
      <canvas ref={canvasRef} width={R} height={R} style={{ position: "absolute", inset: 0, width: "100%", height: "100%", imageRendering: "pixelated", background: THEME.bg }} />
      <svg
        ref={svgRef}
        viewBox={"0 0 " + MAP + " " + MAP}
        role="img"
        aria-label={ariaLabel || "City map"}
        onPointerDown={down}
        onPointerMove={move}
        onPointerUp={up}
        onPointerCancel={up}
        onPointerLeave={leave}
        style={{ position: "absolute", inset: 0, width: "100%", height: "100%", touchAction: "none", border: "1px solid " + THEME.grid }}
      >
        {!plain && gridLines}
        {!plain && (
          <g fontFamily={MONO} fontSize={10} fill={THEME.muted}>
            <text x={6} y={MAP - 6}>0 km</text>
            <text x={MAP - 6} y={MAP - 6} textAnchor="end">
              10 km east
            </text>
            <text x={6} y={14}>10 km north</text>
          </g>
        )}
        {(lines || []).map((L, i) => (
          <MapLine key={L.id || i} L={L} />
        ))}
        {(points || []).map((p, i) => (
          <MapPoint key={i} p={p} wrong={pred ? (pred(p.x, p.y) >= 0.5 ? 1 : 0) !== p.s : false} dim={p.dim} />
        ))}
        {overlay}
        {(handles || []).map((h) => (
          <MapHandle key={h.id} h={h} />
        ))}
      </svg>
    </div>
  );
}
function countWrong(points, pred) {
  let w = 0;
  for (const p of points) if ((pred(p.x, p.y) >= 0.5 ? 1 : 0) !== p.s) w++;
  return w;
}
// @common-end
