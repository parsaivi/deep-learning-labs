import { useState, useMemo, useCallback, useRef, useEffect } from "react";

// Shared by every lesson (the build prepends this file to each one).
// Theme: Blueprint. Paths always carry a second cue: a label and a marker shape or dash style.

const THEME = {
  bg: "#071528",
  surface: "#0b1f3a",
  text: "#e6f0ff",
  muted: "#9db3d1",
  accent: "#4cc9f0",
  highlight: "#ffc857",
  grid: "#24497a",
  warn: "#ff8fa3",
  ok: "#7ee0a1",
};
const MONO = "'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
const SANS = "'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif";
const DISPLAY = "'IBM Plex Sans Condensed', 'Arial Narrow', system-ui, sans-serif";
const SCREENS = ["Puzzle", "Explore", "Name", "Challenge"];
const REDUCED =
  typeof window !== "undefined" && !!window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// @math-start
function clamp(v, a, b) {
  return v < a ? a : v > b ? b : v;
}
function fmt(v, d) {
  if (d === undefined) d = 2;
  if (!isFinite(v)) return "∞";
  if (Math.abs(v) >= 1e5) return v.toExponential(1);
  const s = v.toFixed(d);
  return /^-0\.?0*$/.test(s) ? s.slice(1) : s;
}
function sgn(v, d) {
  return (v >= 0 ? "+" : "−") + fmt(Math.abs(v), d);
}
// numeric derivatives so any surface only needs f
function grad2(f, w) {
  const h = 1e-5;
  return [(f([w[0] + h, w[1]]) - f([w[0] - h, w[1]])) / (2 * h), (f([w[0], w[1] + h]) - f([w[0], w[1] - h])) / (2 * h)];
}
function hess2(f, w) {
  const h = 1e-4;
  const a = grad2(f, [w[0] + h, w[1]]);
  const b = grad2(f, [w[0] - h, w[1]]);
  const c = grad2(f, [w[0], w[1] + h]);
  const d = grad2(f, [w[0], w[1] - h]);
  const hxy = ((a[1] - b[1]) / (2 * h) + (c[0] - d[0]) / (2 * h)) / 2;
  return [
    [(a[0] - b[0]) / (2 * h), hxy],
    [hxy, (c[1] - d[1]) / (2 * h)],
  ];
}
function eig2(M) {
  // symmetric 2x2: eigenvalues (big first) and the unit eigenvector of the big one
  const a = M[0][0];
  const b = M[0][1];
  const d = M[1][1];
  const m = (a + d) / 2;
  const r = Math.sqrt(((a - d) / 2) * ((a - d) / 2) + b * b);
  const l1 = m + r;
  const l2 = m - r;
  let v = Math.abs(b) > 1e-12 ? [l1 - d, b] : a >= d ? [1, 0] : [0, 1];
  const n = Math.hypot(v[0], v[1]);
  v = [v[0] / n, v[1] / n];
  return { l1, l2, v1: v, v2: [-v[1], v[0]] };
}
// quadratic bowl 1/2 (w - c)^T A (w - c)
function quad(A, c) {
  return function (w) {
    const x = w[0] - c[0];
    const y = w[1] - c[1];
    return 0.5 * (A[0][0] * x * x + 2 * A[0][1] * x * y + A[1][1] * y * y);
  };
}
function rotA(l1, l2, deg) {
  const t = (deg * Math.PI) / 180;
  const c = Math.cos(t);
  const s = Math.sin(t);
  return [
    [l1 * c * c + l2 * s * s, (l1 - l2) * c * s],
    [(l1 - l2) * c * s, l1 * s * s + l2 * c * c],
  ];
}
// o: { kind: gd | momentum | nesterov | adagrad | rmsprop | adam | newton, eta, beta, beta2, eps, bias, mu, sched }
// returns { pts, steps: [[dx, dy]], vel: [[vx, vy]], scale: [[sx, sy]], diverged }
function runOpt(f, start, o, n, box) {
  const lim = box || 1e3;
  let w = [start[0], start[1]];
  const pts = [w.slice()];
  const steps = [];
  const vel = [];
  const scale = [];
  let D = [0, 0];
  let m = [0, 0];
  let v = [0, 0];
  let diverged = false;
  const eps = o.eps === undefined ? 1e-8 : o.eps;
  for (let k = 1; k <= n; k++) {
    const eta = o.sched ? o.sched(k - 1) : o.eta;
    let d;
    let sc = [eta, eta];
    if (o.kind === "momentum" || o.kind === "nesterov") {
      const at = o.kind === "nesterov" ? [w[0] - o.beta * D[0], w[1] - o.beta * D[1]] : w;
      const g = grad2(f, at);
      D = [o.beta * D[0] + eta * g[0], o.beta * D[1] + eta * g[1]];
      d = [-D[0], -D[1]];
    } else if (o.kind === "adagrad" || o.kind === "rmsprop" || o.kind === "adam") {
      const g = grad2(f, w);
      if (o.kind === "adagrad") v = [v[0] + g[0] * g[0], v[1] + g[1] * g[1]];
      else {
        const b2 = o.kind === "adam" ? o.beta2 : o.beta;
        v = [b2 * v[0] + (1 - b2) * g[0] * g[0], b2 * v[1] + (1 - b2) * g[1] * g[1]];
      }
      let num = g;
      let vh = v;
      if (o.kind === "adam") {
        m = [o.beta * m[0] + (1 - o.beta) * g[0], o.beta * m[1] + (1 - o.beta) * g[1]];
        num = m;
        if (o.bias) {
          const c1 = 1 - Math.pow(o.beta, k);
          const c2 = 1 - Math.pow(o.beta2, k);
          num = [m[0] / c1, m[1] / c1];
          vh = [v[0] / c2, v[1] / c2];
        }
      }
      sc = [eta / (Math.sqrt(vh[0]) + eps), eta / (Math.sqrt(vh[1]) + eps)];
      d = [-sc[0] * num[0], -sc[1] * num[1]];
    } else if (o.kind === "newton") {
      const g = grad2(f, w);
      const H = hess2(f, w);
      const mu = o.mu || 0;
      const a = H[0][0] + mu;
      const b = H[0][1];
      const c = H[1][1] + mu;
      const det = a * c - b * b;
      if (Math.abs(det) < 1e-10) d = [0, 0];
      else d = [(-eta * (c * g[0] - b * g[1])) / det, (-eta * (-b * g[0] + a * g[1])) / det];
    } else {
      const g = grad2(f, w);
      d = [-eta * g[0], -eta * g[1]];
    }
    w = [w[0] + d[0], w[1] + d[1]];
    if (!isFinite(w[0]) || !isFinite(w[1]) || Math.abs(w[0]) > lim || Math.abs(w[1]) > lim) {
      diverged = true;
      break;
    }
    pts.push(w.slice());
    steps.push(d);
    vel.push(D.slice());
    scale.push(sc);
  }
  return { pts, steps, vel, scale, diverged };
}
// first step k at which every later point stays within tol of target (distance), or -1
function settleStep(pts, target, tol) {
  let k = -1;
  for (let i = pts.length - 1; i >= 0; i--) {
    if (Math.hypot(pts[i][0] - target[0], pts[i][1] - target[1]) <= tol) k = i;
    else break;
  }
  return k;
}

// ---------- Optimization II: orders, sampling, SGD ----------
function rng(seed) {
  let s = seed >>> 0 || 1;
  return function () {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}
function gauss(r) {
  const u = Math.max(r(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r());
}
// n orders: km in [0.5, 8], fee = base0 + rate0 * km + noise. The model uses x = km - 4, so w = [rate, fee of a 4 km order].
function makeOrders(seed, n, rate0, base0, noise) {
  const r = rng(seed);
  const km = new Float64Array(n);
  const x = new Float64Array(n);
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    km[i] = 0.5 + 7.5 * r();
    x[i] = km[i] - 4;
    y[i] = base0 + rate0 * km[i] + noise * gauss(r);
  }
  let Sx = 0, Sxx = 0, Sy = 0, Sxy = 0, Syy = 0;
  for (let i = 0; i < n; i++) {
    Sx += x[i]; Sxx += x[i] * x[i]; Sy += y[i]; Sxy += x[i] * y[i]; Syy += y[i] * y[i];
  }
  const st = { n, Sx: Sx / n, Sxx: Sxx / n, Sy: Sy / n, Sxy: Sxy / n, Syy: Syy / n };
  const det = st.Sxx - st.Sx * st.Sx;
  const rate = (st.Sxy - st.Sx * st.Sy) / det;
  const best = [rate, st.Sy - rate * st.Sx];
  const o = { n, km, x, y, st, best };
  o.loss = function (w) {
    const a = w[0], b = w[1];
    return 0.5 * (st.Syy + a * a * st.Sxx + b * b - 2 * a * st.Sxy - 2 * b * st.Sy + 2 * a * b * st.Sx);
  };
  o.lossBest = o.loss(best);
  return o;
}
// visiting order for one epoch
function epochOrder(n, mode, r, fixedPerm, sortedIdx) {
  if (mode === "sorted") return sortedIdx;
  if (mode === "fixed") return fixedPerm;
  const p = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(r() * (i + 1));
    const t = p[i]; p[i] = p[j]; p[j] = t;
  }
  return p;
}
// mini-batch SGD on the orders. opt: { b, eta | sched(k), order, steps, seed, every }
// returns pts (every `every` updates, plus epoch ends), epochEnd point indices, diverged
function runSGD(o, start, opt) {
  const n = o.n;
  const b = Math.min(opt.b || 1, n);
  const r = rng(opt.seed || 7);
  const sortedIdx = Array.from({ length: n }, (_, i) => i).sort((i, j) => o.km[i] - o.km[j]);
  const fixedPerm = epochOrder(n, "shuffle", rng(99), null, null);
  const every = opt.every || 1;
  let w = [start[0], start[1]];
  const pts = [w.slice()];
  const ks = [0];
  const idx = [];
  const epochEnd = [];
  let perm = epochOrder(n, opt.order || "shuffle", r, fixedPerm, sortedIdx);
  let pos = 0;
  let diverged = false;
  for (let k = 0; k < opt.steps; k++) {
    let g0 = 0, g1 = 0;
    for (let j = 0; j < b; j++) {
      if (pos >= n) {
        perm = epochOrder(n, opt.order || "shuffle", r, fixedPerm, sortedIdx);
        pos = 0;
      }
      const i = perm[pos++];
      if (b === 1) idx.push(i);
      const res = w[1] + w[0] * o.x[i] - o.y[i];
      g0 += res * o.x[i];
      g1 += res;
    }
    const eta = opt.sched ? opt.sched(k) : opt.eta;
    w = [w[0] - (eta * g0) / b, w[1] - (eta * g1) / b];
    if (!isFinite(w[0]) || Math.abs(w[0]) + Math.abs(w[1]) > 1e4) {
      diverged = true;
      pts.push(w.slice());
      ks.push(k + 1);
      break;
    }
    const endEpoch = pos >= n;
    if ((k + 1) % every === 0 || endEpoch || k === opt.steps - 1) {
      pts.push(w.slice());
      ks.push(k + 1);
      if (endEpoch) epochEnd.push(pts.length - 1);
    }
  }
  return { pts, ks, idx, epochEnd, diverged, w };
}
function dist(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}
// @math-end

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
function Pills({ options, value, onPick }) {
  return (
    <div style={{ display: "flex", gap: 6, flexWrap: "wrap", margin: "2px 0 8px" }}>
      {options.map((o) => (
        <Pill key={o[0]} value={o[0]} active={value === o[0]} onPick={onPick}>
          {o[1]}
        </Pill>
      ))}
    </div>
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
function Toggle({ label, value, onChange }) {
  const flip = useCallback(() => onChange(!value), [onChange, value]);
  return (
    <button
      onClick={flip}
      aria-pressed={value}
      style={{
        fontFamily: MONO,
        fontSize: 12.5,
        padding: "5px 10px",
        borderRadius: 3,
        cursor: "pointer",
        border: "1px solid " + (value ? THEME.highlight : THEME.grid),
        background: value ? rgba(THEME.highlight, 0.16) : THEME.bg,
        color: THEME.text,
        margin: "2px 6px 6px 0",
      }}
    >
      {value ? "■ " : "□ "}
      {label}
    </button>
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
function Eq({ children, size }) {
  return (
    <div
      style={{
        fontFamily: MONO,
        fontSize: size || 14,
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
function Readout({ rows }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "auto auto", justifyContent: "space-between", gap: "2px 12px", fontFamily: MONO, fontSize: 13, lineHeight: 1.6 }}>
      {rows.map((r) => (
        <React.Fragment key={r[0]}>
          <span style={{ color: THEME.muted }}>{r[0]}</span>
          <span style={{ color: r[2] || THEME.text, textAlign: "right", fontVariantNumeric: "tabular-nums" }}>{r[1]}</span>
        </React.Fragment>
      ))}
    </div>
  );
}
// a goal line: ◆ when met, ◇ when not, so the state never depends on colour alone
function Goal({ met, children }) {
  return (
    <div style={{ fontFamily: MONO, fontSize: 13, color: met ? THEME.ok : THEME.muted, margin: "3px 0" }}>
      {met ? "◆ " : "◇ "}
      {children}
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
function LessonFrame({ screen, onScreen, title, question, stage, side, below }) {
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
        <div style={{ flex: "1 1 460px", minWidth: 0, maxWidth: 680 }}>{stage}</div>
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

// ---------- the loss map: contour shading on canvas, paths and handles in SVG ----------
// surf: { f, xr: [x0, x1], yr: [y0, y1], xl, yl }. Lower loss = brighter. viewBox 400 x 400.
const VB = 400;
function Marker({ x, y, shape, color, r }) {
  const s = r || 5;
  if (shape === "square") return <rect x={x - s} y={y - s} width={2 * s} height={2 * s} fill={color} stroke={THEME.bg} strokeWidth={1.2} />;
  if (shape === "diamond") return <polygon points={x + "," + (y - s - 1) + " " + (x + s + 1) + "," + y + " " + x + "," + (y + s + 1) + " " + (x - s - 1) + "," + y} fill={color} stroke={THEME.bg} strokeWidth={1.2} />;
  if (shape === "triangle") return <polygon points={x + "," + (y - s - 1) + " " + (x + s + 1) + "," + (y + s) + " " + (x - s - 1) + "," + (y + s)} fill={color} stroke={THEME.bg} strokeWidth={1.2} />;
  return <circle cx={x} cy={y} r={s} fill={color} stroke={THEME.bg} strokeWidth={1.2} />;
}
function PathMark({ path, P }) {
  const pts = path.pts;
  if (!pts.length) return null;
  const xy = pts.map((q) => P(q[0], q[1]));
  const last = xy[xy.length - 1];
  const dots = xy.length <= 70 ? xy : [];
  return (
    <g>
      <polyline points={xy.map((q) => q[0].toFixed(1) + "," + q[1].toFixed(1)).join(" ")} fill="none" stroke={path.color} strokeWidth={path.width || 1.8} strokeDasharray={path.dash} strokeLinejoin="round" />
      {dots.map((q, i) => (
        <circle key={i} cx={q[0]} cy={q[1]} r={1.9} fill={path.color} />
      ))}
      <Marker x={last[0]} y={last[1]} shape={path.shape} color={path.color} r={5.5} />
      {path.label && (
        <text x={last[0] > VB - 100 ? last[0] - 9 : last[0] + 9} y={last[1] < 30 ? last[1] + 18 : last[1] - 8} textAnchor={last[0] > VB - 100 ? "end" : "start"} fontFamily={MONO} fontSize={11} fill={path.color} stroke={THEME.bg} strokeWidth={3} paintOrder="stroke">
          {path.label}
        </text>
      )}
    </g>
  );
}
function LossMap({ surf, paths, handle, onHandle, overlay, maxWidth, ariaLabel, target }) {
  const canvasRef = useRef(null);
  const svgRef = useRef(null);
  const dragRef = useRef(false);
  const R = 150;
  const field = useMemo(() => {
    const v = new Float32Array(R * R);
    let lo = Infinity;
    for (let j = 0; j < R; j++)
      for (let i = 0; i < R; i++) {
        const x = surf.xr[0] + ((surf.xr[1] - surf.xr[0]) * (i + 0.5)) / R;
        const y = surf.yr[1] - ((surf.yr[1] - surf.yr[0]) * (j + 0.5)) / R;
        const val = surf.f([x, y]);
        v[j * R + i] = val;
        if (val < lo) lo = val;
      }
    const t = new Float32Array(R * R);
    let hi = -Infinity;
    for (let k = 0; k < R * R; k++) {
      t[k] = Math.log(v[k] - lo + (surf.floor || 0.02));
      if (t[k] > hi) hi = t[k];
    }
    const t0 = Math.log(surf.floor || 0.02);
    for (let k = 0; k < R * R; k++) t[k] = (t[k] - t0) / (hi - t0 || 1);
    return t;
  }, [surf]);
  useEffect(() => {
    const ctx = canvasRef.current.getContext("2d");
    const img = ctx.createImageData(R, R);
    const [ar, ag, ab] = hexRGB(THEME.accent);
    const [br, bg, bb] = hexRGB(THEME.bg);
    const B = 11;
    for (let j = 0; j < R; j++)
      for (let i = 0; i < R; i++) {
        const k = j * R + i;
        const u = field[k];
        const band = Math.floor(u * B);
        const edge = (i + 1 < R && Math.floor(field[k + 1] * B) !== band) || (j + 1 < R && Math.floor(field[k + R] * B) !== band);
        const lum = (1 - clamp(u, 0, 1)) * 0.62 + (edge ? 0.22 : 0);
        img.data[4 * k] = Math.round(br + (ar - br) * lum);
        img.data[4 * k + 1] = Math.round(bg + (ag - bg) * lum);
        img.data[4 * k + 2] = Math.round(bb + (ab - bb) * lum);
        img.data[4 * k + 3] = 255;
      }
    ctx.putImageData(img, 0, 0);
  }, [field]);
  const P = useCallback(
    (x, y) => [((x - surf.xr[0]) / (surf.xr[1] - surf.xr[0])) * VB, ((surf.yr[1] - y) / (surf.yr[1] - surf.yr[0])) * VB],
    [surf]
  );
  const down = useCallback((e) => {
    if (!e.target.getAttribute || e.target.getAttribute("data-hid") !== "start") return;
    dragRef.current = true;
    svgRef.current.setPointerCapture(e.pointerId);
    e.preventDefault();
  }, []);
  const move = useCallback(
    (e) => {
      if (!dragRef.current || !onHandle) return;
      const p = svgPoint(svgRef.current, e);
      const x = surf.xr[0] + (clamp(p.x, 4, VB - 4) / VB) * (surf.xr[1] - surf.xr[0]);
      const y = surf.yr[1] - (clamp(p.y, 4, VB - 4) / VB) * (surf.yr[1] - surf.yr[0]);
      onHandle(x, y);
    },
    [onHandle, surf]
  );
  const up = useCallback(() => {
    dragRef.current = false;
  }, []);
  let hx = 0;
  let hy = 0;
  if (handle) [hx, hy] = P(handle[0], handle[1]);
  let tx = 0;
  let ty = 0;
  if (target) [tx, ty] = P(target[0], target[1]);
  return (
    <div style={{ position: "relative", width: "100%", maxWidth: maxWidth || 520, aspectRatio: "1 / 1", margin: "0 auto" }}>
      <canvas ref={canvasRef} width={R} height={R} style={{ position: "absolute", inset: 0, width: "100%", height: "100%", background: THEME.bg }} />
      <svg
        ref={svgRef}
        viewBox={"0 0 " + VB + " " + VB}
        role="img"
        aria-label={ariaLabel || "Loss map"}
        onPointerDown={down}
        onPointerMove={move}
        onPointerUp={up}
        onPointerCancel={up}
        style={{ position: "absolute", inset: 0, width: "100%", height: "100%", touchAction: "none", border: "1px solid " + THEME.grid }}
      >
        <g fontFamily={MONO} fontSize={11} fill={THEME.text} stroke={THEME.bg} strokeWidth={3} paintOrder="stroke">
          <text x={VB - 6} y={VB - 8} textAnchor="end">
            {surf.xl} →
          </text>
          <text x={8} y={16}>
            ↑ {surf.yl}
          </text>
          <text x={8} y={VB - 8}>
            brighter = lower loss
          </text>
        </g>
        {target && (
          <g>
            <line x1={tx - 9} x2={tx + 9} y1={ty} y2={ty} stroke={THEME.text} strokeWidth={1.5} />
            <line x1={tx} x2={tx} y1={ty - 9} y2={ty + 9} stroke={THEME.text} strokeWidth={1.5} />
          </g>
        )}
        {overlay && overlay(P)}
        {(paths || []).map((p, i) => (
          <PathMark key={p.key || i} path={p} P={P} />
        ))}
        {handle && (
          <g style={{ cursor: "grab" }}>
            <circle data-hid="start" cx={hx} cy={hy} r={15} fill="transparent" />
            <circle data-hid="start" cx={hx} cy={hy} r={7} fill={THEME.bg} stroke={THEME.highlight} strokeWidth={2.2} />
            <text x={hx + 10} y={hy + 16} fontFamily={MONO} fontSize={10.5} fill={THEME.highlight} stroke={THEME.bg} strokeWidth={3} paintOrder="stroke" pointerEvents="none">
              start (drag)
            </text>
          </g>
        )}
      </svg>
    </div>
  );
}
function Arrow({ x1, y1, x2, y2, color, width, dash }) {
  const a = Math.atan2(y2 - y1, x2 - x1);
  const L = Math.hypot(x2 - x1, y2 - y1);
  const h = Math.min(8, L * 0.5);
  return (
    <g>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth={width || 2} strokeDasharray={dash} />
      {L > 2 && (
        <polygon
          points={x2 + "," + y2 + " " + (x2 - h * Math.cos(a - 0.4)) + "," + (y2 - h * Math.sin(a - 0.4)) + " " + (x2 - h * Math.cos(a + 0.4)) + "," + (y2 - h * Math.sin(a + 0.4))}
          fill={color}
        />
      )}
    </g>
  );
}

// ---------- a 1-D loss curve with the walker's steps ----------
// f: x -> loss; xr, yr ranges; pts: visited x values; curves: [{fn, color, dash}]
const PX0 = 48;
const PX1 = 600;
function Curve1D({ f, xr, yr, pts, curves, xl, label, height, marks, vlines }) {
  const H = height || 230;
  const top = 22;
  const bot = H - 30;
  const X = (x) => PX0 + ((x - xr[0]) / (xr[1] - xr[0])) * (PX1 - PX0);
  const Y = (y) => bot - ((clamp(y, yr[0] - (yr[1] - yr[0]) * 0.05, yr[1] + (yr[1] - yr[0]) * 0.08) - yr[0]) / (yr[1] - yr[0])) * (bot - top);
  const line = (fn) => {
    const a = [];
    for (let i = 0; i <= 320; i++) {
      const x = xr[0] + ((xr[1] - xr[0]) * i) / 320;
      a.push(X(x).toFixed(1) + "," + Y(fn(x)).toFixed(1));
    }
    return a.join(" ");
  };
  const ticks = [];
  for (let i = 0; i <= 5; i++) ticks.push(xr[0] + ((xr[1] - xr[0]) * i) / 5);
  const shown = (pts || []).slice(0, 80);
  return (
    <svg viewBox={"0 0 620 " + H} style={{ width: "100%", display: "block" }} role="img" aria-label={label || "Loss curve"}>
      <defs>
        <clipPath id={"clip-" + H}>
          <rect x={PX0} y={top - 8} width={PX1 - PX0} height={bot - top + 10} />
        </clipPath>
      </defs>
      <rect x={PX0} y={top - 8} width={PX1 - PX0} height={bot - top + 10} fill={THEME.bg} stroke={THEME.grid} />
      {ticks.map((t) => (
        <text key={t} x={X(t)} y={H - 12} textAnchor="middle" fontFamily={MONO} fontSize={10.5} fill={THEME.muted}>
          {fmt(t, Math.abs(xr[1] - xr[0]) < 3 ? 2 : 1)}
        </text>
      ))}
      <text x={PX1} y={H - 1} textAnchor="end" fontFamily={MONO} fontSize={10.5} fill={THEME.muted}>
        {xl}
      </text>
      {label && (
        <text x={PX0 + 6} y={top + 6} fontFamily={MONO} fontSize={11} fill={THEME.muted}>
          {label}
        </text>
      )}
      <g clipPath={"url(#clip-" + H + ")"}>
        {(vlines || []).map((v, i) => (
          <g key={i}>
            <line x1={X(v.x)} x2={X(v.x)} y1={top - 8} y2={bot + 2} stroke={v.color} strokeDasharray="4 3" strokeWidth={1.3} />
            <text x={X(v.x) + 4} y={bot - 4} fontFamily={MONO} fontSize={10.5} fill={v.color}>
              {v.label}
            </text>
          </g>
        ))}
        {(curves || []).map((c, i) => (
          <polyline key={i} points={line(c.fn)} fill="none" stroke={c.color} strokeWidth={c.width || 1.5} strokeDasharray={c.dash} />
        ))}
        <polyline points={line(f)} fill="none" stroke={THEME.accent} strokeWidth={2.4} />
        {shown.length > 1 &&
          shown.slice(1).map((x, i) => {
            const x0 = shown[i];
            return (
              <path
                key={i}
                d={"M" + X(x0) + "," + Y(f(x0)) + " Q" + (X(x0) + X(x)) / 2 + "," + (Math.min(Y(f(x0)), Y(f(x))) - 18) + " " + X(x) + "," + Y(f(x))}
                fill="none"
                stroke={THEME.highlight}
                strokeWidth={1.3}
                opacity={0.4 + (0.6 * (i + 1)) / shown.length}
              />
            );
          })}
        {shown.map((x, i) => (
          <circle key={"p" + i} cx={X(x)} cy={Y(f(x))} r={i === shown.length - 1 ? 6.5 : 3} fill={THEME.highlight} stroke={THEME.bg} strokeWidth={1.2} />
        ))}
        {(marks || []).map((m, i) => (
          <g key={"m" + i}>
            <Marker x={X(m.x)} y={Y(m.y)} shape={m.shape} color={m.color} r={5} />
            {m.label && (
              <text x={X(m.x) + 8} y={Y(m.y) - 8} fontFamily={MONO} fontSize={10.5} fill={m.color} stroke={THEME.bg} strokeWidth={3} paintOrder="stroke">
                {m.label}
              </text>
            )}
          </g>
        ))}
      </g>
    </svg>
  );
}

// ---------- values over iterations ----------
// series: [{ vals, color, dash, label }]; log: plot log10 |v|
function Trace({ series, n, yr, label, height, log, zero }) {
  const H = height || 150;
  const top = 18;
  const bot = H - 22;
  const X = (k) => PX0 + (k / Math.max(1, n)) * (PX1 - PX0);
  const tr = (v) => (log ? Math.log10(Math.max(Math.abs(v), 1e-6)) : v);
  const Y = (v) => bot - ((clamp(tr(v), yr[0], yr[1]) - yr[0]) / (yr[1] - yr[0])) * (bot - top);
  return (
    <svg viewBox={"0 0 620 " + H} style={{ width: "100%", display: "block" }} role="img" aria-label={label || "Values over iterations"}>
      <rect x={PX0} y={top - 4} width={PX1 - PX0} height={bot - top + 8} fill={THEME.bg} stroke={THEME.grid} />
      {zero && <line x1={PX0} x2={PX1} y1={Y(0)} y2={Y(0)} stroke={THEME.grid} strokeWidth={1.2} />}
      <text x={PX0 - 6} y={top + 4} textAnchor="end" fontFamily={MONO} fontSize={10} fill={THEME.muted}>
        {log ? "1e" + yr[1] : fmt(yr[1], 1)}
      </text>
      <text x={PX0 - 6} y={bot + 4} textAnchor="end" fontFamily={MONO} fontSize={10} fill={THEME.muted}>
        {log ? "1e" + yr[0] : fmt(yr[0], 1)}
      </text>
      <text x={PX1} y={H - 6} textAnchor="end" fontFamily={MONO} fontSize={10} fill={THEME.muted}>
        step {n}
      </text>
      <text x={PX0} y={H - 6} fontFamily={MONO} fontSize={10} fill={THEME.muted}>
        0
      </text>
      {label && (
        <text x={PX0 + 6} y={top + 10} fontFamily={MONO} fontSize={10.5} fill={THEME.muted}>
          {label}
        </text>
      )}
      {series.map((s, i) => (
        <g key={i}>
          <polyline
            points={s.vals
              .slice(0, n + 1)
              .map((v, k) => X(k).toFixed(1) + "," + Y(v).toFixed(1))
              .join(" ")}
            fill="none"
            stroke={s.color}
            strokeWidth={1.8}
            strokeDasharray={s.dash}
          />
          {s.label && s.vals.length > 0 && (
            <text
              x={Math.min(X(Math.min(n, s.vals.length - 1)) + 4, PX1 - 4)}
              y={Y(s.vals[Math.min(n, s.vals.length - 1)]) - 5}
              textAnchor={X(Math.min(n, s.vals.length - 1)) > PX1 - 80 ? "end" : "start"}
              fontFamily={MONO}
              fontSize={10.5}
              fill={s.color}
              stroke={THEME.bg}
              strokeWidth={3}
              paintOrder="stroke"
            >
              {s.label}
            </text>
          )}
        </g>
      ))}
    </svg>
  );
}

// ---------- a general x/y chart: series [{ xs, ys, color, dash, label, width }] ----------
function niceLog(v) {
  const e = Math.round(v);
  return Math.abs(e) <= 3 ? fmt(Math.pow(10, e), e < 0 ? -e : 0) : "1e" + e;
}
function Plot({ series, xr, yr, logX, logY, xl, yl, label, height, hlines, vlines, marks, legend }) {
  const H = height || 190;
  const top = 18;
  const bot = H - 30;
  const tx = (v) => (logX ? Math.log10(Math.max(v, 1e-12)) : v);
  const ty = (v) => (logY ? Math.log10(Math.max(v, 1e-12)) : v);
  const X = (v) => PX0 + ((clamp(tx(v), xr[0], xr[1]) - xr[0]) / (xr[1] - xr[0])) * (PX1 - PX0);
  const Y = (v) => bot - ((clamp(ty(v), yr[0] - (yr[1] - yr[0]) * 0.04, yr[1] + (yr[1] - yr[0]) * 0.04) - yr[0]) / (yr[1] - yr[0])) * (bot - top);
  const xt = [];
  const yt = [];
  if (logX) for (let e = Math.ceil(xr[0]); e <= xr[1]; e++) xt.push([X(Math.pow(10, e)), niceLog(e)]);
  const dig = (r) => (r < 0.1 ? 3 : r < 1 ? 2 : r < 5 ? 1 : 0);
  if (!logX) for (let i = 0; i <= 5; i++) { const v = xr[0] + ((xr[1] - xr[0]) * i) / 5; xt.push([X(v), fmt(v, dig(Math.abs(xr[1] - xr[0])))]); }
  if (logY) for (let e = Math.ceil(yr[0]); e <= yr[1]; e++) yt.push([Y(Math.pow(10, e)), niceLog(e)]);
  else for (let i = 0; i <= 4; i++) { const v = yr[0] + ((yr[1] - yr[0]) * i) / 4; yt.push([Y(v), fmt(v, dig(Math.abs(yr[1] - yr[0])))]); }
  const cid = "pc-" + H + "-" + (label || "").length;
  return (
    <svg viewBox={"0 0 620 " + H} style={{ width: "100%", display: "block" }} role="img" aria-label={label || yl || "chart"}>
      <defs>
        <clipPath id={cid}>
          <rect x={PX0} y={top - 6} width={PX1 - PX0} height={bot - top + 12} />
        </clipPath>
      </defs>
      <rect x={PX0} y={top - 6} width={PX1 - PX0} height={bot - top + 12} fill={THEME.bg} stroke={THEME.grid} />
      {xt.map((t, i) => (
        <g key={"x" + i}>
          <line x1={t[0]} x2={t[0]} y1={bot + 6} y2={bot + 2} stroke={THEME.grid} />
          <text x={t[0]} y={bot + 17} textAnchor="middle" fontFamily={MONO} fontSize={10} fill={THEME.muted}>
            {t[1]}
          </text>
        </g>
      ))}
      {yt.map((t, i) => (
        <text key={"y" + i} x={PX0 - 5} y={t[0] + 3.5} textAnchor="end" fontFamily={MONO} fontSize={10} fill={THEME.muted}>
          {t[1]}
        </text>
      ))}
      <text x={PX1} y={H - 2} textAnchor="end" fontFamily={MONO} fontSize={10.5} fill={THEME.muted}>
        {xl}
      </text>
      {label && (
        <text x={PX0 + 6} y={top + 8} fontFamily={MONO} fontSize={10.5} fill={THEME.muted}>
          {label}
        </text>
      )}
      <g clipPath={"url(#" + cid + ")"}>
        {(hlines || []).map((h, i) => (
          <g key={"h" + i}>
            <line x1={PX0} x2={PX1} y1={Y(h.y)} y2={Y(h.y)} stroke={h.color} strokeDasharray={h.dash || "4 3"} strokeWidth={1.3} />
            {h.label && (
              <text x={PX1 - 4} y={Y(h.y) - 4} textAnchor="end" fontFamily={MONO} fontSize={10} fill={h.color} stroke={THEME.bg} strokeWidth={3} paintOrder="stroke">
                {h.label}
              </text>
            )}
          </g>
        ))}
        {(vlines || []).map((v, i) => (
          <g key={"v" + i}>
            <line x1={X(v.x)} x2={X(v.x)} y1={top - 6} y2={bot + 6} stroke={v.color} strokeDasharray={v.dash || "4 3"} strokeWidth={1.3} />
            {v.label && (
              <text x={X(v.x) + 4} y={top + 22 + (v.dy || 0)} fontFamily={MONO} fontSize={10} fill={v.color} stroke={THEME.bg} strokeWidth={3} paintOrder="stroke">
                {v.label}
              </text>
            )}
          </g>
        ))}
        {series.map((s, i) => {
          const n = Math.min(s.xs.length, s.ys.length);
          if (n < 1) return null;
          const step = Math.max(1, Math.floor(n / 700));
          const a = [];
          for (let k = 0; k < n; k += step) if (isFinite(s.ys[k])) a.push(X(s.xs[k]).toFixed(1) + "," + Y(s.ys[k]).toFixed(1));
          if (step > 1 && isFinite(s.ys[n - 1])) a.push(X(s.xs[n - 1]).toFixed(1) + "," + Y(s.ys[n - 1]).toFixed(1));
          return <polyline key={i} points={a.join(" ")} fill="none" stroke={s.color} strokeWidth={s.width || 1.8} strokeDasharray={s.dash} strokeLinejoin="round" />;
        })}
        {(marks || []).map((m, i) => (
          <Marker key={"m" + i} x={X(m.x)} y={Y(m.y)} shape={m.shape} color={m.color} r={4.5} />
        ))}
      </g>
      {series.map((s, i) => {
        const n = Math.min(s.xs.length, s.ys.length);
        if (!s.label || n < 1) return null;
        let k = n - 1;
        while (k > 0 && !isFinite(s.ys[k])) k--;
        const x = X(s.xs[k]);
        const right = x > PX1 - 110;
        return (
          <text key={"l" + i} x={right ? x - 5 : x + 5} y={Y(s.ys[k]) - 5 + (s.ldy || 0)} textAnchor={right ? "end" : "start"} fontFamily={MONO} fontSize={10.5} fill={s.color} stroke={THEME.bg} strokeWidth={3} paintOrder="stroke">
            {s.label}
          </text>
        );
      })}
      {legend && legend.length > 0 && (
        <g>
          <rect x={PX1 - 178} y={top - 3} width={175} height={legend.length * 14 + 6} fill={THEME.bg} opacity={0.88} stroke={THEME.grid} />
          {legend.map((l, i) => (
            <g key={"lg" + i}>
              <line x1={PX1 - 170} x2={PX1 - 146} y1={top + 7 + i * 14} y2={top + 7 + i * 14} stroke={l.color} strokeWidth={l.width || 2} strokeDasharray={l.dash} />
              <text x={PX1 - 140} y={top + 11 + i * 14} fontFamily={MONO} fontSize={10.5} fill={l.color}>
                {l.text}
              </text>
            </g>
          ))}
        </g>
      )}
      {(marks || []).map((m, i) =>
        m.label ? (
          <text key={"ml" + i} x={X(m.x) + 7} y={Y(m.y) - 7} fontFamily={MONO} fontSize={10.5} fill={m.color} stroke={THEME.bg} strokeWidth={3} paintOrder="stroke">
            {m.label}
          </text>
        ) : null
      )}
    </svg>
  );
}

// ---------- the orders: km on x, fee on y, fitted lines on top ----------
// lines: [{ w: [rate, base], color, dash, label }]; hi: index of an order to ring; ticks: [{ km, lo, hi, ok }] tolerance whiskers
function FeePlot({ orders, lines, hi, ticks, height, label }) {
  const H = height || 210;
  const top = 14;
  const bot = H - 28;
  let ylo = Infinity;
  let yhi = -Infinity;
  for (let i = 0; i < orders.n; i++) {
    ylo = Math.min(ylo, orders.y[i]);
    yhi = Math.max(yhi, orders.y[i]);
  }
  ylo = Math.floor(ylo - 0.5);
  yhi = Math.ceil(yhi + 0.5);
  const X = (km) => PX0 + ((km - 0) / 8.5) * (PX1 - PX0);
  const Y = (v) => bot - ((clamp(v, ylo - 3, yhi + 3) - ylo) / (yhi - ylo)) * (bot - top);
  const fee = (w, km) => w[1] + w[0] * (km - 4);
  const kms = [0, 2, 4, 6, 8];
  return (
    <svg viewBox={"0 0 620 " + H} style={{ width: "100%", display: "block" }} role="img" aria-label={label || "Orders: distance against fee, with fitted fee lines"}>
      <defs>
        <clipPath id={"fee-" + H}>
          <rect x={PX0} y={top - 4} width={PX1 - PX0} height={bot - top + 8} />
        </clipPath>
      </defs>
      <rect x={PX0} y={top - 4} width={PX1 - PX0} height={bot - top + 8} fill={THEME.bg} stroke={THEME.grid} />
      {kms.map((k) => (
        <text key={k} x={X(k)} y={bot + 16} textAnchor="middle" fontFamily={MONO} fontSize={10} fill={THEME.muted}>
          {k}
        </text>
      ))}
      {[ylo, (ylo + yhi) / 2, yhi].map((v) => (
        <text key={v} x={PX0 - 5} y={Y(v) + 3.5} textAnchor="end" fontFamily={MONO} fontSize={10} fill={THEME.muted}>
          €{fmt(v, 0)}
        </text>
      ))}
      <text x={PX1} y={H - 2} textAnchor="end" fontFamily={MONO} fontSize={10.5} fill={THEME.muted}>
        distance (km) →
      </text>
      <text x={PX0 + 6} y={top + 8} fontFamily={MONO} fontSize={10.5} fill={THEME.muted}>
        {label || "each dot is one order · lines are fee rules"}
      </text>
      <g clipPath={"url(#fee-" + H + ")"}>
        {Array.from(orders.km).map((km, i) => (
          <circle key={i} cx={X(km)} cy={Y(orders.y[i])} r={2.6} fill={THEME.muted} opacity={0.75} />
        ))}
        {(lines || []).map((l, i) => (
          <line key={i} x1={X(0)} x2={X(8.5)} y1={Y(fee(l.w, 0))} y2={Y(fee(l.w, 8.5))} stroke={l.color} strokeWidth={l.width || 2} strokeDasharray={l.dash} />
        ))}
        {(ticks || []).map((t, i) => (
          <g key={"t" + i}>
            <line x1={X(t.km)} x2={X(t.km)} y1={Y(t.lo)} y2={Y(t.hi)} stroke={t.ok ? THEME.ok : THEME.warn} strokeWidth={2} />
            <line x1={X(t.km) - 6} x2={X(t.km) + 6} y1={Y(t.lo)} y2={Y(t.lo)} stroke={t.ok ? THEME.ok : THEME.warn} strokeWidth={2} />
            <line x1={X(t.km) - 6} x2={X(t.km) + 6} y1={Y(t.hi)} y2={Y(t.hi)} stroke={t.ok ? THEME.ok : THEME.warn} strokeWidth={2} />
          </g>
        ))}
        {hi !== undefined && hi !== null && hi >= 0 && <circle cx={X(orders.km[hi])} cy={Y(orders.y[hi])} r={7} fill="none" stroke={THEME.highlight} strokeWidth={2} />}
      </g>
      {(lines || []).map((l, i) =>
        l.label ? (
          <text key={"l" + i} x={PX1 - 6} y={Y(fee(l.w, 8.2)) + (l.ldy || -6)} textAnchor="end" fontFamily={MONO} fontSize={10.5} fill={l.color} stroke={THEME.bg} strokeWidth={3} paintOrder="stroke">
            {l.label}
          </text>
        ) : null
      )}
    </svg>
  );
}
// a slider over log10 of a value
function LogSlider({ id, label, value, lo, hi, onChange, digits, stepSize }) {
  const change = useCallback((v) => onChange(Math.pow(10, v)), [onChange]);
  const d = digits === undefined ? Math.max(0, -Math.floor(Math.log10(value)) + 1) : digits;
  return <Slider id={id} label={label} value={Math.log10(value)} min={lo} max={hi} stepSize={stepSize || 0.005} onChange={change} display={value < 1e-3 ? value.toExponential(1) : fmt(value, d)} />;
}
// @common-end
