"""Interactive 3D preview of the meshed design, for the Streamlit GUI.

Why hand-rolled WebGL rather than a viewer package: the studio is meant to run on a
lab machine with `pip install streamlit` and nothing else (see docs/gui_guide.md), so
this adds no dependency and fetches nothing from a CDN. Everything below is inlined
into one `srcdoc` iframe.

Two things worth knowing before changing this file:

1. The preview mesh is NOT the exported mesh. Marching cubes gives 45k-113k triangles
   per typology, which is a few megabytes over the websocket on every slider move. We
   vertex-cluster down to a face budget for display only; `Download STL` still writes
   `r["_mesh"]` at full resolution. `mesh_payload` reports both counts so the UI can
   say so.
2. Positions are quantised to 16 bits and normals to 8, which is what keeps a 50k-face
   body around half a megabyte. Both are far finer than the voxel pitch the geometry
   was marched at, so nothing visible is lost. The viewer also offers a *Facets* mode
   that ignores the normals and derives a flat one per fragment — that is the honest
   view of the marching-cubes discretisation the scores are computed on, and it is a
   toggle rather than the default because sub-pixel triangles make it speckle.
"""
from __future__ import annotations

import base64
import json

import numpy as np
import streamlit as st

#: Display face budget. Above this the preview is vertex-clustered. Chosen so the
#: payload stays under ~1 MB base64, which is re-sent on every slider move.
MAX_PREVIEW_FACES = 50_000


# ------------------------------------------------------------------ decimation
def _cluster(verts: np.ndarray, faces: np.ndarray, cell: float):
    """Vertex clustering: weld every vertex in a `cell`-sized grid box to their mean.

    Suits marching-cubes output, where large coplanar patches collapse cleanly. It can
    fold thin features, which is acceptable for a preview and never touches the export.
    """
    ijk = np.floor((verts - verts.min(axis=0)) / cell).astype(np.int64)
    # flatten to one key so np.unique runs on a 1-D array: `unique(axis=0)` on 56k rows
    # costs ~0.1 s, which is felt on every slider move, and this is ~50x faster
    nx, ny = int(ijk[:, 0].max()) + 1, int(ijk[:, 1].max()) + 1
    _, inv = np.unique(ijk[:, 0] + nx * (ijk[:, 1] + ny * ijk[:, 2]),
                       return_inverse=True)
    inv = inv.ravel()
    n = int(inv.max()) + 1
    rep = np.zeros((n, 3), dtype=np.float64)
    for ax in range(3):
        rep[:, ax] = np.bincount(inv, weights=verts[:, ax], minlength=n)
    rep /= np.bincount(inv, minlength=n)[:, None]

    f = inv[faces]
    keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    f = f[keep]
    if len(f):                       # drop duplicate triangles (same 3 clusters)
        _, uniq = np.unique(np.sort(f, axis=1), axis=0, return_index=True)
        f = f[np.sort(uniq)]
    return rep, f


def _surface_area(verts: np.ndarray, faces: np.ndarray) -> float:
    t = verts[faces]
    return float(0.5 * np.linalg.norm(
        np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum())


def decimate(verts: np.ndarray, faces: np.ndarray, max_faces: int):
    """Reduce to roughly `max_faces` triangles. Returns (verts, faces) unchanged if
    already under budget."""
    if len(faces) <= max_faces or len(faces) == 0:
        return verts, faces
    # a cell-sized patch of clustered surface carries ~2 triangles, so aim at
    # cell = sqrt(2 * area / budget) and grow it until the count actually lands
    span = float(np.ptp(verts, axis=0).max())
    cell = max(np.sqrt(2.0 * _surface_area(verts, faces) / max_faces), span / 2000.0)
    best = (verts, faces)
    for _ in range(12):
        v, f = _cluster(verts, faces, cell)
        if len(f) == 0:
            break
        best = (v, f)
        if len(f) <= max_faces:
            break
        cell *= max(1.25, np.sqrt(len(f) / max_faces))
    return best


# ------------------------------------------------------------------ payload
def vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals. The unnormalised face cross product is already
    proportional to area, so summing it per vertex weights by area for free."""
    t = verts[faces]
    fn = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    n = np.zeros_like(verts)
    for ax in range(3):
        n[:, ax] = np.bincount(faces.ravel(),
                               weights=np.repeat(fn[:, ax], 3),
                               minlength=len(verts))
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.where(ln > 0, ln, 1.0)


def mesh_payload(mesh, max_faces: int = MAX_PREVIEW_FACES) -> dict:
    """Quantised, indexed geometry ready for the viewer.

    Positions go over as unsigned shorts normalised across the bounding box (16 bits
    over a 400 mm block is 6 um, far below the voxel pitch) and normals as signed
    bytes (~1 deg). Both are padded to 4 components so the vertex strides stay
    4-byte aligned.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    n_full = int(len(faces))
    verts, faces = decimate(verts, faces, max_faces)

    lo = verts.min(axis=0) if len(verts) else np.zeros(3)
    hi = verts.max(axis=0) if len(verts) else np.ones(3)
    span = np.maximum(hi - lo, 1e-9)

    q = np.rint((verts - lo) / span * 65535.0).astype(np.uint16)
    padded = np.zeros((len(q), 4), dtype=np.uint16)
    padded[:, :3] = q

    nrm = np.zeros((len(verts), 4), dtype=np.int8)
    nrm[:, :3] = np.clip(np.rint(vertex_normals(verts, faces) * 127.0), -127, 127)

    idx_bits = 16 if len(verts) <= 65535 else 32
    idx = faces.astype(np.uint16 if idx_bits == 16 else np.uint32).ravel()

    return {
        "pos": base64.b64encode(padded.tobytes()).decode(),
        "nrm": base64.b64encode(nrm.tobytes()).decode(),
        "idx": base64.b64encode(idx.tobytes()).decode(),
        "idxBits": idx_bits,
        "nIdx": int(idx.size),
        "origin": lo.tolist(),
        "span": span.tolist(),
        "center": ((lo + hi) / 2).tolist(),
        "radius": float(np.linalg.norm(span) / 2),
        "dims": span.tolist(),
        "nFaces": int(len(faces)),
        "nFacesFull": n_full,
    }


# ------------------------------------------------------------------ component
_HTML = r"""
<style>
  :root { color-scheme: light dark; }
  html,body { margin:0; padding:0; overflow:hidden;
              font-family:"Source Sans Pro","Segoe UI",system-ui,sans-serif; }
  #wrap { position:relative; width:100%; height:__H__px; border-radius:8px;
          overflow:hidden; background:radial-gradient(120% 120% at 30% 15%,
          rgba(140,150,165,.20), rgba(90,95,105,.10) 60%, rgba(60,60,66,.14)); }
  canvas { display:block; width:100%; height:100%; cursor:grab; }
  canvas.drag { cursor:grabbing; }
  #bar { position:absolute; left:8px; bottom:8px; right:8px; display:flex;
         gap:6px; align-items:center; flex-wrap:wrap; font-size:12px; }
  button, select { font:inherit; font-size:12px; padding:3px 8px; border-radius:6px;
          border:1px solid rgba(128,128,128,.45); background:rgba(255,255,255,.72);
          color:#111; cursor:pointer; }
  button.on { background:#2e7d32; border-color:#2e7d32; color:#fff; }
  input[type=range] { width:110px; accent-color:#b8860b; }
  #hud { position:absolute; left:10px; top:8px; font-size:11px; line-height:1.45;
         opacity:.72; pointer-events:none; font-variant-numeric:tabular-nums; }
  #err { position:absolute; inset:0; display:none; place-content:center; padding:18px;
         font-size:13px; text-align:center; }
  .pill { padding:2px 6px; border-radius:5px; background:rgba(128,128,128,.18); }
</style>
<div id="wrap">
  <canvas id="cv"></canvas>
  <div id="hud"></div>
  <div id="bar">
    <button id="spin" title="Toggle turntable">Spin</button>
    <button id="facets" title="Shade each triangle flat, showing the voxel steps the
geometry was marched at">Facets</button>
    <button id="reset" title="Back to the isometric view">Reset view</button>
    <span class="pill">Section</span>
    <select id="axis">
      <option value="-1">off</option>
      <option value="0">X</option>
      <option value="1">Y</option>
      <option value="2">Z</option>
    </select>
    <input id="cut" type="range" min="0" max="1" step="0.005" value="0.5" disabled>
    <button id="flip" title="Cut from the other side" disabled>Flip</button>
  </div>
  <div id="err"></div>
</div>
<script>
const M = __PAYLOAD__;
const wrap = document.getElementById("wrap"), cv = document.getElementById("cv");
const hud = document.getElementById("hud"), errBox = document.getElementById("err");

function fail(msg) {
  cv.style.display = "none";
  document.getElementById("bar").style.display = "none";
  errBox.style.display = "grid";
  errBox.textContent = msg;
}

const gl = cv.getContext("webgl2", {antialias: true, alpha: true, stencil: true});
if (!gl) fail("This browser has no WebGL2, so the 3D preview cannot draw. " +
              "The STL download and every number on this page are unaffected.");

function b64(s) {
  const bin = atob(s), out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

const VS = `#version 300 es
in vec4 aQ;
in vec4 aN;
uniform mat4 uProj, uView;
uniform vec3 uOrigin, uSpan;
out vec3 vView, vWorld, vN;
void main() {
  vWorld = uOrigin + aQ.xyz * uSpan;
  vec4 p = uView * vec4(vWorld, 1.0);
  vView = p.xyz;
  vN = mat3(uView) * aN.xyz;                   // uView is rigid, so no inverse needed
  gl_Position = uProj * p;
}`;

const FS = `#version 300 es
precision highp float;
in vec3 vView, vWorld, vN;
uniform int uAxis, uFacets;
uniform float uCut, uSign;
uniform vec3 uSurface, uCutColor;
out vec4 frag;
void main() {
  if (uAxis >= 0) {
    float w = uAxis == 0 ? vWorld.x : (uAxis == 1 ? vWorld.y : vWorld.z);
    if (w * uSign > uCut * uSign) discard;
  }
  vec3 n = uFacets == 1 ? normalize(cross(dFdx(vView), dFdy(vView))) : normalize(vN);
  if (!gl_FrontFacing) n = -n;                 // cut faces light like real surfaces
  vec3 base = gl_FrontFacing ? uSurface : uCutColor;
  vec3 L1 = normalize(vec3(-0.35, 0.55, 0.80));
  vec3 L2 = normalize(vec3(0.70, -0.40, 0.35));
  float d = 0.78 * max(dot(n, L1), 0.0) + 0.30 * max(dot(n, L2), 0.0);
  vec3 V = normalize(-vView);
  float spec = pow(max(dot(reflect(-L1, n), V), 0.0), 26.0) * 0.22;
  float rim = pow(1.0 - max(dot(n, V), 0.0), 3.0) * 0.16;
  frag = vec4(base * (0.30 + d) + spec + rim, 1.0);
}`;

// The cut face. Without it the clip plane shows straight through to whatever inner
// surfaces lie behind, and a 22 mm wall reads as a hole. Standard parity cap: count
// front minus back faces of the clipped body along each view ray in the stencil, then
// fill the plane wherever that count is non-zero, i.e. wherever solid was severed.
const CAP_VS = `#version 300 es
in vec3 aP;
uniform mat4 uProj, uView;
void main() { gl_Position = uProj * uView * vec4(aP, 1.0); }`;

const CAP_FS = `#version 300 es
precision highp float;
uniform vec3 uColor;
uniform vec3 uN;                                 // cut normal, already in view space
out vec4 frag;
void main() {
  vec3 n = uN.z < 0.0 ? -uN : uN;
  vec3 L1 = normalize(vec3(-0.35, 0.55, 0.80));
  vec3 L2 = normalize(vec3(0.70, -0.40, 0.35));
  float d = 0.78 * max(dot(n, L1), 0.0) + 0.30 * max(dot(n, L2), 0.0);
  frag = vec4(uColor * (0.34 + d), 1.0);
}`;

function shader(type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS))
    throw new Error(gl.getShaderInfoLog(s));
  return s;
}

function link(vs, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, shader(gl.VERTEX_SHADER, vs));
  gl.attachShader(p, shader(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS))
    throw new Error(gl.getProgramInfoLog(p));
  return p;
}

let prog, capProg, capVao, capBuf, U = {}, C = {}, vao;
let nIdx = M.nIdx, idxType;
try {
  prog = link(VS, FS);
  gl.useProgram(prog);

  vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const vbo = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, b64(M.pos), gl.STATIC_DRAW);
  const locQ = gl.getAttribLocation(prog, "aQ");
  gl.enableVertexAttribArray(locQ);
  gl.vertexAttribPointer(locQ, 4, gl.UNSIGNED_SHORT, true, 8, 0);

  const nbo = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, nbo);
  gl.bufferData(gl.ARRAY_BUFFER, b64(M.nrm), gl.STATIC_DRAW);
  const locN = gl.getAttribLocation(prog, "aN");
  gl.enableVertexAttribArray(locN);
  gl.vertexAttribPointer(locN, 4, gl.BYTE, true, 4, 0);

  const ibo = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, b64(M.idx), gl.STATIC_DRAW);
  idxType = M.idxBits === 16 ? gl.UNSIGNED_SHORT : gl.UNSIGNED_INT;

  for (const k of ["uProj", "uView", "uOrigin", "uSpan", "uAxis", "uCut", "uSign",
                   "uSurface", "uCutColor", "uFacets"])
    U[k] = gl.getUniformLocation(prog, k);
  gl.uniform3fv(U.uOrigin, M.origin);
  gl.uniform3fv(U.uSpan, M.span);
  gl.uniform3fv(U.uSurface, [0.78, 0.72, 0.63]);   // dry stone
  gl.uniform3fv(U.uCutColor, [0.62, 0.27, 0.19]);  // seen-from-inside, the section red

  capProg = link(CAP_VS, CAP_FS);
  capVao = gl.createVertexArray();
  gl.bindVertexArray(capVao);
  capBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, capBuf);
  gl.bufferData(gl.ARRAY_BUFFER, 18 * 4, gl.DYNAMIC_DRAW);
  const locP = gl.getAttribLocation(capProg, "aP");
  gl.enableVertexAttribArray(locP);
  gl.vertexAttribPointer(locP, 3, gl.FLOAT, false, 0, 0);
  for (const k of ["uProj", "uView", "uColor", "uN"])
    C[k] = gl.getUniformLocation(capProg, k);

  gl.enable(gl.DEPTH_TEST);
} catch (e) { if (gl) fail("3D preview failed to initialise: " + e.message); }

// ---------------------------------------------------------------- camera
const HOME = {yaw: -0.62, pitch: 0.48, dist: M.radius / Math.sin(0.42) * 1.12,
              px: 0, py: 0};
// Streamlit rebuilds this iframe on every rerun, i.e. on every slider move, so the
// camera has to live outside the document or the view you just set up is thrown away
// each time the score updates. localStorage outlives the reload; window.name is the
// fallback for embeds where storage is blocked.
const KEY = "biocast.viewer.camera";
const store = {
  get() {
    try { return localStorage.getItem(KEY) || window.name; }
    catch (e) { return window.name; }
  },
  set(s) {
    try { localStorage.setItem(KEY, s); } catch (e) {}
    try { window.name = s; } catch (e) {}
  },
};

let cam = Object.assign({}, HOME), saved = {};
try {
  const raw = store.get();
  if (raw) {
    const s = saved = JSON.parse(raw);
    // rescale the distance when the body has changed size, or a 200 mm tile inherits
    // the framing of a 390 mm block and lands off screen
    if (s.r > 0 && Math.abs(s.r - M.radius) / M.radius > 0.01) {
      s.dist *= M.radius / s.r;
      s.px = s.py = 0;
    }
    for (const k of ["yaw", "pitch", "dist", "px", "py"])
      if (typeof s[k] === "number" && isFinite(s[k])) cam[k] = s[k];
  }
} catch (e) {}
const save = () => store.set(JSON.stringify(Object.assign(
    {r: M.radius, axis, cutFrac, sign, facets, spinning}, cam)));

const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                         a[0] * b[1] - a[1] * b[0]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1;
                      return [a[0] / l, a[1] / l, a[2] / l]; };

// Z-up world. `eyeDir` points from the target back to the eye; pitch is clamped short
// of the poles so cross(worldUp, eyeDir) never degenerates.
function eyeDir() {
  const cp = Math.cos(cam.pitch);
  return norm([cp * Math.sin(cam.yaw), -cp * Math.cos(cam.yaw), Math.sin(cam.pitch)]);
}

function view() {
  const z = eyeDir();
  const x = norm(cross([0, 0, 1], z));
  const y = cross(z, x);
  const t = [0, 1, 2].map((i) => M.center[i] + x[i] * cam.px + y[i] * cam.py);
  const eye = [0, 1, 2].map((i) => t[i] + z[i] * cam.dist);
  return [x[0], y[0], z[0], 0,
          x[1], y[1], z[1], 0,
          x[2], y[2], z[2], 0,
          -dot(x, eye), -dot(y, eye), -dot(z, eye), 1];
}

function proj(aspect) {
  const fov = 0.84, near = Math.max(cam.dist - M.radius * 3, M.radius * 0.02),
        far = cam.dist + M.radius * 4;
  const t = 1 / Math.tan(fov / 2);
  return [t / aspect, 0, 0, 0,
          0, t, 0, 0,
          0, 0, (far + near) / (near - far), -1,
          0, 0, 2 * far * near / (near - far), 0];
}

// ---------------------------------------------------------------- controls
// The section survives a rerun like the camera does: the point of a cut is usually to
// watch a wall thickness change while dragging that wall's slider.
const num = (v, d) => (typeof v === "number" && isFinite(v) ? v : d);
let axis = [0, 1, 2].includes(saved.axis) ? saved.axis : -1,
    cutFrac = Math.min(Math.max(num(saved.cutFrac, 0.5), 0), 1),
    sign = saved.sign === -1 ? -1 : 1,
    spinning = saved.spinning === true,
    facets = saved.facets === true,
    dirty = true;

const elAxis = document.getElementById("axis"), elCut = document.getElementById("cut"),
      elFlip = document.getElementById("flip"), elSpin = document.getElementById("spin"),
      elFacets = document.getElementById("facets");
elAxis.value = String(axis);
elCut.value = String(cutFrac);
elCut.disabled = elFlip.disabled = axis < 0;
elSpin.classList.toggle("on", spinning);
elFacets.classList.toggle("on", facets);

elAxis.onchange = () => {
  axis = parseInt(elAxis.value, 10);
  elCut.disabled = elFlip.disabled = axis < 0;
  // keep the half on the far side of the plane, so the cut faces the camera rather
  // than showing an apparently untouched body from outside. Flip overrides.
  if (axis >= 0) sign = eyeDir()[axis] > 0 ? 1 : -1;
  save(); dirty = true;
};
elCut.oninput = () => { cutFrac = parseFloat(elCut.value); save(); dirty = true; };
elFlip.onclick = () => { sign = -sign; save(); dirty = true; };
elSpin.onclick = () => {
  spinning = !spinning; elSpin.classList.toggle("on", spinning); save(); dirty = true;
};
elFacets.onclick = () => {
  facets = !facets; elFacets.classList.toggle("on", facets); save(); dirty = true;
};
document.getElementById("reset").onclick = () => {
  cam = Object.assign({}, HOME); save(); dirty = true;
};

let drag = null;
cv.addEventListener("pointerdown", (e) => {
  drag = {x: e.clientX, y: e.clientY, pan: e.shiftKey || e.button === 2 || e.button === 1};
  cv.setPointerCapture(e.pointerId); cv.classList.add("drag");
});
cv.addEventListener("pointermove", (e) => {
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  drag.x = e.clientX; drag.y = e.clientY;
  if (drag.pan) {
    const k = cam.dist * 0.0016;
    cam.px -= dx * k; cam.py += dy * k;
  } else {
    cam.yaw += dx * 0.008;
    cam.pitch = Math.max(-1.5, Math.min(1.5, cam.pitch + dy * 0.008));
  }
  save(); dirty = true;
});
const stop = (e) => { drag = null; cv.classList.remove("drag"); };
cv.addEventListener("pointerup", stop);
cv.addEventListener("pointercancel", stop);
cv.addEventListener("contextmenu", (e) => e.preventDefault());
cv.addEventListener("wheel", (e) => {
  e.preventDefault();
  cam.dist = Math.max(M.radius * 0.25,
                      Math.min(M.radius * 12, cam.dist * Math.exp(e.deltaY * 0.0012)));
  save(); dirty = true;
}, {passive: false});

// ---------------------------------------------------------------- draw
const AX = ["X", "Y", "Z"];
function hudText() {
  const d = M.dims.map((v) => v.toFixed(0)).join(" × ");
  let t = `${d} mm bounding box<br>${M.nFaces.toLocaleString()} triangles shown`;
  if (M.nFacesFull > M.nFaces)
    t += ` of ${M.nFacesFull.toLocaleString()} (preview only)`;
  if (axis >= 0) {
    const lo = M.origin[axis], hi = lo + M.span[axis];
    t += `<br>cut at ${AX[axis]} = ${(lo + cutFrac * (hi - lo)).toFixed(1)} mm`;
  }
  hud.innerHTML = t;
}

const cutAt = () => M.origin[Math.max(axis, 0)] +
                    cutFrac * M.span[Math.max(axis, 0)];

function capQuad(P, V) {
  // the clip plane, sized to the bounding box and nudged a hair inside the solid so
  // it never z-fights with the surface it terminates
  const eps = M.span[axis] * 1e-4 * sign;
  const u = (axis + 1) % 3, v = (axis + 2) % 3;
  const lo = [0, 1, 2].map((i) => M.origin[i] - M.span[i] * 0.02);
  const hi = [0, 1, 2].map((i) => M.origin[i] + M.span[i] * 1.02);
  const pt = (su, sv) => {
    const p = [0, 0, 0];
    p[axis] = cutAt() - eps;
    p[u] = su ? hi[u] : lo[u];
    p[v] = sv ? hi[v] : lo[v];
    return p;
  };
  const q = [pt(0, 0), pt(1, 0), pt(1, 1), pt(0, 0), pt(1, 1), pt(0, 1)].flat();

  gl.useProgram(capProg);
  gl.bindVertexArray(capVao);
  gl.bindBuffer(gl.ARRAY_BUFFER, capBuf);
  gl.bufferSubData(gl.ARRAY_BUFFER, 0, new Float32Array(q));
  gl.uniformMatrix4fv(C.uProj, false, P);
  gl.uniformMatrix4fv(C.uView, false, V);
  gl.uniform3fv(C.uColor, [0.66, 0.31, 0.22]);
  // the outward cut normal is the axis itself, rotated by the view: that is column
  // `axis` of the column-major view matrix, times the side we kept
  gl.uniform3fv(C.uN, [sign * V[4 * axis], sign * V[4 * axis + 1],
                       sign * V[4 * axis + 2]]);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
}

function frame() {
  const w = wrap.clientWidth, h = wrap.clientHeight;
  const dpr = Math.min(devicePixelRatio || 1, 2);
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
    cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); dirty = true;
  }
  if (spinning) { cam.yaw += 0.005; dirty = true; }
  if (dirty && prog) {
    const P = proj(w / Math.max(h, 1)), V = view();
    gl.viewport(0, 0, cv.width, cv.height);
    gl.clearColor(0, 0, 0, 0);
    gl.clearStencil(0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT | gl.STENCIL_BUFFER_BIT);

    gl.useProgram(prog);
    gl.bindVertexArray(vao);
    gl.uniformMatrix4fv(U.uProj, false, P);
    gl.uniformMatrix4fv(U.uView, false, V);
    gl.uniform1i(U.uAxis, axis);
    gl.uniform1i(U.uFacets, facets ? 1 : 0);
    gl.uniform1f(U.uSign, sign);
    gl.uniform1f(U.uCut, cutAt());

    if (axis >= 0) {
      // pass 1: parity into the stencil only, depth off so every crossing counts
      gl.enable(gl.STENCIL_TEST);
      gl.colorMask(false, false, false, false);
      gl.depthMask(false);
      gl.disable(gl.DEPTH_TEST);
      gl.stencilFunc(gl.ALWAYS, 0, 0xff);
      gl.stencilOpSeparate(gl.FRONT, gl.KEEP, gl.KEEP, gl.INCR_WRAP);
      gl.stencilOpSeparate(gl.BACK, gl.KEEP, gl.KEEP, gl.DECR_WRAP);
      gl.drawElements(gl.TRIANGLES, nIdx, idxType, 0);

      // pass 2: fill the plane where the count says solid was severed
      gl.colorMask(true, true, true, true);
      gl.depthMask(true);
      gl.enable(gl.DEPTH_TEST);
      gl.stencilFunc(gl.NOTEQUAL, 0, 0xff);
      gl.stencilOp(gl.KEEP, gl.KEEP, gl.KEEP);
      capQuad(P, V);
      gl.disable(gl.STENCIL_TEST);
      gl.useProgram(prog);
      gl.bindVertexArray(vao);
    }

    // pass 3: the surface itself
    gl.drawElements(gl.TRIANGLES, nIdx, idxType, 0);
    hudText();
    dirty = false;
  }
  requestAnimationFrame(frame);
}
if (prog) { hudText(); frame(); }
</script>
"""


def _embed(html: str, height: int) -> None:
    """`st.iframe` from Streamlit 1.51; `components.v1.html` before it (the package
    floor is 1.30) — that call is deprecated and warns on every rerun from 1.60."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:                                                      # pragma: no cover
        import streamlit.components.v1 as components
        components.html(html, height=height, scrolling=False)


def stl_viewer(mesh, *, height: int = 430, max_faces: int = MAX_PREVIEW_FACES) -> dict:
    """Draw `mesh` in an interactive WebGL canvas. Returns the payload metadata so the
    caller can report the preview/export face counts."""
    payload = mesh_payload(mesh, max_faces=max_faces)
    html = (_HTML.replace("__PAYLOAD__", json.dumps(payload))
                 .replace("__H__", str(int(height))))
    _embed(html, height + 8)
    return payload
