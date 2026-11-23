"use strict";
// Case review page. The video is the clock. The 3D view is ray cast at full voxel resolution:
// each voxel knows its structure and when it was drilled, so any moment is drawn exactly.

const $ = (id) => document.getElementById(id);
const clock = (s) => { s = Math.max(0, Math.floor(s)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; };
const fine = (s) => { const m = Math.floor(s / 60), r = s - m * 60; return `${String(m).padStart(2, "0")}:${r.toFixed(3).padStart(6, "0")}`; };
const number = (v) => v === null || v === undefined || !Number.isFinite(v) ? "-" : Math.abs(v) >= 1000 ? Math.round(v).toLocaleString() : v.toFixed(Math.abs(v) >= 10 ? 1 : 2);
const PATH_SECONDS = 20, BRICK = 8, SHAFT_MM = 60;
const BURR_COLORS = { 1: "#7fd1ae", 2: "#e7c66b", 4: "#6fa8dc", 6: "#d98cb3" };
const CURVE_COLORS = ["#6fc3e8", "#f0b35e", "#8fd694", "#e58fb1", "#b9a3ec", "#e8dd7a", "#7fd8c8", "#f08f7f"];
const DEFAULT_CURVES = ["speed", "force", "voxels", "burr"];
const ON_OFF = new Set(["on", "device_clutch", "cam_clutch", "paused"]);
let info = null, scene = null, target = null, curves = {}, shownCurves = [], comments = [], mark = [null, null];

const VERT = `#version 300 es
out vec2 uv; void main() { vec2 p = vec2(gl_VertexID == 1 ? 3. : -1., gl_VertexID == 2 ? 3. : -1.); uv = p; gl_Position = vec4(p, 0., 1.); }`;

const FRAG = `#version 300 es
precision highp float; precision highp int; precision highp usampler3D;
in vec2 uv; out vec4 color;
uniform usampler3D uLabels, uTimes, uBricks0, uBricks1; uniform sampler2D uPalette;
uniform ivec3 uSize; uniform uint uNow, uLimit0; uniform bool uShowRemoved, uShowDrill;
uniform vec3 uEye, uRight, uUp, uForward, uTip, uAxis; uniform float uTan, uAspect, uBurrR, uShaftR, uShaftLen;
const int BRICK = ${BRICK};
const vec3 REMOVED = vec3(.25, .65, .95), BG = vec3(.02, .05, .08);
vec4 pal(uint id) { return texelFetch(uPalette, ivec2(int(id), 0), 0); }
bool inside(ivec3 q) { return all(greaterThanEqual(q, ivec3(0))) && all(lessThan(q, uSize)); }
bool occ(ivec3 q, int mode) {
  if (!inside(q)) return false;
  uint l = texelFetch(uLabels, q, 0).r; if (l == 0u) return false;
  if (texelFetch(uTimes, q, 0).r > uNow) { float a = pal(l).a; return a > 0. && (mode == 0 || a > .995); }
  return mode == 0 && uShowRemoved;
}
float field(vec3 x, int mode) {
  vec3 g = x - .5; ivec3 b = ivec3(floor(g)); vec3 f = g - vec3(b); float s = 0.;
  for (int i = 0; i < 8; i++) { ivec3 c = ivec3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
    if (occ(b + c, mode)) { vec3 w = mix(1. - f, f, vec3(c)); s += w.x * w.y * w.z; } }
  return s;
}
vec4 surface(vec3 x, int mode) {
  vec3 g = x - .5; ivec3 b = ivec3(floor(g)); vec3 f = g - vec3(b), sum = vec3(0.); float total = 0., best = -1., alpha = 1.;
  for (int i = 0; i < 8; i++) { ivec3 c = ivec3(i & 1, (i >> 1) & 1, (i >> 2) & 1), q = b + c;
    if (!occ(q, mode)) continue;
    vec3 v = mix(1. - f, f, vec3(c)); float w = v.x * v.y * v.z + 1e-4;
    uint l = texelFetch(uLabels, q, 0).r; bool alive = texelFetch(uTimes, q, 0).r > uNow;
    vec4 p = alive ? pal(l) : vec4(REMOVED, .25);
    sum += p.rgb * w; total += w; if (w > best) { best = w; alpha = p.a; }
  }
  return total > 0. ? vec4(sum / total, alpha) : vec4(.7, .7, .7, 1.);
}
vec3 gradient(vec3 x, int mode) {
  const float h = 1.15; vec3 e = vec3(h, 0., 0.);
  return vec3(field(x + e.xyy, mode) - field(x - e.xyy, mode), field(x + e.yxy, mode) - field(x - e.yxy, mode),
              field(x + e.yyx, mode) - field(x - e.yyx, mode));
}
bool march(vec3 o, vec3 rd, float t, float end, int mode, out float hit) {
  vec3 dir = vec3(abs(rd.x) < 1e-5 ? 1e-5 : rd.x, abs(rd.y) < 1e-5 ? 1e-5 : rd.y, abs(rd.z) < 1e-5 ? 1e-5 : rd.z);
  vec3 up = step(0., dir); float start = t; uint limit = mode == 0 ? uLimit0 : uNow;
  for (int k = 0; k < 1800 && t < end; k++) {
    vec3 x = o + rd * t; ivec3 p = clamp(ivec3(floor(x)), ivec3(0), uSize - 1), b = p / BRICK;
    uint top = mode == 0 ? texelFetch(uBricks0, b, 0).r : texelFetch(uBricks1, b, 0).r;
    if (top <= limit) {
      vec3 lo = vec3(b * BRICK), plane = (lo + up * float(BRICK) - o) / dir;
      t = max(min(plane.x, min(plane.y, plane.z)), t) + 2e-3; continue;
    }
    if (occ(p, mode)) {
      float a = max(t - 1.2, start), c = min(t + .7, end);
      if (field(o + rd * a, mode) >= .5) { hit = a; return true; }
      float previous = a;
      for (int i = 1; i <= 4; i++) {
        float m = mix(a, c, float(i) / 4.);
        if (field(o + rd * m, mode) >= .5) {
          float l = previous, h = m;
          for (int j = 0; j < 5; j++) { float q = .5 * (l + h); if (field(o + rd * q, mode) >= .5) h = q; else l = q; }
          hit = h; return true;
        }
        previous = m;
      }
      hit = t + .3; return true;   // a one-voxel structure: its field stays under the surface level
    }
    vec3 plane = (vec3(p) + up - o) / dir;
    t = max(min(plane.x, min(plane.y, plane.z)), t) + 2e-3;
  }
  return false;
}
vec3 light(vec3 base, vec3 n, vec3 rd, float ao) {
  vec3 key = normalize(-rd + uUp * .45 + uRight * .25), h = normalize(key - rd);
  float d = max(dot(n, key), 0.), head = max(dot(n, -rd), 0.);
  return base * (.20 * ao + (.62 * d + .24 * head) * mix(.55, 1., ao)) + vec3(.13) * pow(max(dot(n, h), 0.), 42.) * ao;
}
vec3 shade(vec3 x, vec3 rd, int mode, out float alpha) {
  vec4 c = surface(x, mode); alpha = c.a;
  vec3 g = gradient(x, mode); vec3 n = dot(g, g) > 1e-4 ? -normalize(g) : -rd;
  if (dot(n, rd) > 0.) n = -n;
  float ao = 1. - clamp(.95 * field(x + n * 2.2, mode) + .75 * field(x + n * 5., mode), 0., .72);
  return light(c.rgb, n, rd, ao);
}
float capsule(vec3 ro, vec3 rd, vec3 a, vec3 b, float r, out vec3 n) {
  vec3 ba = b - a, oa = ro - a; float baba = dot(ba, ba), bard = dot(ba, rd), baoa = dot(ba, oa), rdoa = dot(rd, oa), oaoa = dot(oa, oa);
  float qa = baba - bard * bard, qb = baba * rdoa - baoa * bard, qc = baba * oaoa - baoa * baoa - r * r * baba, h = qb * qb - qa * qc;
  if (h < 0.) return -1.;
  float t = (-qb - sqrt(h)) / qa, y = baoa + t * bard;
  if (y <= 0. || y >= baba) { vec3 cap = y <= 0. ? a : b, oc = ro - cap; float bq = dot(rd, oc), cq = dot(oc, oc) - r * r, hq = bq * bq - cq;
    if (hq < 0.) return -1.; t = -bq - sqrt(hq); n = normalize(ro + rd * t - cap); return t; }
  n = normalize(oa + t * rd - ba * y / baba); return t;
}
void main() {
  vec3 rd = normalize(uForward + uRight * uv.x * uTan * uAspect + uUp * uv.y * uTan);
  vec3 size = vec3(uSize), inv = 1. / vec3(abs(rd.x) < 1e-6 ? 1e-6 : rd.x, abs(rd.y) < 1e-6 ? 1e-6 : rd.y, abs(rd.z) < 1e-6 ? 1e-6 : rd.z);
  vec3 a = -uEye * inv, b = (size - uEye) * inv, lo = min(a, b), hi = max(a, b);
  float t0 = max(max(lo.x, lo.y), max(lo.z, 0.)) + 1e-3, t1 = min(hi.x, min(hi.y, hi.z));
  vec3 result = BG; float first = 1e9, hit;
  if (t0 < t1 && march(uEye, rd, t0, t1, 0, hit)) {
    first = hit; float alpha; vec3 front = shade(uEye + rd * hit, rd, 0, alpha);
    if (alpha > .995) result = front;
    else {
      vec3 behind = BG; float second, ignored;
      if (march(uEye, rd, hit + .05, t1, 1, second)) behind = shade(uEye + rd * second, rd, 1, ignored);
      result = mix(behind, front, alpha);
    }
  }
  if (uShowDrill) {
    vec3 n; vec3 oc = uEye - uTip; float bq = dot(rd, oc), hq = bq * bq - dot(oc, oc) + uBurrR * uBurrR;
    float tShaft = capsule(uEye, rd, uTip + uAxis * uBurrR * .6, uTip + uAxis * uShaftLen, uShaftR, n);
    if (tShaft > 0.) { vec3 lit = light(vec3(.62, .68, .74), n, rd, 1.); result = tShaft < first ? lit : mix(result, lit, .13); if (tShaft < first) first = tShaft; }
    if (hq > 0.) { float tBurr = -bq - sqrt(hq); if (tBurr > 0.) { n = normalize(uEye + rd * tBurr - uTip);
      vec3 lit = light(vec3(.95, .72, .25), n, rd, 1.); result = tBurr < first ? lit : mix(result, lit, .34); } }
  }
  color = vec4(pow(result, vec3(.92)), 1.);
}`;

const LINE_VERT = `#version 300 es
in vec3 point; uniform vec3 uEye, uRight, uUp, uForward; uniform float uTan, uAspect, uScale;
void main() { vec3 d = point / uScale - uEye; float z = dot(d, uForward);
  gl_Position = vec4(dot(d, uRight) / (uTan * uAspect), dot(d, uUp) / uTan, 0., z); }`;
const LINE_FRAG = `#version 300 es
precision mediump float; uniform vec4 uColor; out vec4 color; void main() { color = uColor; }`;

const caseFile = (name) => `/case/${name}?v=${encodeURIComponent(info.version || "")}`;
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const unit = (a) => scale(a, 1 / Math.hypot(...a));
const turn = (v, axis, angle) => add(add(scale(v, Math.cos(angle)), scale(cross(axis, v), Math.sin(angle))),
  scale(axis, dot(axis, v) * (1 - Math.cos(angle))));

class Scene {
  constructor(canvas, caseInfo) {
    this.canvas = canvas; this.info = caseInfo; this.t = 0; this.res = 1; this.size = caseInfo.shape; this.quality = 1;
    const gl = canvas.getContext("webgl2", { antialias: false, alpha: false, powerPreference: "high-performance" });
    if (!gl) throw new Error("This browser cannot show the 3D view (WebGL 2 is off).");
    this.gl = gl; this.volume = this.program(VERT, FRAG); this.lines = this.program(LINE_VERT, LINE_FRAG);
    this.vao = gl.createVertexArray();
    this.hidden = new Set(); this.boneAlpha = 1; this.showRemoved = false; this.showPath = false; this.showDrill = true;
    this.needsDraw = true; this.lastInput = 0; this.lastDraw = 0; this.resetView(); this.bindInput();
    new ResizeObserver(() => this.invalidate()).observe(canvas);
    canvas.addEventListener("webglcontextlost", (e) => { e.preventDefault(); $("scene-note").hidden = false;
      $("scene-note").textContent = "The 3D view stopped. Reload the page."; this.lost = true; });
  }
  program(vs, fs) {
    const gl = this.gl, id = gl.createProgram();
    for (const [type, text] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
      const shader = gl.createShader(type); gl.shaderSource(shader, text); gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
      gl.attachShader(id, shader);
    }
    gl.linkProgram(id);
    if (!gl.getProgramParameter(id, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(id));
    const uniforms = {}, count = gl.getProgramParameter(id, gl.ACTIVE_UNIFORMS);
    for (let k = 0; k < count; k++) { const name = gl.getActiveUniform(id, k).name; uniforms[name] = gl.getUniformLocation(id, name); }
    return { id, u: uniforms };
  }
  texture(unit, target, filter = true) {
    const gl = this.gl, t = gl.createTexture(); gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(target, t);
    for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(target, p, gl.NEAREST);
    for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T]) gl.texParameteri(target, p, gl.CLAMP_TO_EDGE);
    if (target === gl.TEXTURE_3D) gl.texParameteri(target, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
    return t;
  }
  async fetchVolume(suffix) {
    const [labels, times] = await Promise.all([`labels${suffix}`, `times${suffix}`].map(async (name) => {
      const reply = await fetch(caseFile(`${name}.bin.gz`)); if (!reply.ok) throw new Error("Could not load the 3D data.");
      return reply.arrayBuffer();
    }));
    return [new Uint8Array(labels), new Uint16Array(times)];
  }
  upload(labels, times, size) {
    const gl = this.gl, [nx, ny, nz] = size; while (gl.getError()) { /* clear */ }
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    this.texture(0, gl.TEXTURE_3D); gl.texImage3D(gl.TEXTURE_3D, 0, gl.R8UI, nx, ny, nz, 0, gl.RED_INTEGER, gl.UNSIGNED_BYTE, labels);
    this.texture(1, gl.TEXTURE_3D); gl.texImage3D(gl.TEXTURE_3D, 0, gl.R16UI, nx, ny, nz, 0, gl.RED_INTEGER, gl.UNSIGNED_SHORT, times);
    return gl.getError() === gl.NO_ERROR;
  }
  async load() {
    const gl = this.gl;
    let [labels, times] = await this.fetchVolume("");
    if (!this.upload(labels, times, this.info.shape)) {   // a phone short of memory: half resolution
      [labels, times] = await this.fetchVolume("_half");
      this.res = 2; this.size = this.info.half_shape;
      if (!this.upload(labels, times, this.size)) throw new Error("This device does not have enough memory for the 3D view.");
      this.resetView();
    }
    this.labels = labels; this.times = times;
    this.brickSize = this.size.map((n) => Math.ceil(n / BRICK));
    const count = this.brickSize[0] * this.brickSize[1] * this.brickSize[2];
    this.bricks0 = new Uint16Array(count); this.bricks1 = new Uint16Array(count);
    for (const [unit, data] of [[3, this.bricks0], [4, this.bricks1]]) {
      this.texture(unit, gl.TEXTURE_3D);
      gl.texImage3D(gl.TEXTURE_3D, 0, gl.R16UI, ...this.brickSize, 0, gl.RED_INTEGER, gl.UNSIGNED_SHORT, data);
    }
    this.texture(2, gl.TEXTURE_2D);
    this.palette = new Uint8Array(256 * 4); this.updatePalette();
    const [track, curveBytes] = await Promise.all(["path", "curves"].map(async (name) => {
      const reply = await fetch(caseFile(`${name}.bin.gz`)); if (!reply.ok) throw new Error("Could not load the session data.");
      return reply.arrayBuffer();
    }));
    this.track = new Float32Array(track); readCurves(curveBytes);
    this.points = new Float32Array(this.track.length / 2);
    for (let k = 0, n = this.track.length / 6; k < n; k++) this.points.set(this.track.subarray(k * 6, k * 6 + 3), k * 3);
    this.pathBuffer = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, this.pathBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, this.points, gl.STATIC_DRAW);
    this.ready = true; this.invalidate();
  }
  updatePalette() {
    const p = this.palette, alpha = new Uint8Array(256); p.fill(0);
    for (const s of this.info.structures) {
      alpha[s.id] = this.hidden.has(s.id) ? 0 : Math.round((s.name === "Bone" ? this.boneAlpha : 1) * 255);
      p.set([s.color[0] * 255, s.color[1] * 255, s.color[2] * 255, alpha[s.id]], s.id * 4);
    }
    const gl = this.gl; gl.activeTexture(gl.TEXTURE2);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, p);
    const key = `${[...alpha].map((a) => a === 0 ? 0 : a >= 254 ? 2 : 1).join("")}${this.showRemoved}`;
    if (key !== this.brickKey) { this.brickKey = key; this.buildBricks(alpha); }
    this.invalidate();
  }
  buildBricks(alpha) {   // lets rays jump over space that holds nothing visible
    const [nx, ny, nz] = this.size, [bx, by] = this.brickSize, b0 = this.bricks0, b1 = this.bricks1;
    const labels = this.labels, times = this.times, removed = this.showRemoved; b0.fill(0); b1.fill(0);
    const xBrick = new Int32Array(nx); for (let x = 0; x < nx; x++) xBrick[x] = x >> 3;
    for (let z = 0, i = 0; z < nz; z++) { const zb = (z >> 3) * bx * by;
      for (let y = 0; y < ny; y++) { const row = zb + (y >> 3) * bx;
        for (let x = 0; x < nx; x++, i++) { const l = labels[i]; if (!l) continue;
          const b = row + xBrick[x], a = alpha[l], v = times[i];
          if ((a > 0 || removed) && v > b0[b]) b0[b] = v;
          if (a >= 254 && v > b1[b]) b1[b] = v;
        } } }
    const gl = this.gl;
    for (const [unit, data] of [[3, b0], [4, b1]]) { gl.activeTexture(gl.TEXTURE0 + unit);
      gl.texSubImage3D(gl.TEXTURE_3D, 0, 0, 0, 0, ...this.brickSize, gl.RED_INTEGER, gl.UNSIGNED_SHORT, data); }
  }
  resetView() {
    const view = this.info.view;
    let forward = view ? unit(view.look) : [0, 0, 1], up = view ? view.up : [0, 1, 0];
    up = unit(add(up, scale(forward, -dot(up, forward))));
    this.forward = forward; this.up = up; this.right = cross(forward, up);
    this.center = scale(this.size, .5); this.fov = 30 * Math.PI / 180;
    this.distance = Math.hypot(...this.size) * .5 / Math.tan(this.fov / 2) * .92;
    this.invalidate();
  }
  invalidate() { this.needsDraw = true; }
  setTime(t) { if (t !== this.t) { this.t = t; this.lastInput = performance.now(); this.invalidate(); } }
  bindInput() {
    const c = this.canvas, pointers = new Map(); let gap = 0, mid = null;
    const midpoint = () => { const v = [...pointers.values()]; return [(v[0].x + v[1].x) / 2, (v[0].y + v[1].y) / 2]; };
    const span = () => { const v = [...pointers.values()]; return Math.hypot(v[0].x - v[1].x, v[0].y - v[1].y); };
    c.addEventListener("pointerdown", (e) => { c.setPointerCapture(e.pointerId); pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (pointers.size === 2) { gap = span(); mid = midpoint(); } });
    const end = (e) => { pointers.delete(e.pointerId); mid = null; this.invalidate(); };
    c.addEventListener("pointerup", end); c.addEventListener("pointercancel", end);
    c.addEventListener("pointermove", (e) => {
      const p = pointers.get(e.pointerId); if (!p) return;
      const dx = e.clientX - p.x, dy = e.clientY - p.y; p.x = e.clientX; p.y = e.clientY;
      this.lastInput = performance.now();
      if (pointers.size === 1 && !e.shiftKey && e.button !== 1) this.orbit(dx, dy);
      else if (pointers.size === 1) this.pan(dx, dy);
      else if (pointers.size === 2) { const s = span(), m = midpoint();
        if (gap > 0) this.zoom(gap / s); if (mid) this.pan(m[0] - mid[0], m[1] - mid[1]); gap = s; mid = m; }
    });
    c.addEventListener("wheel", (e) => { e.preventDefault(); this.lastInput = performance.now(); this.zoom(Math.exp(e.deltaY * .001)); }, { passive: false });
  }
  orbit(dx, dy) {
    const a = -dx * .006, b = -dy * .006;
    this.forward = turn(this.forward, this.up, a); this.right = turn(this.right, this.up, a);
    this.forward = unit(turn(this.forward, this.right, b)); this.up = unit(turn(this.up, this.right, b));
    this.right = unit(cross(this.forward, this.up)); this.invalidate();
  }
  pan(dx, dy) {
    const k = 2 * this.distance * Math.tan(this.fov / 2) / Math.max(this.canvas.clientHeight, 1);
    this.center = add(this.center, add(scale(this.right, -dx * k), scale(this.up, dy * k))); this.invalidate();
  }
  zoom(f) { this.distance = Math.min(Math.max(this.distance * f, 12), Math.hypot(...this.size) * 6); this.invalidate(); }
  camera(p, aspect) {
    const gl = this.gl, u = p.u, eye = add(this.center, scale(this.forward, -this.distance));
    gl.uniform3fv(u.uEye, eye); gl.uniform3fv(u.uRight, this.right); gl.uniform3fv(u.uUp, this.up); gl.uniform3fv(u.uForward, this.forward);
    gl.uniform1f(u.uTan, Math.tan(this.fov / 2)); gl.uniform1f(u.uAspect, aspect);
  }
  draw() {
    if (!this.ready || this.lost || !this.needsDraw || !this.canvas.clientWidth) return;
    const clockNow = performance.now(), moving = clockNow - this.lastInput < 180;
    if (moving && this.lastDraw) {   // keep motion fluid: fewer pixels while moving, full detail at rest
      const cost = clockNow - this.lastDraw;
      if (cost > 45) this.quality = Math.max(this.quality * .88, .38); else if (cost < 24) this.quality = Math.min(this.quality * 1.05, 1);
    }
    const ratio = window.devicePixelRatio || 1;
    const pixel = moving ? Math.min(ratio, 1.5) * this.quality : Math.min(Math.max(ratio, 1.5), 2.5);
    const w = Math.max(Math.round(this.canvas.clientWidth * pixel), 2), h = Math.max(Math.round(this.canvas.clientHeight * pixel), 2);
    if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
    this.needsDraw = moving; this.lastDraw = moving ? clockNow : 0;
    const gl = this.gl, aspect = w / h, u = this.volume.u; gl.viewport(0, 0, w, h);
    gl.useProgram(this.volume.id); gl.bindVertexArray(this.vao);
    gl.uniform1i(u.uLabels, 0); gl.uniform1i(u.uTimes, 1); gl.uniform1i(u.uPalette, 2); gl.uniform1i(u.uBricks0, 3); gl.uniform1i(u.uBricks1, 4);
    gl.uniform3iv(u.uSize, this.size);
    const now = Math.min(Math.round(this.t / this.info.duration * 65533) + 1, 65534);
    gl.uniform1ui(u.uNow, now); gl.uniform1ui(u.uLimit0, this.showRemoved ? 0 : now);
    gl.uniform1i(u.uShowRemoved, this.showRemoved ? 1 : 0); gl.uniform1i(u.uShowDrill, this.showDrill ? 1 : 0);
    const count = this.points.length / 3, last = Math.min(Math.floor(this.t * this.info.path_hz), count - 1);
    const voxel = this.info.spacing_mm[0] * this.res, burr = burrAt(this.t);
    const k = last * 6, tr = this.track, burrR = (burr ? burr.size_mm : 4) * .5 / voxel;
    gl.uniform3f(u.uTip, tr[k] / this.res, tr[k + 1] / this.res, tr[k + 2] / this.res);
    gl.uniform3f(u.uAxis, tr[k + 3], tr[k + 4], tr[k + 5]);
    gl.uniform1f(u.uBurrR, burrR); gl.uniform1f(u.uShaftR, Math.max(burrR * .44, .3 / voxel)); gl.uniform1f(u.uShaftLen, SHAFT_MM / voxel);
    this.camera(this.volume, aspect); gl.drawArrays(gl.TRIANGLES, 0, 3);
    const first = Math.max(last - Math.round(PATH_SECONDS * this.info.path_hz), 0);
    if (this.showPath && last > first) {
      gl.useProgram(this.lines.id); this.camera(this.lines, aspect); gl.uniform1f(this.lines.u.uScale, this.res);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.pathBuffer);
      const at = gl.getAttribLocation(this.lines.id, "point"); gl.enableVertexAttribArray(at); gl.vertexAttribPointer(at, 3, gl.FLOAT, false, 0, 0);
      gl.uniform4f(this.lines.u.uColor, .3, .85, 1, 1); gl.drawArrays(gl.LINE_STRIP, first, last - first + 1);
      gl.disableVertexAttribArray(at);
    }
  }
}

function readCurves(buffer) {
  curves = {};
  for (const c of info.curves) {
    curves[c.key] = { ...c, t: new Float32Array(buffer, c.offset, c.count), y: new Float32Array(buffer, c.offset + 4 * c.count, c.count) };
  }
  renderCurveList(); drawCurvesBase();
}
function curveValue(c, t) {
  if (!c || !c.t.length || t < c.t[0]) return null;
  let lo = 0, hi = c.t.length - 1;
  while (lo < hi) { const m = (lo + hi + 1) >> 1; if (c.t[m] <= t) lo = m; else hi = m - 1; }
  if (c.step || lo === c.t.length - 1) return c.y[lo];
  const span = c.t[lo + 1] - c.t[lo]; return span > 0 ? c.y[lo] + (c.y[lo + 1] - c.y[lo]) * (t - c.t[lo]) / span : c.y[lo];
}
function burrAt(t) { let burr = info.burrs[0] || null; for (const b of info.burrs) if (b.t <= t) burr = b; return burr; }

async function post(url, body = {}) {
  const reply = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const value = await reply.json().catch(() => ({}));
  if (!reply.ok) throw new Error(value.error || "That did not go through.");
  return value;
}

// ---- time ----
const video = $("video");
const now = () => target !== null ? target : Math.min(video.currentTime || 0, info.duration);
function seek(t) { t = Math.max(0, Math.min(t, info.duration)); target = t; video.currentTime = t; scene?.setTime(t); }
video.addEventListener("seeked", () => { if (target !== null && Math.abs(video.currentTime - target) < .12) target = null; });
const range = () => [mark[0] ?? 0, mark[1] ?? info.duration];
function togglePlay() {
  if (!video.paused) { video.pause(); return; }
  const [start, stop] = range(); if (now() >= stop - .02) seek(start);
  video.play();
}
function step(amount) {
  video.pause();
  if (amount === "frame-next") seek(now() + 1 / info.video_fps);
  else if (amount === "frame-back") seek(now() - 1 / info.video_fps);
  else seek(now() + Number(amount));
}
function setMark(which) {
  mark[which] = now();
  if (mark[0] !== null && mark[1] !== null && mark[0] > mark[1]) mark = [mark[1], mark[0]];
  drawTimeline(now());
}

// ---- drawing the timeline and curves ----
function fitCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1, w = Math.round(canvas.clientWidth * ratio), h = Math.round(canvas.clientHeight * ratio);
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; return true; }
  return false;
}
function drawTimeline(t) {
  const c = $("timeline"); if (!c.clientWidth) return; fitCanvas(c);
  const g = c.getContext("2d"), ratio = window.devicePixelRatio || 1, W = c.clientWidth, H = c.clientHeight, d = info.duration;
  const phone = document.body.dataset.version === "phone", pad = phone ? 8 : 10, x = (v) => pad + v / d * (W - 2 * pad);
  const burrY = 4, burrH = phone ? 15 : 20, strokeY = burrY + burrH + 3, strokeH = phone ? 11 : 14, tickY = strokeY + strokeH + 4;
  g.setTransform(ratio, 0, 0, ratio, 0, 0); g.fillStyle = "#0f2433"; g.fillRect(0, 0, W, H);
  g.font = "11px system-ui, sans-serif"; g.textAlign = "center"; g.textBaseline = "middle";
  for (const b of info.burrs) { const a = x(b.t), e = x(b.stop); g.fillStyle = BURR_COLORS[b.size_mm] || "#9bb0c3"; g.fillRect(a, burrY, Math.max(e - a, 1), burrH);
    if (e - a > 62) { g.fillStyle = "#0b1c2c"; g.fillText(`${b.size_mm} mm ${b.type}`, (a + e) / 2, burrY + burrH / 2 + 1); } }
  g.fillStyle = "#12283a"; g.fillRect(pad, strokeY, W - 2 * pad, strokeH); g.fillStyle = "#2f8fbf";
  for (const [a, b] of info.stroke_spans) g.fillRect(x(a), strokeY, Math.max(x(b) - x(a), 1.5), strokeH);
  g.fillStyle = "#6f879a"; g.strokeStyle = "#6f879a"; g.lineWidth = 1;
  const every = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600].find((s) => d / s <= (phone ? 6 : 12)) || 1200;
  for (let v = 0; v <= d; v += every) { const px = x(v); g.beginPath(); g.moveTo(px, tickY); g.lineTo(px, tickY + 4); g.stroke(); g.fillText(clock(v), Math.max(Math.min(px, W - 16), 16), tickY + 11); }
  if (mark[0] !== null || mark[1] !== null) { const [a, b] = range().map(x); g.fillStyle = "rgba(240,179,94,.17)"; g.fillRect(a, 0, b - a, H);
    g.strokeStyle = "#f0b35e"; g.lineWidth = 1.5; for (const px of [a, b]) { g.beginPath(); g.moveTo(px, 0); g.lineTo(px, H); g.stroke(); } }
  g.fillStyle = "#f0b35e"; for (const m of comments) { g.beginPath(); g.arc(x(m.t), strokeY + strokeH / 2, 4, 0, 2 * Math.PI); g.fill(); }
  g.strokeStyle = "#fff"; g.lineWidth = 2; g.beginPath(); g.moveTo(x(t), 0); g.lineTo(x(t), H); g.stroke();
  c.setAttribute("aria-valuemax", String(Math.round(d))); c.setAttribute("aria-valuenow", String(Math.round(t))); c.setAttribute("aria-valuetext", clock(t));
}
const GUTTER = 190;
let curveBase = null, curveKey = "";
function curveSpan(t) {
  const w = Number($("curve-window").value), d = info.duration;
  if (!w || w >= d) return [0, d];
  const start = Math.max(0, Math.min(t - w / 2, d - w)); return [start, start + w];
}
function drawCurvesBase(span = [0, info.duration]) {
  const c = $("curves"); if (!c.clientWidth || !c.clientHeight) return; fitCanvas(c);
  const ratio = window.devicePixelRatio || 1, W = c.clientWidth, H = c.clientHeight;
  curveBase = curveBase || document.createElement("canvas"); curveBase.width = c.width; curveBase.height = c.height;
  const g = curveBase.getContext("2d"); g.setTransform(ratio, 0, 0, ratio, 0, 0); g.fillStyle = "#0f2433"; g.fillRect(0, 0, W, H);
  const keys = shownCurves.filter((k) => curves[k]); if (!keys.length) return;
  const [a, b] = span, x = (v) => GUTTER + (v - a) / (b - a) * (W - GUTTER - 8), lane = H / keys.length;
  g.fillStyle = "rgba(47,143,191,.10)";
  for (const [s, e] of info.stroke_spans) if (e >= a && s <= b) g.fillRect(x(Math.max(s, a)), 0, Math.max(x(Math.min(e, b)) - x(Math.max(s, a)), 1), H);
  keys.forEach((key, index) => {
    const ch = curves[key], top = index * lane; g.strokeStyle = "#1e3a4f"; g.lineWidth = 1; g.beginPath(); g.moveTo(0, top + lane); g.lineTo(W, top + lane); g.stroke();
    let i0 = 0, i1 = ch.t.length; while (i0 < i1 - 1 && ch.t[i0 + 1] < a) i0++; while (i1 > i0 + 1 && ch.t[i1 - 2] > b) i1--;
    let low = Infinity, high = -Infinity; for (let i = i0; i < i1; i++) { const v = ch.y[i]; if (v < low) low = v; if (v > high) high = v; }
    if (!Number.isFinite(low)) return; if (high - low < 1e-9) { low -= .5; high += .5; }
    const y = (v) => top + lane - 5 - (v - low) / (high - low) * (lane - 10);
    g.strokeStyle = CURVE_COLORS[index % CURVE_COLORS.length]; g.lineWidth = 1.1; g.beginPath();
    for (let i = i0; i < i1; i++) { const px = x(Math.max(ch.t[i], a));
      if (i === i0) g.moveTo(px, y(ch.y[i])); else { if (ch.step) g.lineTo(px, y(ch.y[i - 1])); g.lineTo(px, y(ch.y[i])); } }
    if (ch.step) g.lineTo(x(b), y(ch.y[i1 - 1])); g.stroke();
    g.fillStyle = "#6f879a"; g.font = "10px system-ui, sans-serif"; g.textAlign = "left"; g.textBaseline = "top";
    g.fillText(number(high), GUTTER + 3, top + 2); g.textBaseline = "bottom"; g.fillText(number(low), GUTTER + 3, top + lane - 2);
  });
}
function drawCurves(t) {
  const c = $("curves"); if (!c.clientWidth || !c.clientHeight || !info.curves) return;
  const resized = fitCanvas(c), span = curveSpan(t), key = `${shownCurves}|${span}|${c.width}|${c.height}|${Object.keys(curves).length}`;
  if (resized || key !== curveKey) { curveKey = key; drawCurvesBase(span); }
  const g = c.getContext("2d"), ratio = window.devicePixelRatio || 1, W = c.clientWidth, H = c.clientHeight;
  g.setTransform(1, 0, 0, 1, 0, 0); if (curveBase) g.drawImage(curveBase, 0, 0); g.setTransform(ratio, 0, 0, ratio, 0, 0);
  const keys = shownCurves.filter((k) => curves[k]);
  if (!keys.length) { g.fillStyle = "#9bb0c3"; g.font = "13px system-ui"; g.textAlign = "center"; g.textBaseline = "middle"; g.fillText("Choose curves on the left", W / 2, H / 2); return; }
  const lane = H / keys.length; g.textAlign = "left"; g.textBaseline = "top";
  keys.forEach((key2, index) => { const ch = curves[key2], v = curveValue(ch, t);
    const text = ON_OFF.has(key2) && v !== null ? (v ? "on" : "off") : `${number(v)}${ch.unit ? ` ${ch.unit}` : ""}`;
    g.fillStyle = CURVE_COLORS[index % CURVE_COLORS.length]; g.font = "12px system-ui, sans-serif"; g.fillText(ch.label, 8, index * lane + 4, GUTTER - 14);
    if (lane > 36) { g.fillStyle = "#e6eef5"; g.font = "600 15px system-ui, sans-serif"; g.fillText(text, 8, index * lane + 20, GUTTER - 14); } });
  if (t >= span[0] && t <= span[1]) { const px = GUTTER + (t - span[0]) / (span[1] - span[0]) * (W - GUTTER - 8);
    g.strokeStyle = "#fff"; g.lineWidth = 1; g.beginPath(); g.moveTo(px, 0); g.lineTo(px, H); g.stroke(); }
}
function updateReadout(t) {
  $("time").textContent = document.body.dataset.version === "phone" ? `${clock(t)} / ${clock(info.duration)}` : `${fine(t)} / ${fine(info.duration)}`;
  $("comment-time").textContent = clock(t);
  const burr = burrAt(t), on = curveValue(curves.on, t), force = curveValue(curves.force, t), speed = curveValue(curves.speed, t);
  const pieces = [burr ? `${burr.size_mm} mm ${burr.type}` : ""];
  if (curves.on) pieces.push(`drill ${on ? "on" : "off"}`);
  if (curves.force) pieces.push(`${number(force)} N`); if (curves.speed) pieces.push(`${number(speed)} mm/s`);
  $("now-readout").textContent = pieces.filter(Boolean).join("  ·  ");
}
function frame() {
  const t = now(), [, stop] = range();
  if (!video.paused && t >= stop) { if ($("loop").checked) seek(range()[0]); else { video.pause(); seek(stop); } }
  updateReadout(t); drawTimeline(t);
  if (document.body.dataset.version === "desktop") drawCurves(t);
  if (scene) { scene.setTime(t); scene.draw(); }
  requestAnimationFrame(frame);
}

// ---- layout ----
let phoneMode = "video", layoutFlags = [1, 1, 1];
function applyView() {
  const phone = document.body.dataset.version === "phone", stage = $("stage");
  const [showVideo, showScene, showCurves] = phone ? [phoneMode !== "3d", phoneMode !== "video", false] : layoutFlags;
  stage.dataset.video = showVideo ? "1" : "0"; stage.dataset.scene = showScene ? "1" : "0";
  document.querySelector(".curves-wrap").dataset.show = showCurves ? "1" : "0";
  for (const b of document.querySelectorAll("button[data-mode]")) b.setAttribute("aria-pressed", String(b.dataset.mode === phoneMode));
  curveKey = ""; scene?.invalidate();
}
function chooseVersion() {
  const phone = window.innerWidth < 820 || (matchMedia("(pointer: coarse)").matches && window.innerWidth < 1024);
  const version = phone ? "phone" : "desktop";
  if (document.body.dataset.version !== version) { $("structures").close?.(); document.body.dataset.version = version; }
  applyView();
}

function renderFacts() {
  const facts = [["Case", info.code], ["Side", info.side ? `${info.side[0].toUpperCase()}${info.side.slice(1)} ear` : "-"], ["Length", clock(info.duration)]];
  $("session-facts").replaceChildren(...facts.flatMap(([k, v]) => { const dt = document.createElement("dt"), dd = document.createElement("dd"); dt.textContent = k; dd.textContent = v; return [dt, dd]; }));
}
function renderCurveList() {
  const list = $("curve-list"); list.replaceChildren(); let group = null;
  shownCurves = DEFAULT_CURVES.filter((k) => curves[k]);
  for (const c of info.curves) {
    if (c.group !== group) { group = c.group; const h = document.createElement("h3"); h.textContent = group; list.append(h); }
    const row = document.createElement("label"); row.className = "check";
    const box = document.createElement("input"); box.type = "checkbox"; box.checked = shownCurves.includes(c.key);
    box.addEventListener("change", () => { shownCurves = info.curves.filter((x) => x.key === c.key ? box.checked : shownCurves.includes(x.key)).map((x) => x.key); curveKey = ""; });
    row.append(box, document.createTextNode(c.label)); list.append(row);
  }
}
function renderStructures() {
  const list = $("structure-list"); list.replaceChildren();
  for (const s of info.structures) {
    const row = document.createElement("label"); row.className = "check";
    const box = document.createElement("input"); box.type = "checkbox"; box.checked = !scene?.hidden.has(s.id);
    box.addEventListener("change", () => { if (box.checked) scene.hidden.delete(s.id); else scene.hidden.add(s.id); scene.updatePalette(); updateStructureCount(); });
    const swatch = document.createElement("span"); swatch.className = "swatch";
    swatch.style.background = `rgb(${s.color.map((c) => Math.round(c * 255)).join(",")})`;
    row.append(box, swatch, document.createTextNode(s.name)); list.append(row);
  }
  updateStructureCount();
}
function updateStructureCount() { const n = info.structures.length; $("open-structures").textContent = `Structures (${n - (scene?.hidden.size || 0)}/${n})`; }
function setAll(on) {
  for (const s of info.structures) { if (on) scene.hidden.delete(s.id); else scene.hidden.add(s.id); }
  for (const box of $("structure-list").querySelectorAll("input")) box.checked = on;
  scene.updatePalette(); updateStructureCount();
}

function renderComments(rows) {
  rows.sort((a, b) => a.t - b.t); comments = rows;
  $("no-comments").hidden = rows.length > 0;
  $("comments").replaceChildren(...rows.map((c) => {
    const li = document.createElement("li"), head = document.createElement("div"), who = document.createElement("strong");
    const jump = document.createElement("button"), text = document.createElement("div");
    head.className = "comment-head"; who.textContent = c.author; jump.type = "button"; jump.className = "link"; jump.textContent = clock(c.t);
    jump.addEventListener("click", () => { video.pause(); seek(c.t); if (document.body.dataset.version === "phone") window.scrollTo({ top: 0, behavior: "smooth" }); });
    text.className = "comment-text"; text.textContent = c.text; head.append(who, jump); li.append(head, text); return li;
  }));
}
const loadComments = async () => { const reply = await fetch("/api/comments"); if (reply.ok) renderComments(await reply.json()); };

function bind() {
  for (const b of document.querySelectorAll("button[data-mode]")) b.addEventListener("click", () => { phoneMode = b.dataset.mode; applyView(); });
  for (const b of document.querySelectorAll("button[data-step]")) b.addEventListener("click", () => step(b.dataset.step));
  $("layout").addEventListener("change", (e) => { layoutFlags = e.target.value.split(",").map(Number); applyView(); });
  $("play").addEventListener("click", togglePlay);
  video.addEventListener("play", () => { $("play").textContent = "Pause"; $("play").setAttribute("aria-label", "Pause"); });
  video.addEventListener("pause", () => { $("play").textContent = "Play"; $("play").setAttribute("aria-label", "Play"); });
  $("speed").addEventListener("change", () => { video.playbackRate = Number($("speed").value); });
  $("mark-in").addEventListener("click", () => setMark(0)); $("mark-out").addEventListener("click", () => setMark(1));
  $("mark-clear").addEventListener("click", () => { mark = [null, null]; });
  const dragSeek = (canvas, toTime) => {
    let down = false;
    canvas.addEventListener("pointerdown", (e) => { down = true; canvas.setPointerCapture(e.pointerId); seek(toTime(e)); });
    canvas.addEventListener("pointermove", (e) => { if (down) seek(toTime(e)); });
    for (const kind of ["pointerup", "pointercancel"]) canvas.addEventListener(kind, () => { down = false; });
  };
  dragSeek($("timeline"), (e) => { const r = $("timeline").getBoundingClientRect(), pad = document.body.dataset.version === "phone" ? 8 : 10;
    return Math.max(0, Math.min((e.clientX - r.left - pad) / (r.width - 2 * pad), 1)) * info.duration; });
  dragSeek($("curves"), (e) => { const r = $("curves").getBoundingClientRect(), [a, b] = curveSpan(now());
    return a + Math.max(0, Math.min((e.clientX - r.left - GUTTER) / (r.width - GUTTER - 8), 1)) * (b - a); });
  $("comment-time").addEventListener("click", () => video.pause());
  $("comment").addEventListener("focus", () => video.pause());
  $("comment-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = $("comment").value.trim(); if (!text) return;
    $("comment-status").textContent = "Posting";
    try { await post("/api/comments", { t: now(), text }); $("comment").value = ""; $("comment-status").textContent = "Posted"; await loadComments(); }
    catch (error) { $("comment-status").textContent = error.message; }
  });
  $("finish").addEventListener("click", async () => {
    try { await post("/api/finished"); $("finish").textContent = "Sent"; $("comment-status").textContent = "Your review has been sent."; }
    catch (error) { $("comment-status").textContent = error.message; }
  });
  const sheet = $("structures");
  $("open-structures").addEventListener("click", () => {
    if (sheet.open) sheet.close();
    else if (document.body.dataset.version === "phone") sheet.showModal();
    else { sheet.show(); sheet.style.left = `${$("open-structures").offsetLeft}px`; }   // stays open while the model is turned
  });
  $("close-structures").addEventListener("click", () => sheet.close());
  $("show-all").addEventListener("click", () => setAll(true)); $("hide-all").addEventListener("click", () => setAll(false));
  $("bone-opacity").addEventListener("input", (e) => { if (scene) { scene.boneAlpha = e.target.value / 100; scene.updatePalette(); } });
  $("show-removed").addEventListener("change", (e) => { if (scene) { scene.showRemoved = e.target.checked; scene.updatePalette(); } });
  $("show-path").addEventListener("change", (e) => { if (scene) { scene.showPath = e.target.checked; scene.invalidate(); } });
  $("show-drill").addEventListener("change", (e) => { if (scene) { scene.showDrill = e.target.checked; scene.invalidate(); } });
  $("reset-view").addEventListener("click", () => scene?.resetView());
  window.addEventListener("resize", chooseVersion);
  window.addEventListener("keydown", (e) => {
    if (document.body.dataset.version !== "desktop" || e.target.closest("textarea, input, select") || e.ctrlKey || e.metaKey || e.altKey) return;
    const actions = { " ": togglePlay, ArrowLeft: () => step(e.shiftKey ? -1 : "frame-back"), ArrowRight: () => step(e.shiftKey ? 1 : "frame-next"),
      i: () => setMark(0), I: () => setMark(0), o: () => setMark(1), O: () => setMark(1), Home: () => seek(0), End: () => seek(info.duration) };
    if (actions[e.key]) { e.preventDefault(); actions[e.key](); }
  });
}

async function start() {
  const reply = await fetch("/api/case");
  if (!reply.ok) { location.reload(); return; }
  info = await reply.json();
  const side = info.side ? `${info.side[0].toUpperCase()}${info.side.slice(1)} ear` : "";
  $("case-code").textContent = info.code; document.title = `${info.code} review`;
  $("case-info").textContent = [side, clock(info.duration), `Reviewing as ${info.reviewer}`].filter(Boolean).join(" · ");
  video.src = caseFile("video.mp4");
  bind(); chooseVersion(); renderFacts();
  try {
    scene = new Scene($("scene"), info); renderStructures(); await scene.load();
    $("scene-note").hidden = true; window.__sceneReady = true;
  } catch (error) { $("scene-note").hidden = false; $("scene-note").textContent = error.message; }
  await loadComments(); setInterval(loadComments, 20000);
  post("/api/opened").catch(() => {});
  requestAnimationFrame(frame);
}
start();
