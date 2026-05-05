import * as THREE from 'three';
import { HandLandmarker, FilesetResolver, DrawingUtils } from '@mediapipe/tasks-vision';

// ============================================================
// 1. SCENE
// ============================================================
const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();
scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
dirLight.position.set(5, 10, 7);
scene.add(dirLight);
const backLight = new THREE.DirectionalLight(0xffffff, 0.3);
backLight.position.set(-5, -3, -5);
scene.add(backLight);

const grid = new THREE.GridHelper(20, 20, 0x7b8b6f, 0x7b8b6f);
grid.position.y = -1.5; grid.material.opacity = 0.4; grid.material.transparent = true;
scene.add(grid);

const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.1, 100);
camera.position.set(0, 0, 6);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(innerWidth, innerHeight);
container.appendChild(renderer.domElement);
const raycaster = new THREE.Raycaster();

// ============================================================
// 2. SHAPES & MESH
// ============================================================
const SHAPE_LABELS = {
    sphere: 'Sphere',
    cube: 'Cube',
    roundedcube: 'Rounded Cube',
    cylinder: 'Cylinder',
    beveledcylinder: 'Beveled Cylinder',
    cone: 'Cone',
    torus: 'Torus',
    helix: 'Helix',
    pyramid: 'Pyramid',
    icosahedron: 'Icosahedron',
    dodecahedron: 'Dodecahedron',
    torusknot: 'Torus Knot',
};

function makeRoundedCubeGeometry(size = 1.5, radius = 0.22, segments = 10) {
    const g = new THREE.BoxGeometry(size, size, size, segments, segments, segments);
    const p = g.attributes.position;
    const innerHalf = size * 0.5 - radius;
    const v = new THREE.Vector3();
    const inner = new THREE.Vector3();
    const delta = new THREE.Vector3();
    for (let i = 0; i < p.count; i++) {
        v.fromBufferAttribute(p, i);
        inner.set(
            THREE.MathUtils.clamp(v.x, -innerHalf, innerHalf),
            THREE.MathUtils.clamp(v.y, -innerHalf, innerHalf),
            THREE.MathUtils.clamp(v.z, -innerHalf, innerHalf),
        );
        delta.copy(v).sub(inner);
        if (delta.lengthSq() < 1e-6) {
            const ax = Math.abs(v.x), ay = Math.abs(v.y), az = Math.abs(v.z);
            if (ax >= ay && ax >= az) delta.set(Math.sign(v.x) || 1, 0, 0);
            else if (ay >= ax && ay >= az) delta.set(0, Math.sign(v.y) || 1, 0);
            else delta.set(0, 0, Math.sign(v.z) || 1);
        }
        delta.normalize().multiplyScalar(radius);
        p.setXYZ(i, inner.x + delta.x, inner.y + delta.y, inner.z + delta.z);
    }
    p.needsUpdate = true;
    g.computeVertexNormals();
    return g;
}

function makeBeveledCylinderGeometry(radius = 0.82, height = 2, bevel = 0.18, radialSeg = 32, heightSeg = 14) {
    const g = new THREE.CylinderGeometry(radius, radius, height, radialSeg, heightSeg, false);
    const p = g.attributes.position;
    const halfH = height * 0.5;
    const innerRad = Math.max(0.05, radius - bevel);
    const innerHalfH = Math.max(0.05, halfH - bevel);
    const v = new THREE.Vector3();
    const inner = new THREE.Vector3();
    const delta = new THREE.Vector3();
    const radial = new THREE.Vector3();
    for (let i = 0; i < p.count; i++) {
        v.fromBufferAttribute(p, i);
        radial.set(v.x, 0, v.z);
        const radialLen = radial.length();
        if (radialLen > 1e-6) radial.multiplyScalar(1 / radialLen);
        else radial.set(1, 0, 0);
        inner.copy(v);
        inner.y = THREE.MathUtils.clamp(v.y, -innerHalfH, innerHalfH);
        if (radialLen > innerRad) {
            inner.x = radial.x * innerRad;
            inner.z = radial.z * innerRad;
        }
        delta.copy(v).sub(inner);
        if (delta.lengthSq() < 1e-6) delta.copy(radial);
        delta.normalize().multiplyScalar(bevel);
        p.setXYZ(i, inner.x + delta.x, inner.y + delta.y, inner.z + delta.z);
    }
    p.needsUpdate = true;
    g.computeVertexNormals();
    return g;
}

const SHAPES = {
    sphere:   () => new THREE.SphereGeometry(1, 32, 24),
    cube:     () => new THREE.BoxGeometry(1.5, 1.5, 1.5, 10, 10, 10),
    roundedcube: () => makeRoundedCubeGeometry(),
    cylinder: () => new THREE.CylinderGeometry(0.8, 0.8, 2, 32, 10),
    beveledcylinder: () => makeBeveledCylinderGeometry(),
    cone:     () => new THREE.ConeGeometry(1, 2, 32, 10),
    torus:    () => new THREE.TorusGeometry(0.8, 0.35, 20, 40),
    helix:    () => new THREE.TorusKnotGeometry(0.7, 0.16, 180, 28, 2, 7),
    pyramid:  () => new THREE.ConeGeometry(1, 1.8, 4, 1),
    icosahedron: () => new THREE.IcosahedronGeometry(1.05, 0),
    dodecahedron: () => new THREE.DodecahedronGeometry(1.0, 0),
    torusknot: () => new THREE.TorusKnotGeometry(0.76, 0.2, 150, 20, 3, 5),
};

let currentShape = 'sphere';
let mesh = null, wireframe = null;
let shapeLabel = null;
let shapeItems = [];
const material = new THREE.MeshStandardMaterial({ color: 0xa8c3e6, roughness: 0.35, metalness: 0.1, flatShading: false, side: THREE.DoubleSide });
const selectionState = {
    kind: 'none',
    active: false,
    pointLocal: new THREE.Vector3(),
    normalLocal: new THREE.Vector3(0, 0, 1),
    radius: 0.55,
    logicalVertexIds: new Set(),
    boundaryVertexIds: new Set(),
    pulseUntil: 0,
};
const interactionState = {
    mode: 'select',
    selection: selectionState,
    activeOperation: null,
    placementSpec: { shape: 'sphere', scale: 0.3, depth: 0.5, blendStrength: 0.7, variant: 'default' },
    lastAddSpec: null,
};
const topologyCache = {
    logicalVertices: [],
    bufferToLogical: [],
    adjacency: [],
    version: 0,
};
const GESTURE_TUNING = {
    modeLockMs: 120000,
    dragPixelDeadZone: 12,
    dragStep: 0.018,
    dragScale: 0.0024,
    axisBiasRatio: 1.35,
    sculptDepthDeadZone: 0.007,
    sculptDepthStep: 0.012,
    sculptDepthScale: 0.75,
    formDepthDeadZone: 0.010,
    formDepthStep: 0.020,
    formDepthScale: 1.0,
    bevelDepthDeadZone: 0.008,
    bevelDepthStep: 0.014,
    bevelDepthScale: 0.75,
    formNormalSnap: 1.0,
    detailNormalSnap: 1.0,
    sculptNormalSnap: 0.72,
};
let modeLockUntil = 0;

// ---- Crochet shader patch ---------------------------------------------------
// Procedural V-stitch pattern in OBJECT space (so the texture is locked to the
// mesh and rotates with it like a real fabric — no sliding when you orbit).
// Triplanar projection blends three planar samples by the object-space normal,
// so it works on any geometry without needing UVs (important: addShapeToMesh
// merges geometry without preserving UVs).
material.userData.uCrochet      = { value: 0.0 };
material.userData.uCrochetScale = { value: 22.0 };  // higher = smaller stitches
material.userData.uCrochetDark  = { value: 0.18 };  // gap darkness (lower = brighter object)
material.userData.uCrochetBump  = { value: 0.22 };  // 3D pop of the stitches
material.onBeforeCompile = (shader) => {
    shader.uniforms.uCrochet      = material.userData.uCrochet;
    shader.uniforms.uCrochetScale = material.userData.uCrochetScale;
    shader.uniforms.uCrochetDark  = material.userData.uCrochetDark;
    shader.uniforms.uCrochetBump  = material.userData.uCrochetBump;

    shader.vertexShader = shader.vertexShader
        .replace('#include <common>', `
            #include <common>
            varying vec3 vCroObjPos;
            varying vec3 vCroObjNrm;
        `)
        .replace('#include <project_vertex>', `
            #include <project_vertex>
            // OBJECT space — locks the pattern to the mesh.
            vCroObjPos = transformed;
            vCroObjNrm = normalize(objectNormal);
        `);

    shader.fragmentShader = `
        uniform float uCrochet;
        uniform float uCrochetScale;
        uniform float uCrochetDark;
        uniform float uCrochetBump;
        varying vec3 vCroObjPos;
        varying vec3 vCroObjNrm;

        float croHash(vec2 p) {
            return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
        }

        // Single V-stitch SDF inside one cell (uv in [0,1]^2).
        // Returns: x = stitch coverage [0..1] (1 = on yarn, 0 = gap)
        //          y = signed distance to nearest yarn centre line (for AA + bump)
        vec2 croStitchCell(vec2 uv) {
            // Brick-stagger every other row (real crochet single-crochet pattern).
            float row = floor(uv.y);
            float xoff = mod(row, 2.0) * 0.5;
            vec2 cellId = vec2(floor(uv.x + xoff), row);
            vec2 cell   = vec2(fract(uv.x + xoff), fract(uv.y));

            // Two arms of a V meeting at bottom-centre. Mirror around x=0.5
            // so we only solve for one arm.
            vec2 p = vec2(abs(cell.x - 0.5), cell.y);
            // Arm endpoints (in mirrored half-cell): (0, 0.05) -> (0.42, 0.95)
            vec2 a = vec2(0.00, 0.05);
            vec2 b = vec2(0.42, 0.95);
            vec2 ab = b - a;
            float t = clamp(dot(p - a, ab) / dot(ab, ab), 0.0, 1.0);
            float d = length(p - a - ab * t);

            // Anti-aliased yarn thickness using screen-space derivative.
            float aa = max(fwidth(d), 0.001);
            float thickness = 0.13;
            float coverage = 1.0 - smoothstep(thickness - aa, thickness + aa, d);

            return vec2(coverage, d);
        }

        // Triplanar blend by object-space normal.
        vec4 croStitchTri(vec3 p, vec3 n) {
            vec3 b = pow(abs(n), vec3(4.0));   // sharper blend = less ghosting on diagonals
            b /= (b.x + b.y + b.z + 1e-5);
            vec2 sx = croStitchCell(p.zy * uCrochetScale);
            vec2 sy = croStitchCell(p.xz * uCrochetScale);
            vec2 sz = croStitchCell(p.xy * uCrochetScale);
            // x = coverage, y = distance (use min-distance across planes for sharper bump)
            float cov = sx.x * b.x + sy.x * b.y + sz.x * b.z;
            float dist = min(sx.y, min(sy.y, sz.y));
            // Per-stitch colour jitter (slight yarn shade variation per cell)
            vec2 cellId = floor(p.xy * uCrochetScale + vec2(0.5));
            float jitter = croHash(cellId) * 0.12 - 0.06;
            return vec4(cov, dist, jitter, 1.0);
        }
    ` + shader.fragmentShader
        .replace('#include <color_fragment>', `
            #include <color_fragment>
            if (uCrochet > 0.5) {
                vec4 cro = croStitchTri(vCroObjPos, normalize(vCroObjNrm));
                // Darken gaps between stitches.
                diffuseColor.rgb *= (1.0 - (1.0 - cro.x) * uCrochetDark);
                // Per-stitch yarn shade jitter (mostly visible on yarn pixels).
                diffuseColor.rgb *= (1.0 + cro.z * cro.x);
            }
        `)
        .replace('#include <roughnessmap_fragment>', `
            #include <roughnessmap_fragment>
            if (uCrochet > 0.5) {
                vec4 cro = croStitchTri(vCroObjPos, normalize(vCroObjNrm));
                // Yarn ridges catch a bit more light; gaps stay fully fuzzy.
                roughnessFactor = mix(1.0, 0.72, cro.x);
            }
        `)
        .replace('#include <normal_fragment_maps>', `
            #include <normal_fragment_maps>
            if (uCrochet > 0.5) {
                // Cheap derivative bump from the coverage field — yarn strands pop.
                float h  = croStitchTri(vCroObjPos, normalize(vCroObjNrm)).x;
                float hx = dFdx(h);
                float hy = dFdy(h);
                vec3 dpx = dFdx(vCroObjPos);
                vec3 dpy = dFdy(vCroObjPos);
                vec3 r1 = cross(dpy, normal);
                vec3 r2 = cross(normal, dpx);
                vec3 grad = (r1 * hx + r2 * hy) /
                            max(abs(dot(dpx, r1)), 1e-4);
                normal = normalize(normal - grad * uCrochetBump);
            }
        `);
};

function setShape(name) {
    if (mesh) { scene.remove(mesh); mesh.geometry.dispose(); }
    const geo = SHAPES[name]().toNonIndexed();
    geo.computeVertexNormals();
    mesh = new THREE.Mesh(geo, material);
    selectionState.active = false;
    selectionState.kind = 'none';
    selectionState.logicalVertexIds = new Set();
    selectionState.boundaryVertexIds = new Set();
    refreshWireframe();
    rebuildTopologyCache();
    scene.add(mesh);
    currentShape = name;
    syncShapeUi(name);
}

function refreshWireframe() {
    if (wireframe) { mesh.remove(wireframe); wireframe.geometry.dispose(); }
    wireframe = new THREE.LineSegments(
        new THREE.EdgesGeometry(mesh.geometry, 15),
        new THREE.LineBasicMaterial({ color: 0x222222, transparent: true, opacity: 0.12 })
    );
    mesh.add(wireframe);
}

function addShapeToMesh(name, ox, oy, oz, scale) {
    const ng = SHAPES[name]();
    // IMPORTANT: scale BEFORE translate so the placement offset stays in local-mesh
    // coordinates. (translate-then-scale would multiply the offset by `scale`, putting
    // the new shape far from the preview position.)
    ng.scale(scale, scale, scale); ng.translate(ox, oy, oz);
    const ni = ng.toNonIndexed();
    const eg = mesh.geometry;
    const mp = new Float32Array(eg.attributes.position.count * 3 + ni.attributes.position.count * 3);
    mp.set(eg.attributes.position.array, 0);
    mp.set(ni.attributes.position.array, eg.attributes.position.count * 3);
    const mn = new Float32Array(eg.attributes.normal.count * 3 + ni.attributes.normal.count * 3);
    mn.set(eg.attributes.normal.array, 0);
    mn.set(ni.attributes.normal.array, eg.attributes.normal.count * 3);
    const mg = new THREE.BufferGeometry();
    mg.setAttribute('position', new THREE.BufferAttribute(mp, 3));
    mg.setAttribute('normal', new THREE.BufferAttribute(mn, 3));
    mesh.geometry.dispose(); mesh.geometry = mg;
    refreshWireframe();
    rebuildTopologyCache();
    ng.dispose(); ni.dispose();
}

// Pre-merge "swell": push existing mesh vertices outward so they wrap the
// bounding sphere of the placement shape. This eliminates the tell-tale
// concave seam where added geometry would otherwise float above a flat surface.
// `centerLocal` and `radius` are in local-mesh coordinates.
function swellMeshAround(centerLocal, radius, blendStrength) {
    const p = mesh.geometry.attributes.position;
    // Halo extends a bit beyond the merge sphere so the transition tapers smoothly.
    const halo = radius * (1.4 + blendStrength * 0.8);
    const halo2 = halo * halo;
    const target = radius * 0.92; // just inside the new shape's surface
    for (let i = 0; i < p.count; i++) {
        const vx = p.getX(i), vy = p.getY(i), vz = p.getZ(i);
        const dx = vx - centerLocal.x, dy = vy - centerLocal.y, dz = vz - centerLocal.z;
        const d2 = dx*dx + dy*dy + dz*dz;
        if (d2 >= halo2) continue;
        const d = Math.sqrt(d2);
        if (d >= target) continue;        // already outside the new shape — leave alone
        if (d < 1e-5) continue;            // skip the singular center point
        // Cosine falloff: full push at center, tapering to zero at halo.
        const t = d / halo;
        const fall = 0.5 * (1 + Math.cos(Math.PI * t));
        // How much to push: bring vertex up to `target` distance, weighted by falloff and blendStrength.
        const push = (target - d) * fall * (0.6 + blendStrength * 0.4);
        const inv = 1 / d;
        p.setXYZ(i,
            vx + dx * inv * push,
            vy + dy * inv * push,
            vz + dz * inv * push,
        );
    }
    p.needsUpdate = true;
    mesh.geometry.computeVertexNormals();
}

function rebuildTopologyCache() {
    if (!mesh?.geometry?.attributes?.position) return;
    const pos = mesh.geometry.attributes.position;
    const norm = mesh.geometry.attributes.normal;
    const vertexMap = new Map();
    const logicalVertices = [];
    const bufferToLogical = new Array(pos.count);
    const keyScale = 1000;

    function makeKey(x, y, z) {
        return `${Math.round(x * keyScale)}:${Math.round(y * keyScale)}:${Math.round(z * keyScale)}`;
    }

    const v = new THREE.Vector3();
    const n = new THREE.Vector3();
    for (let i = 0; i < pos.count; i++) {
        v.fromBufferAttribute(pos, i);
        const key = makeKey(v.x, v.y, v.z);
        let logicalId = vertexMap.get(key);
        if (logicalId === undefined) {
            logicalId = logicalVertices.length;
            vertexMap.set(key, logicalId);
            logicalVertices.push({
                id: logicalId,
                position: v.clone(),
                normal: new THREE.Vector3(),
                bufferIndices: [],
            });
        }
        const logical = logicalVertices[logicalId];
        logical.bufferIndices.push(i);
        if (norm) {
            n.fromBufferAttribute(norm, i);
            logical.normal.add(n);
        }
        bufferToLogical[i] = logicalId;
    }

    const adjacency = logicalVertices.map(() => new Set());
    for (let i = 0; i < pos.count; i += 3) {
        const a = bufferToLogical[i];
        const b = bufferToLogical[i + 1];
        const c = bufferToLogical[i + 2];
        adjacency[a].add(b); adjacency[a].add(c);
        adjacency[b].add(a); adjacency[b].add(c);
        adjacency[c].add(a); adjacency[c].add(b);
    }
    logicalVertices.forEach(vtx => {
        if (vtx.normal.lengthSq() < 1e-6) vtx.normal.set(0, 0, 1);
        else vtx.normal.normalize();
    });

    topologyCache.logicalVertices = logicalVertices;
    topologyCache.bufferToLogical = bufferToLogical;
    topologyCache.adjacency = adjacency;
    topologyCache.version++;
}

function refreshSelectionFromTopology() {
    if (!selectionState.active || selectionState.logicalVertexIds.size === 0) {
        selectionState.kind = 'none';
        selectionState.active = false;
        return;
    }
    const nextIds = new Set();
    let px = 0, py = 0, pz = 0;
    let nx = 0, ny = 0, nz = 0;
    for (const logicalId of selectionState.logicalVertexIds) {
        const logical = topologyCache.logicalVertices[logicalId];
        if (!logical) continue;
        nextIds.add(logicalId);
        px += logical.position.x; py += logical.position.y; pz += logical.position.z;
        nx += logical.normal.x; ny += logical.normal.y; nz += logical.normal.z;
    }
    if (!nextIds.size) {
        selectionState.kind = 'none';
        selectionState.active = false;
        return;
    }
    selectionState.logicalVertexIds = nextIds;
    selectionState.pointLocal.set(px / nextIds.size, py / nextIds.size, pz / nextIds.size);
    selectionState.normalLocal.set(nx, ny, nz).normalize();
    const boundary = new Set();
    for (const logicalId of nextIds) {
        const neighbors = topologyCache.adjacency[logicalId] || [];
        for (const neighborId of neighbors) {
            if (!nextIds.has(neighborId)) {
                boundary.add(logicalId);
                break;
            }
        }
    }
    selectionState.boundaryVertexIds = boundary;
    selectionState.kind = 'region';
    selectionState.active = true;
}

setShape('sphere');

// ============================================================
// 1b. WORKSHOPS — top-bar themed presets (changes scene material + grid + lights)
// ============================================================
const WORKSHOPS = {
    default: { matColor: 0xa8c3e6, metalness: 0.10, roughness: 0.35, gridColor: 0x7b8b6f, ambient: 0xffffff, dir: 0xffffff, crochet: 0 },
    metal:   { matColor: 0x8a93a3, metalness: 0.92, roughness: 0.12, gridColor: 0x00e5ff, ambient: 0x202833, dir: 0xffd4f0, crochet: 0 },
    wood:    { matColor: 0xb07a47, metalness: 0.05, roughness: 0.85, gridColor: 0x8b5a2b, ambient: 0xffeacc, dir: 0xfff0d4, crochet: 0 },
    // Crochet base colour: pure white. Combined with the shader's mild gap
    // darkening (uCrochetDark = 0.18), the perceived surface lands around
    // eggshell white (~#f5f5f5).
    // Lights kept neutral white so the colour reads true on the object — the
    // pink cottagecore *room* still comes from the CSS background.
    crochet: { matColor: 0xffffff, metalness: 0.00, roughness: 0.95, gridColor: 0xd28aae, ambient: 0xffffff, dir: 0xffffff, crochet: 1 },
};
let currentWorkshop = 'default';
function applyWorkshop(name) {
    if (!WORKSHOPS[name]) return;
    currentWorkshop = name;
    document.body.dataset.workshop = name;
    const w = WORKSHOPS[name];
    material.color.setHex(w.matColor);
    material.metalness = w.metalness;
    material.roughness = w.roughness;
    material.transparent = false;
    material.opacity = 1;
    material.wireframe = false;
    // Toggle the crochet shader uniform (no recompile — the shader checks the uniform at runtime).
    material.userData.uCrochet.value = w.crochet ? 1.0 : 0.0;
    material.needsUpdate = true;
    scene.traverse(o => {
        if (o.isAmbientLight) o.color.setHex(w.ambient);
        else if (o.isDirectionalLight) o.color.setHex(w.dir);
    });
    grid.material.color.setHex(w.gridColor);
    document.querySelectorAll('.ws-btn').forEach(b =>
        b.classList.toggle('active', b.dataset.workshop === name));
}
// Wired up at the bottom (after saveUndo is defined)

// Shape Palette UI
const palette = document.createElement('div');
palette.id = 'shape-palette';
palette.style.cssText = 'position:absolute;top:86px;left:18px;z-index:15;display:flex;flex-direction:column;gap:8px;width:210px;';
palette.innerHTML = '<div style="font-weight:700;font-size:12px;letter-spacing:0.08em;text-transform:uppercase;color:#5a6652;padding-left:6px;">Objects</div>';
const emojis = { sphere:'🔴', cube:'🟦', cylinder:'🫙', cone:'🔺', torus:'🍩' };
const PALETTE_SHAPES = ['sphere', 'cube', 'cylinder', 'cone', 'torus'];
for (const name of PALETTE_SHAPES) {
    const row = document.createElement('div');
    row.style.cssText = 'display:flex;gap:8px;align-items:stretch;';
    const btn = document.createElement('button');
    btn.dataset.shapeSwap = name;
    btn.innerHTML = `<span style="font-size:15px;line-height:1">${emojis[name]}</span><span style="display:flex;flex-direction:column;align-items:flex-start;gap:2px;"><strong style="font-size:12px;line-height:1">${SHAPE_LABELS[name] || name}</strong><span style="font-size:10px;opacity:0.72;line-height:1">click to swap</span></span>`;
    btn.className = 'pal-btn';
    btn.style.cssText = 'flex:1;display:flex;align-items:center;gap:8px;background:rgba(248,238,219,0.92);border:1.5px solid #7b8b6f;border-radius:14px;padding:10px 12px;cursor:pointer;font-size:12px;font-family:inherit;color:#2b332b;text-align:left;box-shadow:0 6px 18px rgba(0,0,0,0.06);';
    btn.addEventListener('click', () => { saveUndo(); setShape(name); });
    if (name==='sphere') btn.style.borderWidth='3px';
    const add = document.createElement('button');
    add.textContent = '➕'; add.title = `Add ${name} (click to enter placement mode, Esc to exit)`;
    add.className = 'pal-add';
    add.dataset.shape = name;
    add.style.cssText = 'min-width:38px;background:rgba(248,238,219,0.95);border:1.5px solid #7b8b6f;border-radius:14px;padding:0 10px;cursor:pointer;font-size:14px;box-shadow:0 6px 18px rgba(0,0,0,0.06);';
    add.addEventListener('click', () => {
        if (placementShape === name) cancelPlacement();
        else startPlacement(name);
    });
    row.appendChild(btn); row.appendChild(add); palette.appendChild(row);
}
document.body.appendChild(palette);

// mark palette for docking
palette.classList.add('left-palette');

// Placement mode — click to set position, drag to scale, release to commit
let placementShape = null;
let placementPreview = null;  // THREE.Mesh preview
let placementPos = null;      // local position {x,y,z}
let placementNormal = null;   // local normal
let placementStartY = 0;      // mouseY at click start
let placementStartX = 0;      // mouseX at click start (for depth control)
let placementDragging = false;

// --- Gesture-driven placement state -----------------------------------------
// When `placementShape` is set + user pinches: preview spawns at the raycast
// hit on the mesh, distance(thumb,index) → size, hand-z (forward/back) → depth.
// Releasing the pinch commits & merges. All values smoothed via lerp to hide
// MediaPipe jitter.
let gesturePlacing = false;
let gesturePinchStartZ = null;     // wrist z at pinch start (relative reference)
let gesturePinchStartDist = null;  // initial finger distance for size baseline
let smoothedPos = new THREE.Vector3();
let smoothedScale = 0.3;
let smoothedDepth = 0.5;
let smoothedBlend = 0.7;

const placementLabel = document.createElement('div');
placementLabel.style.cssText = 'display:none;position:absolute;bottom:172px;left:50%;transform:translateX(-50%);z-index:20;background:rgba(248,238,219,0.96);border:1.5px solid #7b8b6f;border-radius:999px;padding:10px 18px;font-size:13px;font-family:inherit;color:#2b332b;pointer-events:none;box-shadow:0 10px 24px rgba(0,0,0,0.12);';
document.body.appendChild(placementLabel);

const previewMat = new THREE.MeshStandardMaterial({ color: 0xffdd57, transparent: true, opacity: 0.4, side: THREE.DoubleSide });

function getPlacementDefaults(shapeName = currentShape) {
    if (interactionState.lastAddSpec && interactionState.lastAddSpec.shape === shapeName) {
        return { ...interactionState.lastAddSpec };
    }
    return { shape: shapeName, scale: 0.3, depth: 0.5, blendStrength: 0.7, variant: 'default' };
}

function capturePlacementSpec(shapeName, scale, depth, blendStrength) {
    const spec = {
        shape: shapeName,
        scale,
        depth,
        blendStrength,
        variant: shapeName,
    };
    interactionState.lastAddSpec = spec;
    interactionState.placementSpec = { ...spec };
}

function startPlacement(n, spec = null) {
    interactionState.placementSpec = { ...getPlacementDefaults(n), ...(spec || {}), shape: n, variant: n };
    placementShape = n;
    smoothedScale = interactionState.placementSpec.scale;
    smoothedDepth = interactionState.placementSpec.depth;
    smoothedBlend = interactionState.placementSpec.blendStrength;
    document.body.style.cursor = 'crosshair';
    placementLabel.textContent = `Placement: ${SHAPE_LABELS[n] || n} · drag to size/depth · pinch to place with hands · Esc cancels`;
    placementLabel.style.display = 'block';
    // Highlight the active palette button so the user can see they're in placement mode.
    document.querySelectorAll('.pal-add').forEach(b => b.classList.toggle('active', b.dataset.shape === n));
}

// Exit placement mode after a successful commit: clears the in-flight preview/drag
// state and disarms `placementShape` so the next shape must be explicitly armed
// again with the + button.
function finishPlacement() {
    placementDragging = false;
    placementPos = null;
    placementNormal = null;
    gesturePlacing = false;
    gesturePinchStartZ = null;
    gesturePinchStartDist = null;
    placementShape = null;
    interactionState.activeOperation = null;
    if (placementPreview) {
        scene.remove(placementPreview);
        placementPreview.geometry.dispose();
        placementPreview = null;
    }
    document.body.style.cursor = '';
    placementLabel.style.display = 'none';
    document.querySelectorAll('.pal-add').forEach(b => b.classList.remove('active'));
    closeBlendPanel();
}

function cancelPlacement() {
    placementShape = null;
    placementDragging = false;
    placementPos = null;
    gesturePlacing = false;
    gesturePinchStartZ = null;
    gesturePinchStartDist = null;
    interactionState.activeOperation = null;
    if (placementPreview) { scene.remove(placementPreview); placementPreview.geometry.dispose(); placementPreview = null; }
    document.body.style.cursor = '';
    placementLabel.style.display = 'none';
    document.querySelectorAll('.pal-add').forEach(b => b.classList.remove('active'));
    closeBlendPanel();
}

// Update the preview from a 2D drag: vertical = size, horizontal = depth.
// Vertical up → bigger. Horizontal right → extrude (depth → +1), left → indent (-1).
// Default depth at click time is 0.5 (half-buried = naturally merged look).
function updatePreviewFromDrag(mouseX, mouseY) {
    if (!placementPreview || !placementPos || !placementNormal) return;
    const baseScale = interactionState.placementSpec?.scale ?? 0.3;
    const baseDepth = interactionState.placementSpec?.depth ?? 0.5;
    const sDelta = (placementStartY - mouseY) * 0.005;
    const dDelta = (mouseX - placementStartX) * 0.004;
    const s = Math.max(0.1, Math.min(2.0, baseScale + sDelta));
    const d = Math.max(-1.0, Math.min(1.0, baseDepth + dDelta));
    blendDepth = d;
    smoothedScale = s;
    smoothedDepth = d;
    placementPreview.scale.setScalar(s);
    const localCenter = new THREE.Vector3(
        placementPos.x + placementNormal.x * s * d,
        placementPos.y + placementNormal.y * s * d,
        placementPos.z + placementNormal.z * s * d,
    );
    placementPreview.position.copy(localCenter.applyMatrix4(mesh.matrixWorld));
    placementPreview.material.color.setHex(d < 0 ? 0xff7eb6 : 0xffdd57);
    placementLabel.textContent =
        `Size ${s.toFixed(2)}  ·  Depth ${d>=0?'+':''}${d.toFixed(2)} ${d<0?'(indent)':'(extrude)'}  ·  release to lock`;
}

// Smoothly lerp the live preview toward target values driven by hand gestures.
// Called every animation frame while `gesturePlacing` is true.
function applyGesturePlacement() {
    if (!placementPreview || !placementPos || !placementNormal) return;
    const s = smoothedScale;
    const d = smoothedDepth;
    blendDepth = d;
    placementPreview.scale.setScalar(s);
    const localCenter = new THREE.Vector3(
        placementPos.x + placementNormal.x * s * d,
        placementPos.y + placementNormal.y * s * d,
        placementPos.z + placementNormal.z * s * d,
    );
    placementPreview.position.copy(localCenter.applyMatrix4(mesh.matrixWorld));
    placementPreview.material.color.setHex(d < 0 ? 0xff7eb6 : 0xffdd57);
    placementLabel.textContent =
        `🖐️  Size ${s.toFixed(2)}  ·  Depth ${d>=0?'+':''}${d.toFixed(2)} ${d<0?'(indent)':'(extrude)'}  ·  release pinch to lock`;
}

// ----- Blend mode (extrude / indent + seamless merge) -----
// After mouseup-to-finish-resize, we enter blend mode instead of committing immediately.
// User adjusts depth (-1=indent, +1=extrude) and blend (0=hard, 1=fully smoothed seam).
// Closed-fist hand gesture moving forward/back also drives depth.
const blendPanel = document.createElement('div');
blendPanel.id = 'blend-panel';
blendPanel.innerHTML = `
    <h3>✨ Blend & Merge</h3>
    <label>Depth (–1 indent · 0 surface · +1 extrude)</label>
    <input type="range" id="bl-depth" min="-1" max="1" step="0.05" value="0.5">
    <div class="row"><span>Indent</span><span id="bl-depth-v">+0.50</span><span>Extrude</span></div>
    <label>Blend (seamless merge)</label>
    <input type="range" id="bl-blend" min="0" max="1" step="0.05" value="0.7">
    <div class="row"><span>Hard</span><span id="bl-blend-v">0.70</span><span>Soft</span></div>
    <div class="btns">
        <button id="bl-cancel">Cancel</button>
        <button id="bl-confirm" class="primary">Confirm ✓</button>
    </div>
    <div class="gesture-hint">Tip: make a fist and move forward/back to control depth</div>
`;
document.body.appendChild(blendPanel);

let blendDepth = 0.5, blendStrength = 0.7;
const blDepthEl = blendPanel.querySelector('#bl-depth');
const blBlendEl = blendPanel.querySelector('#bl-blend');
const blDepthV  = blendPanel.querySelector('#bl-depth-v');
const blBlendV  = blendPanel.querySelector('#bl-blend-v');
function fmtSigned(n) { return (n>=0?'+':'') + n.toFixed(2); }
blDepthEl.addEventListener('input', e => { blendDepth = parseFloat(e.target.value); blDepthV.textContent = fmtSigned(blendDepth); updateBlendPreview(); });
blBlendEl.addEventListener('input', e => { blendStrength = parseFloat(e.target.value); blBlendV.textContent = blendStrength.toFixed(2); });
blendPanel.querySelector('#bl-cancel').addEventListener('click', cancelPlacement);
blendPanel.querySelector('#bl-confirm').addEventListener('click', commitPlacementBlended);

function openBlendPanel() {
    blendDepth = interactionState.placementSpec?.depth ?? 0.5;
    blendStrength = interactionState.placementSpec?.blendStrength ?? 0.7;
    blDepthEl.value = blendDepth; blDepthV.textContent = fmtSigned(blendDepth);
    blBlendEl.value = blendStrength; blBlendV.textContent = blendStrength.toFixed(2);
    blendPanel.classList.add('open');
    placementLabel.style.display = 'none';
    updateBlendPreview();
}
function closeBlendPanel() {
    blendPanel.classList.remove('open');
}
function updateBlendPreview() {
    if (!placementPreview || !placementPos || !placementNormal) return;
    const s = placementPreview.scale.x;
    // placementPos is the surface hit point in LOCAL coords. Center of new shape sits
    // at anchor + normal * (depth * s).
    const off = blendDepth * s;
    const local = new THREE.Vector3(
        placementPos.x + placementNormal.x * off,
        placementPos.y + placementNormal.y * off,
        placementPos.z + placementNormal.z * off,
    );
    placementPreview.position.copy(local.applyMatrix4(mesh.matrixWorld));
    if (blendDepth < 0) placementPreview.material.color.setHex(0xff7eb6);
    else placementPreview.material.color.setHex(0xffdd57);
}

// Commit gesture placement cleanly — shape sits on surface without distortion
function commitGesturePlacement() {
    if (!placementPreview || !placementPos || !placementNormal) { finishPlacement(); return; }
    saveUndo();

    const s = smoothedScale;  // Use the smoothed scale from gesture
    const anchor = { x: placementPos.x, y: placementPos.y, z: placementPos.z };
    const localNormal = placementNormal.clone();

    // Keep the inserted shape's original form intact.
    // Only bias it slightly into the surface so it reads as attached, not floating.
    const offset = smoothedDepth * s * 0.2;
    const center = {
        x: anchor.x + localNormal.x * offset,
        y: anchor.y + localNormal.y * offset,
        z: anchor.z + localNormal.z * offset,
    };

    // Add the new shape geometry cleanly
    addShapeToMesh(placementShape, center.x, center.y, center.z, s);
    capturePlacementSpec(placementShape, s, smoothedDepth, blendStrength);

    // Keep placement mode sticky for rapid repeat-adding
    finishPlacement();
    if(isDebug) console.log('✅ Gesture placement committed');
}

function commitPlacementBlended() {
    if (!placementPreview || !placementPos || !placementNormal) { cancelPlacement(); return; }
    saveUndo();

    const s = placementPreview.scale.x;
    // Anchor = the actual surface hit point in local-mesh coords (placementPos is now
    // stored without any offset baked in).
    const anchor = { x: placementPos.x, y: placementPos.y, z: placementPos.z };
    const localNormal = placementNormal.clone();

    if (blendDepth < 0) {
        // INDENT: push original mesh vertices inward at the anchor
        const indentRadius = s * 1.5;
        const indentDepth  = Math.abs(blendDepth) * s * 1.5;
        const p = mesh.geometry.attributes.position;
        const r2 = indentRadius * indentRadius;
        for (let i = 0; i < p.count; i++) {
            const vx = p.getX(i), vy = p.getY(i), vz = p.getZ(i);
            const dx = vx - anchor.x, dy = vy - anchor.y, dz = vz - anchor.z;
            const d2 = dx*dx + dy*dy + dz*dz;
            if (d2 < r2) {
                const f = 0.5 * (1 + Math.cos(Math.PI * Math.sqrt(d2) / indentRadius));
                p.setXYZ(i,
                    vx - localNormal.x * indentDepth * f,
                    vy - localNormal.y * indentDepth * f,
                    vz - localNormal.z * indentDepth * f);
            }
        }
        p.needsUpdate = true;
        mesh.geometry.computeVertexNormals();
        // Smoothing pass for seamless seam
        if (blendStrength > 0) {
            const oldR = brushRadius;
            brushRadius = indentRadius * (1.2 + blendStrength * 0.8);
            const wp = new THREE.Vector3(anchor.x, anchor.y, anchor.z).applyMatrix4(mesh.matrixWorld);
            const passes = Math.round(2 + blendStrength * 4);
            for (let k = 0; k < passes; k++) smoothAt(wp);
            brushRadius = oldR;
        }
        refreshWireframe();
        rebuildTopologyCache();
    } else {
        // POSITIVE PLACEMENT: keep the added primitive rigid and recognizable.
        // Mouse placement should behave like stamping a clean shape onto the surface,
        // not like re-sculpting or melting that shape into the host mesh.
        const offset = blendDepth * s;
        const center = {
            x: anchor.x + localNormal.x * offset,
            y: anchor.y + localNormal.y * offset,
            z: anchor.z + localNormal.z * offset,
        };

        // Slightly bias the shape into the surface so it reads as attached, while
        // preserving the primitive's own silhouette.
        center.x -= localNormal.x * s * 0.08;
        center.y -= localNormal.y * s * 0.08;
        center.z -= localNormal.z * s * 0.08;

        addShapeToMesh(placementShape, center.x, center.y, center.z, s);
    }

    capturePlacementSpec(placementShape, s, blendDepth, blendStrength);
    // Sticky mode: keep `placementShape` selected for rapid repeat-adding.
    finishPlacement();
}

window.addEventListener('keydown', e => {
    if (e.key === 'Escape') {
        if (interactionState.activeOperation) OperationController.cancel();
        else if (placementShape || placementPreview) cancelPlacement();
        else if (gestureSelection.active) SelectionController.clear();
    }
    if (e.key === 'x' && !e.target.closest('input') && !splitActive) { saveUndo(); splitMesh(0.5); setTimeout(finalizeSplit, 500); }
});

// ============================================================
// 3. SCULPT TOOLS
// ============================================================
let brushRadius = 0.4, sculptStrength = 0.04;
let activeTool = 'pull';
let symmetryX = false;

// Tool panel
const toolPanel = document.createElement('div');
toolPanel.style.cssText = 'position:absolute;top:320px;left:20px;z-index:15;background:rgba(248,238,219,0.95);border:1.5px solid #7b8b6f;border-radius:18px;padding:12px 14px;font-size:12px;font-family:inherit;color:#2b332b;width:175px;box-shadow:0 12px 30px rgba(0,0,0,0.10);';
toolPanel.innerHTML = `
    <div style="font-weight:700;margin-bottom:6px;letter-spacing:0.08em;text-transform:uppercase;font-size:11px;color:#5a6652;">Sculpt Tools</div>
    <div id="tool-btns" style="display:flex;flex-wrap:wrap;gap:5px;margin-bottom:10px;"></div>
    <div style="font-weight:600;margin-bottom:4px;">Brush Size</div>
    <input type="range" id="br" min="0.1" max="1.5" step="0.05" value="0.4" style="width:100%;">
    <div style="display:flex;justify-content:space-between;font-size:10px;"><span>Small</span><span id="brv">0.40</span><span>Big</span></div>
    <div style="font-weight:600;margin:6px 0 4px;">Brush Strength</div>
    <input type="range" id="sr" min="0.01" max="0.12" step="0.005" value="0.04" style="width:100%;">
    <div style="display:flex;justify-content:space-between;font-size:10px;"><span>Soft</span><span id="srv">0.040</span><span>Hard</span></div>
    <div style="margin-top:8px;display:flex;gap:4px;flex-wrap:wrap;">
        <button id="sym-btn" style="flex:1;padding:4px;border:1.5px solid #7b8b6f;border-radius:6px;background:#fff;cursor:pointer;font-size:11px;">🪞 Sym</button>
        <button id="undo-btn" style="flex:1;padding:4px;border:1.5px solid #7b8b6f;border-radius:6px;background:#fff;cursor:pointer;font-size:11px;">↩ Undo</button>
        <button id="export-btn" style="flex:1;padding:4px;border:1.5px solid #7b8b6f;border-radius:6px;background:#fff;cursor:pointer;font-size:11px;">💾 STL</button>
    </div>
`;
document.body.appendChild(toolPanel);

// mark tool panel for docking
toolPanel.classList.add('left-tools');

// Create a left dock to group palette and tools and a toggle to collapse them
;(function createLeftDock(){
    const leftDock = document.createElement('div');
    leftDock.id = 'left-dock';
    document.body.appendChild(leftDock);

    // Move existing palette and toolPanel into dock
    if (palette) leftDock.appendChild(palette);
    if (toolPanel) leftDock.appendChild(toolPanel);

    // Add small toggle button to show/hide dock
    const toggle = document.createElement('button');
    toggle.id = 'left-dock-toggle';
    toggle.title = 'Toggle tools';
    toggle.textContent = '☰';
    document.body.appendChild(toggle);
    toggle.addEventListener('click', () => leftDock.classList.toggle('collapsed'));

    // Auto-collapse on small screens for clarity
    function adaptDock() {
        if (innerWidth <= 900) leftDock.classList.add('collapsed');
        else leftDock.classList.remove('collapsed');
    }
    adaptDock();
    window.addEventListener('resize', adaptDock);
})();

const TOOLS = [
    { id:'pull', em:'🔼', tip:'Pull out' },
    { id:'push', em:'🔽', tip:'Push in' },
    { id:'smooth', em:'🧈', tip:'Smooth' },
    { id:'flatten', em:'🫓', tip:'Flatten' },
    { id:'inflate', em:'🎈', tip:'Inflate' },
    { id:'grab', em:'✊', tip:'Grab & move' },
    { id:'crease', em:'✂️', tip:'Crease/sharp line' },
    { id:'pinch', em:'🤏', tip:'Pinch vertices together' },
];

const toolButtonsById = new Map();

function refreshToolButtonUi() {
    for (const [id, btn] of toolButtonsById.entries()) {
        btn.style.background = id === activeTool ? '#7b8b6f' : '#fff';
        btn.style.color = id === activeTool ? '#fff' : '#2b332b';
    }
}

function setActiveTool(nextTool) {
    if (!TOOLS.some(t => t.id === nextTool)) return;
    activeTool = nextTool;
    refreshToolButtonUi();
}

const tbc = document.getElementById('tool-btns');
for (const t of TOOLS) {
    const b = document.createElement('button');
    b.textContent = t.em; b.title = t.tip; b.dataset.tool = t.id;
    b.style.cssText = 'padding:7px 10px;border:1.5px solid #7b8b6f;border-radius:10px;background:#fff;cursor:pointer;font-size:15px;transition:all 0.1s;box-shadow:0 4px 10px rgba(0,0,0,0.06);';
    b.addEventListener('click', () => setActiveTool(t.id));
    toolButtonsById.set(t.id, b);
    tbc.appendChild(b);
}
refreshToolButtonUi();

document.getElementById('br').addEventListener('input', e => { brushRadius=parseFloat(e.target.value); document.getElementById('brv').textContent=brushRadius.toFixed(2); });
document.getElementById('sr').addEventListener('input', e => { sculptStrength=parseFloat(e.target.value); document.getElementById('srv').textContent=sculptStrength.toFixed(3); });

const symBtn = document.getElementById('sym-btn');
symBtn.addEventListener('click', () => { symmetryX=!symmetryX; symBtn.style.background=symmetryX?'#7b8b6f':'#fff'; symBtn.style.color=symmetryX?'#fff':'#2b332b'; });

const formActionPanel = document.createElement('div');
formActionPanel.style.cssText = 'margin-top:10px;padding-top:10px;border-top:1px dashed rgba(123,139,111,0.35);display:flex;flex-direction:column;gap:6px;';
formActionPanel.innerHTML = `
    <div style="font-weight:700;letter-spacing:0.08em;text-transform:uppercase;font-size:11px;color:#5a6652;">Form Ops</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">
        <button id="dup-last-btn" style="padding:6px;border:1.5px solid #7b8b6f;border-radius:8px;background:#fff;cursor:pointer;font-size:11px;">⧉ Duplicate</button>
        <button id="clear-patch-btn" style="padding:6px;border:1.5px solid #7b8b6f;border-radius:8px;background:#fff;cursor:pointer;font-size:11px;">◌ Clear</button>
        <button id="extrude-lite-btn" style="padding:6px;border:1.5px solid #7b8b6f;border-radius:8px;background:#fff;cursor:pointer;font-size:11px;">⇧ Extrude</button>
        <button id="bevel-lite-btn" style="padding:6px;border:1.5px solid #7b8b6f;border-radius:8px;background:#fff;cursor:pointer;font-size:11px;">⬓ Bevel</button>
    </div>
    <div id="selection-readout" style="font-size:11px;line-height:1.35;color:#5a6652;">Patch: none. Click or pinch a surface to select a region.</div>
`;
toolPanel.appendChild(formActionPanel);

const duplicateLastBtn = document.getElementById('dup-last-btn');
const clearPatchBtn = document.getElementById('clear-patch-btn');
const extrudeLiteBtn = document.getElementById('extrude-lite-btn');
const bevelLiteBtn = document.getElementById('bevel-lite-btn');
const selectionReadout = document.getElementById('selection-readout');

duplicateLastBtn.addEventListener('click', () => {
    if (!beginDuplicatePlacement()) setCommandStatus('Duplicate Last needs a placed shape first');
});
clearPatchBtn.addEventListener('click', () => {
    SelectionController.clear();
    if (selectionReadout) selectionReadout.textContent = 'Patch: none. Click or pinch a surface to select a region.';
    setCommandStatus('Selection cleared');
});
extrudeLiteBtn.addEventListener('click', () => {
    if (!gestureSelection.active) {
        setCommandStatus('Select a patch before extruding');
        return;
    }
    OperationController.nudge('extrude-lite', 0.08);
});
bevelLiteBtn.addEventListener('click', () => {
    if (!gestureSelection.active) {
        setCommandStatus('Select a patch before beveling');
        return;
    }
    OperationController.nudge('bevel-lite', 0.06);
});

// Undo — full snapshot of geometry + material + transform so any action can be reversed
const undoStack = [];
function snapshot() {
    const pos = mesh.geometry.attributes.position;
    const norm = mesh.geometry.attributes.normal;
    return {
        positions: new Float32Array(pos.array),
        normals: norm ? new Float32Array(norm.array) : null,
        material: {
            color: material.color.getHex(),
            metalness: material.metalness,
            roughness: material.roughness,
            opacity: material.opacity,
            transparent: material.transparent,
            wireframe: material.wireframe,
        },
        scale: mesh.scale.toArray(),
        position: mesh.position.toArray(),
        rotation: [mesh.rotation.x, mesh.rotation.y, mesh.rotation.z],
    };
}
function saveUndo() {
    undoStack.push(snapshot());
    if (undoStack.length > 20) undoStack.shift();
}
function undo() {
    if (!undoStack.length) return;
    const s = undoStack.pop();
    // Replace geometry buffers — handles shape changes, splits, adds (different vertex counts)
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(s.positions, 3));
    if (s.normals) g.setAttribute('normal', new THREE.BufferAttribute(s.normals, 3));
    mesh.geometry.dispose();
    mesh.geometry = g;
    mesh.geometry.computeVertexNormals();
    refreshWireframe();
    rebuildTopologyCache();
    // Restore material
    material.color.setHex(s.material.color);
    material.metalness = s.material.metalness;
    material.roughness = s.material.roughness;
    material.opacity = s.material.opacity;
    material.transparent = s.material.transparent;
    material.wireframe = s.material.wireframe;
    material.needsUpdate = true;
    // Restore transform
    mesh.scale.fromArray(s.scale);
    mesh.position.fromArray(s.position);
    mesh.rotation.set(s.rotation[0], s.rotation[1], s.rotation[2]);
    SelectionController.refreshFromGeometry();
}
document.getElementById('undo-btn').addEventListener('click', undo);
window.addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
        // Don't hijack native text undo when typing in chat input
        if (e.target.closest('input, textarea')) return;
        e.preventDefault();
        undo();
    }
});

// STL Export
document.getElementById('export-btn').addEventListener('click', () => {
    const pos = mesh.geometry.attributes.position;
    const triangles = pos.count / 3;
    const buf = new ArrayBuffer(84 + triangles * 50);
    const dv = new DataView(buf);
    dv.setUint32(80, triangles, true);
    let offset = 84;
    const vA = new THREE.Vector3(), vB = new THREE.Vector3(), vC = new THREE.Vector3(), norm = new THREE.Vector3();
    for (let i = 0; i < triangles; i++) {
        const i3 = i * 3;
        vA.fromBufferAttribute(pos, i3); vB.fromBufferAttribute(pos, i3+1); vC.fromBufferAttribute(pos, i3+2);
        vA.applyMatrix4(mesh.matrixWorld); vB.applyMatrix4(mesh.matrixWorld); vC.applyMatrix4(mesh.matrixWorld);
        const edge1 = vB.clone().sub(vA), edge2 = vC.clone().sub(vA);
        norm.crossVectors(edge1, edge2).normalize();
        dv.setFloat32(offset, norm.x, true); dv.setFloat32(offset+4, norm.y, true); dv.setFloat32(offset+8, norm.z, true); offset+=12;
        dv.setFloat32(offset, vA.x, true); dv.setFloat32(offset+4, vA.y, true); dv.setFloat32(offset+8, vA.z, true); offset+=12;
        dv.setFloat32(offset, vB.x, true); dv.setFloat32(offset+4, vB.y, true); dv.setFloat32(offset+8, vB.z, true); offset+=12;
        dv.setFloat32(offset, vC.x, true); dv.setFloat32(offset+4, vC.y, true); dv.setFloat32(offset+8, vC.z, true); offset+=12;
        dv.setUint16(offset, 0, true); offset+=2;
    }
    const blob = new Blob([buf], { type: 'application/octet-stream' });
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'sculpture.stl'; a.click();
});

// Brush viz
const brushVizGeo = new THREE.RingGeometry(0.95, 1, 48);
const brushVizMat = new THREE.MeshBasicMaterial({ color: 0xffdd57, transparent: true, opacity: 0.5, side: THREE.DoubleSide, depthTest: false });
const brushViz = new THREE.Mesh(brushVizGeo, brushVizMat);
brushViz.visible = false; brushViz.renderOrder = 999;
scene.add(brushViz);
function showBrush(pt, nm) { brushViz.position.copy(pt); brushViz.scale.setScalar(brushRadius); if(nm){const t=pt.clone().add(nm);brushViz.lookAt(t);} brushViz.visible=true; }
function hideBrush() { brushViz.visible = false; }

// Gesture hover/select visuals (post-placement sculpt intent)
const hoverVizMat = new THREE.MeshBasicMaterial({ color: 0xfff08a, transparent: true, opacity: 0.95, side: THREE.DoubleSide, depthTest: false, toneMapped: false });
const hoverViz = new THREE.Mesh(new THREE.RingGeometry(0.90, 1.0, 48), hoverVizMat);
hoverViz.visible = false;
hoverViz.renderOrder = 1000;
scene.add(hoverViz);

const gestureSelectRings = [];
const selectRingScales = [0.88, 1.12, 1.38, 1.68];
for (let i = 0; i < selectRingScales.length; i++) {
    const m = new THREE.MeshBasicMaterial({ color: 0xfff3a1, transparent: true, opacity: Math.max(0.12, 0.38 - i * 0.07), side: THREE.DoubleSide, depthTest: false, toneMapped: false });
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.90, 1.0, 48), m);
    ring.visible = false;
    ring.renderOrder = 1001 + i;
    scene.add(ring);
    gestureSelectRings.push(ring);
}

const gestureSelectionGlow = new THREE.Mesh(
    new THREE.CircleGeometry(1.0, 48),
    new THREE.MeshBasicMaterial({ color: 0xfff4b8, transparent: true, opacity: 0.12, side: THREE.DoubleSide, depthTest: false, toneMapped: false })
);
gestureSelectionGlow.visible = false;
gestureSelectionGlow.renderOrder = 1000;
scene.add(gestureSelectionGlow);

const gestureSelection = selectionState;
let pinchSelectLatch = false;
let pinchSculptLastZ = null;
let pinchSculptUndoArmed = false;

function orientRingAt(ring, pt, nm, scale) {
    ring.position.copy(pt);
    ring.scale.setScalar(scale);
    const look = pt.clone().add(nm);
    ring.lookAt(look);
    ring.visible = true;
}

function showGestureHover(ptLocal, nmLocal, radius) {
    const worldPt = ptLocal.clone().applyMatrix4(mesh.matrixWorld);
    const worldNormal = nmLocal.clone().transformDirection(mesh.matrixWorld).normalize();
    orientRingAt(hoverViz, worldPt, worldNormal, radius);
}

function hideGestureHover() {
    hoverViz.visible = false;
}

function updateGestureSelectionViz() {
    if (!gestureSelection.active) {
        gestureSelectRings.forEach(r => r.visible = false);
        gestureSelectionGlow.visible = false;
        if (selectionReadout) selectionReadout.textContent = 'Patch: none. Click or pinch a surface to select a region.';
        return;
    }
    const worldPt = gestureSelection.pointLocal.clone().applyMatrix4(mesh.matrixWorld);
    const worldNormal = gestureSelection.normalLocal.clone().transformDirection(mesh.matrixWorld).normalize();

    gestureSelectionGlow.position.copy(worldPt);
    const pulseActive = performance.now() < gestureSelection.pulseUntil;
    const pulseScale = pulseActive ? 1.16 + Math.sin(performance.now() * 0.03) * 0.03 : 1.05;
    gestureSelectionGlow.scale.setScalar(gestureSelection.radius * pulseScale);
    gestureSelectionGlow.material.opacity = pulseActive ? 0.20 : 0.12;
    gestureSelectionGlow.lookAt(worldPt.clone().add(worldNormal));
    gestureSelectionGlow.visible = true;

    for (let i = 0; i < gestureSelectRings.length; i++) {
        orientRingAt(
            gestureSelectRings[i],
            worldPt,
            worldNormal,
            gestureSelection.radius * selectRingScales[i],
        );
    }
    if (selectionReadout) {
        selectionReadout.textContent = `Patch: ${gestureSelection.logicalVertexIds.size} verts · boundary ${gestureSelection.boundaryVertexIds.size} · ${interactionState.mode === 'detail' ? 'detail/bevel ready' : interactionState.mode === 'form' ? 'form/extrude ready' : 'mold ready'}`;
    }
}

function clearGestureSelection() {
    gestureSelection.active = false;
    gestureSelection.kind = 'none';
    gestureSelection.logicalVertexIds = new Set();
    gestureSelection.boundaryVertexIds = new Set();
    pinchSelectLatch = false;
    pinchSculptLastZ = null;
    pinchSculptUndoArmed = false;
    gestureSelection.pulseUntil = 0;
    hideGestureHover();
    gestureSelectionGlow.visible = false;
    gestureSelectRings.forEach(r => r.visible = false);
    if (selectionReadout) selectionReadout.textContent = 'Patch: none. Click or pinch a surface to select a region.';
}

function logicalVertexFromHit(hit) {
    if (!hit?.face) return null;
    const faceIndices = [hit.face.a, hit.face.b, hit.face.c].filter(idx => idx !== undefined);
    if (!faceIndices.length) return null;
    let best = topologyCache.bufferToLogical[faceIndices[0]];
    let bestDist = Infinity;
    const p = mesh.geometry.attributes.position;
    const v = new THREE.Vector3();
    for (const idx of faceIndices) {
        v.fromBufferAttribute(p, idx);
        const d = v.distanceToSquared(hit.point.clone().applyMatrix4(mesh.matrixWorld.clone().invert()));
        if (d < bestDist) {
            bestDist = d;
            best = topologyCache.bufferToLogical[idx];
        }
    }
    return best ?? null;
}

const SelectionController = {
    clear() {
        clearGestureSelection();
    },
    selectHit(hit, radius = Math.max(0.34, Math.min(1.35, brushRadius * 1.8))) {
        if (!hit) return;
        const inv = mesh.matrixWorld.clone().invert();
        const pointLocal = hit.point.clone().applyMatrix4(inv);
        const normalLocal = getHitNormal(hit).transformDirection(inv).normalize();
        const seedLogicalId = logicalVertexFromHit(hit);
        const selected = new Set();
        const queue = [];
        if (seedLogicalId !== null) queue.push(seedLogicalId);
        const radius2 = radius * radius * 1.2;
        while (queue.length) {
            const logicalId = queue.shift();
            if (selected.has(logicalId)) continue;
            const logical = topologyCache.logicalVertices[logicalId];
            if (!logical) continue;
            if (logical.position.distanceToSquared(pointLocal) > radius2) continue;
            selected.add(logicalId);
            for (const neighborId of topologyCache.adjacency[logicalId] || []) {
                if (!selected.has(neighborId)) queue.push(neighborId);
            }
        }
        if (!selected.size) return;
        gestureSelection.active = true;
        gestureSelection.kind = 'region';
        gestureSelection.pointLocal.copy(pointLocal);
        gestureSelection.normalLocal.copy(normalLocal);
        gestureSelection.radius = radius;
        gestureSelection.logicalVertexIds = selected;
        gestureSelection.boundaryVertexIds = new Set();
        gestureSelection.pulseUntil = performance.now() + 220;
        refreshSelectionFromTopology();
        updateGestureSelectionViz();
    },
    refreshFromGeometry() {
        if (!gestureSelection.active) return;
        refreshSelectionFromTopology();
        updateGestureSelectionViz();
    },
};

function selectGestureRegion(hit) {
    SelectionController.selectHit(hit);
}

function syncSelectionAfterGeometryChange(refreshEdges = false) {
    mesh.geometry.computeVertexNormals();
    rebuildTopologyCache();
    if (gestureSelection.active) SelectionController.refreshFromGeometry();
    if (refreshEdges) refreshWireframe();
}

function snapScalar(value, step) {
    if (!step) return value;
    return Math.round(value / step) * step;
}

function signedDominantAxis(vec) {
    const absX = Math.abs(vec.x);
    const absY = Math.abs(vec.y);
    const absZ = Math.abs(vec.z);
    if (absX >= absY && absX >= absZ) return new THREE.Vector3(Math.sign(vec.x) || 1, 0, 0);
    if (absY >= absX && absY >= absZ) return new THREE.Vector3(0, Math.sign(vec.y) || 1, 0);
    return new THREE.Vector3(0, 0, Math.sign(vec.z) || 1);
}

function getSnappedSelectionNormal(mode = interactionState.mode) {
    const base = gestureSelection.normalLocal.clone().normalize();
    const snapped = signedDominantAxis(base);
    const blend = mode === 'detail'
        ? GESTURE_TUNING.detailNormalSnap
        : mode === 'form'
            ? GESTURE_TUNING.formNormalSnap
            : GESTURE_TUNING.sculptNormalSnap;
    return base.lerp(snapped, blend).normalize();
}

function getSelectionFrame(mode = interactionState.mode) {
    const normal = getSnappedSelectionNormal(mode);
    const cameraRight = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0).normalize();
    const tangentX = cameraRight.clone().sub(normal.clone().multiplyScalar(cameraRight.dot(normal)));
    if (tangentX.lengthSq() < 1e-6) tangentX.copy(new THREE.Vector3(1, 0, 0).sub(normal.clone().multiplyScalar(normal.x)));
    tangentX.normalize();
    const tangentY = new THREE.Vector3().crossVectors(normal, tangentX).normalize();
    return { normal, tangentX, tangentY };
}

function regularizeSelectedPatch({ planarStrength = 0.6, quantizeTangent = false, tangentStep = GESTURE_TUNING.dragStep, boundaryStrength = 0.4, mode = interactionState.mode } = {}) {
    if (!gestureSelection.active || !gestureSelection.logicalVertexIds.size) return;
    const p = mesh.geometry.attributes.position;
    const { normal, tangentX, tangentY } = getSelectionFrame(mode);
    let avgDepth = 0;
    const ids = [...gestureSelection.logicalVertexIds];
    for (const logicalId of ids) {
        const logical = topologyCache.logicalVertices[logicalId];
        if (!logical) continue;
        avgDepth += logical.position.dot(normal);
    }
    avgDepth /= Math.max(ids.length, 1);

    for (const logicalId of ids) {
        const logical = topologyCache.logicalVertices[logicalId];
        if (!logical) continue;
        const offset = logical.position.clone().sub(gestureSelection.pointLocal);
        const tx = offset.dot(tangentX);
        const ty = offset.dot(tangentY);
        const quantizedTx = quantizeTangent ? snapScalar(tx, tangentStep) : tx;
        const quantizedTy = quantizeTangent ? snapScalar(ty, tangentStep) : ty;
        const target = gestureSelection.pointLocal.clone()
            .addScaledVector(tangentX, quantizedTx)
            .addScaledVector(tangentY, quantizedTy)
            .addScaledVector(normal, avgDepth - gestureSelection.pointLocal.dot(normal));
        const strength = gestureSelection.boundaryVertexIds.has(logicalId) ? planarStrength * boundaryStrength : planarStrength;
        setLogicalVertexPosition(logicalId, logical.position.clone().lerp(target, strength));
    }
    p.needsUpdate = true;
}

function normalizeOperationDelta(id, rawDelta) {
    if (id === 'bevel-lite') {
        if (Math.abs(rawDelta) < GESTURE_TUNING.bevelDepthDeadZone) return 0;
        return snapScalar(THREE.MathUtils.clamp(rawDelta * GESTURE_TUNING.bevelDepthScale, -0.045, 0.045), GESTURE_TUNING.bevelDepthStep);
    }
    if (id === 'extrude-lite') {
        if (Math.abs(rawDelta) < GESTURE_TUNING.formDepthDeadZone) return 0;
        return snapScalar(THREE.MathUtils.clamp(rawDelta * GESTURE_TUNING.formDepthScale, -0.06, 0.06), GESTURE_TUNING.formDepthStep);
    }
    return rawDelta;
}

const OperationController = {
    begin(id, meta = {}) {
        if (!gestureSelection.active || !gestureSelection.logicalVertexIds.size) return false;
        if (!interactionState.activeOperation || interactionState.activeOperation.id !== id) {
            saveUndo();
            interactionState.activeOperation = { id, lastValue: 0, meta };
        }
        return true;
    },
    update(nextValue) {
        const op = interactionState.activeOperation;
        if (!op) return;
        const delta = normalizeOperationDelta(op.id, nextValue - op.lastValue);
        if (Math.abs(delta) < 0.0005) return;
        if (op.id === 'extrude-lite') extrudeSelectedPatch(delta);
        if (op.id === 'bevel-lite') bevelSelectedPatch(delta);
        op.lastValue += delta;
        syncSelectionAfterGeometryChange(false);
    },
    commit() {
        if (!interactionState.activeOperation) return;
        syncSelectionAfterGeometryChange(true);
        setCommandStatus(interactionState.activeOperation.id === 'bevel-lite' ? 'Selection beveled' : 'Selection extruded');
        interactionState.activeOperation = null;
    },
    cancel() {
        if (!interactionState.activeOperation) return;
        undo();
        interactionState.activeOperation = null;
        SelectionController.refreshFromGeometry();
    },
    nudge(id, amount) {
        if (!this.begin(id, { source: 'button' })) return;
        this.update(amount);
        this.commit();
    },
};

function sculptSelectedRegion(pinchZ) {
    if (!gestureSelection.active) return;
    if (interactionState.mode === 'form' || interactionState.mode === 'detail') {
        const opId = interactionState.mode === 'detail' ? 'bevel-lite' : 'extrude-lite';
        if (!OperationController.begin(opId, { source: 'gesture' })) return;
        if (pinchSculptLastZ !== null) {
            const dz = pinchZ - pinchSculptLastZ;
            const nextValue = interactionState.activeOperation.lastValue + dz * 2.1;
            OperationController.update(nextValue);
        }
        pinchSculptLastZ = pinchZ;
        return;
    }
    if (!pinchSculptUndoArmed) {
        saveUndo();
        pinchSculptUndoArmed = true;
    }
    if (pinchSculptLastZ !== null) {
        const dz = pinchZ - pinchSculptLastZ;
        if (Math.abs(dz) < GESTURE_TUNING.sculptDepthDeadZone) {
            pinchSculptLastZ = pinchZ;
            return;
        }
        const pushPull = snapScalar(Math.max(-0.045, Math.min(0.045, dz * GESTURE_TUNING.sculptDepthScale)), GESTURE_TUNING.sculptDepthStep);
        if (Math.abs(pushPull) > 0.0008) {
            extrudeSelectedPatch(pushPull * 0.55);
            regularizeSelectedPatch({ planarStrength: 0.46, quantizeTangent: true, tangentStep: GESTURE_TUNING.dragStep, boundaryStrength: 0.55, mode: 'sculpt' });
            syncSelectionAfterGeometryChange(false);
        }
    }
    pinchSculptLastZ = pinchZ;
}

// Sculpt functions — reusable temps
const _v = new THREE.Vector3(), _lp = new THREE.Vector3(), _ln = new THREE.Vector3(), _n = new THREE.Vector3();

function getHitNormal(hit) {
    if (hit?.face?.normal) return hit.face.normal.clone();
    if (hit?.normal) return hit.normal.clone();
    return new THREE.Vector3(0, 0, 1);
}

function sculptAt(point, normal, strength) {
    const p = mesh.geometry.attributes.position;
    const inv = mesh.matrixWorld.clone().invert();
    _lp.copy(point).applyMatrix4(inv); _ln.copy(normal).transformDirection(inv).normalize();
    const r2 = brushRadius*brushRadius; let mod=false;
    for (let i=0;i<p.count;i++) {
        _v.fromBufferAttribute(p,i);
        const dx=_v.x-_lp.x, dy=_v.y-_lp.y, dz=_v.z-_lp.z, d2=dx*dx+dy*dy+dz*dz;
        if (d2<r2) { const f=0.5*(1+Math.cos(Math.PI*Math.sqrt(d2)/brushRadius)); p.setXYZ(i,_v.x+_ln.x*strength*f,_v.y+_ln.y*strength*f,_v.z+_ln.z*strength*f); mod=true; }
    }
    if(mod){p.needsUpdate=true;mesh.geometry.computeVertexNormals();}
}

function smoothAt(point) {
    const p=mesh.geometry.attributes.position;
    const inv=mesh.matrixWorld.clone().invert();
    _lp.copy(point).applyMatrix4(inv);
    const r2=brushRadius*brushRadius, aff=[];
    for(let i=0;i<p.count;i++){_v.fromBufferAttribute(p,i);const dx=_v.x-_lp.x,dy=_v.y-_lp.y,dz=_v.z-_lp.z,d2=dx*dx+dy*dy+dz*dz;if(d2<r2)aff.push({i,x:_v.x,y:_v.y,z:_v.z,d:Math.sqrt(d2)});}
    if(aff.length<2)return;
    let ax=0,ay=0,az=0;for(const a of aff){ax+=a.x;ay+=a.y;az+=a.z;}ax/=aff.length;ay/=aff.length;az/=aff.length;
    for(const a of aff){const f=0.4*(0.5*(1+Math.cos(Math.PI*a.d/brushRadius)));p.setXYZ(a.i,a.x+(ax-a.x)*f,a.y+(ay-a.y)*f,a.z+(az-a.z)*f);}
    p.needsUpdate=true;mesh.geometry.computeVertexNormals();
}

function flattenAt(point, normal) {
    const p=mesh.geometry.attributes.position;
    const inv=mesh.matrixWorld.clone().invert();
    _lp.copy(point).applyMatrix4(inv);_ln.copy(normal).transformDirection(inv).normalize();
    const r2=brushRadius*brushRadius;
    for(let i=0;i<p.count;i++){_v.fromBufferAttribute(p,i);const dx=_v.x-_lp.x,dy=_v.y-_lp.y,dz=_v.z-_lp.z,d2=dx*dx+dy*dy+dz*dz;
    if(d2<r2){const d=Math.sqrt(d2),f=0.5*(0.5*(1+Math.cos(Math.PI*d/brushRadius))),dot=dx*_ln.x+dy*_ln.y+dz*_ln.z;p.setXYZ(i,_v.x-_ln.x*dot*f,_v.y-_ln.y*dot*f,_v.z-_ln.z*dot*f);}}
    p.needsUpdate=true;mesh.geometry.computeVertexNormals();
}

function inflateAt(point, strength) {
    const p=mesh.geometry.attributes.position,nm=mesh.geometry.attributes.normal;
    const inv=mesh.matrixWorld.clone().invert();
    _lp.copy(point).applyMatrix4(inv);const r2=brushRadius*brushRadius;
    for(let i=0;i<p.count;i++){_v.fromBufferAttribute(p,i);const dx=_v.x-_lp.x,dy=_v.y-_lp.y,dz=_v.z-_lp.z,d2=dx*dx+dy*dy+dz*dz;
    if(d2<r2){_n.fromBufferAttribute(nm,i).normalize();const f=0.5*(1+Math.cos(Math.PI*Math.sqrt(d2)/brushRadius));p.setXYZ(i,_v.x+_n.x*strength*f,_v.y+_n.y*strength*f,_v.z+_n.z*strength*f);}}
    p.needsUpdate=true;mesh.geometry.computeVertexNormals();
}

function grabAt(point, dx, dy) {
    const p=mesh.geometry.attributes.position;
    const inv=mesh.matrixWorld.clone().invert();
    _lp.copy(point).applyMatrix4(inv);
    // Convert screen delta to world delta
    const cam = camera.position.clone();
    const right = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0).normalize();
    const up = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 1).normalize();
    const worldDx = right.multiplyScalar(dx * 0.003).add(up.multiplyScalar(-dy * 0.003));
    const localDx = worldDx.transformDirection(inv);
    const r2=brushRadius*brushRadius;
    for(let i=0;i<p.count;i++){_v.fromBufferAttribute(p,i);const ddx=_v.x-_lp.x,ddy=_v.y-_lp.y,ddz=_v.z-_lp.z,d2=ddx*ddx+ddy*ddy+ddz*ddz;
    if(d2<r2){const f=0.5*(1+Math.cos(Math.PI*Math.sqrt(d2)/brushRadius));p.setXYZ(i,_v.x+localDx.x*f,_v.y+localDx.y*f,_v.z+localDx.z*f);}}
    p.needsUpdate=true;mesh.geometry.computeVertexNormals();
}

function creaseAt(point, normal, strength) {
    // Moves vertices toward the stroke line (creates a sharp ridge/valley)
    const p=mesh.geometry.attributes.position;
    const inv=mesh.matrixWorld.clone().invert();
    _lp.copy(point).applyMatrix4(inv);_ln.copy(normal).transformDirection(inv).normalize();
    const r2=brushRadius*brushRadius;
    for(let i=0;i<p.count;i++){_v.fromBufferAttribute(p,i);const dx=_v.x-_lp.x,dy=_v.y-_lp.y,dz=_v.z-_lp.z,d2=dx*dx+dy*dy+dz*dz;
    if(d2<r2){const d=Math.sqrt(d2),f=0.5*(1+Math.cos(Math.PI*d/brushRadius));
    // Pull toward center line + push along normal
    const toCenter_x=_lp.x-_v.x,toCenter_y=_lp.y-_v.y,toCenter_z=_lp.z-_v.z;
    const proj=toCenter_x*_ln.x+toCenter_y*_ln.y+toCenter_z*_ln.z;
    const tangent_x=toCenter_x-_ln.x*proj,tangent_y=toCenter_y-_ln.y*proj,tangent_z=toCenter_z-_ln.z*proj;
    p.setXYZ(i,_v.x+tangent_x*strength*f*0.5+_ln.x*strength*f,_v.y+tangent_y*strength*f*0.5+_ln.y*strength*f,_v.z+tangent_z*strength*f*0.5+_ln.z*strength*f);}}
    p.needsUpdate=true;mesh.geometry.computeVertexNormals();
}

function pinchAt(point) {
    // Pulls vertices toward the brush center
    const p=mesh.geometry.attributes.position;
    const inv=mesh.matrixWorld.clone().invert();
    _lp.copy(point).applyMatrix4(inv);const r2=brushRadius*brushRadius;
    for(let i=0;i<p.count;i++){_v.fromBufferAttribute(p,i);const dx=_v.x-_lp.x,dy=_v.y-_lp.y,dz=_v.z-_lp.z,d2=dx*dx+dy*dy+dz*dz;
    if(d2<r2){const d=Math.sqrt(d2),f=sculptStrength*0.3*(0.5*(1+Math.cos(Math.PI*d/brushRadius)));
    p.setXYZ(i,_v.x-dx*f,_v.y-dy*f,_v.z-dz*f);}}
    p.needsUpdate=true;mesh.geometry.computeVertexNormals();
}

function applyTool(hit, dx, dy) {
    const hitNormal = getHitNormal(hit);
    switch(activeTool) {
        case 'pull': sculptAt(hit.point, hitNormal, sculptStrength); break;
        case 'push': sculptAt(hit.point, hitNormal, -sculptStrength); break;
        case 'smooth': smoothAt(hit.point); break;
        case 'flatten': flattenAt(hit.point, hitNormal); break;
        case 'inflate': inflateAt(hit.point, sculptStrength); break;
        case 'grab': grabAt(hit.point, dx||0, dy||0); break;
        case 'crease': creaseAt(hit.point, hitNormal, sculptStrength); break;
        case 'pinch': pinchAt(hit.point); break;
    }
    if (symmetryX) {
        const inv=mesh.matrixWorld.clone().invert();
        const lp=hit.point.clone().applyMatrix4(inv); lp.x=-lp.x;
        const mp=lp.applyMatrix4(mesh.matrixWorld);
        const mn=getHitNormal(hit); mn.x=-mn.x;
        raycaster.set(mp.clone().addScaledVector(mn,0.5), mn.clone().negate());
        const mh=raycaster.intersectObject(mesh);
        if(mh.length) {
            const m=mh[0];
            const mNormal = getHitNormal(m);
            switch(activeTool) {
                case 'pull': sculptAt(m.point,mNormal,sculptStrength); break;
                case 'push': sculptAt(m.point,mNormal,-sculptStrength); break;
                case 'smooth': smoothAt(m.point); break;
                case 'flatten': flattenAt(m.point,mNormal); break;
                case 'inflate': inflateAt(m.point,sculptStrength); break;
                case 'grab': grabAt(m.point,dx?-dx:0,dy||0); break;
                case 'crease': creaseAt(m.point,mNormal,sculptStrength); break;
                case 'pinch': pinchAt(m.point); break;
            }
        }
    }
}

function applyToolToSelection(dx = 0, dy = 0) {
    if (!gestureSelection.active) return false;
    const worldPoint = gestureSelection.pointLocal.clone().applyMatrix4(mesh.matrixWorld);
    const worldNormal = gestureSelection.normalLocal.clone().transformDirection(mesh.matrixWorld).normalize();
    const oldR = brushRadius;
    brushRadius = gestureSelection.radius;
    switch (activeTool) {
        case 'pull': sculptAt(worldPoint, worldNormal, sculptStrength); break;
        case 'push': sculptAt(worldPoint, worldNormal, -sculptStrength); break;
        case 'smooth': smoothAt(worldPoint); break;
        case 'flatten': flattenAt(worldPoint, worldNormal); break;
        case 'inflate': inflateAt(worldPoint, sculptStrength); break;
        case 'grab': moveSelectedPatchByScreenDelta(dx, dy); break;
        case 'crease': creaseAt(worldPoint, worldNormal, sculptStrength); break;
        case 'pinch': pinchAt(worldPoint); break;
        default:
            brushRadius = oldR;
            return false;
    }
    brushRadius = oldR;
    syncSelectionAfterGeometryChange(false);
    return true;
}

function setLogicalVertexPosition(logicalId, nextPosition) {
    const p = mesh.geometry.attributes.position;
    const logical = topologyCache.logicalVertices[logicalId];
    if (!logical) return;
    for (const bufferIndex of logical.bufferIndices) {
        p.setXYZ(bufferIndex, nextPosition.x, nextPosition.y, nextPosition.z);
    }
}

function getSelectionWeight(logicalId) {
    const logical = topologyCache.logicalVertices[logicalId];
    if (!logical) return 0;
    const dist = logical.position.distanceTo(gestureSelection.pointLocal);
    const radial = THREE.MathUtils.clamp(1 - dist / Math.max(gestureSelection.radius, 0.001), 0, 1);
    let weight = 0.22 + radial * 0.78;
    if (gestureSelection.boundaryVertexIds.has(logicalId)) weight *= 0.42;
    return weight;
}

function softenSelectionBoundary(amount = 0.18) {
    if (!gestureSelection.boundaryVertexIds.size) return;
    const updates = [];
    for (const logicalId of gestureSelection.boundaryVertexIds) {
        const logical = topologyCache.logicalVertices[logicalId];
        if (!logical) continue;
        const neighbors = [...(topologyCache.adjacency[logicalId] || [])];
        if (!neighbors.length) continue;
        const avg = new THREE.Vector3();
        let count = 0;
        for (const neighborId of neighbors) {
            const neighbor = topologyCache.logicalVertices[neighborId];
            if (!neighbor) continue;
            avg.add(neighbor.position);
            count++;
        }
        if (!count) continue;
        avg.multiplyScalar(1 / count);
        updates.push([logicalId, logical.position.clone().lerp(avg, amount)]);
    }
    const p = mesh.geometry.attributes.position;
    for (const [logicalId, nextPosition] of updates) {
        setLogicalVertexPosition(logicalId, nextPosition);
    }
    p.needsUpdate = true;
}

function extrudeSelectedPatch(amount) {
    const p = mesh.geometry.attributes.position;
    const moveNormal = getSnappedSelectionNormal('form');
    for (const logicalId of gestureSelection.logicalVertexIds) {
        const logical = topologyCache.logicalVertices[logicalId];
        if (!logical) continue;
        const weight = gestureSelection.boundaryVertexIds.has(logicalId) ? 0.58 : 1.0;
        const nextPosition = logical.position.clone().addScaledVector(moveNormal, amount * weight * 1.05);
        setLogicalVertexPosition(logicalId, nextPosition);
    }
    p.needsUpdate = true;
    regularizeSelectedPatch({ planarStrength: 0.84, quantizeTangent: true, tangentStep: GESTURE_TUNING.formDepthStep, boundaryStrength: 0.52, mode: 'form' });
    softenSelectionBoundary(0.16);
}

function bevelSelectedPatch(amount) {
    const p = mesh.geometry.attributes.position;
    const center = gestureSelection.pointLocal.clone();
    const { normal, tangentX, tangentY } = getSelectionFrame('detail');
    for (const logicalId of gestureSelection.logicalVertexIds) {
        const logical = topologyCache.logicalVertices[logicalId];
        if (!logical) continue;
        const weight = gestureSelection.boundaryVertexIds.has(logicalId) ? 0.62 : 1.0;
        const offset = logical.position.clone().sub(center);
        const tx = snapScalar(offset.dot(tangentX), GESTURE_TUNING.bevelDepthStep);
        const ty = snapScalar(offset.dot(tangentY), GESTURE_TUNING.bevelDepthStep);
        const tangent = tangentX.clone().multiplyScalar(tx).addScaledVector(tangentY, ty);
        const tangentShrink = tangent.multiplyScalar(-amount * weight * 0.32);
        const lift = normal.clone().multiplyScalar(amount * weight * 0.66);
        const nextPosition = logical.position.clone().add(tangentShrink).add(lift);
        setLogicalVertexPosition(logicalId, nextPosition);
    }
    p.needsUpdate = true;
    regularizeSelectedPatch({ planarStrength: 0.9, quantizeTangent: true, tangentStep: GESTURE_TUNING.bevelDepthStep, boundaryStrength: 0.58, mode: 'detail' });
    softenSelectionBoundary(0.26);
}

function moveSelectedPatchByScreenDelta(dx, dy) {
    if (Math.abs(dx) < GESTURE_TUNING.dragPixelDeadZone && Math.abs(dy) < GESTURE_TUNING.dragPixelDeadZone) return;
    const p = mesh.geometry.attributes.position;
    const inv = mesh.matrixWorld.clone().invert();
    const { tangentX, tangentY } = getSelectionFrame(interactionState.mode === 'detail' ? 'detail' : interactionState.mode === 'form' ? 'form' : 'sculpt');
    let mx = dx * GESTURE_TUNING.dragScale;
    let my = -dy * GESTURE_TUNING.dragScale;
    if (Math.abs(mx) > Math.abs(my) * GESTURE_TUNING.axisBiasRatio) my = 0;
    else if (Math.abs(my) > Math.abs(mx) * GESTURE_TUNING.axisBiasRatio) mx = 0;
    mx = snapScalar(mx, GESTURE_TUNING.dragStep);
    my = snapScalar(my, GESTURE_TUNING.dragStep);
    const worldDelta = tangentX.clone().multiplyScalar(mx).addScaledVector(tangentY, my);
    const localDelta = worldDelta.transformDirection(inv);
    for (const logicalId of gestureSelection.logicalVertexIds) {
        const logical = topologyCache.logicalVertices[logicalId];
        if (!logical) continue;
        const weight = gestureSelection.boundaryVertexIds.has(logicalId) ? 0.64 : 0.94;
        const nextPosition = logical.position.clone().addScaledVector(localDelta, weight);
        setLogicalVertexPosition(logicalId, nextPosition);
    }
    p.needsUpdate = true;
    regularizeSelectedPatch({
        planarStrength: interactionState.mode === 'sculpt' ? 0.42 : 0.68,
        quantizeTangent: interactionState.mode !== 'sculpt',
        tangentStep: GESTURE_TUNING.dragStep,
        boundaryStrength: 0.54,
        mode: interactionState.mode,
    });
    softenSelectionBoundary(0.12);
}

function beginDuplicatePlacement() {
    if (!interactionState.lastAddSpec) return false;
    setInteractionMode('add', 'ui');
    startPlacement(interactionState.lastAddSpec.shape, interactionState.lastAddSpec);
    setCommandStatus(`Duplicate ready: ${SHAPE_LABELS[interactionState.lastAddSpec.shape] || interactionState.lastAddSpec.shape}`);
    return true;
}

// ============================================================
// 4. MEDIAPIPE
// ============================================================
const video = document.getElementById('webcam');
let handLandmarker, lastVideoTime = -1;
const debugOverlay=document.getElementById('debug-overlay'), debugCanvas=document.getElementById('debug-canvas');
const debugCtx=debugCanvas.getContext('2d'), debugText=document.getElementById('debug-gesture-text');
const debugCommandText=document.getElementById('debug-command-text');
const drawUtils=new DrawingUtils(debugCtx), debugBtn=document.getElementById('debug-toggle-btn');
let isDebug=false;
debugBtn.addEventListener('click',()=>{isDebug=!isDebug;debugBtn.textContent=isDebug?'🛠 Debug On':'🛠 Debug Off';debugBtn.classList.toggle('active',isDebug);debugOverlay.style.display=isDebug?'block':'none';if(isDebug){debugCanvas.width=innerWidth;debugCanvas.height=innerHeight;}else debugCtx.clearRect(0,0,debugCanvas.width,debugCanvas.height);});

// Hand tracking on/off — useful when user is doing other things (scratching face, on phone, etc.)
const handsBtn = document.getElementById('hands-toggle-btn');
let handsEnabled = true;
handsBtn.addEventListener('click', () => {
    handsEnabled = !handsEnabled;
    handsBtn.textContent = handsEnabled ? '✋ Hands On' : '✋ Hands Off';
    handsBtn.classList.toggle('disabled', !handsEnabled);
    handsBtn.classList.toggle('active', handsEnabled);
    if (!handsEnabled) {
        rotVX = 0; rotVY = 0;
        last2HDist = null;
        lastPalm = [null, null];
        lastCuppedPos = [null,null];
        resetGestureState();
        debugText && (debugText.textContent = 'Hands disabled');
    }
});
handsBtn.classList.add('active');

async function initHands() {
    const v=await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm");
    handLandmarker=await HandLandmarker.createFromOptions(v,{baseOptions:{modelAssetPath:"https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",delegate:"GPU"},runningMode:"VIDEO",numHands:2});
    navigator.mediaDevices.getUserMedia({video:{width:640,height:480}}).then(s=>{video.srcObject=s;video.addEventListener("loadeddata",loop);}).catch(e=>console.error("Camera:",e));
}

window.addEventListener('resize',()=>{renderer.setSize(innerWidth,innerHeight);camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();if(isDebug){debugCanvas.width=innerWidth;debugCanvas.height=innerHeight;}});

// ============================================================
// 5. GESTURES (hands = manipulation only)
// ============================================================
function isOpenPalm(lm){const w=lm[0],t=[8,12,16,20],p=[6,10,14,18];let e=0;for(let i=0;i<4;i++)if(Math.hypot(lm[t[i]].x-w.x,lm[t[i]].y-w.y)>Math.hypot(lm[p[i]].x-w.x,lm[p[i]].y-w.y)*1.15)e++;return e>=3&&Math.hypot(lm[4].x-lm[5].x,lm[4].y-lm[5].y)>0.06;}
function isFist(lm){const w=lm[0],t=[8,12,16,20],p=[6,10,14,18];let c=0;for(let i=0;i<4;i++)if(Math.hypot(lm[t[i]].x-w.x,lm[t[i]].y-w.y)<Math.hypot(lm[p[i]].x-w.x,lm[p[i]].y-w.y))c++;return c>=4;}
function isPinching(lm){return Math.hypot(lm[4].x-lm[8].x,lm[4].y-lm[8].y,lm[4].z-lm[8].z)<0.05&&Math.hypot(lm[3].x-lm[7].x,lm[3].y-lm[7].y,lm[3].z-lm[7].z)<0.12;}
// C-shaped (cupped) hand: fingers curled inward forming a cup, thumb extended outward.
// SUPER PERMISSIVE: anything that's not a flat open palm AND not a fist AND not a pinch.
// This is meant to catch real hand-molding gestures in placement mode.
function isCupped(lm) {
    const w = lm[0];  // wrist
    const t = [8, 12, 16, 20];  // finger tips
    const p = [6, 10, 14, 18];  // finger MCPs (middle knuckles)

    // Check if hand is NOT fully open (not all tips far from wrist)
    let extendedCount = 0;
    for (let i = 0; i < 4; i++) {
        const tipDist = Math.hypot(lm[t[i]].x - w.x, lm[t[i]].y - w.y);
        const mcpDist = Math.hypot(lm[p[i]].x - w.x, lm[p[i]].y - w.y);
        // If tip is far from wrist (more than 1.5x the MCP distance), it's extended
        if (tipDist > mcpDist * 1.5) extendedCount++;
    }

    // Thumb must be somewhat separated (not pinching against index)
    const thumbIndexDist = Math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y, lm[4].z - lm[8].z);
    const notPinching = thumbIndexDist > 0.085;

    // SUPER PERMISSIVE: if NOT all fingers extended AND not pinching → it's a cup
    const isNotFullyOpen = extendedCount <= 1;

    if(isDebug) console.log('isCupped check:', {extendedCount, notPinching, isNotFullyOpen, thumbIndexDist});

    return isNotFullyOpen && notPinching;
}

// Dwell-time gesture confirmation — gesture must persist for N consecutive frames
// before activating, and must be absent for M frames before releasing. This stops
// passive hand motion (face scratching, phone, hand-on-head) from triggering moves.
const CONFIRM_FRAMES = 5;   // ~5 video frames (~150ms at 30fps) of consistent gesture
const RELEASE_FRAMES = 3;   // ~3 frames of inconsistency before deactivation
const gestureState = [
    { active: null, candidate: null, candidateCount: 0, releaseCount: 0 },
    { active: null, candidate: null, candidateCount: 0, releaseCount: 0 },
];
function classifyHand(lm) {
    // Hand must be reasonably positioned in frame center band — ignore hands at top
    // of frame (often near face/hair) or bottom edge (often resting/off-screen).
    const wristY = lm[0].y;
    if (wristY < 0.05 || wristY > 0.95) return null;
    if (isFist(lm))    return 'fist';
    if (isPinching(lm)) return 'pinch';
    if (isCupped(lm) && (placementShape || gestureSelection.active || interactionState.mode === 'sculpt' || interactionState.mode === 'form' || interactionState.mode === 'detail')) return 'cupped';
    if (isOpenPalm(lm)) return 'palm';
    return null; // ambiguous = no gesture
}
function stableGesture(hand, idx) {
    const raw = classifyHand(hand);
    const s = gestureState[idx];
    if(isDebug && raw) console.log(`Hand ${idx} raw gesture:`, raw, 'active:', s.active, 'candidate:', s.candidate, 'candidateCount:', s.candidateCount);
    if (raw && raw === s.active) {
        s.candidate = null;
        s.candidateCount = 0;
        s.releaseCount = 0;
        return s.active;
    }
    if (s.active && raw !== s.active) {
        // Active gesture is no longer detected — count toward release.
        s.releaseCount++;
        // But also start considering the new (or null) candidate for next activation.
        if (raw && raw === s.candidate) s.candidateCount++;
        else { s.candidate = raw; s.candidateCount = raw ? 1 : 0; }
        if (s.releaseCount >= RELEASE_FRAMES) {
            s.active = null;
            s.releaseCount = 0;
        }
        return s.active;
    }
    // No active gesture yet — must reach CONFIRM_FRAMES of the same candidate.
    if (raw && raw === s.candidate) {
        s.candidateCount++;
        if(isDebug) console.log(`Hand ${idx} ${raw} candidate count: ${s.candidateCount}/${CONFIRM_FRAMES}`);
        if (s.candidateCount >= CONFIRM_FRAMES) {
            if(isDebug) console.log(`Hand ${idx} CONFIRMING gesture: ${raw}`);
            s.active = s.candidate;
            s.candidate = null;
            s.candidateCount = 0;
            s.releaseCount = 0;
            return s.active;
        }
    } else {
        s.candidate = raw;
        s.candidateCount = raw ? 1 : 0;
    }
    return null;
}
function resetGestureState() {
    gestureState.forEach(s => { s.active=null; s.candidate=null; s.candidateCount=0; s.releaseCount=0; });
}

function summarizeGestureCommand(gestures, handCount) {
    if (placementShape && (gestures[0] === 'pinch' || gestures[0] === 'cupped')) return `Command: place ${placementShape} (grab & mold)`;
    if (gestureSelection.active && handCount === 1 && gestures[0] === 'pinch') {
        if (interactionState.mode === 'detail') return 'Command: bevel selected patch';
        if (interactionState.mode === 'form') return 'Command: extrude selected patch';
        return 'Command: mold selected patch';
    }
    if (handCount === 2) {
        if (gestures[0] === 'fist' && gestures[1] === 'cupped') return 'Command: freeze + mold (single-side)';
        if (gestures[0] === 'cupped' && gestures[1] === 'fist') return 'Command: freeze + mold (single-side)';
        if (gestures[0] === 'cupped' && gestures[1] === 'cupped') return 'Command: dual-mold (symmetric)';
        if (gestures[0] === 'pinch' && gestures[1] === 'pinch') return 'Command: split / move';
        if (gestures[0] === 'palm' && gestures[1] === 'palm') return 'Command: scale';
    }
    if (handCount === 1) {
        if (gestures[0] === 'cupped') return 'Command: grab & mold';
        if (gestures[0] === 'pinch') return 'Command: move / translate';
        if (gestures[0] === 'palm') return 'Command: rotate';
        if (gestures[0] === 'fist') return 'Command: freeze';
    }
    return 'Command: idle';
}

const GestureInterpreter = {
    translateDeadZone: 0.015,
    scaleDeadZone: 0.005,
    depthDeadZone: 0.006,
    intentForHand(hand, gesture) {
        if (!hand || !gesture) return null;
        const thumb = hand[4];
        const index = hand[8];
        const wrist = hand[0];
        const anchorNdc = new THREE.Vector2(
            -((((thumb.x + index.x) * 0.5) * 2) - 1),
            -((((thumb.y + index.y) * 0.5) * 2) - 1),
        );
        const spread = Math.hypot(thumb.x - index.x, thumb.y - index.y, thumb.z - index.z);
        const depthDelta = (thumb.z + index.z) * 0.5 - wrist.z;
        return { gesture, confidence: 1, handCount: 1, anchorNdc, spread, depthDelta };
    },
    filteredScaleDelta(previousDistance, nextDistance) {
        const delta = previousDistance - nextDistance;
        return Math.abs(delta) < this.scaleDeadZone ? 0 : delta;
    },
    filteredDepthDelta(delta) {
        return Math.abs(delta) < this.depthDeadZone ? 0 : delta;
    },
};

// ============================================================
// 6. SPLIT SYSTEM — two pinches apart splits mesh in half
// ============================================================
let splitActive = false;
let splitLeft = null, splitRight = null;
let splitStartDist = 0, splitOriginX = 0;

function splitMesh(handDist) {
    const pos = mesh.geometry.attributes.position;
    const norm = mesh.geometry.attributes.normal;
    const inv = mesh.matrixWorld.clone().invert();

    // Collect triangles into left (x<0) and right (x>=0) based on triangle centroid
    const leftVerts = [], leftNorms = [], rightVerts = [], rightNorms = [];
    for (let i = 0; i < pos.count; i += 3) {
        const ax=pos.getX(i), ay=pos.getY(i), az=pos.getZ(i);
        const bx=pos.getX(i+1), by=pos.getY(i+1), bz=pos.getZ(i+1);
        const cx=pos.getX(i+2), cy=pos.getY(i+2), cz=pos.getZ(i+2);
        const centX = (ax+bx+cx)/3;

        const target = centX < 0 ? leftVerts : rightVerts;
        const nTarget = centX < 0 ? leftNorms : rightNorms;
        target.push(ax,ay,az, bx,by,bz, cx,cy,cz);
        nTarget.push(norm.getX(i),norm.getY(i),norm.getZ(i),
                     norm.getX(i+1),norm.getY(i+1),norm.getZ(i+1),
                     norm.getX(i+2),norm.getY(i+2),norm.getZ(i+2));
    }

    // Hide original mesh
    mesh.visible = false;

    // Create two half meshes
    function makeHalf(verts, norms) {
        const g = new THREE.BufferGeometry();
        g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(verts), 3));
        g.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(norms), 3));
        g.computeVertexNormals();
        const m = new THREE.Mesh(g, material.clone());
        m.position.copy(mesh.position);
        m.rotation.copy(mesh.rotation);
        m.scale.copy(mesh.scale);
        scene.add(m);
        return m;
    }

    splitLeft = makeHalf(leftVerts, leftNorms);
    splitRight = makeHalf(rightVerts, rightNorms);
    splitStartDist = handDist;
    splitOriginX = mesh.position.x;
    splitActive = true;

    // Immediately offset halves so both are visible
    splitLeft.position.x = splitOriginX - 0.8;
    splitRight.position.x = splitOriginX + 0.8;
}

function finalizeSplit() {
    if (!splitActive) return;
    scene.remove(mesh);
    mesh.geometry.dispose();

    // Right half = active sculpting mesh (blue)
    mesh = splitRight;
    mesh.material = material;
    refreshWireframe();
    rebuildTopologyCache();
    SelectionController.clear();

    // Left half stays in scene as a grey ghost — clickable later
    const leftMat = material.clone();
    leftMat.color.setHex(0xd0d0d0);
    leftMat.opacity = 0.6;
    leftMat.transparent = true;
    splitLeft.material = leftMat;

    splitActive = false;
    splitLeft = null;
    splitRight = null;
}

// ============================================================
// 7. RENDER LOOP — hands for rotate/scale/translate/split
// ============================================================
let lastPalm=[null,null], lastCuppedPos=[null,null], last2HDist=null, rotVX=0, rotVY=0;
let lastFistZ=null;  // for blend-mode depth control via fist forward/back
const _recenterTarget = new THREE.Vector3(0, 0, 0);

function loop() {
    requestAnimationFrame(loop);
    let interacting=false, holding=false;
    // Tracks whether the current frame's gestures are actively positioning/scaling
    // the mesh. When false, the mesh drifts back to (0, 0, 0) so it stays centred
    // on screen instead of stranded wherever the last translate left it.
    let posActive = false;

    if(handLandmarker && handsEnabled && video.currentTime!==lastVideoTime) {
        lastVideoTime=video.currentTime;
        const det=handLandmarker.detectForVideo(video,performance.now());

        // Reset gesture state for hands no longer detected
        if (det.landmarks.length < 2) gestureState[1] = { active: null, candidate: null, candidateCount: 0, releaseCount: 0 };
        if (det.landmarks.length < 1) gestureState[0] = { active: null, candidate: null, candidateCount: 0, releaseCount: 0 };

        // Debug
        if(isDebug){debugCanvas.width=innerWidth;debugCanvas.height=innerHeight;debugCtx.clearRect(0,0,debugCanvas.width,debugCanvas.height);debugCtx.save();debugCtx.translate(debugCanvas.width,0);debugCtx.scale(-1,1);debugCtx.globalAlpha=0.35;debugCtx.drawImage(video,0,0,debugCanvas.width,debugCanvas.height);debugCtx.restore();debugCtx.globalAlpha=1;
        if(det.landmarks.length){let lb='';const activeGestures=[];det.landmarks.forEach((h,i)=>{const m=h.map(l=>({...l,x:1-l.x}));drawUtils.drawConnectors(m,HandLandmarker.HAND_CONNECTIONS,{color:i?'#e66767':'#7b8b6f',lineWidth:3});drawUtils.drawLandmarks(m,{color:i?'#fffaec':'#f8eedb',fillColor:i?'#e66767':'#7b8b6f',radius:4});const g=gestureState[i].active;activeGestures.push(g);if(g==='fist')lb+='✊ Freeze  ';else if(g==='pinch')lb+='🤏 Move  ';else if(g==='cupped')lb+='🖐️ Grab  ';else if(g==='palm')lb+='🤚 Rotate  ';else lb+='⏳ idle  ';});debugText.textContent=lb.trim();debugCommandText.textContent=summarizeGestureCommand(activeGestures, det.landmarks.length);}else {debugText.textContent='No hands';debugCommandText.textContent='Command: idle';}}

        if(det.landmarks.length) {
            // Run stability classifier on each detected hand BEFORE branching
            const gestures = det.landmarks.map((h, i) => stableGesture(h, i));
            adaptModeFromGestures(gestures, det.landmarks.length);
            const anyActive = gestures.some(g => g !== null);
            let hoverHit = null;
            if (anyActive) interacting = true;

            // ============================================================
            // GESTURE-DRIVEN SHAPE PLACEMENT
            // When `placementShape` is set (user clicked ➕):
            // - PINCH: thumb-index pinch for precise placement
            // - CUPPED: C-shaped grab for intuitive molding (spread of all fingers → size)
            //   • finger spread (hand opening within cup) → size
            //   • hand z (forward/back) relative to start → depth
            //   • release gesture → instant commit
            // While placing, all other gestures are suppressed.
            // ============================================================
            if (interactionMode === 'freeze') {
                holding = true;
                lastPalm = [null, null];
                last2HDist = null;
                clearGestureSelection();
            } else if (placementShape && det.landmarks.length >= 1 && (gestures[0] === 'pinch' || gestures[0] === 'cupped')) {
                interacting = true; posActive = true;
                hideGestureHover();
                gestureSelectRings.forEach(r => r.visible = false);
                if(isDebug) console.log('🎯 PLACEMENT ACTIVE - gesture:', gestures[0], 'shape:', placementShape);
                const hand = det.landmarks[0];

                // Get hand center and finger spread differently for pinch vs cupped
                let px, py, pz, fingerDist;
                if (gestures[0] === 'pinch') {
                    // Pinch: use thumb-index midpoint and their distance
                    const thumb = hand[4], index = hand[8];
                    px = (thumb.x + index.x) / 2;
                    py = (thumb.y + index.y) / 2;
                    pz = (thumb.z + index.z) / 2;
                    fingerDist = Math.hypot(thumb.x - index.x, thumb.y - index.y, thumb.z - index.z);
                } else {
                    // Cupped: use palm center (wrist + middle finger middle knuckle average)
                    // and use average finger spread as size control
                    const wrist = hand[0], midFinger = hand[12];
                    px = (wrist.x + midFinger.x) / 2;
                    py = (wrist.y + midFinger.y) / 2;
                    pz = (wrist.z + midFinger.z) / 2;
                    // For cupped, measure spread: distance between thumb tip and pinky tip
                    const thumb = hand[4], pinky = hand[20];
                    fingerDist = Math.hypot(thumb.x - pinky.x, thumb.y - pinky.y, thumb.z - pinky.z);
                }

                // Raycast through the hand center to find the target surface point.
                const ndc = new THREE.Vector2(-(px * 2 - 1), -(py * 2 - 1));
                raycaster.setFromCamera(ndc, camera);
                const hits = raycaster.intersectObject(mesh);
                if(isDebug) console.log('🎯 Raycast hits:', hits.length, 'gesturePlacing:', gesturePlacing);

                if (!gesturePlacing) {
                    // First frame of gesture → spawn preview at the hit (if any).
                    if (hits.length) {
                        if(isDebug) console.log('🎯✨ SPAWNING PREVIEW at hit point');
                        const inv = mesh.matrixWorld.clone().invert();
                        const lp = hits[0].point.clone().applyMatrix4(inv);
                        const ln = getHitNormal(hits[0]).transformDirection(inv).normalize();
                        placementPos = { x: lp.x, y: lp.y, z: lp.z };
                        placementNormal = ln;
                        gesturePinchStartZ = pz;
                        gesturePinchStartDist = fingerDist;
                        smoothedScale = interactionState.placementSpec?.scale ?? 0.3;
                        smoothedDepth = interactionState.placementSpec?.depth ?? 0.5;
                        const previewGeo = SHAPES[placementShape]();
                        if (placementPreview) {
                            scene.remove(placementPreview);
                            placementPreview.geometry.dispose();
                        }
                        placementPreview = new THREE.Mesh(previewGeo, previewMat);
                        placementPreview.scale.setScalar(smoothedScale);
                        scene.add(placementPreview);
                        gesturePlacing = true;
                    }
                } else {
                    // While gesturing: re-raycast to slide along surface (smoothed).
                    if (hits.length) {
                        const inv = mesh.matrixWorld.clone().invert();
                        const lp = hits[0].point.clone().applyMatrix4(inv);
                        const ln = getHitNormal(hits[0]).transformDirection(inv).normalize();
                        // Lerp the local-anchor toward the new hit so the preview slides smoothly.
                        placementPos.x += (lp.x - placementPos.x) * 0.25;
                        placementPos.y += (lp.y - placementPos.y) * 0.25;
                        placementPos.z += (lp.z - placementPos.z) * 0.25;
                        // Normal is also lerped & re-normalized to avoid pops at face boundaries.
                        placementNormal.lerp(ln, 0.25).normalize();
                    }

                    // Size scaling from finger spread
                    if (gestures[0] === 'pinch') {
                        // Pinch: ~0.04 = closed, ~0.30 = spread. Scale factor 5.0
                        const targetScale = Math.max(0.1, Math.min(1.5, fingerDist * 5.0));
                        smoothedScale += (targetScale - smoothedScale) * 0.25;
                    } else {
                        // Cupped: ~0.15 = partially closed, ~0.45 = wide spread. Scale factor 3.0
                        const targetScale = Math.max(0.1, Math.min(1.5, fingerDist * 3.0));
                        smoothedScale += (targetScale - smoothedScale) * 0.25;
                    }

                    // Target depth from hand z relative to gesture-start z.
                    // Smaller z = closer to camera. Hand forward → push out → extrude.
                    const dz = gesturePinchStartZ - pz;       // positive = moved forward
                    const targetDepth = Math.max(-1, Math.min(1, 0.5 + dz * 12.0));
                    smoothedDepth += (targetDepth - smoothedDepth) * 0.20;
                    applyGesturePlacement();
                }
                // Skip the rest of the gesture branches while placing.
                lastPalm = [null, null];
                last2HDist = null;
            } else {
                // Gesture released or no longer in placement mode → commit if we were placing.
                if (gesturePlacing) {
                    if(isDebug) console.log('🎯 Gesture released → committing clean placement');
                    commitGesturePlacement();
                }
                if (interactionState.activeOperation && gestures[0] !== 'pinch') {
                    OperationController.commit();
                }

                // Gesture hover + pinch-once selection (only outside + placement mode).
                if (!placementShape && det.landmarks.length === 1) {
                    const h = det.landmarks[0];
                    const g = gestures[0];
                    const hx = (g === 'pinch') ? (h[4].x + h[8].x) * 0.5 : h[8].x;
                    const hy = (g === 'pinch') ? (h[4].y + h[8].y) * 0.5 : h[8].y;
                    const ndcHover = new THREE.Vector2(-(hx * 2 - 1), -(hy * 2 - 1));
                    raycaster.setFromCamera(ndcHover, camera);
                    const hoverHits = raycaster.intersectObject(mesh);
                    if (hoverHits.length) {
                        hoverHit = hoverHits[0];
                        const r = Math.max(0.34, Math.min(1.4, brushRadius * 1.55));
                        const inv = mesh.matrixWorld.clone().invert();
                        const localPoint = hoverHit.point.clone().applyMatrix4(inv);
                        const localNormal = getHitNormal(hoverHit).transformDirection(inv).normalize();
                        showGestureHover(localPoint, localNormal, r);
                    } else {
                        hideGestureHover();
                    }

                    const pinchNow = (g === 'pinch');
                    if (pinchNow && !pinchSelectLatch && hoverHit) {
                        selectGestureRegion(hoverHit);
                    }
                    if (!pinchNow && !gestureSelection.active) hideGestureHover();
                    pinchSelectLatch = pinchNow;
                } else {
                    hideGestureHover();
                    pinchSelectLatch = false;
                    pinchSculptLastZ = null;
                    pinchSculptUndoArmed = false;
                    if (!gestureSelection.active) gestureSelectRings.forEach(r => r.visible = false);
                }

            // ----- Blend-mode depth control via fist forward/back (single-hand) -----
            if (blendPanel.classList.contains('open') && det.landmarks.length === 1 && gestures[0] === 'fist') {
                // Use wrist z (negative = closer to camera in MediaPipe coords)
                const wz = det.landmarks[0][0].z;
                if (lastFistZ !== null) {
                    const dz = wz - lastFistZ;          // negative = moving forward
                    blendDepth = Math.max(-1, Math.min(1, blendDepth - dz * 6.0));
                    blDepthEl.value = blendDepth;
                    blDepthV.textContent = fmtSigned(blendDepth);
                    updateBlendPreview();
                }
                lastFistZ = wz;
            } else {
                lastFistZ = null;
            }

            // Two hands
            if(det.landmarks.length===2) {
                const h1=det.landmarks[0],h2=det.landmarks[1];
                const g1=gestures[0], g2=gestures[1];
                const d=Math.hypot(h1[9].x-h2[9].x,h1[9].y-h2[9].y);

                // Both open → SCALE
                if(g1==='palm' && g2==='palm' && last2HDist!==null){
                    const dd = GestureInterpreter.filteredScaleDelta(last2HDist, d);
                    if(dd){let s=Math.max(0.2,Math.min(5,mesh.scale.x-dd*0.8));mesh.scale.setScalar(THREE.MathUtils.lerp(mesh.scale.x, s, 0.45));rotVX=0;rotVY=0;}
                    posActive = true;
                }

                // ============= TWO-HAND MOLDING PATTERNS =============

                // Freeze + Cupped: One hand holds (fist), other hand sculpts (cupped)
                // → Single-sided molding: place/sculpt on one region while anchored by freeze hand
                if ((g1 === 'fist' && g2 === 'cupped') || (g1 === 'cupped' && g2 === 'fist')) {
                    rotVX = 0; rotVY = 0;  // Freeze rotation
                    posActive = true;
                    // The cupped hand drives placement/sculpting; fist hand acts as anchor
                    const cuppedHand = (g1 === 'cupped') ? h1 : h2;
                    const wrist = cuppedHand[0], midFinger = cuppedHand[12];
                    const px = (wrist.x + midFinger.x) / 2;
                    const py = (wrist.y + midFinger.y) / 2;
                    const thumb = cuppedHand[4], pinky = cuppedHand[20];
                    const fingerSpread = Math.hypot(thumb.x - pinky.x, thumb.y - pinky.y, thumb.z - pinky.z);

                    // Raycast for sculpting position
                    const ndc = new THREE.Vector2(-(px * 2 - 1), -(py * 2 - 1));
                    raycaster.setFromCamera(ndc, camera);
                    const hits = raycaster.intersectObject(mesh);
                    if (hits.length) {
                        // Motion-driven molding: drag the clay with the cupped hand.
                        const prev = lastCuppedPos[g1 === 'cupped' ? 0 : 1];
                        const oldR = brushRadius;
                        brushRadius = Math.max(0.2, Math.min(1.0, fingerSpread * 2.5));
                        if (prev) {
                            const dx = (px - prev.x) * innerWidth;
                            const dy = (py - prev.y) * innerHeight;
                            if (gestureSelection.active) {
                                moveSelectedPatchByScreenDelta(dx, dy);
                                syncSelectionAfterGeometryChange(false);
                            } else {
                                grabAt(hits[0].point, dx, dy);
                            }
                        } else {
                            smoothAt(hits[0].point);
                        }
                        brushRadius = oldR;
                    }
                    lastCuppedPos[g1 === 'cupped' ? 0 : 1] = { x: px, y: py };
                    // Visual feedback: debug command shows freeze+mold pattern
                }

                // Dual Cupped: Both hands cupped → Symmetric molding on both sides
                // → Multi-region molding: sculpt mirror-symmetric or independent regions
                else if (g1 === 'cupped' && g2 === 'cupped') {
                    rotVX = 0; rotVY = 0;  // Freeze rotation for stability
                    posActive = true;

                    // Both hands drive sculpting on their respective regions
                    for (let h = 0; h < 2; h++) {
                        const hand = (h === 0) ? h1 : h2;
                        const wrist = hand[0], midFinger = hand[12];
                        const px = (wrist.x + midFinger.x) / 2;
                        const py = (wrist.y + midFinger.y) / 2;
                        const thumb = hand[4], pinky = hand[20];
                        const fingerSpread = Math.hypot(thumb.x - pinky.x, thumb.y - pinky.y, thumb.z - pinky.z);

                        const ndc = new THREE.Vector2(-(px * 2 - 1), -(py * 2 - 1));
                        raycaster.setFromCamera(ndc, camera);
                        const hits = raycaster.intersectObject(mesh);
                        if (hits.length) {
                            // Each cupped hand deforms independently based on its motion.
                            const prev = lastCuppedPos[h];
                            const oldR = brushRadius;
                            brushRadius = Math.max(0.2, Math.min(1.0, fingerSpread * 2.5));
                            if (prev) {
                                const dx = (px - prev.x) * innerWidth;
                                const dy = (py - prev.y) * innerHeight;
                                if (gestureSelection.active) {
                                    moveSelectedPatchByScreenDelta(dx, dy);
                                    syncSelectionAfterGeometryChange(false);
                                } else {
                                    grabAt(hits[0].point, dx, dy);
                                }
                            } else {
                                smoothAt(hits[0].point);
                            }
                            brushRadius = oldR;
                        }
                        lastCuppedPos[h] = { x: px, y: py };
                    }
                    // Visual feedback: debug command shows dual-mold pattern
                }

                // Both pinch → TRANSLATE or SPLIT
                else if(g1==='pinch' && g2==='pinch'){
                    if(splitActive){
                        // Move halves apart based on hand distance
                        const spread=(d-splitStartDist)*3.0;
                        if(splitLeft) splitLeft.position.x=splitOriginX-Math.max(0,spread);
                        if(splitRight) splitRight.position.x=splitOriginX+Math.max(0,spread);
                    }
                    else if(last2HDist!==null && d>0.3 && d-last2HDist>0.03) {
                        // Hands spreading apart fast enough → SPLIT
                        saveUndo();
                        splitMesh(d);
                    }
                    else if(!splitActive) {
                        // Normal translate
                        const px=(h1[4].x+h1[8].x+h2[4].x+h2[8].x)/4;
                        const py=(h1[4].y+h1[8].y+h2[4].y+h2[8].y)/4;
                        const intent = GestureInterpreter.intentForHand({ 4: { x: px, y: py, z: 0 }, 8: { x: px, y: py, z: 0 }, 0: { z: 0 } }, 'pinch');
                        const v=new THREE.Vector3(intent.anchorNdc.x,intent.anchorNdc.y,0.5).unproject(camera);
                        v.sub(camera.position).normalize();
                        const t=(mesh.position.z-camera.position.z)/v.z;
                        mesh.position.lerp(camera.position.clone().add(v.multiplyScalar(t)),0.2);
                        rotVX=0;rotVY=0;
                        posActive = true;
                    }
                } else if(splitActive) {
                    // Released pinch while split — finalize
                    finalizeSplit();
                }
                last2HDist=d;
            } else last2HDist=null;

            // Single hand
            det.landmarks.forEach((hand,idx)=>{
                if(idx>0&&det.landmarks.length===2)return;
                if(idx>1)return;
                const palm=hand[9];
                const g = gestures[idx];

                if(g==='fist'){holding=true;lastPalm[idx]=null;return;}
                if(det.landmarks.length===2)return;

                // Single pinch → TRANSLATE
                if(g==='pinch'){
                    const intent = GestureInterpreter.intentForHand(hand, g);
                    const px=(hand[4].x+hand[8].x)/2,py=(hand[4].y+hand[8].y)/2;
                    const pinchZ = (hand[4].z + hand[8].z) * 0.5;

                    if (gestureSelection.active && idx === 0) {
                        sculptSelectedRegion(pinchZ);

                        if (hoverHit) {
                            const inv = mesh.matrixWorld.clone().invert();
                            const nextPoint = hoverHit.point.clone().applyMatrix4(inv);
                            const nextNormal = getHitNormal(hoverHit).transformDirection(inv).normalize();
                            gestureSelection.pointLocal.lerp(nextPoint, 0.22);
                            gestureSelection.normalLocal.lerp(nextNormal, 0.22).normalize();
                        }
                        updateGestureSelectionViz();
                        lastPalm[idx]=null;rotVX=0;rotVY=0;
                        posActive = true;
                    } else {
                        const ndc=intent.anchorNdc;
                        const v=new THREE.Vector3(ndc.x,ndc.y,0.5).unproject(camera);
                        v.sub(camera.position).normalize();
                        const t=(mesh.position.z-camera.position.z)/v.z;
                        mesh.position.lerp(camera.position.clone().add(v.multiplyScalar(t)),0.15);
                        pinchSculptLastZ = null;
                        pinchSculptUndoArmed = false;
                        lastPalm[idx]=null;rotVX=0;rotVY=0;
                        posActive = true;
                    }
                }
                // Single cupped → GENTLE SCULPT / MOLD
                else if(g==='cupped'){
                    // Use cupped hand to sculpt softly (smoothing, molding)
                    const wrist = hand[0], midFinger = hand[12];
                    const px = (wrist.x + midFinger.x) / 2;
                    const py = (wrist.y + midFinger.y) / 2;
                    const thumb = hand[4], pinky = hand[20];
                    const fingerSpread = Math.hypot(thumb.x - pinky.x, thumb.y - pinky.y, thumb.z - pinky.z);

                    const ndc = new THREE.Vector2(-(px * 2 - 1), -(py * 2 - 1));
                    raycaster.setFromCamera(ndc, camera);
                    const hits = raycaster.intersectObject(mesh);
                    if (hits.length) {
                        // Drag the surface with the cupped hand so molding is visible.
                        const prev = lastCuppedPos[idx];
                        const oldR = brushRadius;
                        brushRadius = Math.max(0.3, Math.min(1.2, fingerSpread * 2.5));
                        if (prev) {
                            const dx = (px - prev.x) * innerWidth;
                            const dy = (py - prev.y) * innerHeight;
                            if (gestureSelection.active) {
                                moveSelectedPatchByScreenDelta(dx, dy);
                                syncSelectionAfterGeometryChange(false);
                            } else {
                                grabAt(hits[0].point, dx, dy);
                            }
                        } else {
                            smoothAt(hits[0].point);
                        }
                        brushRadius = oldR;
                    }
                    lastCuppedPos[idx] = { x: px, y: py };
                    lastPalm[idx]=null;rotVX=0;rotVY=0;
                    posActive = true;
                }
                // Open palm → ROTATE
                else if(g==='palm'){
                    pinchSculptLastZ = null;
                    pinchSculptUndoArmed = false;
                    if (interactionState.activeOperation) OperationController.commit();
                    lastCuppedPos[idx] = null;
                    if(lastPalm[idx]){const dx=palm.x-lastPalm[idx].x,dy=palm.y-lastPalm[idx].y;rotVX-=dx*5;rotVY+=dy*5;}
                    lastPalm[idx]=palm;
                } else {
                    pinchSculptLastZ = null;
                    pinchSculptUndoArmed = false;
                    if (interactionState.activeOperation) OperationController.commit();
                    lastPalm[idx]=null; lastCuppedPos[idx]=null;
                    if (gestureSelection.active && g === null) hideGestureHover();
                }
            });
            }  // ← close `else` of placement-mode interception
        } else {
            // No hands detected. If we were mid-placement, commit the current preview
            // so the user doesn't get stuck with a floating ghost shape.
            if (gesturePlacing) {
                gesturePlacing = false;
                commitGesturePlacement();
            }
            if (interactionState.activeOperation) OperationController.commit();
            lastPalm=[null,null];
            lastCuppedPos=[null,null];
            resetGestureState();
        }
    }

    if(holding || interactionMode === 'freeze'){rotVX=0;rotVY=0;}
    mesh.rotation.y+=rotVX;mesh.rotation.x+=rotVY;
    rotVX*=0.85;rotVY*=0.85;
    if(!interacting&&Math.abs(rotVX)<0.01)mesh.rotation.y+=0.002;
    // Drift mesh back to scene origin when not actively translating/scaling, so the
    // object stays centred on screen instead of stranded wherever the last gesture
    // left it. Skipped during split (two separate meshes are being positioned).
    if (!posActive && !splitActive && !mDown) {
        mesh.position.lerp(_recenterTarget, 0.06);
    }
    const tz=Math.max(4,mesh.scale.x*3);camera.position.z+=(tz-camera.position.z)*0.08;
    renderer.render(scene,camera);
}

// ============================================================
// 8. MOUSE SCULPTING
// ============================================================
let mDown=false, mBtn=-1, mLast={x:0,y:0}, mOnMesh=false, mUndoSaved=false;
let mouseOperationStartY = 0;
window.addEventListener('contextmenu',e=>e.preventDefault());

window.addEventListener('mousedown',e=>{
    // Placement mode: click on mesh to start placing.
    if(placementShape&&e.button===0&&!gesturePlacing){
        const nx=(e.clientX/innerWidth)*2-1,ny=-(e.clientY/innerHeight)*2+1;
        raycaster.setFromCamera(new THREE.Vector2(nx,ny),camera);
        const h=raycaster.intersectObject(mesh);
        if(h.length){
            const inv=mesh.matrixWorld.clone().invert();
            const lp=h[0].point.clone().applyMatrix4(inv);
            const ln=getHitNormal(h[0]).transformDirection(inv).normalize();
            placementPos = { x: lp.x, y: lp.y, z: lp.z };
            placementNormal = ln;
            placementStartY = e.clientY;
            placementStartX = e.clientX;
            placementDragging = true;
            blendDepth = interactionState.placementSpec?.depth ?? 0.5;
            blendStrength = interactionState.placementSpec?.blendStrength ?? 0.7;
            // Create preview using the current placement spec.
            const previewGeo = SHAPES[placementShape]();
            placementPreview = new THREE.Mesh(previewGeo, previewMat);
            const initS = interactionState.placementSpec?.scale ?? 0.3;
            const initDepth = interactionState.placementSpec?.depth ?? 0.5;
            const offWorld = getHitNormal(h[0]).normalize().multiplyScalar(initS * initDepth);
            placementPreview.position.copy(h[0].point).add(offWorld);
            placementPreview.scale.setScalar(initS);
            scene.add(placementPreview);
            placementLabel.textContent = `Drag ↑↓ size  ←→ depth  — release to lock`;
        }
        return;
    }
    mDown=true;mBtn=e.button;mUndoSaved=false;mLast={x:e.clientX,y:e.clientY};
    const nx=(e.clientX/innerWidth)*2-1,ny=-(e.clientY/innerHeight)*2+1;
    raycaster.setFromCamera(new THREE.Vector2(nx,ny),camera);
    const hits = raycaster.intersectObject(mesh);
    mOnMesh=hits.length>0;
    if (mOnMesh && mBtn === 0 && (interactionMode === 'form' || interactionMode === 'detail')) {
        SelectionController.selectHit(hits[0], Math.max(0.34, Math.min(1.45, brushRadius * 1.65)));
        mouseOperationStartY = e.clientY;
        const opId = interactionMode === 'detail' ? 'bevel-lite' : 'extrude-lite';
        OperationController.begin(opId, { source: 'mouse' });
        return;
    }
    if(mOnMesh&&mBtn!==1){saveUndo();mUndoSaved=true;}
});

window.addEventListener('mousemove',e=>{
    // Placement drag: resize preview AND adjust depth
    if(placementDragging&&placementPreview){
        updatePreviewFromDrag(e.clientX, e.clientY);
        return;
    }
    const nx=(e.clientX/innerWidth)*2-1,ny=-(e.clientY/innerHeight)*2+1;
    raycaster.setFromCamera(new THREE.Vector2(nx,ny),camera);
    const hits=raycaster.intersectObject(mesh);
    if(!mDown){
        if(hits.length){
            showBrush(hits[0].point,getHitNormal(hits[0]));
            if (interactionMode === 'form' || interactionMode === 'detail') {
                const inv = mesh.matrixWorld.clone().invert();
                showGestureHover(hits[0].point.clone().applyMatrix4(inv), getHitNormal(hits[0]).transformDirection(inv).normalize(), Math.max(0.34, Math.min(1.4, brushRadius * 1.45)));
            }
        } else {
            hideBrush();
            if (!gestureSelection.active) hideGestureHover();
        }
        return;
    }
    const dx=e.clientX-mLast.x,dy=e.clientY-mLast.y;mLast={x:e.clientX,y:e.clientY};
    if (interactionState.activeOperation && (interactionMode === 'form' || interactionMode === 'detail')) {
        const nextValue = (mouseOperationStartY - e.clientY) * 0.006;
        OperationController.update(nextValue);
        return;
    }
    if(!mOnMesh||mBtn===1){rotVX+=dx*0.003;rotVY+=dy*0.003;}
    else if(hits.length){
        if (interactionMode === 'freeze') return;
        if (interactionMode === 'select') {
            rotVX += dx * 0.003;
            rotVY += dy * 0.003;
            return;
        }
        showBrush(hits[0].point,getHitNormal(hits[0]));
        if(e.shiftKey){
            if (gestureSelection.active) applyToolToSelection(dx, dy);
            else smoothAt(hits[0].point);
        }
        else{
            const saved=activeTool;
            if(mBtn===2){if(activeTool==='pull')activeTool='push';else if(activeTool==='push')activeTool='pull';}
            if (gestureSelection.active && interactionMode === 'sculpt') applyToolToSelection(dx, dy);
            else applyTool(hits[0],dx,dy);
            activeTool=saved;
        }
    }
});

window.addEventListener('mouseup',()=>{
    if(placementDragging){
        // Lock and commit immediately on release (no extra panel step).
        placementDragging = false;
        commitPlacementBlended();
        return;
    }
    if (interactionState.activeOperation) OperationController.commit();
    mDown=false;mBtn=-1;mOnMesh=false;mUndoSaved=false;
});
window.addEventListener('wheel',e=>{brushRadius=Math.max(0.1,Math.min(1.5,brushRadius-e.deltaY*0.001));document.getElementById('br').value=brushRadius;document.getElementById('brv').textContent=brushRadius.toFixed(2);});

initHands();

// ============================================================
// 8b. SPLICE-STYLE TOOLBAR WIRING
// ============================================================
let interactionMode = 'select';
let gestureAdaptiveMode = true;
const modeHintEl = document.getElementById('mode-hint');
const modeButtons = Array.from(document.querySelectorAll('.splice-mode-btn'));
const spliceAddBtn = document.getElementById('splice-add-btn');
const spliceDuplicateBtn = document.getElementById('splice-duplicate-btn');
const shapeTrigger = document.getElementById('splice-shape-trigger');
const shapeMenu = document.getElementById('splice-shape-menu');
shapeLabel = document.getElementById('splice-shape-label');
shapeItems = Array.from(document.querySelectorAll('.splice-shape-item'));

const MODE_LABELS = {
    select: 'Navigate',
    sculpt: 'Sculpt',
    form: 'Form',
    detail: 'Detail',
    freeze: 'Freeze',
    add: 'Place',
};
const MODE_ICONS = {
    select: '↖',
    sculpt: '✋',
    form: '✍',
    detail: '◌',
    freeze: '✊',
    add: '▷',
};

const MODE_TOOL = {
    sculpt: 'pull',
    form: 'grab',
    detail: 'crease',
};

function syncShapeUi(name) {
    if (shapeLabel) {
        shapeLabel.textContent = SHAPE_LABELS[name] || name;
    }
    shapeItems.forEach(item => item.classList.toggle('active', item.dataset.shape === name));
    document.querySelectorAll('[data-shape-swap]').forEach(btn => {
        btn.style.borderWidth = btn.dataset.shapeSwap === name ? '3px' : '1.5px';
    });
}

function syncModeUi(source = 'ui') {
    modeButtons.forEach(btn => btn.classList.toggle('active', btn.dataset.mode === interactionMode));
    if (modeHintEl) {
        const locked = performance.now() < modeLockUntil;
        const lead = source === 'gesture'
            ? 'Mode: ' + MODE_LABELS[interactionMode] + ' · by gesture'
            : 'Mode: ' + MODE_LABELS[interactionMode] + (locked ? ' · locked' : ' · gesture adaptive');
        modeHintEl.textContent = lead;
    }
    if (gestureSelection.active) updateGestureSelectionViz();
}

function setInteractionMode(mode, source = 'ui') {
    if (!MODE_LABELS[mode]) return;
    if (source === 'gesture' && performance.now() < modeLockUntil) return;
    if (interactionMode === mode) {
        if (mode === 'add' && !placementShape) startPlacement(currentShape);
        return;
    }
    interactionMode = mode;
    interactionState.mode = mode;
    if (source === 'ui') modeLockUntil = performance.now() + GESTURE_TUNING.modeLockMs;

    if (mode === 'add') {
        if (!placementShape) startPlacement(currentShape);
    } else if (placementShape && !gesturePlacing) {
        cancelPlacement();
    }

    if (MODE_TOOL[mode]) setActiveTool(MODE_TOOL[mode]);
    if (mode === 'freeze') setCommandStatus('Freeze mode: motion locked');
    else setCommandStatus('Mode: ' + MODE_LABELS[mode]);
    syncModeUi(source);
}

modeButtons.forEach(btn => {
    const mode = btn.dataset.mode;
    btn.innerHTML = `<span class="splice-mode-icon">${MODE_ICONS[mode] || ''}</span><span class="splice-mode-name">${MODE_LABELS[mode] || mode}</span>`;
});
interactionState.mode = interactionMode;

function adaptModeFromGestures(gestures, handCount) {
    if (!gestureAdaptiveMode || !handsEnabled || placementShape || blendPanel.classList.contains('open')) return;
    if (handCount === 2 && gestures[0] === 'palm' && gestures[1] === 'palm') {
        setInteractionMode('select', 'gesture');
        return;
    }
    if (gestures.includes('fist')) {
        setInteractionMode('freeze', 'gesture');
        return;
    }
    if (gestures.includes('cupped')) {
        setInteractionMode('sculpt', 'gesture');
        return;
    }
    if (gestures.includes('pinch')) {
        setInteractionMode('form', 'gesture');
        return;
    }
    if (gestures.includes('palm')) {
        setInteractionMode('select', 'gesture');
    }
}

if (shapeTrigger && shapeMenu) {
    shapeTrigger.addEventListener('click', (e) => {
        e.stopPropagation();
        const willOpen = !shapeMenu.classList.contains('open');
        shapeMenu.classList.toggle('open', willOpen);
        shapeTrigger.setAttribute('aria-expanded', willOpen ? 'true' : 'false');
    });
    document.addEventListener('click', () => {
        shapeMenu.classList.remove('open');
        shapeTrigger.setAttribute('aria-expanded', 'false');
    });
    shapeMenu.addEventListener('click', (e) => e.stopPropagation());
}

shapeItems.forEach(item => {
    item.addEventListener('click', () => {
        const nextShape = item.dataset.shape;
        if (!SHAPES[nextShape]) return;
        saveUndo();
        setShape(nextShape);
        shapeMenu.classList.remove('open');
        shapeTrigger.setAttribute('aria-expanded', 'false');
        if (interactionMode === 'add') {
            cancelPlacement();
            startPlacement(nextShape);
        }
        setCommandStatus('Picked up: ' + (SHAPE_LABELS[nextShape] || nextShape));
    });
});

if (spliceAddBtn) {
    spliceAddBtn.addEventListener('click', () => {
        setInteractionMode('add', 'ui');
    });
}

if (spliceDuplicateBtn) {
    spliceDuplicateBtn.addEventListener('click', () => {
        if (!beginDuplicatePlacement()) setCommandStatus('Duplicate Last needs a placed shape first');
    });
}

modeButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        setInteractionMode(btn.dataset.mode, 'ui');
    });
});

syncModeUi('ui');
syncShapeUi(currentShape);

// ============================================================
// 9. CAT AI (Boba)
// ============================================================
const chatInput=document.getElementById('chat-input'),sendBtn=document.getElementById('send-btn'),micBtn=document.getElementById('mic-btn');
const catDialogue=document.getElementById('cat-dialogue'),chatHistoryContainer=document.getElementById('chat-history-container');
const chatHistory=document.getElementById('chat-history'),toggleHistoryBtn=document.getElementById('toggle-history-btn');
const bgMusic=document.getElementById('bg-music'),musicToggleBtn=document.getElementById('music-toggle-btn');
const quickCommandButtons=document.querySelectorAll('[data-quick-command]');
const commandStatus=document.getElementById('command-status');
let isMusicPlaying=false;bgMusic.volume=0.2;
musicToggleBtn.addEventListener('click',()=>{if(isMusicPlaying){bgMusic.pause();musicToggleBtn.textContent='🔇 Music Off';}else{bgMusic.play().catch(()=>{});musicToggleBtn.textContent='🎵 Music On';}isMusicPlaying=!isMusicPlaying;});
let isHistoryOpen=false;

function setCommandStatus(text) {
    if (commandStatus) commandStatus.textContent = text;
}

function runQuickCommand(commandText) {
    chatInput.value = commandText;
    catCmd(commandText);
    chatInput.value = '';
}

quickCommandButtons.forEach(button => {
    button.addEventListener('click', () => runQuickCommand(button.dataset.quickCommand));
});

function catCmd(cmd){
    const lc=cmd.toLowerCase();let resp='',done=false;
    for(const n of Object.keys(SHAPES)){if(lc.includes(n)){saveUndo();setShape(n);resp=`Switched to ${n}! 🎨`;done=true;setCommandStatus(`Picked up: ${n}`);break;}}
    if(!done&&/\b(reset|new|restart|clear)\b/i.test(lc)){saveUndo();resp="Fresh! 🆕";setShape(currentShape);done=true;setCommandStatus('Picked up: reset');}
    else if(!done&&/\b(bigger|larger|grow|expand)\b/i.test(lc)){saveUndo();resp="Growing!";mesh.scale.multiplyScalar(1.3);done=true;setCommandStatus('Picked up: scale up');}
    else if(!done&&/\b(smaller|tiny|shrink)\b/i.test(lc)){saveUndo();resp="Shrinking!";mesh.scale.multiplyScalar(0.7);done=true;setCommandStatus('Picked up: scale down');}
    else if(!done&&/\b(red|pink)\b/i.test(lc)){saveUndo();material.color.setHex(0xe66767);resp="Red! 🍓";done=true;setCommandStatus('Picked up: red');}
    else if(!done&&/\b(blue|azure)\b/i.test(lc)){saveUndo();material.color.setHex(0xa8c3e6);resp="Blue! 🫐";done=true;setCommandStatus('Picked up: blue');}
    else if(!done&&/\b(green|sage)\b/i.test(lc)){saveUndo();material.color.setHex(0x9cb89c);resp="Green! 🌿";done=true;setCommandStatus('Picked up: sage green');}
    else if(!done&&/\b(yellow|gold)\b/i.test(lc)){saveUndo();material.color.setHex(0xf4d03f);resp="Yellow! ☀️";done=true;setCommandStatus('Picked up: yellow');}
    else if(!done&&/\b(white|cream)\b/i.test(lc)){saveUndo();material.color.setHex(0xfffaec);resp="White! 🥛";done=true;setCommandStatus('Picked up: cream');}
    else if(!done&&/\b(dark|black)\b/i.test(lc)){saveUndo();material.color.setHex(0x333333);resp="Dark! 🐈‍⬛";done=true;setCommandStatus('Picked up: dark');}
    else if(!done&&/\b(metal|shiny)\b/i.test(lc)){saveUndo();material.metalness=0.8;material.roughness=0.1;material.needsUpdate=true;resp="Shiny! ✨";done=true;setCommandStatus('Picked up: metal');}
    else if(!done&&/\b(matte|clay)\b/i.test(lc)){saveUndo();material.metalness=0.1;material.roughness=0.8;material.needsUpdate=true;resp="Matte! 🧱";done=true;setCommandStatus('Picked up: matte clay');}
    else if(!done&&/\b(glass|transparent)\b/i.test(lc)){saveUndo();material.transparent=true;material.opacity=0.5;material.metalness=1;material.roughness=0;material.needsUpdate=true;resp="Glass! 🧊";done=true;setCommandStatus('Picked up: glass');}
    else if(!done&&/\b(wireframe|lines)\b/i.test(lc)){saveUndo();material.wireframe=!material.wireframe;material.needsUpdate=true;resp="Wireframe! 📐";done=true;setCommandStatus('Picked up: wireframe');}
    else if(!done&&/\b(export|stl|print)\b/i.test(lc)){document.getElementById('export-btn').click();resp="Exporting STL! 💾";done=true;setCommandStatus('Picked up: export STL');}
    else if(!done&&(/\b(help|how)\b/i.test(lc)||cmd.length>25)){resp="Mouse: left=pull, right=push, shift=smooth. Hands: palm=rotate, pinch=move, two palms=scale, two pinches apart=split. Try shapes, colors, 'export stl'!";done=true;setCommandStatus('Picked up: help');}
    if(!done){resp="Meow? Try shapes (sphere,cube,torus), colors, 'export stl', or 'reset'!";setCommandStatus('Command idle');}
    catDialogue.style.opacity=0;setTimeout(()=>{catDialogue.style.opacity=1;catDialogue.innerHTML=`<em>"${resp}"</em>`;const u=document.createElement("li");u.className="history-user";u.textContent=`You: "${cmd}"`;const b=document.createElement("li");b.className="history-cat";b.textContent=`Cat: "${resp}"`;chatHistory.appendChild(u);chatHistory.appendChild(b);chatHistoryContainer.scrollTop=chatHistoryContainer.scrollHeight;},200);
}

toggleHistoryBtn.addEventListener('click',()=>{isHistoryOpen=!isHistoryOpen;chatHistoryContainer.style.display=isHistoryOpen?'block':'none';});
sendBtn.addEventListener('click',()=>{if(chatInput.value.trim()){catCmd(chatInput.value);chatInput.value='';}});
chatInput.addEventListener('keypress',e=>{if(e.key==='Enter'&&chatInput.value.trim()){catCmd(chatInput.value);chatInput.value='';}});

const catChar=document.getElementById('cat-character'),chatUI=document.getElementById('chat-interface');
const eyeOpen=document.getElementById('right-eye-open'),eyeWink=document.getElementById('right-eye-wink');
let chatVis=true;
catChar.addEventListener('click',()=>{chatVis=!chatVis;chatUI.style.display=chatVis?'flex':'none';eyeOpen.style.display='none';eyeWink.style.display='block';if(chatVis){catDialogue.style.opacity=0;setTimeout(()=>{catDialogue.innerHTML='<em>"Meow! ✨"</em>';catDialogue.style.opacity=1;},150);}setTimeout(()=>{eyeOpen.style.display='block';eyeWink.style.display='none';},450);});

const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
if(SR){const r=new SR();r.continuous=false;r.lang='en-US';r.onstart=()=>{micBtn.classList.add('recording');catDialogue.innerHTML='<em>Listening...</em>';};r.onresult=e=>{const t=e.results[0][0].transcript;chatInput.value=t;catCmd(t);chatInput.value='';};r.onerror=()=>micBtn.classList.remove('recording');r.onend=()=>micBtn.classList.remove('recording');micBtn.addEventListener('click',()=>r.start());}else micBtn.style.display='none';

// ============================================================
// 10. WORKSHOP TOP-BAR WIRING (saveUndo() now exists)
// ============================================================
document.querySelectorAll('.ws-btn').forEach(b => {
    b.addEventListener('click', () => {
        if (b.dataset.workshop === currentWorkshop) return;
        saveUndo();
        applyWorkshop(b.dataset.workshop);
    });
});

// ============================================================
// 11. OBJECT SCANNING — webcam silhouette → 3D voxel mesh
// ============================================================
// Approach: capture a frame, extract the foreground silhouette by comparing each
// pixel's color to the background color (sampled from the corners), downsample
// the binary mask to a coarse grid, then build a voxel-style mesh from the grid.
// The result becomes the active sculptable mesh — user can refine in any workshop.

(function setupScan() {
    const scanBtn = document.getElementById('scan-btn');

    // Build modal lazily
    const modal = document.createElement('div');
    modal.id = 'scan-modal';
    modal.innerHTML = `
        <div id="scan-modal-inner">
            <h2>📷 Object Scanner</h2>
            <p>Hold a real object in front of the camera against a plain, uniformly-lit background.
               Move the object closer to the camera and keep it centered. Adjust the threshold so only the object stays pink in the preview, then capture.</p>
            <div id="scan-video-wrap">
                <video id="scan-video" autoplay playsinline muted></video>
                <canvas id="scan-preview"></canvas>
                <canvas id="scan-frame"></canvas>
                <div id="scan-guide">
                    <span>Keep the object and both hands inside this frame</span>
                </div>
            </div>
            <div id="scan-threshold-row">
                <label for="scan-threshold">Threshold</label>
                <input type="range" id="scan-threshold" min="10" max="120" step="2" value="45">
                <span id="scan-threshold-v">45</span>
            </div>
            <div id="scan-controls">
                <button id="scan-capture" class="primary">📸 Capture & Build Mesh</button>
                <button id="scan-close">Close</button>
            </div>
            <div id="scan-status">Ready.</div>
        </div>
    `;
    document.body.appendChild(modal);

    const scanVideo = modal.querySelector('#scan-video');
    const scanPreview = modal.querySelector('#scan-preview');
    const scanFrame = modal.querySelector('#scan-frame');
    const scanGuide = modal.querySelector('#scan-guide');
    const scanThresh = modal.querySelector('#scan-threshold');
    const scanThreshV = modal.querySelector('#scan-threshold-v');
    const scanStatus = modal.querySelector('#scan-status');
    const scanCapture = modal.querySelector('#scan-capture');
    const scanClose = modal.querySelector('#scan-close');

    // Verify guide is visible
    if (scanGuide) {
        setTimeout(() => {
            const rect = scanGuide.getBoundingClientRect();
            if (rect.width === 0 || rect.height === 0) {
                scanGuide.style.opacity = '1';
                scanGuide.style.visibility = 'visible';
                scanGuide.style.display = 'block';
            }
        }, 100);
    }

    let scanStream = null;
    let previewRAF = 0;
    let threshold = 45;

    scanThresh.addEventListener('input', e => {
        threshold = parseInt(e.target.value, 10);
        scanThreshV.textContent = String(threshold);
    });

    scanBtn.addEventListener('click', () => openScanner());
    scanClose.addEventListener('click', () => closeScanner());
    scanCapture.addEventListener('click', () => doCapture());

    async function openScanner() {
        modal.classList.add('open');
        scanStatus.textContent = 'Starting camera…';
        try {
            scanStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
            scanVideo.srcObject = scanStream;
            await scanVideo.play();
            scanPreview.width = scanVideo.videoWidth || 640;
            scanPreview.height = scanVideo.videoHeight || 480;
            scanFrame.width = scanPreview.width;
            scanFrame.height = scanPreview.height;
            scanStatus.textContent = 'Adjust the threshold so the object glows pink, then capture.';
            startPreviewLoop();
        } catch (err) {
            scanStatus.textContent = 'Camera error: ' + (err.message || err);
        }
    }
    function closeScanner() {
        cancelAnimationFrame(previewRAF);
        if (scanStream) {
            scanStream.getTracks().forEach(t => t.stop());
            scanStream = null;
        }
        scanVideo.srcObject = null;
        modal.classList.remove('open');
    }

    // Grab background color estimate from the four corners of the captured frame.
    function sampleBackground(imgData) {
        const { data, width, height } = imgData;
        const samples = [];
        const sz = 8;
        for (const [cx, cy] of [[0,0],[width-sz,0],[0,height-sz],[width-sz,height-sz]]) {
            for (let dy = 0; dy < sz; dy++) for (let dx = 0; dx < sz; dx++) {
                const i = ((cy + dy) * width + (cx + dx)) * 4;
                samples.push([data[i], data[i+1], data[i+2]]);
            }
        }
        let r=0, g=0, b=0;
        for (const s of samples) { r += s[0]; g += s[1]; b += s[2]; }
        return [r / samples.length, g / samples.length, b / samples.length];
    }

    function getScanFocusRegion(videoEl, width, height) {
        if (!handLandmarker || !videoEl || videoEl.readyState < 2) return null;
        try {
            const detected = handLandmarker.detectForVideo(videoEl, performance.now());
            if (!detected || !detected.landmarks || !detected.landmarks.length) return null;
            let minX = 1, minY = 1, maxX = 0, maxY = 0;
            for (const hand of detected.landmarks) {
                for (const lm of hand) {
                    if (lm.x < minX) minX = lm.x;
                    if (lm.y < minY) minY = lm.y;
                    if (lm.x > maxX) maxX = lm.x;
                    if (lm.y > maxY) maxY = lm.y;
                }
            }
            const padX = 0.18;
            const padY = 0.22;
            return {
                x0: Math.max(0, Math.floor((minX - padX) * width)),
                y0: Math.max(0, Math.floor((minY - padY) * height)),
                x1: Math.min(width - 1, Math.ceil((maxX + padX) * width)),
                y1: Math.min(height - 1, Math.ceil((maxY + padY) * height)),
            };
        } catch (err) {
            return null;
        }
    }

    // Build foreground mask by scoring pixels against the estimated background.
    // Center pixels are weighted higher so distant background is less likely to survive.
    function buildMask(imgData, thresh, focusRegion = null) {
        const { data, width, height } = imgData;
        const [br, bg, bb] = sampleBackground(imgData);
        const scores = new Float32Array(width * height);
        const left = Math.floor(width * 0.12);
        const right = Math.floor(width * 0.88);
        const top = Math.floor(height * 0.10);
        const bottom = Math.floor(height * 0.92);
        const roi = focusRegion || {
            x0: Math.floor(width * 0.18),
            y0: Math.floor(height * 0.18),
            x1: Math.floor(width * 0.82),
            y1: Math.floor(height * 0.84),
        };
        const cx = width * 0.5;
        const cy = height * 0.52;
        const maxR = Math.hypot(width * 0.5, height * 0.5);
        let sum = 0;
        let sum2 = 0;
        let count = 0;

        for (let i = 0, p = 0; i < data.length; i += 4, p++) {
            const y = (p / width) | 0;
            const x = p - y * width;
            if (x < left || x > right || y < top || y > bottom) continue;
            if (x < roi.x0 || x > roi.x1 || y < roi.y0 || y > roi.y1) continue;
            const dr = data[i] - br, dg = data[i+1] - bg, db = data[i+2] - bb;
            const colorDist = Math.hypot(dr, dg, db);
            const dx = x - cx;
            const dy = y - cy;
            const radial = Math.min(1, Math.hypot(dx, dy) / maxR);
            const centerWeight = 1 - Math.pow(radial, 1.9);
            const score = colorDist * (0.55 + centerWeight * 0.85);
            scores[p] = score;
            sum += score;
            sum2 += score * score;
            count++;
        }

        const mean = count ? (sum / count) : 0;
        const variance = count ? Math.max(0, sum2 / count - mean * mean) : 0;
        const std = Math.sqrt(variance);
        const baseThreshold = mean + std * 0.55;
        const manualBias = Math.max(0.8, Math.min(1.7, thresh / 45));
        const adaptiveThreshold = baseThreshold * manualBias;

        const mask = new Uint8Array(width * height);
        for (let p = 0; p < scores.length; p++) {
            if (scores[p] > adaptiveThreshold) mask[p] = 1;
        }
        return mask;
    }

    function smoothMask(mask, width, height) {
        const out = new Uint8Array(width * height);
        for (let y = 1; y < height - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                let count = 0;
                for (let oy = -1; oy <= 1; oy++) {
                    for (let ox = -1; ox <= 1; ox++) {
                        count += mask[(y + oy) * width + (x + ox)];
                    }
                }
                out[y * width + x] = count >= 5 ? 1 : 0;
            }
        }
        return out;
    }

    function erodeMask(mask, width, height) {
        const out = new Uint8Array(width * height);
        for (let y = 1; y < height - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                let keep = 1;
                for (let oy = -1; oy <= 1 && keep; oy++) {
                    for (let ox = -1; ox <= 1; ox++) {
                        if (!mask[(y + oy) * width + (x + ox)]) { keep = 0; break; }
                    }
                }
                out[y * width + x] = keep;
            }
        }
        return out;
    }

    // Keep only the largest connected component (rejects tiny background-noise blobs).
    function keepLargestComponent(mask, width, height) {
        const visited = new Uint8Array(width * height);
        const out = new Uint8Array(width * height);
        let bestSize = 0;
        const stack = new Int32Array(width * height);
        let stackTop = 0;
        let bestStart = -1;
        for (let i = 0; i < mask.length; i++) {
            if (!mask[i] || visited[i]) continue;
            // Flood fill
            stackTop = 0;
            stack[stackTop++] = i;
            visited[i] = 1;
            const region = [];
            while (stackTop > 0) {
                const idx = stack[--stackTop];
                region.push(idx);
                const x = idx % width, y = (idx / width) | 0;
                const nbrs = [];
                if (x > 0) nbrs.push(idx - 1);
                if (x < width-1) nbrs.push(idx + 1);
                if (y > 0) nbrs.push(idx - width);
                if (y < height-1) nbrs.push(idx + width);
                for (const n of nbrs) {
                    if (mask[n] && !visited[n]) { visited[n] = 1; stack[stackTop++] = n; }
                }
            }
            if (region.length > bestSize) {
                bestSize = region.length;
                bestStart = i;
                // Lazily fill `out` only for the eventual best — re-flood at end.
            }
        }
        if (bestStart < 0) return out;
        // Re-flood from bestStart (cheaper than tracking every region's pixels).
        const visited2 = new Uint8Array(width * height);
        stackTop = 0;
        stack[stackTop++] = bestStart;
        visited2[bestStart] = 1;
        while (stackTop > 0) {
            const idx = stack[--stackTop];
            out[idx] = 1;
            const x = idx % width, y = (idx / width) | 0;
            const nbrs = [];
            if (x > 0) nbrs.push(idx - 1);
            if (x < width-1) nbrs.push(idx + 1);
            if (y > 0) nbrs.push(idx - width);
            if (y < height-1) nbrs.push(idx + width);
            for (const n of nbrs) {
                if (mask[n] && !visited2[n]) { visited2[n] = 1; stack[stackTop++] = n; }
            }
        }
        return out;
    }

    // Downsample mask to coarse grid for a sculptable triangle count.
    function downsampleMask(mask, w, h, targetW) {
        const aspect = h / w;
        const dw = targetW;
        const dh = Math.max(8, Math.round(targetW * aspect));
        const out = new Uint8Array(dw * dh);
        const sx = w / dw, sy = h / dh;
        for (let dy = 0; dy < dh; dy++) {
            for (let dx = 0; dx < dw; dx++) {
                const x0 = Math.floor(dx * sx), x1 = Math.floor((dx+1) * sx);
                const y0 = Math.floor(dy * sy), y1 = Math.floor((dy+1) * sy);
                let on = 0, total = 0;
                for (let yy = y0; yy < y1; yy++) for (let xx = x0; xx < x1; xx++) {
                    if (mask[yy * w + xx]) on++;
                    total++;
                }
                out[dy * dw + dx] = (on / total > 0.5) ? 1 : 0;
            }
        }
        return { mask: out, w: dw, h: dh };
    }

    // Build a voxel-style 3D mesh by walking each foreground cell of the coarse mask.
    function buildVoxelGeometry(mask, w, h) {
        const verts = [];
        const norms = [];
        const fg = (x, y) => x >= 0 && y >= 0 && x < w && y < h && mask[y * w + x];

        // Compute mask centroid + scale so the resulting mesh fits a ~2-unit box.
        let cx = 0, cy = 0, n = 0;
        for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
            if (mask[y*w+x]) { cx += x + 0.5; cy += y + 0.5; n++; }
        }
        if (!n) return null;
        cx /= n; cy /= n;
        const maxDim = Math.max(w, h);
        const sxy = 2.5 / maxDim;
        const depth = (Math.min(w, h) * sxy) * 0.45;
        const halfD = depth / 2;

        function addQuad(p1, p2, p3, p4, nx, ny, nz) {
            verts.push(p1[0],p1[1],p1[2], p2[0],p2[1],p2[2], p3[0],p3[1],p3[2],
                       p1[0],p1[1],p1[2], p3[0],p3[1],p3[2], p4[0],p4[1],p4[2]);
            for (let k = 0; k < 6; k++) norms.push(nx, ny, nz);
        }

        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                if (!mask[y*w+x]) continue;
                const wx0 = (x       - cx) * sxy;
                const wx1 = (x + 1   - cx) * sxy;
                const wy0 = -(y      - cy) * sxy; // image y grows downward; world y grows upward
                const wy1 = -(y + 1  - cy) * sxy;
                // Front face (+z)
                addQuad([wx0, wy1, halfD], [wx1, wy1, halfD], [wx1, wy0, halfD], [wx0, wy0, halfD], 0,0,1);
                // Back face (-z)
                addQuad([wx0, wy0, -halfD], [wx1, wy0, -halfD], [wx1, wy1, -halfD], [wx0, wy1, -halfD], 0,0,-1);
                // Side walls only where neighbour is empty
                if (!fg(x-1, y)) addQuad([wx0, wy1, -halfD], [wx0, wy1, halfD], [wx0, wy0, halfD], [wx0, wy0, -halfD], -1,0,0);
                if (!fg(x+1, y)) addQuad([wx1, wy0, -halfD], [wx1, wy0, halfD], [wx1, wy1, halfD], [wx1, wy1, -halfD], 1,0,0);
                if (!fg(x, y-1)) addQuad([wx0, wy0, -halfD], [wx0, wy0, halfD], [wx1, wy0, halfD], [wx1, wy0, -halfD], 0,1,0);
                if (!fg(x, y+1)) addQuad([wx1, wy1, -halfD], [wx1, wy1, halfD], [wx0, wy1, halfD], [wx0, wy1, -halfD], 0,-1,0);
            }
        }

        const g = new THREE.BufferGeometry();
        g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(verts), 3));
        g.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(norms), 3));
        g.computeVertexNormals();
        return g;
    }

    function drawScanFrameOverlay(ctx, width, height, focusRegion) {
        const frame = focusRegion || {
            x0: Math.floor(width * 0.18),
            y0: Math.floor(height * 0.18),
            x1: Math.floor(width * 0.82),
            y1: Math.floor(height * 0.84),
        };
        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = 'rgba(15, 18, 20, 0.18)';
        ctx.fillRect(0, 0, width, height);
        ctx.clearRect(frame.x0, frame.y0, frame.x1 - frame.x0, frame.y1 - frame.y0);
        ctx.strokeStyle = 'rgba(255, 130, 200, 0.95)';
        ctx.lineWidth = Math.max(2, Math.round(Math.min(width, height) * 0.006));
        ctx.strokeRect(frame.x0, frame.y0, frame.x1 - frame.x0, frame.y1 - frame.y0);
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
        ctx.lineWidth = Math.max(1, Math.round(Math.min(width, height) * 0.0035));
        const corner = Math.max(16, Math.round(Math.min(width, height) * 0.04));
        const corners = [
            [frame.x0, frame.y0, 1, 1],
            [frame.x1, frame.y0, -1, 1],
            [frame.x0, frame.y1, 1, -1],
            [frame.x1, frame.y1, -1, -1],
        ];
        for (const [x, y, sx, sy] of corners) {
            ctx.beginPath();
            ctx.moveTo(x, y);
            ctx.lineTo(x + sx * corner, y);
            ctx.moveTo(x, y);
            ctx.lineTo(x, y + sy * corner);
            ctx.stroke();
        }
        ctx.fillStyle = 'rgba(255, 255, 255, 0.88)';
        ctx.font = `${Math.max(12, Math.round(Math.min(width, height) * 0.03))}px sans-serif`;
        ctx.fillText('Keep object and hands inside the frame', frame.x0 + 12, Math.max(18, frame.y0 - 10));
    }

    function startPreviewLoop() {
        const w = scanPreview.width, h = scanPreview.height;
        const ctx = scanPreview.getContext('2d', { willReadFrequently: true });
        const frameCtx = scanFrame.getContext('2d');
        const tick = () => {
            if (!scanStream) return;
            // Mirror the video onto our canvas to match what user sees.
            ctx.save();
            ctx.translate(w, 0);
            ctx.scale(-1, 1);
            ctx.drawImage(scanVideo, 0, 0, w, h);
            ctx.restore();
            const img = ctx.getImageData(0, 0, w, h);
            const focusRegion = getScanFocusRegion(scanVideo, w, h);
            let mask = buildMask(img, threshold, focusRegion);
            mask = erodeMask(mask, w, h);
            mask = smoothMask(mask, w, h);
            // Tint the FG pink in the preview overlay.
            const out = ctx.createImageData(w, h);
            for (let i = 0; i < out.data.length; i += 4) {
                out.data[i] = 255;
                out.data[i+1] = 255;
                out.data[i+2] = 255;
                out.data[i+3] = 255;
            }
            for (let i = 0, p = 0; i < out.data.length; i += 4, p++) {
                if (mask[p]) {
                    out.data[i] = 255; out.data[i+1] = 130; out.data[i+2] = 200; out.data[i+3] = 235;
                }
            }
            ctx.putImageData(out, 0, 0);
            drawScanFrameOverlay(frameCtx, w, h, focusRegion);
            previewRAF = requestAnimationFrame(tick);
        };
        previewRAF = requestAnimationFrame(tick);
    }

    function doCapture() {
        if (!scanStream) { scanStatus.textContent = 'Camera not ready.'; return; }
        const w = scanVideo.videoWidth, h = scanVideo.videoHeight;
        const tmp = document.createElement('canvas');
        tmp.width = w; tmp.height = h;
        const tctx = tmp.getContext('2d');
        tctx.save(); tctx.translate(w, 0); tctx.scale(-1, 1);
        tctx.drawImage(scanVideo, 0, 0, w, h);
        tctx.restore();
        const img = tctx.getImageData(0, 0, w, h);
        scanStatus.textContent = 'Building mask…';
        const focusRegion = getScanFocusRegion(scanVideo, w, h);
        let mask = buildMask(img, threshold, focusRegion);
        mask = erodeMask(mask, w, h);
        mask = smoothMask(mask, w, h);
        mask = keepLargestComponent(mask, w, h);
        const fgCount = mask.reduce((a, v) => a + v, 0);
        if (fgCount < w * h * 0.01) {
            scanStatus.textContent = 'No object detected — try adjusting threshold.';
            return;
        }
        const ds = downsampleMask(mask, w, h, 56); // ~56px wide is sculptable
        scanStatus.textContent = `Building mesh (${ds.w}×${ds.h} grid)…`;
        const geo = buildVoxelGeometry(ds.mask, ds.w, ds.h);
        if (!geo) { scanStatus.textContent = 'Mesh build failed.'; return; }

        // Replace active mesh with the scanned one. Save undo first so user can revert.
        saveUndo();
        if (mesh) { scene.remove(mesh); mesh.geometry.dispose(); }
        mesh = new THREE.Mesh(geo, material);
        // Reset transform so scanned object appears centered.
        mesh.position.set(0, 0, 0);
        mesh.rotation.set(0, 0, 0);
        mesh.scale.set(1, 1, 1);
        refreshWireframe();
        rebuildTopologyCache();
        SelectionController.clear();
        scene.add(mesh);

        scanStatus.textContent = `Done — ${(geo.attributes.position.count / 3) | 0} triangles. You can now sculpt, color, and export!`;
        // Brief delay then close so user sees the success message.
        setTimeout(closeScanner, 900);
    }
})();
