import * as THREE from 'three';
import { HandLandmarker, FilesetResolver, DrawingUtils } from '@mediapipe/tasks-vision';

// ============================================================
// 1. SCENE SETUP
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
grid.position.y = -1.5;
grid.material.opacity = 0.4;
grid.material.transparent = true;
scene.add(grid);

const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.1, 100);
camera.position.set(0, 0, 6);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
container.appendChild(renderer.domElement);
const raycaster = new THREE.Raycaster();

// --- Pinch touch point visuals ---
const touchPointMat = new THREE.MeshBasicMaterial({
    color: 0xffdd57, transparent: true, opacity: 0.9, depthTest: false, toneMapped: false,
});
const touchPointRingMat = new THREE.MeshBasicMaterial({
    color: 0xffdd57, transparent: true, opacity: 0.4, side: THREE.DoubleSide, depthTest: false, toneMapped: false,
});
const touchPointDot = new THREE.Mesh(new THREE.SphereGeometry(0.04, 12, 8), touchPointMat);
touchPointDot.renderOrder = 1000;
touchPointDot.visible = false;
scene.add(touchPointDot);

const touchPointRing = new THREE.Mesh(new THREE.RingGeometry(0.06, 0.09, 32), touchPointRingMat);
touchPointRing.renderOrder = 999;
touchPointRing.visible = false;
scene.add(touchPointRing);

// Pulsing outer ring
const touchPointPulse = new THREE.Mesh(
    new THREE.RingGeometry(0.09, 0.11, 32),
    new THREE.MeshBasicMaterial({ color: 0xffdd57, transparent: true, opacity: 0.2, side: THREE.DoubleSide, depthTest: false, toneMapped: false })
);
touchPointPulse.renderOrder = 998;
touchPointPulse.visible = false;
scene.add(touchPointPulse);

let touchPointTime = 0;
const _touchNdc = new THREE.Vector2();
const _touchPt = new THREE.Vector3();
const _touchNm = new THREE.Vector3();
const _touchDir = new THREE.Vector3();

function showTouchPoint(screenX, screenY) {
    // Raycast from pinch screen position to find 3D point
    _touchNdc.set(
        (screenX / innerWidth) * 2 - 1,
        -(screenY / innerHeight) * 2 + 1
    );
    raycaster.setFromCamera(_touchNdc, camera);
    const hits = raycaster.intersectObject(mesh);
    if (hits.length) {
        _touchPt.copy(hits[0].point);
        _touchNm.set(0, 0, 1);
        if (hits[0].face?.normal) _touchNm.copy(hits[0].face.normal).transformDirection(mesh.matrixWorld).normalize();
        touchPointDot.position.copy(_touchPt).addScaledVector(_touchNm, 0.02);
        touchPointRing.position.copy(touchPointDot.position);
        touchPointPulse.position.copy(touchPointDot.position);
        touchPointRing.lookAt(_touchPt.copy(touchPointDot.position).add(_touchNm));
        touchPointPulse.lookAt(_touchPt.copy(touchPointDot.position).add(_touchNm));
        touchPointDot.visible = true;
        touchPointRing.visible = true;
        touchPointPulse.visible = true;
        // Pulse animation
        touchPointTime += 0.15;
        const pulse = 1 + Math.sin(touchPointTime * 3) * 0.25;
        touchPointPulse.scale.setScalar(pulse);
        touchPointPulse.material.opacity = 0.12 + Math.sin(touchPointTime * 3) * 0.08;
    } else {
        // No hit — show at a fixed depth in front of camera
        raycaster.ray.direction.normalize();
        _touchDir.copy(raycaster.ray.direction).multiplyScalar(4);
        touchPointDot.position.copy(raycaster.ray.origin).add(_touchDir);
        touchPointRing.position.copy(touchPointDot.position);
        touchPointPulse.position.copy(touchPointDot.position);
        touchPointRing.lookAt(camera.position);
        touchPointPulse.lookAt(camera.position);
        touchPointDot.visible = true;
        touchPointRing.visible = true;
        touchPointPulse.visible = true;
    }
}

function hideTouchPoint() {
    touchPointDot.visible = false;
    touchPointRing.visible = false;
    touchPointPulse.visible = false;
}

window.addEventListener('resize', () => {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
});

// ============================================================
// 2. SHAPES & MESH
// ============================================================
const SHAPE_LABELS = {
    sphere: 'Sphere', cube: 'Cube', cylinder: 'Cylinder', cone: 'Cone',
    torus: 'Torus', pyramid: 'Pyramid', icosahedron: 'Icosahedron',
    dodecahedron: 'Dodecahedron', torusknot: 'Torus Knot', helix: 'Helix',
    apple: 'Apple', heart: 'Heart', star: 'Star', mushroom: 'Mushroom',
    banana: 'Banana', berry: 'Berry', clover: 'Clover', pear: 'Pear', strawberry: 'Strawberry',
};

// Organic shape generator: deform a sphere with math
function makeOrganicShape(deformFn, segs = 48, rings = 36) {
    const geo = new THREE.SphereGeometry(1, segs, rings);
    const pos = geo.attributes.position;
    for (let i = 0; i < pos.count; i++) {
        const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
        const len = Math.sqrt(x*x + y*y + z*z) || 1;
        const nx = x/len, ny = y/len, nz = z/len;
        const theta = Math.atan2(nz, nx); // longitude
        const phi = Math.acos(Math.max(-1, Math.min(1, ny))); // latitude from top
        const r = deformFn(nx, ny, nz, theta, phi, len);
        pos.setXYZ(i, nx*r, ny*r, nz*r);
    }
    pos.needsUpdate = true;
    geo.computeVertexNormals();
    return geo;
}

const SHAPES = {
    sphere:       () => new THREE.SphereGeometry(1, 32, 24),
    cube:         () => new THREE.BoxGeometry(1.5, 1.5, 1.5, 10, 10, 10),
    cylinder:     () => new THREE.CylinderGeometry(0.8, 0.8, 2, 32, 10),
    cone:         () => new THREE.ConeGeometry(1, 2, 32, 10),
    torus:        () => new THREE.TorusGeometry(0.8, 0.35, 20, 40),
    pyramid:      () => new THREE.ConeGeometry(1, 1.8, 4, 1),
    icosahedron:  () => new THREE.IcosahedronGeometry(1.05, 0),
    dodecahedron: () => new THREE.DodecahedronGeometry(1.0, 0),
    torusknot:    () => new THREE.TorusKnotGeometry(0.76, 0.2, 150, 20, 3, 5),
    helix:        () => new THREE.TorusKnotGeometry(0.7, 0.16, 180, 28, 2, 7),

    // Organic templates — direct vertex manipulation for accurate shapes
    apple: () => {
        const geo = new THREE.SphereGeometry(1, 48, 36);
        const pos = geo.attributes.position;
        for (let i = 0; i < pos.count; i++) {
            let x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
            const r = Math.sqrt(x*x + z*z);
            const theta = Math.atan2(z, x);
            // Body: wider at middle, narrower at top/bottom
            const bodyScale = 1.0 + 0.15 * Math.cos(y * Math.PI * 0.5);
            x *= bodyScale; z *= bodyScale;
            // Top indent (stem dimple)
            if (y > 0.7) {
                const t = (y - 0.7) / 0.3;
                const indent = t * t * 0.35;
                const squeeze = 1.0 - t * 0.5;
                x *= squeeze; z *= squeeze;
                y -= indent;
            }
            // Bottom taper — gentle roundness
            if (y < -0.6) {
                const t = (-0.6 - y) / 0.4;
                x *= 1.0 - t * 0.4;
                z *= 1.0 - t * 0.4;
            }
            // Subtle five-fold symmetry (apple lobes)
            const lobe = 1.0 + 0.04 * Math.cos(theta * 5);
            x *= lobe; z *= lobe;
            pos.setXYZ(i, x, y, z);
        }
        pos.needsUpdate = true; geo.computeVertexNormals(); return geo;
    },
    heart: () => {
        // Parametric heart using vertex manipulation
        const geo = new THREE.SphereGeometry(1, 48, 36);
        const pos = geo.attributes.position;
        for (let i = 0; i < pos.count; i++) {
            let x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
            const len = Math.sqrt(x*x + y*y + z*z) || 1;
            const nx = x/len, ny = y/len, nz = z/len;
            const theta = Math.atan2(nz, nx);
            // Heart cross-section: r = 1 - sin(theta) style
            if (ny > -0.1) {
                // Upper part: two lobes
                const lobeR = 0.9 + 0.4 * Math.pow(Math.abs(Math.sin(theta)), 0.7);
                const vertScale = 1.0 - 0.2 * ny * ny;
                x = nx * lobeR * vertScale;
                z = nz * lobeR * vertScale;
                // Push lobes up and apart
                y = ny * 0.7 + 0.2;
                // Crease between lobes at front/back
                const crease = Math.pow(Math.max(0, Math.cos(theta)), 8) * 0.2;
                y -= crease * Math.max(0, ny);
            } else {
                // Lower part: taper to point
                const t = Math.min(1, (-0.1 - ny) / 0.9);
                const taper = 1.0 - t;
                x = nx * 0.8 * taper * taper;
                z = nz * 0.8 * taper * taper;
                y = -0.1 - t * 1.2;
            }
            pos.setXYZ(i, x, y, z);
        }
        pos.needsUpdate = true; geo.computeVertexNormals(); return geo;
    },
    star: () => {
        // Flat 5-pointed star extruded to have thickness
        const geo = new THREE.CylinderGeometry(1, 1, 0.3, 60, 2, false);
        const pos = geo.attributes.position;
        for (let i = 0; i < pos.count; i++) {
            let x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
            const theta = Math.atan2(z, x);
            const r = Math.sqrt(x*x + z*z);
            // Star shape: alternating spikes and valleys
            const angle5 = ((theta % (Math.PI * 2)) + Math.PI * 2) % (Math.PI * 2);
            const segment = angle5 / (Math.PI * 2) * 5;
            const frac = segment % 1;
            // Triangle wave between inner radius (0.4) and outer (1.0)
            const starR = frac < 0.5
                ? 0.4 + (1.0 - 0.4) * (frac / 0.5)
                : 0.4 + (1.0 - 0.4) * ((1.0 - frac) / 0.5);
            const scale = r > 0.01 ? starR / r : 1;
            x *= scale; z *= scale;
            // Round the edges slightly
            const edgeRound = 1.0 - 0.15 * Math.abs(y) / 0.15;
            x *= Math.max(0.9, edgeRound); z *= Math.max(0.9, edgeRound);
            pos.setXYZ(i, x, y, z);
        }
        pos.needsUpdate = true; geo.computeVertexNormals(); return geo;
    },
    mushroom: () => {
        // Cap on top + cylindrical stem below
        const geo = new THREE.SphereGeometry(1, 48, 40);
        const pos = geo.attributes.position;
        for (let i = 0; i < pos.count; i++) {
            let x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
            const len = Math.sqrt(x*x + y*y + z*z) || 1;
            const nx = x/len, nz = z/len;
            const horizR = Math.sqrt(nx*nx + nz*nz);
            if (y > -0.15) {
                // Cap: dome shape, wide and flattened
                const capY = Math.max(0, y);
                const capR = 1.15 * Math.sqrt(Math.max(0, 1.0 - capY * capY * 0.8));
                x = nx * capR * 1.1;
                z = nz * capR * 1.1;
                y = capY * 0.45 + 0.15; // flatten
                // Underside curl
                if (y < 0.2 && horizR > 0.6) {
                    y -= (horizR - 0.6) * 0.3;
                }
            } else {
                // Stem: narrow tapered cylinder
                const stemT = Math.min(1, (-0.15 - y) / 0.85);
                const stemR = 0.22 + 0.06 * Math.sin(stemT * 3) - stemT * 0.04;
                x = nx * stemR / Math.max(0.01, horizR);
                z = nz * stemR / Math.max(0.01, horizR);
                y = -0.15 - stemT * 1.1;
                // Slight bulge at base
                if (stemT > 0.8) { const b = (stemT - 0.8) / 0.2; x *= 1.0 + b * 0.3; z *= 1.0 + b * 0.3; }
            }
            pos.setXYZ(i, x, y, z);
        }
        pos.needsUpdate = true; geo.computeVertexNormals(); return geo;
    },
    banana: () => {
        // Elongated curved cylinder tapered at ends
        const geo = new THREE.SphereGeometry(1, 32, 56);
        const pos = geo.attributes.position;
        for (let i = 0; i < pos.count; i++) {
            let x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
            const len = Math.sqrt(x*x + y*y + z*z) || 1;
            // Normalize
            const ny = y / len;
            // Map Y from -1..1 to banana length
            const t = (ny + 1) / 2; // 0 = bottom, 1 = top
            // Cross-section radius: tapers at both ends
            const crossR = 0.28 * Math.sin(t * Math.PI) * (1.0 - 0.3 * Math.pow(Math.abs(t - 0.5) * 2, 3));
            // Banana curve: arc in XY plane
            const curveAngle = (t - 0.5) * Math.PI * 0.7;
            const cx = Math.sin(curveAngle) * 0.8;
            const cy = Math.cos(curveAngle) * 1.4 - 0.7;
            // Angular position around cross-section
            const nx2 = x / len, nz2 = z / len;
            const angR = Math.sqrt(nx2*nx2 + nz2*nz2) || 0.01;
            x = cx + (nx2 / angR) * crossR;
            z = (nz2 / angR) * crossR;
            y = cy;
            // Slight pentagon cross-section (banana ridges)
            const cTheta = Math.atan2(nz2, nx2);
            const ridge = 1.0 + 0.08 * Math.cos(cTheta * 5);
            x = cx + (nx2 / angR) * crossR * ridge;
            z = (nz2 / angR) * crossR * ridge;
            pos.setXYZ(i, x, y, z);
        }
        pos.needsUpdate = true; geo.computeVertexNormals(); return geo;
    },
    berry: () => {
        const geo = new THREE.SphereGeometry(0.85, 48, 36);
        const pos = geo.attributes.position;
        for (let i = 0; i < pos.count; i++) {
            let x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
            const len = Math.sqrt(x*x + y*y + z*z) || 1;
            const theta = Math.atan2(z, x);
            const phi = Math.acos(Math.max(-1, Math.min(1, y / len)));
            // Individual drupelets (bumpy spheres packed together)
            const bump = 0.07 * Math.sin(theta * 7 + 0.5) * Math.sin(phi * 6)
                       + 0.04 * Math.cos(theta * 11 + 1.2) * Math.cos(phi * 9);
            const r = len + bump;
            // Slightly flatten top for calyx
            const topFlatten = y > 0.6 ? 1.0 - (y - 0.6) * 0.3 : 1.0;
            pos.setXYZ(i, x / len * r, y / len * r * topFlatten, z / len * r);
        }
        pos.needsUpdate = true; geo.computeVertexNormals(); return geo;
    },
    clover: () => {
        // Three heart-shaped leaves on a stem
        const geo = new THREE.SphereGeometry(1, 64, 36);
        const pos = geo.attributes.position;
        for (let i = 0; i < pos.count; i++) {
            let x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
            const len = Math.sqrt(x*x + y*y + z*z) || 1;
            const nx = x/len, ny = y/len, nz = z/len;
            const theta = Math.atan2(nz, nx);
            if (ny > -0.3) {
                // Leaves: 3 lobes using cardioid-like function
                const lobeAngle = theta * 1.5; // 3 lobes
                const lobeR = 0.5 + 0.5 * Math.pow(Math.abs(Math.sin(lobeAngle)), 0.6);
                // Flatten leaves vertically
                const flatY = ny * 0.25 + 0.15;
                const leafScale = 1.0 - Math.pow(Math.max(0, ny), 2) * 0.5;
                x = nx * lobeR * leafScale;
                z = nz * lobeR * leafScale;
                y = flatY;
            } else {
                // Stem
                const stemT = (-0.3 - ny) / 0.7;
                x = nx * 0.08;
                z = nz * 0.08;
                y = -0.3 - stemT * 1.0;
            }
            pos.setXYZ(i, x, y, z);
        }
        pos.needsUpdate = true; geo.computeVertexNormals(); return geo;
    },
    pear: () => {
        const geo = new THREE.SphereGeometry(1, 48, 36);
        const pos = geo.attributes.position;
        for (let i = 0; i < pos.count; i++) {
            let x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
            const len = Math.sqrt(x*x + y*y + z*z) || 1;
            const ny = y / len;
            // Pear profile: narrow neck at top, wide bottom
            let bodyR;
            if (ny > 0.3) {
                // Neck narrows toward top
                const t = (ny - 0.3) / 0.7;
                bodyR = 0.55 - t * 0.25;
            } else {
                // Wide bottom belly
                const t = (0.3 - ny) / 1.3;
                bodyR = 0.55 + 0.45 * Math.sin(t * Math.PI);
            }
            // Top dimple
            if (ny > 0.85) {
                bodyR *= 1.0 - (ny - 0.85) * 2;
            }
            x = x / len * bodyR;
            z = z / len * bodyR;
            // Stretch vertically for pear proportions
            y *= 1.2;
            pos.setXYZ(i, x, y, z);
        }
        pos.needsUpdate = true; geo.computeVertexNormals(); return geo;
    },
    strawberry: () => {
        const geo = new THREE.SphereGeometry(1, 48, 36);
        const pos = geo.attributes.position;
        for (let i = 0; i < pos.count; i++) {
            let x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
            const len = Math.sqrt(x*x + y*y + z*z) || 1;
            const ny = y / len;
            const theta = Math.atan2(z, x);
            const phi = Math.acos(Math.max(-1, Math.min(1, ny)));
            // Strawberry: wide shoulders, tapers to point at bottom
            let bodyR;
            if (ny > 0.2) {
                // Wide top with slight shoulder roundness
                bodyR = 0.85 + 0.1 * Math.cos((ny - 0.2) * 2);
            } else {
                // Taper to point
                const t = (0.2 - ny) / 1.2;
                bodyR = 0.85 * (1.0 - t * t);
                bodyR = Math.max(0.04, bodyR);
            }
            // Seed dimples
            const seeds = 0.03 * Math.sin(theta * 8 + phi * 0.5) * Math.sin(phi * 7);
            bodyR += seeds;
            x = x / len * bodyR;
            z = z / len * bodyR;
            // Elongate slightly
            y *= 1.15;
            pos.setXYZ(i, x, y, z);
        }
        pos.needsUpdate = true; geo.computeVertexNormals(); return geo;
    },
};

let currentShape = 'sphere';
let mesh = null;
let wireframe = null;
const material = new THREE.MeshStandardMaterial({
    color: 0xa8c3e6, roughness: 0.35, metalness: 0.1,
    flatShading: false, side: THREE.DoubleSide,
});

// Track last added shape for array duplication
let lastAddedRange = null; // { start: vertexIndex, count: numVertices }

// ============================================================
// MULTI-MESH: separate objects that can be individually selected/moved
// ============================================================
const sceneObjects = []; // array of { mesh, name, wireframe }
let selectedObject = null;
let selectedOutline = null;
const outlineMat = new THREE.MeshBasicMaterial({
    color: 0xffdd57, transparent: true, opacity: 0.3,
    side: THREE.BackSide, depthWrite: false,
});

function addSceneObject(shapeName, worldPos, scale, rotation) {
    const geo = SHAPES[shapeName]().toNonIndexed();
    geo.computeVertexNormals();
    const mat = material.clone();
    const obj = new THREE.Mesh(geo, mat);
    // Convert world position to main mesh's local space so object rotates with mesh
    const localPos = worldPos.clone().applyMatrix4(new THREE.Matrix4().copy(mesh.matrixWorld).invert());
    obj.position.copy(localPos);
    if (scale) obj.scale.setScalar(scale);
    if (rotation) obj.rotation.copy(rotation);
    mesh.add(obj); // child of main mesh — moves/rotates together
    const wire = new THREE.LineSegments(
        new THREE.EdgesGeometry(geo, 15),
        new THREE.LineBasicMaterial({ color: 0x222222, transparent: true, opacity: 0.12 })
    );
    obj.add(wire);
    const entry = { mesh: obj, name: shapeName, wireframe: wire };
    sceneObjects.push(entry);
    return entry;
}

function removeSceneObject(entry) {
    if (!entry) return;
    mesh.remove(entry.mesh); // remove from main mesh parent
    entry.mesh.geometry.dispose();
    entry.mesh.material.dispose();
    if (entry.wireframe) entry.wireframe.geometry.dispose();
    const idx = sceneObjects.indexOf(entry);
    if (idx !== -1) sceneObjects.splice(idx, 1);
    if (selectedObject === entry) deselectObject();
}

let selectedBoxHelper = null; // bounding box wireframe
let selectedAxesHelper = null; // RGB axis arrows

function selectObject(entry) {
    deselectObject();
    selectedObject = entry;

    // Yellow outline
    const outlineGeo = entry.mesh.geometry.clone();
    selectedOutline = new THREE.Mesh(outlineGeo, outlineMat);
    entry.mesh.updateMatrixWorld(true);
    selectedOutline.matrixAutoUpdate = false;
    selectedOutline.matrix.copy(entry.mesh.matrixWorld).scale(new THREE.Vector3(1.05, 1.05, 1.05));
    scene.add(selectedOutline);

    // Bounding box wireframe (dark orange)
    selectedBoxHelper = new THREE.BoxHelper(entry.mesh, 0xe07020);
    scene.add(selectedBoxHelper);

    // Axis helper (R=X, G=Y, B=Z) — extends well outside the object
    entry.mesh.geometry.computeBoundingSphere();
    const axisSize = (entry.mesh.geometry.boundingSphere?.radius || 1) * entry.mesh.scale.x * 2.5;
    selectedAxesHelper = new THREE.AxesHelper(axisSize);
    selectedAxesHelper.material.depthTest = false;
    selectedAxesHelper.renderOrder = 999;
    entry.mesh.add(selectedAxesHelper);

    setCommandStatus(`Selected: ${SHAPE_LABELS[entry.name] || entry.name} — drag to move, scroll to resize, Delete to remove`);
}

function deselectObject() {
    if (selectedOutline) {
        scene.remove(selectedOutline);
        selectedOutline.geometry.dispose();
        selectedOutline = null;
    }
    if (selectedBoxHelper) {
        scene.remove(selectedBoxHelper);
        selectedBoxHelper.dispose();
        selectedBoxHelper = null;
    }
    if (selectedAxesHelper && selectedObject) {
        selectedObject.mesh.remove(selectedAxesHelper);
        selectedAxesHelper.dispose();
        selectedAxesHelper = null;
    }
    selectedObject = null;
}

function updateSelectedOutline() {
    if (!selectedObject) return;
    selectedObject.mesh.updateMatrixWorld(true);
    if (selectedOutline) {
        selectedOutline.matrix.copy(selectedObject.mesh.matrixWorld).scale(new THREE.Vector3(1.05, 1.05, 1.05));
    }
    if (selectedBoxHelper) {
        selectedBoxHelper.update();
    }
    // Update axes size if scale changed
    if (selectedAxesHelper) {
        selectedObject.mesh.geometry.computeBoundingSphere();
        const newSize = (selectedObject.mesh.geometry.boundingSphere?.radius || 1) * selectedObject.mesh.scale.x * 2.5;
        selectedObject.mesh.remove(selectedAxesHelper);
        selectedAxesHelper.dispose();
        selectedAxesHelper = new THREE.AxesHelper(newSize);
        selectedAxesHelper.material.depthTest = false;
        selectedAxesHelper.renderOrder = 999;
        selectedObject.mesh.add(selectedAxesHelper);
    }
}

function mergeAllObjects() {
    if (sceneObjects.length === 0) return;
    saveUndo();
    for (const entry of [...sceneObjects]) {
        // Since objects are children of mesh, use their local matrix directly
        const objGeo = entry.mesh.geometry.clone();
        objGeo.applyMatrix4(entry.mesh.matrix);
        const niPos = objGeo.attributes.position;
        const eg = mesh.geometry;
        const mp = new Float32Array(eg.attributes.position.count * 3 + niPos.count * 3);
        mp.set(eg.attributes.position.array, 0);
        mp.set(niPos.array, eg.attributes.position.count * 3);
        const mg = new THREE.BufferGeometry();
        mg.setAttribute('position', new THREE.BufferAttribute(mp, 3));
        mg.computeVertexNormals();
        mesh.geometry.dispose();
        mesh.geometry = mg;
        objGeo.dispose();
        mesh.remove(entry.mesh);
        entry.mesh.geometry.dispose();
        entry.mesh.material.dispose();
    }
    sceneObjects.length = 0;
    deselectObject();
    refreshWireframe();
    setCommandStatus('All objects merged into main mesh!');
}

// Texture presets
const TEXTURES = {
    matte:     { roughness: 0.9,  metalness: 0.0,  transparent: false, opacity: 1,    wireframe: false },
    glossy:    { roughness: 0.15, metalness: 0.1,  transparent: false, opacity: 1,    wireframe: false },
    metallic:  { roughness: 0.2,  metalness: 0.8,  transparent: false, opacity: 1,    wireframe: false },
    wireframe: { roughness: 0.5,  metalness: 0.0,  transparent: false, opacity: 1,    wireframe: true  },
    clay:      { roughness: 0.35, metalness: 0.1,  transparent: false, opacity: 1,    wireframe: false },
    glass:     { roughness: 0.05, metalness: 0.1,  transparent: true,  opacity: 0.4,  wireframe: false },
};

function applyTexture(name) {
    const t = TEXTURES[name];
    if (!t) return;
    Object.assign(material, t);
    material.needsUpdate = true;
}

function setShape(name) {
    if (mesh) { scene.remove(mesh); mesh.geometry.dispose(); }
    const geo = SHAPES[name]().toNonIndexed();
    geo.computeVertexNormals();
    mesh = new THREE.Mesh(geo, material);
    refreshWireframe();
    scene.add(mesh);
    currentShape = name;
    syncShapeUI(name);
}

function refreshWireframe() {
    if (wireframe) { mesh.remove(wireframe); wireframe.geometry.dispose(); }
    wireframe = new THREE.LineSegments(
        new THREE.EdgesGeometry(mesh.geometry, 15),
        new THREE.LineBasicMaterial({ color: 0x222222, transparent: true, opacity: 0.12 })
    );
    mesh.add(wireframe);
}

function addShapeToMesh(name, ox, oy, oz, scale, localNormal, userRotation) {
    const ng = SHAPES[name]();
    ng.scale(scale, scale, scale);

    // Align shape's "up" (Y) with the surface normal at placement point
    const up = new THREE.Vector3(0, 1, 0);
    const norm = localNormal ? localNormal.clone().normalize() : up.clone();
    const rotQ = new THREE.Quaternion();
    rotQ.setFromUnitVectors(up, norm);

    // Apply user rotation on top of normal alignment
    if (userRotation) {
        const userQ = new THREE.Quaternion().setFromEuler(userRotation);
        rotQ.multiply(userQ);
    }

    ng.applyQuaternion(rotQ);

    // Translate to placement point
    ng.translate(ox, oy, oz);

    const ni = ng.toNonIndexed();

    // Merge geometries — shape maintains its original form
    const niPos = ni.attributes.position;
    const eg = mesh.geometry;
    const prevCount = eg.attributes.position.count;
    const mp = new Float32Array(prevCount * 3 + niPos.count * 3);
    mp.set(eg.attributes.position.array, 0);
    mp.set(niPos.array, prevCount * 3);

    const mg = new THREE.BufferGeometry();
    mg.setAttribute('position', new THREE.BufferAttribute(mp, 3));
    mg.computeVertexNormals();
    mesh.geometry.dispose();
    mesh.geometry = mg;
    refreshWireframe();

    // Track last added shape vertices for array duplication
    lastAddedRange = { start: prevCount, count: niPos.count };

    ng.dispose(); ni.dispose();
}

// ============================================================
// 2c. PATTERN WRAP — duplicate a shape around the object surface
// ============================================================
function patternWrap(shapeName, scale, count, mode) {
    // mode: 'ring' (horizontal), 'vertical', 'all'
    saveUndo();
    const pos = mesh.geometry.attributes.position;

    // Find mesh center
    let cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < pos.count; i++) {
        cx += pos.getX(i); cy += pos.getY(i); cz += pos.getZ(i);
    }
    cx /= pos.count; cy /= pos.count; cz /= pos.count;
    const center = new THREE.Vector3(cx, cy, cz).applyMatrix4(mesh.matrixWorld);

    // Helper: raycast from center outward to find actual surface point
    const patternRay = new THREE.Raycaster();
    function findSurfacePoint(direction) {
        const dir = direction.clone().normalize();
        // Start ray from well outside the mesh, pointing inward
        const origin = center.clone().addScaledVector(dir, 10);
        patternRay.set(origin, dir.clone().negate());
        const hits = patternRay.intersectObject(mesh);
        if (hits.length > 0) {
            const hit = hits[0];
            const norm = hit.face?.normal
                ? hit.face.normal.clone().transformDirection(mesh.matrixWorld).normalize()
                : dir.clone();
            // Place slightly above surface
            const surfacePos = hit.point.clone().addScaledVector(norm, scale * 0.3);
            // Convert to mesh local space
            const invMatrix = new THREE.Matrix4().copy(mesh.matrixWorld).invert();
            const localPos = surfacePos.applyMatrix4(invMatrix);
            const localNorm = norm.transformDirection(invMatrix).normalize();
            return { pos: localPos, norm: localNorm };
        }
        return null;
    }

    let placed = 0;

    if (mode === 'ring') {
        for (let i = 0; i < count; i++) {
            const angle = (i / count) * Math.PI * 2;
            const dir = new THREE.Vector3(Math.cos(angle), 0, Math.sin(angle));
            const hit = findSurfacePoint(dir);
            if (hit) { addShapeToMesh(shapeName, hit.pos.x, hit.pos.y, hit.pos.z, scale, hit.norm); placed++; }
        }
    } else if (mode === 'vertical') {
        for (let i = 0; i < count; i++) {
            const t = (i / (count - 1 || 1)) * Math.PI;
            const dir = new THREE.Vector3(Math.sin(t), Math.cos(t), 0);
            const hit = findSurfacePoint(dir);
            if (hit) { addShapeToMesh(shapeName, hit.pos.x, hit.pos.y, hit.pos.z, scale, hit.norm); placed++; }
        }
    } else {
        // 'all' — Fibonacci spiral distribution
        const goldenAngle = Math.PI * (3 - Math.sqrt(5));
        for (let i = 0; i < count; i++) {
            const y = 1 - (i / (count - 1 || 1)) * 2;
            const radiusAtY = Math.sqrt(1 - y * y);
            const angle = goldenAngle * i;
            const dir = new THREE.Vector3(Math.cos(angle) * radiusAtY, y, Math.sin(angle) * radiusAtY);
            const hit = findSurfacePoint(dir);
            if (hit) { addShapeToMesh(shapeName, hit.pos.x, hit.pos.y, hit.pos.z, scale, hit.norm); placed++; }
        }
    }
    refreshWireframe();
    setCommandStatus(`Pattern: ${placed}× ${SHAPE_LABELS[shapeName] || shapeName} (${mode})`);
}

// ============================================================
// 2b. ADD SHAPE PREVIEW SYSTEM
// ============================================================
const addShapePreviewMat = new THREE.MeshStandardMaterial({
    color: 0xffdd57,
    transparent: true,
    opacity: 0.35,
    roughness: 0.5,
    metalness: 0.0,
    side: THREE.DoubleSide,
    depthWrite: false,
});
const addShapePreviewWireMat = new THREE.LineBasicMaterial({
    color: 0xffdd57,
    transparent: true,
    opacity: 0.7,
});

let addShapePreviewMesh = null;
let addShapePreviewWire = null;
let addShapeMode = false;        // true when "Add" tool is active
let addShapeType = 'sphere';     // which shape to add
let addShapePosition = new THREE.Vector3(); // world position of preview
let addShapeHitNormal = new THREE.Vector3(0, 1, 0); // surface normal at placement
let addShapeScale = 0.5;         // current preview scale
let addShapeBaseRadius = 1;      // geometry bounding sphere radius at scale 1
let addShapeBaseHandScale = null; // baseline hand scale for cupped resizing
let addShapeVisible = false;     // is there a valid placement point?
let addShapeSurfaceHit = new THREE.Vector3(); // raw surface hit point (before offset)
let addShapeRotation = new THREE.Euler(0, 0, 0); // user rotation of preview

function createAddShapePreview(shapeName) {
    // Remove old preview
    removeAddShapePreview();
    const geo = SHAPES[shapeName]();
    geo.computeBoundingSphere();
    addShapeBaseRadius = geo.boundingSphere ? geo.boundingSphere.radius : 1;
    addShapePreviewMesh = new THREE.Mesh(geo, addShapePreviewMat);
    addShapePreviewMesh.renderOrder = 500;
    addShapePreviewMesh.visible = false;
    scene.add(addShapePreviewMesh);

    addShapePreviewWire = new THREE.LineSegments(
        new THREE.EdgesGeometry(geo, 15),
        addShapePreviewWireMat
    );
    addShapePreviewWire.renderOrder = 501;
    addShapePreviewMesh.add(addShapePreviewWire);
}

function removeAddShapePreview() {
    if (addShapePreviewMesh) {
        scene.remove(addShapePreviewMesh);
        addShapePreviewMesh.geometry.dispose();
        if (addShapePreviewWire) addShapePreviewWire.geometry.dispose();
        addShapePreviewMesh = null;
        addShapePreviewWire = null;
    }
}

function updateAddShapePreview(worldPos, scale) {
    if (!addShapePreviewMesh) return;
    addShapePreviewMesh.position.copy(worldPos);
    addShapePreviewMesh.scale.setScalar(scale);
    // Apply user rotation on top of normal-alignment
    const normalQ = new THREE.Quaternion();
    normalQ.setFromUnitVectors(new THREE.Vector3(0, 1, 0), addShapeHitNormal.clone().normalize());
    const userQ = new THREE.Quaternion().setFromEuler(addShapeRotation);
    normalQ.multiply(userQ);
    addShapePreviewMesh.quaternion.copy(normalQ);
    addShapePreviewMesh.visible = true;
    addShapeVisible = true;
}

// Recompute preview position from stored surface hit point at current scale
function recomputeAddShapePosition() {
    if (!addShapeVisible || addShapeHitNormal.lengthSq() === 0) return;
    const surfaceOffset = addShapeBaseRadius * addShapeScale * 0.35;
    addShapePosition.copy(addShapeSurfaceHit).addScaledVector(addShapeHitNormal, surfaceOffset);
    if (addShapePreviewMesh) {
        addShapePreviewMesh.position.copy(addShapePosition);
    }
}

function hideAddShapePreview() {
    if (addShapePreviewMesh) addShapePreviewMesh.visible = false;
    addShapeVisible = false;
}

function enterAddShapeMode(shapeName) {
    addShapeMode = true;
    addShapeType = shapeName || 'sphere';
    addShapeBaseHandScale = null;
    addShapeRotation.set(0, 0, 0);
    createAddShapePreview(addShapeType);
    // Normalize default scale: target ~0.3 units radius regardless of shape
    addShapeScale = addShapeBaseRadius > 0.01 ? 0.3 / addShapeBaseRadius : 0.5;
    setCommandStatus(`Add Shape: ${SHAPE_LABELS[addShapeType] || addShapeType} — hover to position, scroll to resize, click to place`);
    const addBtn = document.getElementById('add-shape-btn');
    if (addBtn) addBtn.classList.add('active');
}

function exitAddShapeMode() {
    addShapeMode = false;
    addShapeBaseHandScale = null;
    gestureState._smoothSpan = null;
    removeAddShapePreview();
    setCommandStatus('Ready');
    const addBtn = document.getElementById('add-shape-btn');
    if (addBtn) addBtn.classList.remove('active');
    const picker = document.getElementById('add-shape-picker');
    if (picker) picker.style.display = 'none';
}

function confirmAddShape() {
    if (!addShapeMode || !addShapeVisible) return;
    // Add as separate scene object instead of merging
    const entry = addSceneObject(addShapeType, addShapePosition, addShapeScale,
        addShapePreviewMesh ? addShapePreviewMesh.rotation : null);
    selectObject(entry);
    setCommandStatus(`Placed ${SHAPE_LABELS[addShapeType] || addShapeType}! Drag to move.`);
    exitAddShapeMode();
}

// ============================================================
// 2d. DUPLICATE — click on mesh to pick up region, drag to place copy
// ============================================================
let dupMode = false;
let dupPreviewMesh = null;
let dupGeometry = null;
let dupHitNormal = new THREE.Vector3();
let dupPosition = new THREE.Vector3();
let dupRadius = 0.5;

function startDuplicate(hitPoint, hitNormal) {
    const p = mesh.geometry.attributes.position;
    const invMatrix = new THREE.Matrix4().copy(mesh.matrixWorld).invert();
    const localHit = hitPoint.clone().applyMatrix4(invMatrix);
    const r2 = dupRadius * dupRadius;
    const verts = [];
    for (let i = 0; i < p.count; i += 3) {
        let inRange = false;
        for (let v = 0; v < 3; v++) {
            const dx = p.getX(i+v) - localHit.x, dy = p.getY(i+v) - localHit.y, dz = p.getZ(i+v) - localHit.z;
            if (dx*dx + dy*dy + dz*dz < r2) { inRange = true; break; }
        }
        if (inRange) {
            for (let v = 0; v < 3; v++) verts.push(p.getX(i+v) - localHit.x, p.getY(i+v) - localHit.y, p.getZ(i+v) - localHit.z);
        }
    }
    if (verts.length < 9) return;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(verts), 3));
    geo.computeVertexNormals();
    dupGeometry = geo;
    if (dupPreviewMesh) { scene.remove(dupPreviewMesh); dupPreviewMesh.geometry.dispose(); }
    dupPreviewMesh = new THREE.Mesh(geo, addShapePreviewMat);
    dupPreviewMesh.renderOrder = 500;
    scene.add(dupPreviewMesh);
    dupHitNormal.copy(hitNormal);
    dupPosition.copy(hitPoint);
    dupPreviewMesh.position.copy(hitPoint);
    dupMode = true;
    setCommandStatus('Drag to place duplicate — release to set');
}

function updateDuplicate(screenX, screenY) {
    if (!dupMode || !dupPreviewMesh) return;
    const ndc = new THREE.Vector2((screenX / innerWidth) * 2 - 1, -(screenY / innerHeight) * 2 + 1);
    raycaster.setFromCamera(ndc, camera);
    const hits = raycaster.intersectObjects([mesh]);
    if (hits.length) {
        const hitNorm = hits[0].face?.normal ? hits[0].face.normal.clone().transformDirection(mesh.matrixWorld).normalize() : new THREE.Vector3(0,0,1);
        dupPosition.copy(hits[0].point).addScaledVector(hitNorm, 0.05);
    } else {
        const dir = new THREE.Vector3(); raycaster.ray.direction.normalize();
        dir.copy(raycaster.ray.direction).multiplyScalar(4);
        dupPosition.copy(raycaster.ray.origin).add(dir);
    }
    dupPreviewMesh.position.copy(dupPosition);
}

function confirmDuplicate() {
    if (!dupMode || !dupPreviewMesh || !dupGeometry) return;
    saveUndo();
    const invMatrix = new THREE.Matrix4().copy(mesh.matrixWorld).invert();
    const localPos = dupPosition.clone().applyMatrix4(invMatrix);
    const srcPos = dupGeometry.attributes.position;
    const newVerts = new Float32Array(srcPos.count * 3);
    for (let i = 0; i < srcPos.count; i++) {
        newVerts[i*3] = srcPos.getX(i) + localPos.x;
        newVerts[i*3+1] = srcPos.getY(i) + localPos.y;
        newVerts[i*3+2] = srcPos.getZ(i) + localPos.z;
    }
    const eg = mesh.geometry;
    const prevCount = eg.attributes.position.count;
    const addCount = newVerts.length / 3;
    const mp = new Float32Array(prevCount * 3 + newVerts.length);
    mp.set(eg.attributes.position.array, 0);
    mp.set(newVerts, prevCount * 3);
    const mg = new THREE.BufferGeometry();
    mg.setAttribute('position', new THREE.BufferAttribute(mp, 3));
    safeComputeNormals();
    mesh.geometry.dispose();
    mesh.geometry = mg;
    safeComputeNormals();
    refreshWireframe();
    lastAddedRange = { start: prevCount, count: addCount };
    setCommandStatus('Duplicate placed!');
    cancelDuplicate();
}

function cancelDuplicate() {
    dupMode = false;
    if (dupPreviewMesh) { scene.remove(dupPreviewMesh); dupPreviewMesh.geometry.dispose(); dupPreviewMesh = null; }
    if (dupGeometry) { dupGeometry.dispose(); dupGeometry = null; }
}

// Start with empty canvas — just create an invisible container mesh
const emptyGeo = new THREE.BufferGeometry();
emptyGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(0), 3));
mesh = new THREE.Mesh(emptyGeo, material);
scene.add(mesh);

// ============================================================
// 3. UNDO SYSTEM
// ============================================================
const undoStack = [];
function snapshot() {
    const pos = mesh.geometry.attributes.position;
    const norm = mesh.geometry.attributes.normal;
    return {
        positions: new Float32Array(pos.array),
        normals: norm ? new Float32Array(norm.array) : null,
        color: material.color.getHex(),
        scale: mesh.scale.toArray(),
        position: mesh.position.toArray(),
        rotation: [mesh.rotation.x, mesh.rotation.y, mesh.rotation.z],
    };
}
function saveUndo() {
    undoStack.push(snapshot());
    if (undoStack.length > 30) undoStack.shift();
}
function undo() {
    if (!undoStack.length) {
        setCommandStatus('Nothing to undo');
        return;
    }
    const s = undoStack.pop();
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(s.positions, 3));
    if (s.normals) g.setAttribute('normal', new THREE.BufferAttribute(s.normals, 3));
    mesh.geometry.dispose();
    mesh.geometry = g;
    safeComputeNormals();
    refreshWireframe();
    material.color.setHex(s.color);
    // Fix black glitch: reset vertexColors if geometry has no color attribute
    if (!g.attributes.color) {
        material.vertexColors = false;
    }
    material.needsUpdate = true;
    mesh.scale.fromArray(s.scale);
    mesh.position.fromArray(s.position);
    mesh.rotation.set(s.rotation[0], s.rotation[1], s.rotation[2]);
    // Re-apply abstract gradient + update rest positions if in abstract mode
    if (currentAppMode === 'abstract') {
        applyBlobGradient();
        if (abstractRestPositions) {
            abstractRestPositions = new Float32Array(mesh.geometry.attributes.position.array);
        }
    }
    setCommandStatus(`Undone (${undoStack.length} left)`);
}

// ============================================================
// 4. SCULPT / MODELING TOOLS
// ============================================================
let brushRadius = 0.6;
let sculptStrength = 0.04;
const _v = new THREE.Vector3(), _lp = new THREE.Vector3(), _ln = new THREE.Vector3(), _n = new THREE.Vector3();
const _moldRight = new THREE.Vector3(), _moldUp = new THREE.Vector3(), _moldLocalD = new THREE.Vector3();
const _invMatrix = new THREE.Matrix4();
let MAX_VERTEX_DISPLACEMENT = 0.15;

// --- Face highlight system ---
let highlightMesh = null;
const highlightMat = new THREE.MeshBasicMaterial({
    color: 0xffdd57, transparent: true, opacity: 0.25,
    side: THREE.DoubleSide, depthWrite: false, depthTest: true,
});
let lastHighlightPoint = null;
let lastHighlightNormal = null;

function highlightFacesAt(point, normal) {
    if (!mesh) return;
    lastHighlightPoint = point.clone();
    lastHighlightNormal = normal.clone();
    const p = mesh.geometry.attributes.position;
    _invMatrix.copy(mesh.matrixWorld).invert();
    _lp.copy(point).applyMatrix4(_invMatrix);
    const r2 = brushRadius * brushRadius;

    // Collect triangles within brush radius
    const verts = [];
    for (let i = 0; i < p.count; i += 3) {
        // Check if any vertex of this triangle is within radius
        let inRange = false;
        for (let v = 0; v < 3; v++) {
            _v.fromBufferAttribute(p, i + v);
            const dx = _v.x - _lp.x, dy = _v.y - _lp.y, dz = _v.z - _lp.z;
            if (dx*dx + dy*dy + dz*dz < r2) { inRange = true; break; }
        }
        if (inRange) {
            for (let v = 0; v < 3; v++) {
                _v.fromBufferAttribute(p, i + v);
                verts.push(_v.x, _v.y, _v.z);
            }
        }
    }

    if (verts.length === 0) {
        if (highlightMesh) highlightMesh.visible = false;
        return;
    }

    // Build or update highlight mesh
    const arr = new Float32Array(verts);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(arr, 3));
    geo.computeVertexNormals();

    if (highlightMesh) {
        highlightMesh.geometry.dispose();
        highlightMesh.geometry = geo;
    } else {
        highlightMesh = new THREE.Mesh(geo, highlightMat);
        highlightMesh.renderOrder = 900;
    }
    // Match the mesh transform
    highlightMesh.position.copy(mesh.position);
    highlightMesh.rotation.copy(mesh.rotation);
    highlightMesh.scale.copy(mesh.scale);
    highlightMesh.visible = true;
    if (!highlightMesh.parent) scene.add(highlightMesh);
}

function hideHighlight() {
    if (highlightMesh) highlightMesh.visible = false;
    lastHighlightPoint = null;
    lastHighlightNormal = null;
} // increased for clay-like sculpting

function getHitNormal(hit) {
    return hit?.face?.normal ? hit.face.normal.clone() : new THREE.Vector3(0, 0, 1);
}

// Safely recompute normals without losing vertex colors
function safeComputeNormals() {
    mesh.geometry.computeVertexNormals();
    // If material expects vertex colors but geometry has none → turn off to prevent black
    if (material.vertexColors && !mesh.geometry.attributes.color) {
        material.vertexColors = false;
        material.needsUpdate = true;
    }
}

function extrudeAt(point, normal, strength) {
    const p = mesh.geometry.attributes.position;
    _invMatrix.copy(mesh.matrixWorld).invert();
    _lp.copy(point).applyMatrix4(_invMatrix);
    _ln.copy(normal).transformDirection(_invMatrix).normalize();
    const r2 = brushRadius * brushRadius;
    let mod = false;
    const clamped = Math.max(-MAX_VERTEX_DISPLACEMENT, Math.min(MAX_VERTEX_DISPLACEMENT, strength));
    for (let i = 0; i < p.count; i++) {
        _v.fromBufferAttribute(p, i);
        const dx = _v.x - _lp.x, dy = _v.y - _lp.y, dz = _v.z - _lp.z;
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < r2) {
            const f = 0.5 * (1 + Math.cos(Math.PI * Math.sqrt(d2) / brushRadius));
            p.setXYZ(i, _v.x + _ln.x * clamped * f, _v.y + _ln.y * clamped * f, _v.z + _ln.z * clamped * f);
            mod = true;
        }
    }
    if (mod) { p.needsUpdate = true; safeComputeNormals(); }
}

function smoothAt(point) {
    const p = mesh.geometry.attributes.position;
    _invMatrix.copy(mesh.matrixWorld).invert();
    _lp.copy(point).applyMatrix4(_invMatrix);
    const r2 = brushRadius * brushRadius;
    const aff = [];
    for (let i = 0; i < p.count; i++) {
        _v.fromBufferAttribute(p, i);
        const dx = _v.x - _lp.x, dy = _v.y - _lp.y, dz = _v.z - _lp.z;
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < r2) aff.push({ i, x: _v.x, y: _v.y, z: _v.z, d: Math.sqrt(d2) });
    }
    if (aff.length < 2) return;
    let ax = 0, ay = 0, az = 0;
    for (const a of aff) { ax += a.x; ay += a.y; az += a.z; }
    ax /= aff.length; ay /= aff.length; az /= aff.length;
    for (const a of aff) {
        const f = 0.4 * (0.5 * (1 + Math.cos(Math.PI * a.d / brushRadius)));
        p.setXYZ(a.i, a.x + (ax - a.x) * f, a.y + (ay - a.y) * f, a.z + (az - a.z) * f);
    }
    p.needsUpdate = true;
    safeComputeNormals();
}

// Smooth entire mesh — fixes edges, cleans up sharp transitions
function smoothAllEdges(passes = 3) {
    saveUndo();
    const p = mesh.geometry.attributes.position;
    const count = p.count;

    for (let pass = 0; pass < passes; pass++) {
        // For each triangle, average each vertex with its neighbors
        for (let i = 0; i < count; i += 3) {
            const ax = p.getX(i), ay = p.getY(i), az = p.getZ(i);
            const bx = p.getX(i+1), by = p.getY(i+1), bz = p.getZ(i+1);
            const cx = p.getX(i+2), cy = p.getY(i+2), cz = p.getZ(i+2);
            const mx = (ax+bx+cx)/3, my = (ay+by+cy)/3, mz = (az+bz+cz)/3;
            const blend = 0.25;
            p.setXYZ(i,   ax + (mx-ax)*blend, ay + (my-ay)*blend, az + (mz-az)*blend);
            p.setXYZ(i+1, bx + (mx-bx)*blend, by + (my-by)*blend, bz + (mz-bz)*blend);
            p.setXYZ(i+2, cx + (mx-cx)*blend, cy + (my-cy)*blend, cz + (mz-cz)*blend);
        }
    }

    p.needsUpdate = true;
    safeComputeNormals();
    refreshWireframe();
    setCommandStatus(`Smoothed edges (${passes} passes)! ✨`);
}

function bevelAt(point, normal, strength) {
    // Soft curved bevel: gentle extrude with wide falloff + multiple smooth passes
    softSculptAt(point, normal, strength);
}

// Soft cloth-like sculpt: wide area, tiny displacement, heavy smoothing
// Produces fluid, fabric-like deformations
function softSculptAt(point, normal, strength) {
    const p = mesh.geometry.attributes.position;
    _invMatrix.copy(mesh.matrixWorld).invert();
    _lp.copy(point).applyMatrix4(_invMatrix);
    _ln.copy(normal).transformDirection(_invMatrix).normalize();

    // Use wider radius than brush for softer falloff
    const softRadius = brushRadius * 1.8;
    const r2 = softRadius * softRadius;
    // Very small displacement per frame — builds up with continuous drag
    const disp = Math.max(-0.02, Math.min(0.02, strength));
    let mod = false;

    for (let i = 0; i < p.count; i++) {
        _v.fromBufferAttribute(p, i);
        const dx = _v.x - _lp.x, dy = _v.y - _lp.y, dz = _v.z - _lp.z;
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < r2) {
            const dist = Math.sqrt(d2) / softRadius;
            // Very smooth quintic falloff — like pressing into soft fabric
            const t = 1.0 - dist;
            const f = t * t * t * (t * (t * 6 - 15) + 10); // smootherstep
            p.setXYZ(i,
                _v.x + _ln.x * disp * f,
                _v.y + _ln.y * disp * f,
                _v.z + _ln.z * disp * f
            );
            mod = true;
        }
    }
    if (mod) {
        p.needsUpdate = true;
        // Multiple smooth passes — blends the displacement into surrounding area
        for (let s = 0; s < 4; s++) smoothAt(point);
        safeComputeNormals();
    }
}

function moldAt(point, dx, dy) {
    // Gentle grab — limited displacement for shape preservation
    const p = mesh.geometry.attributes.position;
    _invMatrix.copy(mesh.matrixWorld).invert();
    _lp.copy(point).applyMatrix4(_invMatrix);
    _moldRight.setFromMatrixColumn(camera.matrixWorld, 0).normalize();
    _moldUp.setFromMatrixColumn(camera.matrixWorld, 1).normalize();
    _moldLocalD.copy(_moldRight).multiplyScalar(dx * 0.002).add(_moldUp.multiplyScalar(-dy * 0.002));
    _moldLocalD.transformDirection(_invMatrix);
    // Clamp displacement
    if (_moldLocalD.length() > MAX_VERTEX_DISPLACEMENT) _moldLocalD.setLength(MAX_VERTEX_DISPLACEMENT);
    const r2 = brushRadius * brushRadius;
    for (let i = 0; i < p.count; i++) {
        _v.fromBufferAttribute(p, i);
        const ddx = _v.x - _lp.x, ddy = _v.y - _lp.y, ddz = _v.z - _lp.z;
        const d2 = ddx * ddx + ddy * ddy + ddz * ddz;
        if (d2 < r2) {
            const f = 0.5 * (1 + Math.cos(Math.PI * Math.sqrt(d2) / brushRadius));
            p.setXYZ(i, _v.x + _moldLocalD.x * f, _v.y + _moldLocalD.y * f, _v.z + _moldLocalD.z * f);
        }
    }
    p.needsUpdate = true;
    safeComputeNormals();
}

// Mirror mesh across an axis
function mirrorMesh(axis = 'x') {
    saveUndo();
    const p = mesh.geometry.attributes.position;
    const count = p.count;
    // Duplicate all vertices mirrored
    const newPos = new Float32Array(count * 6);
    const newNorm = new Float32Array(count * 6);
    const n = mesh.geometry.attributes.normal;
    newPos.set(p.array, 0);
    newNorm.set(n.array, 0);
    for (let i = 0; i < count; i++) {
        const idx = i * 3;
        newPos[count * 3 + idx] = axis === 'x' ? -p.array[idx] : p.array[idx];
        newPos[count * 3 + idx + 1] = axis === 'y' ? -p.array[idx + 1] : p.array[idx + 1];
        newPos[count * 3 + idx + 2] = axis === 'z' ? -p.array[idx + 2] : p.array[idx + 2];
        newNorm[count * 3 + idx] = axis === 'x' ? -n.array[idx] : n.array[idx];
        newNorm[count * 3 + idx + 1] = axis === 'y' ? -n.array[idx + 1] : n.array[idx + 1];
        newNorm[count * 3 + idx + 2] = axis === 'z' ? -n.array[idx + 2] : n.array[idx + 2];
    }
    const mg = new THREE.BufferGeometry();
    mg.setAttribute('position', new THREE.BufferAttribute(newPos, 3));
    mg.setAttribute('normal', new THREE.BufferAttribute(newNorm, 3));
    mesh.geometry.dispose();
    mesh.geometry = mg;
    safeComputeNormals();
    refreshWireframe();
    setCommandStatus('Mirrored across ' + axis.toUpperCase());
}

// Array: duplicate N times along a direction
function arrayMesh(count, direction) {
    saveUndo();
    const p = mesh.geometry.attributes.position;
    const srcCount = p.count;

    const newVerts = [];
    for (let c = 1; c <= count; c++) {
        const offset = direction.clone().multiplyScalar(c * 2.0);
        for (let i = 0; i < srcCount; i++) {
            const si = i * 3;
            newVerts.push(
                p.array[si] + offset.x,
                p.array[si + 1] + offset.y,
                p.array[si + 2] + offset.z
            );
        }
    }

    const existArr = p.array;
    const addArr = new Float32Array(newVerts);
    const mp = new Float32Array(existArr.length + addArr.length);
    mp.set(existArr, 0);
    mp.set(addArr, existArr.length);

    const mg = new THREE.BufferGeometry();
    mg.setAttribute('position', new THREE.BufferAttribute(mp, 3));
    mesh.geometry.dispose();
    mesh.geometry = mg;
    safeComputeNormals();
    refreshWireframe();
    setCommandStatus(`Arrayed ${count} copies`);
}

// Split mesh cleanly in half along an axis through the center.
// Clips triangles that cross the split plane and caps the cut face.
function splitMesh(splitPos = 0, axis = 'x') {
    saveUndo();
    const p = mesh.geometry.attributes.position;

    // Find mesh center on the split axis for a true midpoint cut
    let axMin = Infinity, axMax = -Infinity;
    const ai = axis === 'x' ? 0 : axis === 'y' ? 1 : 2;
    for (let i = 0; i < p.count; i++) {
        const v = ai === 0 ? p.getX(i) : ai === 1 ? p.getY(i) : p.getZ(i);
        if (v < axMin) axMin = v;
        if (v > axMax) axMax = v;
    }
    const splitAt = (axMin + axMax) / 2;

    const getAxis = (idx) => ai === 0 ? p.getX(idx) : ai === 1 ? p.getY(idx) : p.getZ(idx);
    const getVert = (idx) => new THREE.Vector3(p.getX(idx), p.getY(idx), p.getZ(idx));

    // Interpolate vertex on the split plane between two vertices
    function lerpToPlane(vA, vB) {
        const aVal = ai === 0 ? vA.x : ai === 1 ? vA.y : vA.z;
        const bVal = ai === 0 ? vB.x : ai === 1 ? vB.y : vB.z;
        const t = (splitAt - aVal) / (bVal - aVal);
        return new THREE.Vector3().lerpVectors(vA, vB, Math.max(0, Math.min(1, t)));
    }

    const newVerts = [];  // kept half triangles
    const capVerts = [];  // vertices on the cut plane for capping

    for (let i = 0; i < p.count; i += 3) {
        const vA = getVert(i), vB = getVert(i + 1), vC = getVert(i + 2);
        const aA = ai === 0 ? vA.x : ai === 1 ? vA.y : vA.z;
        const aB = ai === 0 ? vB.x : ai === 1 ? vB.y : vB.z;
        const aC = ai === 0 ? vC.x : ai === 1 ? vC.y : vC.z;
        const sA = aA >= splitAt, sB = aB >= splitAt, sC = aC >= splitAt;
        const keepCount = (sA ? 1 : 0) + (sB ? 1 : 0) + (sC ? 1 : 0);

        if (keepCount === 3) {
            // Entire triangle on kept side
            newVerts.push(vA, vB, vC);
        } else if (keepCount === 0) {
            // Entire triangle on discarded side — skip
        } else if (keepCount === 2) {
            // Two verts on kept side — clip to produce 2 triangles
            // Sort so the lone discarded vert is first
            let lone, k1, k2;
            if (!sA) { lone = vA; k1 = vB; k2 = vC; }
            else if (!sB) { lone = vB; k1 = vA; k2 = vC; }
            else { lone = vC; k1 = vA; k2 = vB; }

            const i1 = lerpToPlane(lone, k1);
            const i2 = lerpToPlane(lone, k2);

            // Two triangles forming the clipped quad
            newVerts.push(k1, k2, i1);
            newVerts.push(i1, k2, i2);

            // Cap edge
            capVerts.push(i1, i2);
        } else {
            // One vert on kept side — clip to produce 1 triangle
            let keep, d1, d2;
            if (sA) { keep = vA; d1 = vB; d2 = vC; }
            else if (sB) { keep = vB; d1 = vA; d2 = vC; }
            else { keep = vC; d1 = vA; d2 = vB; }

            const i1 = lerpToPlane(d1, keep);
            const i2 = lerpToPlane(d2, keep);

            newVerts.push(keep, i1, i2);

            // Cap edge
            capVerts.push(i1, i2);
        }
    }

    // Build hollow bowl — no cap face.
    // Instead, create an inner wall by duplicating the outer shell offset inward,
    // then connect the rim edges to form a visible wall thickness.

    const WALL_THICKNESS = 0.08;

    // 1. Build the inner shell — offset each vertex inward toward mesh center
    const innerVerts = [];
    for (const v of newVerts) {
        const dir = v.clone().normalize(); // direction from center
        const inner = v.clone().sub(dir.multiplyScalar(WALL_THICKNESS));
        innerVerts.push(inner);
    }

    // 2. Add inner shell with reversed winding (faces point inward)
    const allVerts = [...newVerts]; // outer shell (faces outward)
    for (let i = 0; i < innerVerts.length; i += 3) {
        // Reverse winding: swap vertex 1 and 2
        allVerts.push(innerVerts[i], innerVerts[i + 2], innerVerts[i + 1]);
    }

    // 3. Connect rim — bridge outer and inner cap edges to form the wall thickness
    if (capVerts.length >= 2) {
        // Collect rim edge pairs (outer → inner)
        const center = new THREE.Vector3();
        for (const v of capVerts) center.add(v);
        center.divideScalar(capVerts.length);

        // Tangent frame on the cut plane
        const normal = new THREE.Vector3(ai === 0 ? 1 : 0, ai === 1 ? 1 : 0, ai === 2 ? 1 : 0);
        const tangent1 = new THREE.Vector3(ai === 0 ? 0 : 1, ai === 1 ? 0 : (ai === 0 ? 1 : 0), ai === 2 ? 0 : (ai === 0 ? 0 : 1)).normalize();
        const tangent2 = new THREE.Vector3().crossVectors(normal, tangent1).normalize();

        // Deduplicate close vertices
        const unique = [capVerts[0]];
        for (let i = 1; i < capVerts.length; i++) {
            let isDupe = false;
            for (const u of unique) {
                if (capVerts[i].distanceTo(u) < 0.001) { isDupe = true; break; }
            }
            if (!isDupe) unique.push(capVerts[i]);
        }

        if (unique.length >= 3) {
            // Sort by angle to form ordered rim loop
            unique.sort((a, b) => {
                const da = a.clone().sub(center);
                const db = b.clone().sub(center);
                return Math.atan2(da.dot(tangent2), da.dot(tangent1)) -
                       Math.atan2(db.dot(tangent2), db.dot(tangent1));
            });

            // For each rim edge, create a quad connecting outer to inner
            for (let i = 0; i < unique.length; i++) {
                const next = (i + 1) % unique.length;
                const outerA = unique[i];
                const outerB = unique[next];
                const dirA = outerA.clone().normalize();
                const dirB = outerB.clone().normalize();
                const innerA = outerA.clone().sub(dirA.multiplyScalar(WALL_THICKNESS));
                const innerB = outerB.clone().sub(dirB.multiplyScalar(WALL_THICKNESS));

                // Two triangles forming the rim quad
                allVerts.push(outerA, outerB, innerA);
                allVerts.push(innerA, outerB, innerB);
            }
        }
    }

    if (allVerts.length < 3) return;

    // Build new geometry
    const posArr = new Float32Array(allVerts.length * 3);
    for (let i = 0; i < allVerts.length; i++) {
        posArr[i * 3] = allVerts[i].x;
        posArr[i * 3 + 1] = allVerts[i].y;
        posArr[i * 3 + 2] = allVerts[i].z;
    }
    const mg = new THREE.BufferGeometry();
    mg.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
    mg.computeVertexNormals();
    mesh.geometry.dispose();
    mesh.geometry = mg;
    refreshWireframe();
    setCommandStatus('Split — clean cut');
}

// Distance measurement
let distancePoints = [];
let distanceLines = [];
function measureDistance(point) {
    distancePoints.push(point.clone());
    if (distancePoints.length === 2) {
        const dist = distancePoints[0].distanceTo(distancePoints[1]);
        setCommandStatus(`Distance: ${dist.toFixed(3)} units`);
        // Draw line
        const lineGeo = new THREE.BufferGeometry().setFromPoints(distancePoints);
        const lineMat = new THREE.LineBasicMaterial({ color: 0xff0000 });
        const line = new THREE.Line(lineGeo, lineMat);
        scene.add(line);
        distanceLines.push(line);
        distancePoints = [];
        setTimeout(() => {
            distanceLines.forEach(l => { scene.remove(l); l.geometry.dispose(); l.material.dispose(); });
            distanceLines = [];
        }, 5000);
    } else {
        setCommandStatus('Click second point to measure distance');
    }
}

// Merge shapes — fill gaps between disconnected parts of the mesh
// Finds vertices near each other from different "islands" and bridges them
// by pulling nearby vertices together and inflating the gap region
function mergeShapes() {
    saveUndo();
    const p = mesh.geometry.attributes.position;
    const count = p.count;

    // 1. Find the mesh center of mass
    let cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < count; i++) {
        cx += p.getX(i); cy += p.getY(i); cz += p.getZ(i);
    }
    cx /= count; cy /= count; cz /= count;

    // 2. For each vertex, find its nearest neighbor that's from a different triangle
    //    and if the gap is small enough, pull them together
    const mergeRadius = 0.6; // max distance to consider for merging
    const pullStrength = 0.7; // how much to close the gap (0=none, 1=fully close)

    // Build a simple spatial grid for faster neighbor lookup
    const cellSize = mergeRadius;
    const grid = new Map();
    const key = (x, y, z) => `${Math.floor(x/cellSize)},${Math.floor(y/cellSize)},${Math.floor(z/cellSize)}`;

    for (let i = 0; i < count; i++) {
        const k = key(p.getX(i), p.getY(i), p.getZ(i));
        if (!grid.has(k)) grid.set(k, []);
        grid.get(k).push(i);
    }

    // 3. For each vertex, look at nearby cells for close vertices from different triangles
    const moved = new Uint8Array(count); // track which vertices were adjusted
    const triOf = (idx) => Math.floor(idx / 3); // which triangle a vertex belongs to

    for (let i = 0; i < count; i++) {
        const xi = p.getX(i), yi = p.getY(i), zi = p.getZ(i);
        const ti = triOf(i);
        let closestDist = mergeRadius;
        let closestJ = -1;
        let cjx = 0, cjy = 0, cjz = 0;

        // Check neighboring cells
        const gx = Math.floor(xi / cellSize), gy = Math.floor(yi / cellSize), gz = Math.floor(zi / cellSize);
        for (let dx = -1; dx <= 1; dx++) {
            for (let dy = -1; dy <= 1; dy++) {
                for (let dz = -1; dz <= 1; dz++) {
                    const cell = grid.get(`${gx+dx},${gy+dy},${gz+dz}`);
                    if (!cell) continue;
                    for (const j of cell) {
                        if (triOf(j) === ti) continue; // same triangle, skip
                        // Don't merge within adjacent triangles (shared edge)
                        if (Math.abs(triOf(j) - ti) <= 1) continue;
                        const xj = p.getX(j), yj = p.getY(j), zj = p.getZ(j);
                        const dist = Math.sqrt((xi-xj)**2 + (yi-yj)**2 + (zi-zj)**2);
                        if (dist < closestDist && dist > 0.001) {
                            closestDist = dist;
                            closestJ = j;
                            cjx = xj; cjy = yj; cjz = zj;
                        }
                    }
                }
            }
        }

        if (closestJ !== -1 && closestDist < mergeRadius) {
            // Pull this vertex toward the closest vertex from another part
            const t = pullStrength * (1.0 - closestDist / mergeRadius); // stronger pull when closer
            const smooth = t * t; // ease-in for natural look
            p.setXYZ(i,
                xi + (cjx - xi) * smooth * 0.5,
                yi + (cjy - yi) * smooth * 0.5,
                zi + (cjz - zi) * smooth * 0.5
            );
            moved[i] = 1;
        }
    }

    // 4. Smooth the merge zone — average moved vertices with their triangle neighbors
    for (let i = 0; i < count; i += 3) {
        if (moved[i] || moved[i+1] || moved[i+2]) {
            const ax = p.getX(i), ay = p.getY(i), az = p.getZ(i);
            const bx = p.getX(i+1), by = p.getY(i+1), bz = p.getZ(i+1);
            const cx2 = p.getX(i+2), cy2 = p.getY(i+2), cz2 = p.getZ(i+2);
            const mx = (ax+bx+cx2)/3, my = (ay+by+cy2)/3, mz = (az+bz+cz2)/3;
            const blend = 0.15; // subtle smoothing
            for (let v = 0; v < 3; v++) {
                if (moved[i+v]) {
                    p.setXYZ(i+v,
                        p.getX(i+v) * (1-blend) + mx * blend,
                        p.getY(i+v) * (1-blend) + my * blend,
                        p.getZ(i+v) * (1-blend) + mz * blend
                    );
                }
            }
        }
    }

    p.needsUpdate = true;
    safeComputeNormals();
    refreshWireframe();

    const movedCount = moved.reduce((a, b) => a + b, 0);
    setCommandStatus(`Merged! ${movedCount} vertices bridged 🔗`);
}

// Check mesh integrity
function checkMesh() {
    const p = mesh.geometry.attributes.position;
    const triCount = p.count / 3;
    let issues = [];
    if (p.count % 3 !== 0) issues.push('Non-triangulated faces');
    // Check for NaN
    for (let i = 0; i < p.array.length; i++) {
        if (isNaN(p.array[i])) { issues.push('NaN vertex detected'); break; }
    }
    if (issues.length === 0) {
        setCommandStatus(`✓ Mesh OK: ${triCount} triangles, ${p.count} vertices`);
    } else {
        setCommandStatus(`⚠ Issues: ${issues.join(', ')}`);
    }
}

// STL Export
function exportSTL() {
    const pos = mesh.geometry.attributes.position;
    const triangles = pos.count / 3;
    const buf = new ArrayBuffer(84 + triangles * 50);
    const dv = new DataView(buf);
    dv.setUint32(80, triangles, true);
    let offset = 84;
    const vA = new THREE.Vector3(), vB = new THREE.Vector3(), vC = new THREE.Vector3(), norm = new THREE.Vector3();
    for (let i = 0; i < triangles; i++) {
        const i3 = i * 3;
        vA.fromBufferAttribute(pos, i3); vB.fromBufferAttribute(pos, i3 + 1); vC.fromBufferAttribute(pos, i3 + 2);
        vA.applyMatrix4(mesh.matrixWorld); vB.applyMatrix4(mesh.matrixWorld); vC.applyMatrix4(mesh.matrixWorld);
        const e1 = vB.clone().sub(vA), e2 = vC.clone().sub(vA);
        norm.crossVectors(e1, e2).normalize();
        dv.setFloat32(offset, norm.x, true); dv.setFloat32(offset + 4, norm.y, true); dv.setFloat32(offset + 8, norm.z, true); offset += 12;
        dv.setFloat32(offset, vA.x, true); dv.setFloat32(offset + 4, vA.y, true); dv.setFloat32(offset + 8, vA.z, true); offset += 12;
        dv.setFloat32(offset, vB.x, true); dv.setFloat32(offset + 4, vB.y, true); dv.setFloat32(offset + 8, vB.z, true); offset += 12;
        dv.setFloat32(offset, vC.x, true); dv.setFloat32(offset + 4, vC.y, true); dv.setFloat32(offset + 8, vC.z, true); offset += 12;
        dv.setUint16(offset, 0, true); offset += 2;
    }
    const blob = new Blob([buf], { type: 'application/octet-stream' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'model.stl';
    a.click();
    URL.revokeObjectURL(a.href);
}

// ============================================================
// 5. APP MODE SYSTEM
// ============================================================
let currentAppMode = 'gallery';
let currentProjectId = null; // ID of currently loaded project for "Save" overwrite
let activeTool = 'move'; // current toolbar tool

function setCommandStatus(text) {
    const el = document.getElementById('command-status');
    if (el) el.textContent = text;
    const hint = document.getElementById('mode-hint');
    if (hint) hint.textContent = text;
}

// Per-mode saved mesh state so switching modes doesn't destroy work
const modeStates = { model: null, abstract: null };

function saveModeState(modeName) {
    if (!mesh) return;
    modeStates[modeName] = {
        positions: new Float32Array(mesh.geometry.attributes.position.array),
        normals: mesh.geometry.attributes.normal ? new Float32Array(mesh.geometry.attributes.normal.array) : null,
        colors: mesh.geometry.attributes.color ? new Float32Array(mesh.geometry.attributes.color.array) : null,
        color: material.color.getHex(),
        scale: mesh.scale.toArray(),
        position: mesh.position.toArray(),
        rotation: [mesh.rotation.x, mesh.rotation.y, mesh.rotation.z],
        vertexColors: material.vertexColors,
    };
}

function restoreModeState(modeName) {
    const s = modeStates[modeName];
    if (!s) return false;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(s.positions), 3));
    if (s.normals) g.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(s.normals), 3));
    if (s.colors) g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(s.colors), 3));
    if (mesh) { scene.remove(mesh); mesh.geometry.dispose(); }
    mesh = new THREE.Mesh(g, material);
    g.computeVertexNormals();
    refreshWireframe();
    scene.add(mesh);
    material.color.setHex(s.color);
    material.vertexColors = s.vertexColors;
    material.needsUpdate = true;
    mesh.scale.fromArray(s.scale);
    mesh.position.fromArray(s.position);
    mesh.rotation.set(s.rotation[0], s.rotation[1], s.rotation[2]);
    return true;
}

function applyAppMode(name) {
    if (name === currentAppMode) return;
    const prevMode = currentAppMode;
    currentAppMode = name;
    document.body.dataset.appmode = name;

    const modelActions = document.getElementById('model-actions');
    const sketchActions = document.getElementById('sketch-actions');
    const sketchCanvas = document.getElementById('sketch-canvas');
    const canvasContainer = document.getElementById('canvas-container');
    const spliceWrap = document.getElementById('splice-toolbar-wrap');
    const guide = document.getElementById('instructions-guide');
    const sketchGuide = document.getElementById('sketch-guide');
    const galleryView = document.getElementById('gallery-view');

    // Hide everything first
    if (modelActions) modelActions.style.display = 'none';
    if (sketchActions) sketchActions.style.display = 'none';
    if (sketchCanvas) sketchCanvas.style.display = 'none';
    if (canvasContainer) canvasContainer.style.display = 'none';
    if (spliceWrap) spliceWrap.style.display = 'none';
    if (guide) guide.style.display = 'none';
    if (sketchGuide) sketchGuide.style.display = 'none';
    if (galleryView) galleryView.style.display = 'none';

    grid.visible = false;

    document.querySelectorAll('.ws-btn').forEach(b =>
        b.classList.toggle('active', b.dataset.appmode === name));

    if (name === 'model') {
        if (modelActions) modelActions.style.display = '';
        if (canvasContainer) canvasContainer.style.display = '';
        if (spliceWrap) spliceWrap.style.display = '';
        if (guide) guide.style.display = '';
        grid.visible = true;
        // Don't restore state — scene is already intact
    } else if (name === 'sketch') {
        if (sketchActions) sketchActions.style.display = '';
        if (sketchCanvas) sketchCanvas.style.display = 'block';
        if (sketchGuide) sketchGuide.style.display = '';
        initSketchCanvas();
    } else if (name === 'gallery') {
        if (galleryView) galleryView.style.display = 'block';
        renderGallery();
    }
}

// ============================================================
// 6. ABSTRACT MODE — fluid organic blob
// ============================================================
let abstractAnimating = false;
let abstractTime = 0;
let abstractRestPositions = null;

function initAbstractBlob() {
    const geo = new THREE.SphereGeometry(1.3, 64, 48).toNonIndexed();
    geo.computeVertexNormals();
    const pos = geo.attributes.position;

    // Organic lava-lamp deformation
    for (let i = 0; i < pos.count; i++) {
        let x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
        const len = Math.sqrt(x * x + y * y + z * z) || 1;
        const nx = x / len, ny = y / len, nz = z / len;
        y *= 1.4; // elongate vertically
        const lump1 = Math.sin(nx * 3.2 + 0.5) * Math.cos(ny * 2.1 + 1.2) * Math.sin(nz * 2.8) * 0.15;
        const lump2 = Math.sin(nx * 5.1) * Math.cos(ny * 4.3 + 2.1) * Math.cos(nz * 3.7) * 0.07;
        const waist = 1.0 - Math.pow(Math.abs(ny), 2.5) * 0.15;
        const r = (len + lump1 + lump2) * waist;
        pos.setXYZ(i, nx * r, y + ny * lump1 * 0.5, nz * r);
    }
    pos.needsUpdate = true;
    geo.computeVertexNormals();

    if (mesh) { scene.remove(mesh); mesh.geometry.dispose(); }
    mesh = new THREE.Mesh(geo, material);
    scene.add(mesh);
    refreshWireframe();

    abstractRestPositions = new Float32Array(mesh.geometry.attributes.position.array);

    // Apply gradient colors
    applyBlobGradient();
}

function applyBlobGradient() {
    if (!mesh) return;
    const pos = mesh.geometry.attributes.position;
    const colors = new Float32Array(pos.count * 3);
    let yMin = Infinity, yMax = -Infinity;
    for (let i = 0; i < pos.count; i++) {
        const y = pos.getY(i);
        if (y < yMin) yMin = y;
        if (y > yMax) yMax = y;
    }
    const yRange = Math.max(yMax - yMin, 0.01);
    const ur = material.color.r, ug = material.color.g, ub = material.color.b;
    for (let i = 0; i < pos.count; i++) {
        const t = (pos.getY(i) - yMin) / yRange;
        // pink → lavender → baby blue
        let r, g, b;
        if (t < 0.5) {
            const s = t * 2;
            r = 0.95 + (0.78 - 0.95) * s;
            g = 0.60 + (0.65 - 0.60) * s;
            b = 0.70 + (0.90 - 0.70) * s;
        } else {
            const s = (t - 0.5) * 2;
            r = 0.78 + (0.60 - 0.78) * s;
            g = 0.65 + (0.78 - 0.65) * s;
            b = 0.90 + (0.95 - 0.90) * s;
        }
        colors[i * 3] = r * 0.6 + ur * 0.4;
        colors[i * 3 + 1] = g * 0.6 + ug * 0.4;
        colors[i * 3 + 2] = b * 0.6 + ub * 0.4;
    }
    mesh.geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    material.vertexColors = true;
    material.needsUpdate = true;
}

function abstractAnimate() {
    if (!abstractAnimating || !mesh || !abstractRestPositions) return;
    abstractTime += 0.012;
    const pos = mesh.geometry.attributes.position;
    const rest = abstractRestPositions;
    for (let i = 0; i < pos.count; i++) {
        const rx = rest[i * 3], ry = rest[i * 3 + 1], rz = rest[i * 3 + 2];
        const len = Math.sqrt(rx * rx + ry * ry + rz * rz) || 1;
        const nx = rx / len, ny = ry / len, nz = rz / len;
        const wave1 = Math.sin(abstractTime * 1.2 + ny * 3.0 + nx * 2.0) * 0.045;
        const wave2 = Math.sin(abstractTime * 2.2 + rx * 4.0 + rz * 3.5) * Math.cos(abstractTime * 1.5 + ry * 3.0) * 0.025;
        const wave3 = Math.sin(abstractTime * 3.8 + nx * 7.0 + nz * 5.5) * Math.cos(abstractTime * 2.8 + ny * 8.0) * 0.012;
        const tangent = Math.sin(abstractTime * 1.0 + ny * 4.0) * 0.015;
        const breathe = wave1 + wave2 + wave3;
        pos.setXYZ(i,
            rx + nx * breathe + nz * tangent,
            ry + ny * breathe,
            rz + nz * breathe - nx * tangent
        );
    }
    pos.needsUpdate = true;
    safeComputeNormals();
}

// Abstract mouse interaction: drag to mold, shift+drag to stretch
let abstractDragging = false;
let abstractDragStart = null;
let abstractHitPoint = null;

// ============================================================
// 7. SKETCH MODE — 2D canvas with brush textures
// ============================================================
let sketchCtx = null;
let sketchColor = '#333333';
let sketchTool = 'pencil';
let isSketchDrawing = false;
let lastSketchPt = null;

const SKETCH_TOOLS = {
    pencil:  { size: 3,  opacity: 0.85, blur: 0,   cap: 'round',  composite: 'source-over', jitter: 0.3 },
    pen:     { size: 2,  opacity: 1.0,  blur: 0,   cap: 'round',  composite: 'source-over', jitter: 0 },
    marker:  { size: 12, opacity: 0.6,  blur: 0,   cap: 'round',  composite: 'source-over', jitter: 0 },
    crayon:  { size: 8,  opacity: 0.55, blur: 1,   cap: 'round',  composite: 'source-over', jitter: 1.5 },
    pastel:  { size: 14, opacity: 0.35, blur: 3,   cap: 'round',  composite: 'source-over', jitter: 2.0 },
    acrylic: { size: 10, opacity: 0.9,  blur: 0.5, cap: 'square', composite: 'source-over', jitter: 0.5 },
    eraser:  { size: 20, opacity: 1.0,  blur: 0,   cap: 'round',  composite: 'destination-out', jitter: 0 },
};

function initSketchCanvas() {
    const canvas = document.getElementById('sketch-canvas');
    if (!canvas) return;
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    sketchCtx = canvas.getContext('2d');
    sketchCtx.fillStyle = '#ffffff';
    sketchCtx.fillRect(0, 0, canvas.width, canvas.height);
}

function sketchStroke(x, y) {
    if (!sketchCtx) return;
    const tool = SKETCH_TOOLS[sketchTool] || SKETCH_TOOLS.pencil;
    sketchCtx.save();
    sketchCtx.globalAlpha = tool.opacity;
    sketchCtx.globalCompositeOperation = tool.composite;
    sketchCtx.lineCap = tool.cap;
    sketchCtx.lineJoin = 'round';
    sketchCtx.lineWidth = tool.size;
    sketchCtx.strokeStyle = sketchTool === 'eraser' ? '#ffffff' : sketchColor;
    if (tool.blur > 0) sketchCtx.filter = `blur(${tool.blur}px)`;
    const jx = tool.jitter ? (Math.random() - 0.5) * tool.jitter : 0;
    const jy = tool.jitter ? (Math.random() - 0.5) * tool.jitter : 0;
    sketchCtx.beginPath();
    if (lastSketchPt) {
        sketchCtx.moveTo(lastSketchPt.x, lastSketchPt.y);
        sketchCtx.lineTo(x + jx, y + jy);
    } else {
        sketchCtx.moveTo(x + jx, y + jy);
        sketchCtx.lineTo(x + jx + 0.5, y + jy + 0.5);
    }
    sketchCtx.stroke();
    sketchCtx.restore();
    lastSketchPt = { x: x + jx, y: y + jy };
}

// Sketch SVG export with laser-cut templates
function exportSketchSVG(template = 'custom') {
    if (!sketchCtx) return;
    const canvas = document.getElementById('sketch-canvas');
    const dataURL = canvas.toDataURL('image/png');

    // Template dimensions (mm for laser cut)
    const templates = {
        custom:   { w: canvas.width, h: canvas.height, label: 'Custom' },
        card:     { w: 350, h: 200, label: 'Card (3.5x2in)' },
        keychain: { w: 60,  h: 60,  label: 'Keychain (60mm)' },
        coaster:  { w: 100, h: 100, label: 'Coaster (100mm)' },
        bookmark: { w: 50,  h: 180, label: 'Bookmark (50x180mm)' },
    };
    const t = templates[template] || templates.custom;

    const svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
     width="${t.w}mm" height="${t.h}mm" viewBox="0 0 ${t.w} ${t.h}">
  <title>Laser Cut - ${t.label}</title>
  <!-- Cut line (red = cut) -->
  <rect x="0" y="0" width="${t.w}" height="${t.h}" fill="none" stroke="red" stroke-width="0.1"/>
  ${template === 'keychain' ? `<circle cx="${t.w / 2}" cy="8" r="3" fill="none" stroke="red" stroke-width="0.1"/>` : ''}
  <!-- Sketch artwork -->
  <image href="${dataURL}" x="2" y="2" width="${t.w - 4}" height="${t.h - 4}" preserveAspectRatio="xMidYMid meet"/>
</svg>`;

    const blob = new Blob([svg], { type: 'image/svg+xml' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `sketch-${template}.svg`;
    a.click();
    URL.revokeObjectURL(a.href);
    setCommandStatus(`Exported ${t.label} as SVG`);
}

// ============================================================
// 8. GESTURE DETECTION SYSTEM
// ============================================================
const GESTURE_SMOOTHING = 0.3; // EMA alpha
const GESTURE_BUFFER_SIZE = 10;
const PINCH_THRESHOLD = 0.04;
const DEAD_ZONE = 0.005;
const MODE_LOCK_MS = 150;
const CUPPED_HOLD_MS = 500;

// Engagement zone — ignore hands that are incidental (resting, scratching, etc.)
const ENGAGE_ZONE_MARGIN = 0.12;
const ENGAGE_MIN_HAND_SIZE = 0.07;

let handLandmarker = null;
let handTrackingActive = false;
let isDebug = false;
let lastGestureTime = 0;
let currentGesture = null;
let gestureLockedUntil = 0;

// Smoothed landmark buffer (pre-allocated)
const smoothedLandmarks = [new Array(21).fill(null).map(() => ({ x: 0, y: 0, z: 0 })),
                           new Array(21).fill(null).map(() => ({ x: 0, y: 0, z: 0 }))];
let landmarkInited = [false, false];

// Rolling motion buffer
const motionBuffer = [[], []]; // per hand, last N frames of wrist position

function smoothLandmark(handIdx, lmIdx, raw) {
    const s = smoothedLandmarks[handIdx][lmIdx];
    if (!landmarkInited[handIdx]) {
        s.x = raw.x; s.y = raw.y; s.z = raw.z;
        return s;
    }
    const a = GESTURE_SMOOTHING;
    s.x = s.x * (1 - a) + raw.x * a;
    s.y = s.y * (1 - a) + raw.y * a;
    s.z = s.z * (1 - a) + raw.z * a;
    return s;
}

function processLandmarks(rawHands) {
    const result = [];
    for (let h = 0; h < Math.min(rawHands.length, 2); h++) {
        const hand = [];
        for (let i = 0; i < 21; i++) {
            hand.push(smoothLandmark(h, i, rawHands[h][i]));
        }
        landmarkInited[h] = true;
        result.push(hand);

        // Update motion buffer
        motionBuffer[h].push({ x: hand[0].x, y: hand[0].y, z: hand[0].z, t: performance.now() });
        if (motionBuffer[h].length > GESTURE_BUFFER_SIZE) motionBuffer[h].shift();
    }
    return result;
}

// --- Gesture classifiers ---
function dist3D(a, b) {
    return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2);
}

// Check if a hand is in the valid interaction zone (basic spatial filter)
function isHandInZone(hand) {
    const wrist = hand[0];
    const midMCP = hand[9];
    const inZoneX = wrist.x > ENGAGE_ZONE_MARGIN && wrist.x < (1 - ENGAGE_ZONE_MARGIN);
    const inZoneY = wrist.y > ENGAGE_ZONE_MARGIN && wrist.y < (1 - ENGAGE_ZONE_MARGIN);
    if (!inZoneX || !inZoneY) return false;
    const handSize = Math.hypot(midMCP.x - wrist.x, midMCP.y - wrist.y);
    if (handSize < ENGAGE_MIN_HAND_SIZE) return false;
    return true;
}

function isPinch(hand) {
    return dist3D(hand[4], hand[8]) < PINCH_THRESHOLD;
}

function isFist(hand) {
    // All fingertips close to palm (wrist)
    const palm = hand[0];
    const tips = [hand[8], hand[12], hand[16], hand[20]];
    return tips.every(t => dist3D(t, palm) < 0.12);
}

function isOpenPalm(hand) {
    const palm = hand[0];
    const tips = [hand[4], hand[8], hand[12], hand[16], hand[20]];
    return tips.every(t => dist3D(t, palm) > 0.12);
}

function isCupped(hand) {
    // Fingertips curled: MCPs farther from palm center than tips
    const palm = hand[0];
    const mcps = [hand[5], hand[9], hand[13], hand[17]];
    const tips = [hand[8], hand[12], hand[16], hand[20]];
    let curledCount = 0;
    for (let i = 0; i < 4; i++) {
        if (dist3D(mcps[i], palm) > dist3D(tips[i], palm) * 0.9) curledCount++;
    }
    return curledCount >= 3 && !isPinch(hand) && !isFist(hand);
}

function isPointing(hand) {
    // Index extended, others curled
    const palm = hand[0];
    const indexExt = dist3D(hand[8], palm) > 0.15;
    const othersCurled = [hand[12], hand[16], hand[20]].every(t => dist3D(t, palm) < 0.12);
    return indexExt && othersCurled && !isPinch(hand);
}

function isLShape(hand) {
    // Thumb + index extended at ~90°, others curled
    const thumbExt = dist3D(hand[4], hand[2]) > 0.06;
    const indexExt = dist3D(hand[8], hand[5]) > 0.06;
    const othersCurled = [hand[12], hand[16], hand[20]].every(t => dist3D(t, hand[0]) < 0.12);
    if (!thumbExt || !indexExt || !othersCurled) return false;
    // Check angle between thumb and index directions
    const thumbDir = { x: hand[4].x - hand[2].x, y: hand[4].y - hand[2].y };
    const indexDir = { x: hand[8].x - hand[5].x, y: hand[8].y - hand[5].y };
    const dot = thumbDir.x * indexDir.x + thumbDir.y * indexDir.y;
    const mag = Math.sqrt(thumbDir.x ** 2 + thumbDir.y ** 2) * Math.sqrt(indexDir.x ** 2 + indexDir.y ** 2);
    const angle = Math.acos(Math.max(-1, Math.min(1, dot / (mag || 1))));
    return angle > Math.PI / 4 && angle < Math.PI * 3 / 4; // 45° to 135°
}

function classifyGesture(hand) {
    if (isPinch(hand)) return 'pinch';
    if (isFist(hand)) return 'fist';
    if (isPointing(hand)) return 'point';
    if (isCupped(hand)) return 'cupped';
    if (isLShape(hand)) return 'lshape';
    if (isOpenPalm(hand)) return 'palm';
    return null;
}

function getMotionVector(handIdx) {
    const buf = motionBuffer[handIdx];
    if (buf.length < 3) return { x: 0, y: 0, z: 0, magnitude: 0 };
    const recent = buf[buf.length - 1];
    const old = buf[buf.length - 3];
    const dx = recent.x - old.x;
    const dy = recent.y - old.y;
    const dz = recent.z - old.z;
    return { x: dx, y: dy, z: dz, magnitude: Math.sqrt(dx * dx + dy * dy + dz * dz) };
}

// ============================================================
// 9. GESTURE → COMMAND MAPPING
// ============================================================
// State for gesture-driven operations
let gestureState = {
    lastPinchPos: [null, null],    // per hand
    lastPalmPos: [null, null],     // per hand — palm center position for rotation
    twoHandDist: null,
    cuppedHoldStart: null,
    addShapePreview: null,
    addShapeScale: 0.3,
    addShapeScaleAtCupStart: 0.5,  // scale when cupped resize began
    lastGestures: [null, null],
    pinchSubMode: null,            // 'translate' | 'extrude' | null — locked once determined
    pinchSubModeFrames: 0,         // frames since pinch started (for initial direction sampling)
    pinchAccumLateral: 0,          // accumulated lateral movement during sampling
    pinchAccumDepth: 0,            // accumulated depth movement during sampling
    scaleUndoArmed: false,
    bevelUndoArmed: false,
    moldUndoArmed: false,
    rotateUndoArmed: false,
};

// Smooth rotation state — EMA-filtered for fluid feel
let smoothRotVelX = 0;
let smoothRotVelY = 0;
const ROT_SMOOTHING = 0.45;
let ROT_SENSITIVITY = 8.0;
let ROT_DAMPING = 0.92;
const ROT_MIN_DELTA = 0.0005;

// Smooth pinch state
let smoothTransVelX = 0;
let smoothTransVelY = 0;
let smoothExtrudeVel = 0;
const PINCH_SMOOTHING = 0.5;
let TRANSLATE_SENSITIVITY = 5.0;
const EXTRUDE_SENSITIVITY = 4.0;
const TRANSLATE_DAMPING = 0.85;
const PINCH_SAMPLE_FRAMES = 6;
const PINCH_LOCK_DEPTH_RATIO = 0.5;

// Gesture transition guard
const GESTURE_SWITCH_FRAMES = 2;  // lowered from 4 for faster response
let gestureConfirmBuffer = [null, null]; // per hand: { gesture, count }
let lastConfirmedGesture = [null, null];

function confirmGesture(handIdx, rawGesture) {
    const buf = gestureConfirmBuffer[handIdx];
    if (buf && buf.gesture === rawGesture) {
        buf.count++;
    } else {
        gestureConfirmBuffer[handIdx] = { gesture: rawGesture, count: 1 };
    }
    // Only switch if we've seen the new gesture for enough frames
    if (gestureConfirmBuffer[handIdx].count >= GESTURE_SWITCH_FRAMES) {
        lastConfirmedGesture[handIdx] = rawGesture;
        return rawGesture;
    }
    // Stick with the last confirmed gesture to prevent flicker
    return lastConfirmedGesture[handIdx];
}

function handleGestures(hands) {
    if (!hands.length || !mesh) return;
    const now = performance.now();
    if (now < gestureLockedUntil) return;

    const rawGestures = hands.map(h => classifyGesture(h));
    // Apply hysteresis — requires N consecutive frames before switching gesture
    const gestures = rawGestures.map((g, i) => confirmGesture(i, g));

    // === ADD SHAPE MODE — dual-hand support ===
    // In add-shape mode: simplified — every hand simultaneously:
    // - Positions preview by where it is on screen
    // - Hand openness controls scale
    // - Pinch (thumb+index touch) = place
    // Second hand palm = rotate object
    if (addShapeMode && currentAppMode === 'model') {
        const hand = hands[0];
        const wrist = hand[0];
        const thumbTip = hand[4], indexTip = hand[8];
        const pinchDist = Math.hypot(thumbTip.x - indexTip.x, thumbTip.y - indexTip.y);

        // --- PINCH = PLACE (thumb touches index) ---
        if (pinchDist < 0.035) {
            if (!gestureState._pinchPlaceStart) {
                gestureState._pinchPlaceStart = now;
            } else if (now - gestureState._pinchPlaceStart > 250) {
                if (addShapeVisible) {
                    confirmAddShape();
                    gestureLockedUntil = now + 600;
                }
                addShapeBaseHandScale = null;
                gestureState._smoothSpan = null;
                gestureState._pinchPlaceStart = null;
                currentGesture = 'addshape-place';
                return;
            }
            setCommandStatus('Hold pinch to place...');
            currentGesture = 'addshape-place';
            return;
        }
        gestureState._pinchPlaceStart = null;

        // --- POSITION preview by hand center ---
        const palmCenter = hand[9];
        const screenX = (1 - palmCenter.x) * innerWidth;
        const screenY = palmCenter.y * innerHeight;
        const ndc = new THREE.Vector2(
            (screenX / innerWidth) * 2 - 1,
            -(screenY / innerHeight) * 2 + 1
        );
        raycaster.setFromCamera(ndc, camera);
        const hits = raycaster.intersectObjects([mesh]);
        if (hits.length) {
            const hitNorm = hits[0].face?.normal
                ? hits[0].face.normal.clone().transformDirection(mesh.matrixWorld).normalize()
                : new THREE.Vector3(0, 0, 1);
            addShapeHitNormal.copy(hitNorm);
            addShapeSurfaceHit.copy(hits[0].point);
            const surfaceOffset = addShapeBaseRadius * addShapeScale * 0.35;
            addShapePosition.copy(hits[0].point).addScaledVector(hitNorm, surfaceOffset);
        } else {
            const dir = new THREE.Vector3();
            raycaster.ray.direction.normalize();
            dir.copy(raycaster.ray.direction).multiplyScalar(4);
            addShapePosition.copy(raycaster.ray.origin).add(dir);
            addShapeSurfaceHit.copy(addShapePosition);
        }
        updateAddShapePreview(addShapePosition, addShapeScale);

        // --- SCALE by hand openness (all fingertips avg distance from wrist) ---
        const tips = [hand[4], hand[8], hand[12], hand[16], hand[20]];
        let totalSpread = 0;
        for (const tip of tips) {
            totalSpread += Math.hypot(tip.x - wrist.x, tip.y - wrist.y);
        }
        const handOpenness = totalSpread / 5;

        if (gestureState._smoothSpan == null) {
            gestureState._smoothSpan = handOpenness;
        } else {
            gestureState._smoothSpan = gestureState._smoothSpan * 0.3 + handOpenness * 0.7;
        }
        if (addShapeBaseHandScale === null) {
            addShapeBaseHandScale = gestureState._smoothSpan;
            gestureState._cupStartScale = addShapeScale;
        }
        const ratio = gestureState._smoothSpan / addShapeBaseHandScale;
        const targetScale = Math.max(0.05, Math.min(3.0, gestureState._cupStartScale * ratio));
        addShapeScale = addShapeScale * 0.2 + targetScale * 0.8;
        if (addShapePreviewMesh) {
            addShapePreviewMesh.scale.setScalar(addShapeScale);
            recomputeAddShapePosition();
        }

        // --- SECOND HAND: rotate object ---
        if (hands.length >= 2) {
            const rHand = hands[1];
            const rPalm = rHand[9];
            if (gestureState.lastPalmPos[1]) {
                const dx = rPalm.x - gestureState.lastPalmPos[1].x;
                const dy = rPalm.y - gestureState.lastPalmPos[1].y;
                if (Math.abs(dx) > ROT_MIN_DELTA || Math.abs(dy) > ROT_MIN_DELTA) {
                    smoothRotVelY = smoothRotVelY * (1 - ROT_SMOOTHING) + (-dx * ROT_SENSITIVITY) * ROT_SMOOTHING;
                    smoothRotVelX = smoothRotVelX * (1 - ROT_SMOOTHING) + (dy * ROT_SENSITIVITY) * ROT_SMOOTHING;
                }
            }
            gestureState.lastPalmPos[1] = { x: rPalm.x, y: rPalm.y };
        }

        // --- FIST = cancel ---
        if (gestures[0] === 'fist') {
            exitAddShapeMode();
            currentGesture = 'freeze';
            return;
        }

        currentGesture = 'addshape';
        setCommandStatus(`Size: ${addShapeScale.toFixed(2)} — open hand = bigger, close = smaller, pinch = place`);
        return; // Consume all gestures in add mode
    }

    // === TWO-HAND GESTURES (priority) ===
    if (hands.length >= 2) {
        const g1 = gestures[0], g2 = gestures[1];

        // Two palms → Scale (apart/together)
        if (g1 === 'palm' && g2 === 'palm') {
            const wrist0 = hands[0][0], wrist1 = hands[1][0];
            const dist = dist3D(wrist0, wrist1);

            // Scale (hands apart/together)
            if (gestureState.twoHandDist !== null) {
                const delta = dist - gestureState.twoHandDist;
                if (Math.abs(delta) > DEAD_ZONE) {
                    if (!gestureState.scaleUndoArmed) { saveUndo(); gestureState.scaleUndoArmed = true; }
                    mesh.scale.multiplyScalar(1 + delta * 2);
                    setCommandStatus('Scaling...');
                }
            }
            gestureState.twoHandDist = dist;
            currentGesture = 'scale';
            gestureLockedUntil = now + MODE_LOCK_MS;
            return;
        }
        gestureState.twoHandDist = null;
        gestureState.scaleUndoArmed = false;

        // Two pinches apart → Split
        if (g1 === 'pinch' && g2 === 'pinch') {
            const p1 = { x: (hands[0][4].x + hands[0][8].x) / 2, y: (hands[0][4].y + hands[0][8].y) / 2 };
            const p2 = { x: (hands[1][4].x + hands[1][8].x) / 2, y: (hands[1][4].y + hands[1][8].y) / 2 };
            const pDist = Math.hypot(p1.x - p2.x, p1.y - p2.y);

            if (pDist > 0.3) {
                const hdx = Math.abs(p1.x - p2.x);
                const hdy = Math.abs(p1.y - p2.y);
                const splitAxis = hdx > hdy ? 'x' : 'y';
                splitMesh(0, splitAxis);
                setCommandStatus(`Split along ${splitAxis.toUpperCase()}`);
                currentGesture = 'split';
                gestureLockedUntil = now + 1000;
            }
            return;
        }

        // Palm + spinning index → Revolve (placeholder)
        if ((g1 === 'palm' && g2 === 'point') || (g1 === 'point' && g2 === 'palm')) {
            setCommandStatus('Revolve mode — draw profile first');
            currentGesture = 'revolve';
            return;
        }
    }

    // === SINGLE-HAND GESTURES ===
    const hand = hands[0];
    const g = gestures[0];

    // Peace sign (✌️) → Mirror at highlighted area
    // If faces were highlighted with point gesture, mirror happens at that spot
    {
        const palm = hand[0];
        const indexExt = dist3D(hand[8], palm) > 0.14;
        const middleExt = dist3D(hand[12], palm) > 0.14;
        const ringCurled = dist3D(hand[16], palm) < 0.11;
        const pinkyCurled = dist3D(hand[20], palm) < 0.11;
        const thumbCurled = dist3D(hand[4], palm) < 0.11;
        if (indexExt && middleExt && ringCurled && pinkyCurled && thumbCurled) {
            if (!gestureState._peaceHoldStart) {
                gestureState._peaceHoldStart = now;
            } else if (now - gestureState._peaceHoldStart > 500) {
                if (lastHighlightPoint && lastHighlightNormal) {
                    // Mirror at the highlighted spot — use the highlight normal to determine axis
                    const invMatrix = new THREE.Matrix4().copy(mesh.matrixWorld).invert();
                    const localNorm = lastHighlightNormal.clone().transformDirection(invMatrix).normalize();
                    // Pick the axis most aligned with the highlight normal
                    const ax = Math.abs(localNorm.x), ay = Math.abs(localNorm.y), az = Math.abs(localNorm.z);
                    const axis = ax > ay && ax > az ? 'x' : ay > az ? 'y' : 'z';
                    // Mirror at the highlight point on that axis
                    const localPt = lastHighlightPoint.clone().applyMatrix4(invMatrix);
                    const splitPos = axis === 'x' ? localPt.x : axis === 'y' ? localPt.y : localPt.z;
                    saveUndo();
                    // Local mirror: duplicate verts mirrored across the split plane at the highlight point
                    const p = mesh.geometry.attributes.position;
                    const count = p.count;
                    const newPos = new Float32Array(count * 6);
                    newPos.set(p.array, 0);
                    for (let i = 0; i < count; i++) {
                        const idx = i * 3;
                        const ai = axis === 'x' ? 0 : axis === 'y' ? 1 : 2;
                        newPos[count * 3 + idx] = ai === 0 ? 2 * splitPos - p.array[idx] : p.array[idx];
                        newPos[count * 3 + idx + 1] = ai === 1 ? 2 * splitPos - p.array[idx + 1] : p.array[idx + 1];
                        newPos[count * 3 + idx + 2] = ai === 2 ? 2 * splitPos - p.array[idx + 2] : p.array[idx + 2];
                    }
                    const mg = new THREE.BufferGeometry();
                    mg.setAttribute('position', new THREE.BufferAttribute(newPos, 3));
                    mesh.geometry.dispose();
                    mesh.geometry = mg;
                    safeComputeNormals();
                    refreshWireframe();
                    setCommandStatus(`Mirrored at highlighted spot (${axis.toUpperCase()}) ✌️`);
                } else {
                    // No highlight — mirror whole mesh at center
                    const wx = hand[0].x;
                    const axis = wx < 0.4 ? 'x' : wx > 0.6 ? 'z' : 'y';
                    mirrorMesh(axis);
                    setCommandStatus(`Mirrored (${axis.toUpperCase()}) ✌️`);
                }
                hideHighlight();
                gestureLockedUntil = now + 1500;
                gestureState._peaceHoldStart = null;
                currentGesture = 'mirror';
                return;
            }
            setCommandStatus(lastHighlightPoint ? 'Hold ✌️ to mirror at highlighted area...' : 'Hold ✌️ to mirror...');
            currentGesture = 'peace';
            return;
        } else {
            gestureState._peaceHoldStart = null;
        }
    }

    // 3 fingers up (index + middle + ring) + push down → Array
    {
        const palm = hand[0];
        const indexExt = dist3D(hand[8], palm) > 0.13;
        const middleExt = dist3D(hand[12], palm) > 0.13;
        const ringExt = dist3D(hand[16], palm) > 0.13;
        const pinkyCurled = dist3D(hand[20], palm) < 0.11;
        const thumbCurled = dist3D(hand[4], palm) < 0.11;
        if (indexExt && middleExt && ringExt && pinkyCurled && thumbCurled) {
            // Track vertical motion
            if (gestureState._threeFingerLastY == null) {
                gestureState._threeFingerLastY = palm.y;
            }
            const dy = palm.y - gestureState._threeFingerLastY;
            gestureState._threeFingerLastY = palm.y;

            // Pushing down (wrist moving down in frame = y increasing)
            if (dy > 0.008) {
                if (!gestureState._threeFingerPushStart) {
                    gestureState._threeFingerPushStart = now;
                } else if (now - gestureState._threeFingerPushStart > 250) {
                    const dir = new THREE.Vector3(1, 0, 0);
                    arrayMesh(2, dir);
                    gestureLockedUntil = now + 2000;
                    gestureState._threeFingerPushStart = null;
                    gestureState._threeFingerLastY = null;
                    currentGesture = 'array';
                    return;
                }
                setCommandStatus('3 fingers push down → array...');
                currentGesture = 'array';
                return;
            } else {
                gestureState._threeFingerPushStart = null;
            }
            setCommandStatus('3 fingers up — push down to array');
            currentGesture = 'threefingers';
            return;
        } else {
            gestureState._threeFingerLastY = null;
            gestureState._threeFingerPushStart = null;
        }
    }

    if (g === 'fist') {
        // Freeze — do nothing
        setCommandStatus('Frozen — fist held');
        currentGesture = 'freeze';
        return;
    }

    if (g === 'palm') {
        // Rotate — palm swipe directly controls rotation direction
        // Hand moves right → object rotates right, hand moves down → object tilts forward
        const palmCenter = hand[9]; // middle finger MCP = stable palm center
        if (gestureState.lastPalmPos[0]) {
            const dx = palmCenter.x - gestureState.lastPalmPos[0].x;
            const dy = palmCenter.y - gestureState.lastPalmPos[0].y;
            if (Math.abs(dx) > ROT_MIN_DELTA || Math.abs(dy) > ROT_MIN_DELTA) {
                if (!gestureState.rotateUndoArmed) { saveUndo(); gestureState.rotateUndoArmed = true; }
                // Globe model: hand touching front surface pushes it in that direction
                // MediaPipe mirrors X (hand moves right → x decreases), so negate
                // Hand right → surface moves right → rotate Y positive
                // Hand down  → surface moves down  → rotate X positive (top tilts toward you)
                const targetVelY = -dx * ROT_SENSITIVITY;
                const targetVelX = dy * ROT_SENSITIVITY;
                // Blend toward target velocity — responsive but not jerky
                smoothRotVelY = smoothRotVelY * (1 - ROT_SMOOTHING) + targetVelY * ROT_SMOOTHING;
                smoothRotVelX = smoothRotVelX * (1 - ROT_SMOOTHING) + targetVelX * ROT_SMOOTHING;
                setCommandStatus('Rotating...');
            }
        }
        gestureState.lastPalmPos[0] = { x: palmCenter.x, y: palmCenter.y };
        gestureState.lastGestures[0] = 'palm';
        currentGesture = 'rotate';
        return;
    } else {
        // When palm released, reset tracking but keep momentum
        gestureState.lastPalmPos[0] = null;
        gestureState.rotateUndoArmed = false;
    }

    if (g === 'pinch') {
        const px = (hand[4].x + hand[8].x) / 2;
        const py = (hand[4].y + hand[8].y) / 2;

        // Hand scale = wrist-to-middle-MCP distance in screen space.
        // Moves toward camera → hand bigger → scale up. Pull away → smaller → scale down.
        // Far more reliable than MediaPipe Z values.
        const handScale = Math.hypot(hand[9].x - hand[0].x, hand[9].y - hand[0].y);

        // First frame — record baseline, no action
        if (gestureState.lastGestures[0] !== 'pinch') {
            gestureState.lastPinchPos[0] = { x: px, y: py };
            gestureState.lastHandScale = handScale;
            gestureState.startHandScale = handScale;  // baseline for total extrude
            gestureState.lastGestures[0] = 'pinch';
            gestureState.extrudeUndoArmed = false;
            gestureState.extrudeRestPositions = null;  // snapshot of geometry before extrude
            gestureState.translateUndoArmed = false;
            smoothTransVelX = 0;
            smoothTransVelY = 0;
            smoothExtrudeVel = 0;
            currentGesture = 'pinch';
            touchPointMat.color.setHex(0xffdd57);     // yellow = pinch idle
            touchPointRingMat.color.setHex(0xffdd57);
            return;
        }

        // Show touch point at pinch position
        const screenX = (1 - px) * innerWidth;  // mirror for camera
        const screenY = py * innerHeight;
        showTouchPoint(screenX, screenY);

        if (gestureState.lastPinchPos[0] && gestureState.lastHandScale !== undefined) {
            const dx = px - gestureState.lastPinchPos[0].x;
            const dy = py - gestureState.lastPinchPos[0].y;
            const scaleDelta = handScale - gestureState.lastHandScale;

            // --- TRANSLATE: lateral pinch movement moves the object ---
            const lateralMag = Math.abs(dx) + Math.abs(dy);
            if (lateralMag > 0.002) {
                if (!gestureState.translateUndoArmed) {
                    saveUndo();
                    gestureState.translateUndoArmed = true;
                }
                const moveX = -dx * TRANSLATE_SENSITIVITY;
                const moveY = dy * TRANSLATE_SENSITIVITY;
                smoothTransVelX = smoothTransVelX * (1 - PINCH_SMOOTHING) + moveX * PINCH_SMOOTHING;
                smoothTransVelY = smoothTransVelY * (1 - PINCH_SMOOTHING) + moveY * PINCH_SMOOTHING;
                mesh.position.x += smoothTransVelX;
                mesh.position.y -= smoothTransVelY;
                currentGesture = 'translate';
                setCommandStatus('Translating...');
                touchPointMat.color.setHex(0x57ddff);     // blue = translate
                touchPointRingMat.color.setHex(0x57ddff);
            }

            // --- EXTRUDE: total hand scale change since pinch started ---
            // Tracks total distance your hand moved toward/away from camera.
            // Applies as absolute stretch from original geometry, so the extrusion
            // length follows your hand and stays when you stop.
            const totalScaleChange = handScale - gestureState.startHandScale;

            if (Math.abs(totalScaleChange) > 0.003) {
                // Snapshot original geometry on first extrude frame
                if (!gestureState.extrudeUndoArmed) {
                    saveUndo();
                    gestureState.extrudeUndoArmed = true;
                    gestureState.extrudeRestPositions = new Float32Array(mesh.geometry.attributes.position.array);
                }

                if (gestureState.extrudeRestPositions) {
                    // Smooth the target stretch amount
                    const targetStretch = totalScaleChange * EXTRUDE_SENSITIVITY * 18;
                    smoothExtrudeVel = smoothExtrudeVel * 0.6 + targetStretch * 0.4;

                    // Get camera forward in mesh local space
                    const camForward = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
                    const invRot = mesh.quaternion.clone().invert();
                    const localAxis = camForward.applyQuaternion(invRot).normalize();

                    // Apply absolute stretch from rest positions
                    const p = mesh.geometry.attributes.position;
                    const rest = gestureState.extrudeRestPositions;
                    for (let i = 0; i < p.count; i++) {
                        const rx = rest[i * 3], ry = rest[i * 3 + 1], rz = rest[i * 3 + 2];
                        const dot = rx * localAxis.x + ry * localAxis.y + rz * localAxis.z;
                        const stretch = dot * smoothExtrudeVel;
                        p.setXYZ(i,
                            rx + localAxis.x * stretch,
                            ry + localAxis.y * stretch,
                            rz + localAxis.z * stretch
                        );
                    }
                    p.needsUpdate = true;
                    safeComputeNormals();
                    currentGesture = 'extrude';
                    setCommandStatus(`Extruding: ${Math.abs(smoothExtrudeVel).toFixed(2)} (${totalScaleChange > 0 ? 'pulling out' : 'pushing in'})`);
                    touchPointMat.color.setHex(0xff6b6b);     // red = extrude
                    touchPointRingMat.color.setHex(0xff6b6b);
                }
            }
        }
        gestureState.lastPinchPos[0] = { x: px, y: py };
        gestureState.lastHandScale = handScale;
        gestureState.lastGestures[0] = 'pinch';
        return;
    } else {
        // Reset all pinch state on release — extrusion stays where it is
        gestureState.lastPinchPos[0] = null;
        gestureState.lastHandScale = undefined;
        gestureState.startHandScale = undefined;
        gestureState.extrudeUndoArmed = false;
        gestureState.extrudeRestPositions = null;
        gestureState.translateUndoArmed = false;
        hideTouchPoint();
        if (gestureState.lastGestures[0] === 'pinch') gestureState.lastGestures[0] = null;
    }

    if (g === 'cupped') {
        // Extrude at highlighted face region
        const wrist = hand[0], mid = hand[9];
        const cx = (wrist.x + mid.x) / 2;
        const cy = (wrist.y + mid.y) / 2;
        const ndc = new THREE.Vector2(-((cx * 2) - 1), -((cy * 2) - 1));
        raycaster.setFromCamera(ndc, camera);
        const hits = raycaster.intersectObject(mesh);
        if (hits.length) {
            if (!gestureState.moldUndoArmed) { saveUndo(); gestureState.moldUndoArmed = true; }
            // Pause abstract animation during sculpting
            if (currentAppMode === 'abstract') abstractAnimating = false;
            const hitNorm = getHitNormal(hits[0]).clone().transformDirection(mesh.matrixWorld).normalize();
            const motion = getMotionVector(0);
            const strength = -motion.y * 3.0;
            extrudeAt(hits[0].point, hitNorm, strength * 0.01);
            highlightFacesAt(hits[0].point, hitNorm);
            // Update abstract rest positions and reapply colors
            if (currentAppMode === 'abstract' && abstractRestPositions) {
                abstractRestPositions = new Float32Array(mesh.geometry.attributes.position.array);
                applyBlobGradient();
            }
            setCommandStatus('Extruding — cupped hand');
        }
        currentGesture = 'extrude';
        return;
    } else {
        gestureState.moldUndoArmed = false;
        // Resume abstract animation when cupped ends
        if (currentAppMode === 'abstract' && !abstractAnimating) abstractAnimating = true;
    }

    if (g === 'point') {
        if (currentAppMode === 'sketch') {
            const tip = hand[8];
            const x = (1 - tip.x) * innerWidth;
            const y = tip.y * innerHeight;
            if (!isSketchDrawing) { isSketchDrawing = true; lastSketchPt = null; }
            sketchStroke(x, y);
            currentGesture = 'sketch';
        } else {
            // Highlight faces under fingertip
            const tip = hand[8];
            const screenX = (1 - tip.x) * innerWidth;
            const screenY = tip.y * innerHeight;
            const ndc = new THREE.Vector2(
                (screenX / innerWidth) * 2 - 1,
                -(screenY / innerHeight) * 2 + 1
            );
            raycaster.setFromCamera(ndc, camera);
            const hits = raycaster.intersectObject(mesh);
            if (hits.length) {
                const hitNorm = hits[0].face?.normal
                    ? hits[0].face.normal.clone().transformDirection(mesh.matrixWorld).normalize()
                    : new THREE.Vector3(0, 0, 1);
                highlightFacesAt(hits[0].point, hitNorm);
                setCommandStatus('Pointing at face — pinch to extrude, cupped to sculpt');
            } else {
                hideHighlight();
            }
            currentGesture = 'select';
        }
        return;
    } else {
        if (isSketchDrawing) { isSketchDrawing = false; lastSketchPt = null; }
        if (g !== 'cupped' && g !== 'pinch') hideHighlight();
    }

    if (g === 'lshape') {
        // Scale preview for add shape
        const thumbTip = hand[4], indexTip = hand[8];
        const spread = dist3D(thumbTip, indexTip);
        gestureState.addShapeScale = Math.max(0.1, Math.min(2.0, spread * 10));
        setCommandStatus(`Scale preview: ${gestureState.addShapeScale.toFixed(2)}`);
        currentGesture = 'scalepreview';
        return;
    }

    // Palm eraser in sketch mode
    if (currentAppMode === 'sketch' && g === 'palm') {
        const wrist = hand[0];
        const cx = (1 - wrist.x) * innerWidth;
        const cy = wrist.y * innerHeight;
        if (sketchCtx) {
            sketchCtx.save();
            sketchCtx.globalCompositeOperation = 'destination-out';
            sketchCtx.beginPath();
            sketchCtx.arc(cx, cy, 50, 0, Math.PI * 2);
            sketchCtx.fill();
            sketchCtx.restore();
        }
        currentGesture = 'erase';
        return;
    }

    currentGesture = null;
}

// ============================================================
// 10. MEDIAPIPE HAND TRACKING SETUP
// ============================================================
let cameraStream = null;
let lastDetectTime = 0;
const DETECT_INTERVAL_MS = 33; // ~30fps for hand detection

async function startCamera() {
    const video = document.getElementById('webcam');
    const status = document.getElementById('camera-status');
    if (cameraStream) return true; // already running
    try {
        if (status) { status.style.display = 'block'; status.textContent = '📷 Starting camera...'; }
        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
        video.srcObject = stream;
        cameraStream = stream;
        await video.play();
        video.classList.add('active');
        if (status) { status.textContent = '📷 Camera active'; setTimeout(() => { status.style.display = 'none'; }, 2000); }
        return true;
    } catch (e) {
        console.error('Camera failed:', e);
        if (status) { status.style.display = 'block'; status.textContent = '❌ Camera unavailable: ' + (e.message || e.name); }
        return false;
    }
}

function stopCamera() {
    const video = document.getElementById('webcam');
    const status = document.getElementById('camera-status');
    if (cameraStream) {
        cameraStream.getTracks().forEach(t => t.stop());
        cameraStream = null;
    }
    if (video) { video.srcObject = null; video.classList.remove('active'); }
    if (status) status.style.display = 'none';
}

async function initHandTracking() {
    try {
        const cameraOk = await startCamera();
        if (!cameraOk) {
            setCommandStatus('Camera unavailable — hand tracking needs a webcam');
            return;
        }

        setCommandStatus('Loading hand tracking model...');
        const vision = await FilesetResolver.forVisionTasks(
            'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm'
        );
        handLandmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
                delegate: 'GPU',
            },
            runningMode: 'VIDEO',
            numHands: 2,
            minHandDetectionConfidence: 0.6,
            minTrackingConfidence: 0.5,
        });

        handTrackingActive = true;
        setCommandStatus('✋ Hand tracking active — gestures ready!');
    } catch (e) {
        console.error('Hand tracking init failed:', e);
        setCommandStatus('Hand tracking failed: ' + (e.message || 'unknown error'));
    }
}

function detectHands() {
    // Hand tracking removed
    return;
}

// ============================================================
// GESTURE POINTER — dwell-to-click UI interaction
// ============================================================
const gesturePointerEl = document.getElementById('gesture-pointer');
const gpRingProgress = gesturePointerEl?.querySelector('.gp-ring-progress');
const DWELL_DELAY_MS = 500;   // time before countdown starts
const DWELL_DURATION_MS = 1500; // countdown duration
let gpVisible = false;
let gpLastX = 0, gpLastY = 0;
let gpSmoothX = 0, gpSmoothY = 0;
let gpHoverTarget = null;
let gpHoverStartTime = 0;
let gpDwellStartTime = 0;
let gpLastClickTime = 0;
const GP_CLICK_COOLDOWN = 800;  // prevent rapid re-clicks
const CIRCUMFERENCE = 2 * Math.PI * 18; // matches SVG circle r=18

let gpFrozen = false; // thumbs-up freezes pointer position

// Detect thumbs-up: thumb extended upward, other fingers curled
function isThumbsUp(hand) {
    const wrist = hand[0];
    const thumbTip = hand[4];
    const indexTip = hand[8];
    const middleTip = hand[12];
    const ringTip = hand[16];
    const pinkyTip = hand[20];
    // Thumb must be significantly above wrist
    const thumbUp = (wrist.y - thumbTip.y) > 0.08;
    // Other fingers must be curled (close to wrist)
    const othersCurled = [indexTip, middleTip, ringTip, pinkyTip].every(
        tip => Math.hypot(tip.x - wrist.x, tip.y - wrist.y) < 0.12
    );
    return thumbUp && othersCurled;
}

function updateGesturePointer(hand) {
    if (!gesturePointerEl) return;

    // Thumbs-up = freeze pointer, hand can move freely
    if (isThumbsUp(hand)) {
        if (!gpFrozen) {
            gpFrozen = true;
            gesturePointerEl.style.opacity = '0.5';
        }
        return; // Don't update position
    }
    if (gpFrozen) {
        gpFrozen = false;
        gesturePointerEl.style.opacity = '1';
    }

    const indexTip = hand[8]; // index fingertip
    // Convert MediaPipe coords (0-1, mirrored) to screen coords
    const screenX = (1 - indexTip.x) * innerWidth;
    const screenY = indexTip.y * innerHeight;

    // Smooth pointer position
    gpSmoothX = gpSmoothX * 0.5 + screenX * 0.5;
    gpSmoothY = gpSmoothY * 0.5 + screenY * 0.5;

    // Show and position pointer
    if (!gpVisible) {
        gesturePointerEl.style.display = 'block';
        gpVisible = true;
        gpSmoothX = screenX;
        gpSmoothY = screenY;
    }
    gesturePointerEl.style.left = gpSmoothX + 'px';
    gesturePointerEl.style.top = gpSmoothY + 'px';

    // Check what's under the pointer
    const elUnder = document.elementFromPoint(gpSmoothX, gpSmoothY);
    const clickable = elUnder ? elUnder.closest('button, [data-shape], select, input, a, .ws-btn, .splice-icon-btn, .splice-shape-item, .tool-btn, .draw-tool-btn, .add-shape-option') : null;

    const now = performance.now();

    if (clickable) {
        gesturePointerEl.classList.add('gp-hover');

        if (clickable !== gpHoverTarget) {
            // Started hovering a new element
            gpHoverTarget = clickable;
            gpHoverStartTime = now;
            gpDwellStartTime = 0;
            resetDwellRing();
        } else {
            const hoverDuration = now - gpHoverStartTime;

            if (hoverDuration > DWELL_DELAY_MS && gpDwellStartTime === 0) {
                // Start countdown
                gpDwellStartTime = now;
            }

            if (gpDwellStartTime > 0) {
                const dwellProgress = Math.min(1, (now - gpDwellStartTime) / DWELL_DURATION_MS);
                updateDwellRing(dwellProgress);

                if (dwellProgress >= 1 && (now - gpLastClickTime) > GP_CLICK_COOLDOWN) {
                    // Countdown complete — trigger click!
                    gpLastClickTime = now;
                    clickable.click();
                    gpHoverTarget = null;
                    gpDwellStartTime = 0;
                    resetDwellRing();

                    // Flash the dot green briefly
                    gesturePointerEl.style.setProperty('--gp-flash', '#44ff44');
                    setTimeout(() => gesturePointerEl.style.removeProperty('--gp-flash'), 300);

                    setCommandStatus(`Selected: ${clickable.textContent.trim().substring(0, 30)}`);
                }
            }
        }
    } else {
        gesturePointerEl.classList.remove('gp-hover');
        gpHoverTarget = null;
        gpDwellStartTime = 0;
        resetDwellRing();
    }
}

function updateDwellRing(progress) {
    if (!gpRingProgress) return;
    const offset = CIRCUMFERENCE * (1 - progress);
    gpRingProgress.style.strokeDashoffset = offset;
}

function resetDwellRing() {
    if (!gpRingProgress) return;
    gpRingProgress.style.strokeDashoffset = CIRCUMFERENCE;
}

function hideGesturePointer() {
    if (!gesturePointerEl) return;
    gesturePointerEl.style.display = 'none';
    gpVisible = false;
    gpHoverTarget = null;
    gpDwellStartTime = 0;
    resetDwellRing();
}

// Debug overlay
function drawDebugLandmarks(landmarks) {
    const canvas = document.getElementById('debug-canvas');
    if (!canvas) return;
    canvas.width = innerWidth;
    canvas.height = innerHeight;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // MediaPipe hand skeleton connections
    // Each finger: MCP → PIP → DIP → TIP, plus palm connections
    const CONNECTIONS = [
        // Thumb
        [0, 1], [1, 2], [2, 3], [3, 4],
        // Index
        [0, 5], [5, 6], [6, 7], [7, 8],
        // Middle
        [0, 9], [9, 10], [10, 11], [11, 12],
        // Ring
        [0, 13], [13, 14], [14, 15], [15, 16],
        // Pinky
        [0, 17], [17, 18], [18, 19], [19, 20],
        // Palm cross-connections
        [5, 9], [9, 13], [13, 17],
    ];

    // Joint types for coloring
    const TIPS = new Set([4, 8, 12, 16, 20]);
    const MCPS = new Set([5, 9, 13, 17]);
    const WRIST = 0;

    const handColors = ['#ff4488', '#44aaff']; // hand 1 = pink, hand 2 = blue

    for (let h = 0; h < landmarks.length; h++) {
        const hand = landmarks[h];
        const color = handColors[h % 2];

        // Helper: get screen coords (mirrored)
        const pt = (idx) => ({
            x: (1 - hand[idx].x) * canvas.width,
            y: hand[idx].y * canvas.height,
        });

        // Draw skeleton wires
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5;
        ctx.globalAlpha = 0.8;
        for (const [a, b] of CONNECTIONS) {
            const pa = pt(a), pb = pt(b);
            ctx.beginPath();
            ctx.moveTo(pa.x, pa.y);
            ctx.lineTo(pb.x, pb.y);
            ctx.stroke();
        }

        // Draw joints
        ctx.globalAlpha = 1.0;
        for (let i = 0; i < 21; i++) {
            const p = pt(i);

            if (TIPS.has(i)) {
                // Fingertips — larger, bright filled circle with glow
                ctx.fillStyle = '#ffffff';
                ctx.shadowColor = color;
                ctx.shadowBlur = 8;
                ctx.beginPath();
                ctx.arc(p.x, p.y, 7, 0, Math.PI * 2);
                ctx.fill();
                ctx.shadowBlur = 0;
                // Inner colored dot
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
                ctx.fill();
            } else if (i === WRIST) {
                // Wrist — square anchor
                ctx.fillStyle = color;
                ctx.shadowColor = color;
                ctx.shadowBlur = 6;
                ctx.fillRect(p.x - 6, p.y - 6, 12, 12);
                ctx.shadowBlur = 0;
            } else if (MCPS.has(i)) {
                // MCP knuckles — medium ring
                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);
                ctx.stroke();
                ctx.fillStyle = 'rgba(255,255,255,0.4)';
                ctx.fill();
            } else {
                // PIP/DIP joints — small dots
                ctx.fillStyle = color;
                ctx.globalAlpha = 0.7;
                ctx.beginPath();
                ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
                ctx.fill();
                ctx.globalAlpha = 1.0;
            }
        }

        // Label fingertips
        ctx.font = 'bold 10px monospace';
        ctx.fillStyle = '#ffffff';
        ctx.textAlign = 'center';
        const tipLabels = { 4: 'THB', 8: 'IDX', 12: 'MID', 16: 'RNG', 20: 'PNK' };
        for (const [idx, label] of Object.entries(tipLabels)) {
            const p = pt(+idx);
            ctx.fillText(label, p.x, p.y - 12);
        }

        // Draw pinch distance line if pinching
        if (isPinch(hand)) {
            const thumb = pt(4), index = pt(8);
            ctx.strokeStyle = '#ffdd57';
            ctx.lineWidth = 3;
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            ctx.moveTo(thumb.x, thumb.y);
            ctx.lineTo(index.x, index.y);
            ctx.stroke();
            ctx.setLineDash([]);
            // Pinch midpoint dot
            const mx = (thumb.x + index.x) / 2, my = (thumb.y + index.y) / 2;
            ctx.fillStyle = '#ffdd57';
            ctx.shadowColor = '#ffdd57';
            ctx.shadowBlur = 10;
            ctx.beginPath();
            ctx.arc(mx, my, 5, 0, Math.PI * 2);
            ctx.fill();
            ctx.shadowBlur = 0;
        }

        // Wrist label with hand index
        const w = pt(0);
        ctx.fillStyle = color;
        ctx.font = 'bold 12px monospace';
        ctx.fillText(`Hand ${h + 1}`, w.x, w.y + 18);
    }

    // Gesture + command readout
    ctx.strokeStyle = 'transparent';
    const gestureText = document.getElementById('debug-gesture-text');
    if (gestureText) gestureText.textContent = `Gesture: ${currentGesture || 'None'}`;
    const commandText = document.getElementById('debug-command-text');
    if (commandText) commandText.textContent = `Command: ${currentGesture || 'Idle'}`;
}

// ============================================================
// 11. MOUSE INTERACTION
// ============================================================
let isDragging = false;
let mouseBtn = -1;
let lastMouse = { x: 0, y: 0 };
let sculptHitPoint = null;
let sculptHitNormal = null;
let mouseUndoArmed = false;
let carveHoldActive = false;  // true when click-holding on mesh to carve in
let carvePoint = null;
let carveNormal = null;
let draggingObject = null; // scene object being dragged
let dragMoved = false; // track if mouse actually moved
let pendingDeselect = false; // deselect on mouseup only if no drag

// Orbit/pan reference point
const orbitRefEl = document.getElementById('orbit-ref');
let orbitRefTimeout = null;
function showOrbitRef() {
    if (!orbitRefEl) return;
    orbitRefEl.style.left = '50%';
    orbitRefEl.style.top = '50%';
    orbitRefEl.style.display = 'block';
    orbitRefEl.classList.add('visible');
    clearTimeout(orbitRefTimeout);
}
function hideOrbitRef() {
    if (!orbitRefEl) return;
    orbitRefEl.classList.remove('visible');
    orbitRefTimeout = setTimeout(() => { orbitRefEl.style.display = 'none'; }, 200);
}

renderer.domElement.addEventListener('contextmenu', e => e.preventDefault());

renderer.domElement.addEventListener('mousedown', e => {
    isDragging = true;
    mouseBtn = e.button;
    lastMouse = { x: e.clientX, y: e.clientY };
    mouseUndoArmed = false;
    dragMoved = false;
    pendingDeselect = false;

    // Add Shape mode — click to place
    if (addShapeMode && currentAppMode === 'model' && e.button === 0) {
        if (addShapeVisible) {
            confirmAddShape();
        }
        isDragging = false;
        return;
    }

    if (currentAppMode === 'model') {
        const ndc = new THREE.Vector2(
            (e.clientX / innerWidth) * 2 - 1,
            -(e.clientY / innerHeight) * 2 + 1
        );
        raycaster.setFromCamera(ndc, camera);

        // Check scene objects first (separate added shapes)
        if (sceneObjects.length > 0 && e.button === 0) {
            const objMeshes = sceneObjects.map(o => o.mesh);
            const objHits = raycaster.intersectObjects(objMeshes, true);
            if (objHits.length > 0) {
                // Walk up to find which scene object was hit (could hit wireframe child)
                let hitObj = objHits[0].object;
                let entry = null;
                while (hitObj) {
                    entry = sceneObjects.find(o => o.mesh === hitObj);
                    if (entry) break;
                    hitObj = hitObj.parent;
                }
                if (entry) {
                    selectObject(entry);
                    draggingObject = entry;
                    isDragging = true;
                    return;
                }
            }
        }

        // Click empty space = mark for deselect (only on mouseup if no drag)
        if (selectedObject && e.button === 0) {
            const mainHits = raycaster.intersectObject(mesh);
            const objMeshes = sceneObjects.map(o => o.mesh);
            const objHits = raycaster.intersectObjects(objMeshes, true);
            if (mainHits.length === 0 && objHits.length === 0) {
                pendingDeselect = true;
            }
        }

        const hits = raycaster.intersectObject(mesh);

        if (hits.length && e.button === 0) {
            sculptHitPoint = hits[0].point.clone();
            sculptHitNormal = getHitNormal(hits[0]).clone();
            const hitNormW = hits[0].face?.normal
                ? hits[0].face.normal.clone().transformDirection(mesh.matrixWorld).normalize()
                : new THREE.Vector3(0, 0, 1);

            // Shift+click = duplicate region
            if (e.shiftKey) {
                startDuplicate(hits[0].point, hitNormW);
                isDragging = true;
                return;
            }

            // Click+hold on mesh (no modifier) = start carve-in
            if (!e.ctrlKey && !e.metaKey && !e.altKey) {
                carveHoldActive = true;
                carvePoint = hits[0].point.clone();
                carveNormal = hitNormW.clone().negate(); // push inward
                if (!mouseUndoArmed) { saveUndo(); mouseUndoArmed = true; }
            }

            if (activeTool === 'distance') {
                measureDistance(hits[0].point.clone());
                isDragging = false;
                return;
            }
        }
    } else if (currentAppMode === 'abstract') {
        abstractDragging = true;
        abstractDragStart = { x: e.clientX, y: e.clientY };
        const ndc = new THREE.Vector2(
            (e.clientX / innerWidth) * 2 - 1,
            -(e.clientY / innerHeight) * 2 + 1
        );
        raycaster.setFromCamera(ndc, camera);
        const hits = raycaster.intersectObject(mesh);
        if (hits.length) abstractHitPoint = hits[0].point.clone();
    }
});

renderer.domElement.addEventListener('mousemove', e => {
    // Add Shape mode — hover to position preview
    if (addShapeMode && currentAppMode === 'model' && !isDragging) {
        const ndc = new THREE.Vector2(
            (e.clientX / innerWidth) * 2 - 1,
            -(e.clientY / innerHeight) * 2 + 1
        );
        raycaster.setFromCamera(ndc, camera);
        const hits = raycaster.intersectObjects([mesh]);
        if (hits.length) {
            const hitNorm = hits[0].face?.normal
                ? hits[0].face.normal.clone().transformDirection(mesh.matrixWorld).normalize()
                : new THREE.Vector3(0, 0, 1);
            addShapeHitNormal.copy(hitNorm);
            addShapeSurfaceHit.copy(hits[0].point);
            const surfaceOffset = addShapeBaseRadius * addShapeScale * 0.35;
            addShapePosition.copy(hits[0].point).addScaledVector(hitNorm, surfaceOffset);
            updateAddShapePreview(addShapePosition, addShapeScale);
        } else {
            const dir = new THREE.Vector3();
            raycaster.ray.direction.normalize();
            dir.copy(raycaster.ray.direction).multiplyScalar(4);
            addShapePosition.copy(raycaster.ray.origin).add(dir);
            addShapeSurfaceHit.copy(addShapePosition);
            updateAddShapePreview(addShapePosition, addShapeScale);
        }
        return;
    }

    // Face highlight on hover (not dragging, model mode)
    if (!isDragging && currentAppMode === 'model' && !addShapeMode) {
        const ndc = new THREE.Vector2(
            (e.clientX / innerWidth) * 2 - 1,
            -(e.clientY / innerHeight) * 2 + 1
        );
        raycaster.setFromCamera(ndc, camera);
        const hits = raycaster.intersectObject(mesh);
        if (hits.length) {
            const hitNorm = hits[0].face?.normal
                ? hits[0].face.normal.clone().transformDirection(mesh.matrixWorld).normalize()
                : new THREE.Vector3(0, 0, 1);
            highlightFacesAt(hits[0].point, hitNorm);
        } else {
            hideHighlight();
        }
        return;
    }

    if (!isDragging) return;

    // Drag selected scene object — Shift+drag = scale along axis, normal drag = move
    if (draggingObject) {
        const dx = e.clientX - lastMouse.x;
        const dy = e.clientY - lastMouse.y;
        lastMouse = { x: e.clientX, y: e.clientY };

        if (e.shiftKey) {
            // Shift+drag = stretch/scale along dominant axis
            const scaleSpeed = 0.005;
            if (Math.abs(dx) > Math.abs(dy)) {
                // Horizontal drag = scale X
                draggingObject.mesh.scale.x = Math.max(0.05, draggingObject.mesh.scale.x + dx * scaleSpeed);
            } else {
                // Vertical drag = scale Y
                draggingObject.mesh.scale.y = Math.max(0.05, draggingObject.mesh.scale.y - dy * scaleSpeed);
            }
            updateSelectedOutline();
            const s = draggingObject.mesh.scale;
            setCommandStatus(`Scale: X=${s.x.toFixed(2)} Y=${s.y.toFixed(2)} Z=${s.z.toFixed(2)} — Shift+drag to deform`);
        } else {
            // Normal drag = move
            const moveSpeed = 0.008;
            const right = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0).normalize();
            const up = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 1).normalize();
            const invMeshRot = new THREE.Matrix4().extractRotation(mesh.matrixWorld).invert();
            right.applyMatrix4(invMeshRot);
            up.applyMatrix4(invMeshRot);
            draggingObject.mesh.position.addScaledVector(right, dx * moveSpeed);
            draggingObject.mesh.position.addScaledVector(up, -dy * moveSpeed);
            updateSelectedOutline();
            setCommandStatus(`Moving ${SHAPE_LABELS[draggingObject.name] || draggingObject.name}...`);
        }
        return;
    }

    // Stop carving when mouse moves (dragging = extrude, not carve)
    if (carveHoldActive) {
        const dx = e.clientX - lastMouse.x;
        const dy = e.clientY - lastMouse.y;
        if (Math.abs(dx) > 3 || Math.abs(dy) > 3) {
            carveHoldActive = false;
        }
    }

    // Duplicate mode — drag moves the preview
    if (dupMode) {
        updateDuplicate(e.clientX, e.clientY);
        return;
    }

    // Hide highlight during drag
    hideHighlight();
    dragMoved = true;
    const dx = e.clientX - lastMouse.x;
    const dy = e.clientY - lastMouse.y;
    lastMouse = { x: e.clientX, y: e.clientY };

    if (currentAppMode === 'model') {
        if (mouseBtn === 2) {
            // Right drag → pan
            if (!mouseUndoArmed) { saveUndo(); mouseUndoArmed = true; }
            showOrbitRef();
            mesh.position.x += dx * 0.01;
            mesh.position.y -= dy * 0.01;
        } else if (mouseBtn === 0) {
            // Left drag → orbit
            if (!mouseUndoArmed) { saveUndo(); mouseUndoArmed = true; }
            showOrbitRef();
            if (selectedObject) {
                selectedObject.mesh.rotation.y += dx * 0.01;
                selectedObject.mesh.rotation.x += dy * 0.01;
                updateSelectedOutline();
                setCommandStatus(`Rotating ${SHAPE_LABELS[selectedObject.name] || selectedObject.name}`);
            } else if (sceneObjects.length > 0) {
                // Orbit around the center of all objects
                const pivot = new THREE.Vector3();
                for (const entry of sceneObjects) {
                    pivot.add(entry.mesh.position);
                }
                pivot.divideScalar(sceneObjects.length);

                // Move mesh so pivot is at origin, rotate, move back
                mesh.position.sub(pivot);
                mesh.rotation.y += dx * 0.01;
                mesh.rotation.x += dy * 0.01;
                mesh.position.add(pivot);
            } else {
                mesh.rotation.y += dx * 0.01;
                mesh.rotation.x += dy * 0.01;
            }
        }
    } else if (currentAppMode === 'abstract') {
        if (abstractDragging && abstractHitPoint) {
            if (!mouseUndoArmed) { saveUndo(); mouseUndoArmed = true; }
            // Pause breathing while sculpting so it doesn't fight
            abstractAnimating = false;
            if (e.shiftKey) {
                const normal = abstractHitPoint.clone().normalize();
                extrudeAt(abstractHitPoint, normal, -dy * 0.005);
            } else {
                extrudeAt(abstractHitPoint, abstractHitPoint.clone().normalize(), -dy * 0.005);
            }
            // Update rest positions so animation incorporates changes
            if (abstractRestPositions) {
                abstractRestPositions = new Float32Array(mesh.geometry.attributes.position.array);
            }
            // Re-apply colors to prevent black
            applyBlobGradient();
        } else {
            mesh.rotation.y += dx * 0.01;
            mesh.rotation.x += dy * 0.01;
        }
    }
});

renderer.domElement.addEventListener('mouseup', () => {
    if (dupMode) {
        confirmDuplicate();
    }
    // Only deselect if it was a click (no drag movement)
    if (pendingDeselect && !dragMoved) {
        deselectObject();
    }
    pendingDeselect = false;
    draggingObject = null;
    dragMoved = false;
    isDragging = false;
    hideOrbitRef();
    mouseBtn = -1;
    sculptHitPoint = null;
    sculptHitNormal = null;
    abstractDragging = false;
    abstractHitPoint = null;
    mouseUndoArmed = false;
    carveHoldActive = false;
    carvePoint = null;
    carveNormal = null;
    if (currentAppMode === 'abstract' && !abstractAnimating) {
        abstractAnimating = true;
    }
});

renderer.domElement.addEventListener('click', e => {
    if (currentAppMode === 'model' && e.altKey) {
        mirrorMesh('x');
    }
});

renderer.domElement.addEventListener('dblclick', () => {
    setCommandStatus('Selection cleared');
});

renderer.domElement.addEventListener('wheel', e => {
    if (addShapeMode && currentAppMode === 'model') {
        // Scroll to resize preview
        e.preventDefault();
        addShapeScale = Math.max(0.1, Math.min(3.0, addShapeScale - e.deltaY * 0.003));
        if (addShapePreviewMesh) {
            addShapePreviewMesh.scale.setScalar(addShapeScale);
            recomputeAddShapePosition();
        }
        setCommandStatus(`Add Shape size: ${addShapeScale.toFixed(2)}`);
        return;
    }
    // Scroll to resize selected object (uniform)
    if (selectedObject && currentAppMode === 'model') {
        e.preventDefault();
        const scaleFactor = 1 - e.deltaY * 0.002;
        const s = selectedObject.mesh.scale;
        s.x = Math.max(0.05, Math.min(5.0, s.x * scaleFactor));
        s.y = Math.max(0.05, Math.min(5.0, s.y * scaleFactor));
        s.z = Math.max(0.05, Math.min(5.0, s.z * scaleFactor));
        updateSelectedOutline();
        setCommandStatus(`Size: ${s.x.toFixed(2)} × ${s.y.toFixed(2)} × ${s.z.toFixed(2)}`);
        return;
    }
    if (currentAppMode === 'abstract') {
        mesh.scale.multiplyScalar(1 - e.deltaY * 0.001);
    } else {
        camera.position.z += e.deltaY * 0.005;
        camera.position.z = Math.max(2, Math.min(20, camera.position.z));
    }
}, { passive: false });
const sketchCanvas = document.getElementById('sketch-canvas');
if (sketchCanvas) {
    sketchCanvas.addEventListener('mousedown', e => {
        if (currentAppMode !== 'sketch') return;
        isSketchDrawing = true;
        lastSketchPt = null;
        sketchStroke(e.clientX, e.clientY);
    });
    sketchCanvas.addEventListener('mousemove', e => {
        if (!isSketchDrawing || currentAppMode !== 'sketch') return;
        sketchStroke(e.clientX, e.clientY);
    });
    sketchCanvas.addEventListener('mouseup', () => { isSketchDrawing = false; lastSketchPt = null; });
    sketchCanvas.addEventListener('mouseleave', () => { isSketchDrawing = false; lastSketchPt = null; });

    // Touch support
    sketchCanvas.addEventListener('touchstart', e => {
        if (currentAppMode !== 'sketch') return;
        e.preventDefault();
        isSketchDrawing = true; lastSketchPt = null;
        sketchStroke(e.touches[0].clientX, e.touches[0].clientY);
    }, { passive: false });
    sketchCanvas.addEventListener('touchmove', e => {
        if (!isSketchDrawing || currentAppMode !== 'sketch') return;
        e.preventDefault();
        sketchStroke(e.touches[0].clientX, e.touches[0].clientY);
    }, { passive: false });
    sketchCanvas.addEventListener('touchend', () => { isSketchDrawing = false; lastSketchPt = null; });
}

// ============================================================
// 12. UI WIRING
// ============================================================

// Mode tabs
document.querySelectorAll('.ws-btn').forEach(btn => {
    btn.addEventListener('click', () => applyAppMode(btn.dataset.appmode));
});

// Shape menu
const shapeMenu = document.getElementById('splice-shape-menu');
const shapeTrigger = document.getElementById('splice-shape-trigger');
if (shapeTrigger) {
    shapeTrigger.addEventListener('click', () => {
        shapeMenu.classList.toggle('open');
    });
}
document.querySelectorAll('.splice-shape-item').forEach(item => {
    item.addEventListener('click', () => {
        saveUndo();
        setShape(item.dataset.shape);
        shapeMenu.classList.remove('open');
    });
});

function syncShapeUI(name) {
    const label = document.getElementById('splice-shape-label');
    if (label) label.textContent = SHAPE_LABELS[name] || name;
}

// Toolbar tool buttons
document.querySelectorAll('.tool-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        // Exit add-shape mode when switching to any other tool
        if (btn.dataset.tool !== 'addshape' && addShapeMode) {
            exitAddShapeMode();
        }
        document.querySelectorAll('.tool-btn').forEach(b => {
            if (b.id !== 'add-shape-btn') b.classList.remove('active');
        });
        if (btn.dataset.tool === 'addshape') {
            const picker = document.getElementById('add-shape-picker');
            if (addShapeMode) {
                exitAddShapeMode();
            } else {
                // Show shape picker
                if (picker) picker.style.display = picker.style.display === 'none' ? 'flex' : 'none';
            }
            return;
        }
        btn.classList.add('active');
        activeTool = btn.dataset.tool;
        setCommandStatus(`Tool: ${activeTool}`);
    });
});

// Add Shape picker options
document.querySelectorAll('.add-shape-option').forEach(btn => {
    btn.addEventListener('click', () => {
        const shape = btn.dataset.addshape;
        document.getElementById('add-shape-picker').style.display = 'none';
        document.querySelectorAll('.tool-btn').forEach(b => {
            if (b.id !== 'add-shape-btn') b.classList.remove('active');
        });
        enterAddShapeMode(shape);
    });
});

// Quick commands
document.querySelectorAll('[data-quick-command]').forEach(btn => {
    btn.addEventListener('click', () => {
        const cmd = btn.dataset.quickCommand;
        if (cmd === 'wireframe') {
            material.wireframe = !material.wireframe;
            material.needsUpdate = true;
        } else if (cmd === 'reset') {
            saveUndo();
            setShape(currentShape);
        }
    });
});

// Undo button
document.getElementById('undo-btn')?.addEventListener('click', undo);

// STL Export
document.getElementById('export-stl-btn')?.addEventListener('click', exportSTL);
document.getElementById('merge-all-btn')?.addEventListener('click', mergeAllObjects);
document.getElementById('save-btn')?.addEventListener('click', saveProject);
document.getElementById('save-as-btn')?.addEventListener('click', saveAsProject);

// Clipboard for copy/cut/paste
let clipboardObject = null; // { name, position, scale, rotation }

// Keyboard shortcuts
window.addEventListener('keydown', e => {
    if (e.target.closest('input, textarea')) return;

    if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
        e.preventDefault();
        undo();
    }

    // Ctrl+S = Save, Ctrl+Shift+S = Save As
    if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault();
        if (e.shiftKey) {
            saveAsProject();
        } else {
            saveProject();
        }
    }

    // Copy: Ctrl+C
    if ((e.ctrlKey || e.metaKey) && e.key === 'c' && selectedObject) {
        e.preventDefault();
        clipboardObject = {
            name: selectedObject.name,
            position: selectedObject.mesh.position.clone(),
            scale: selectedObject.mesh.scale.x,
            rotation: selectedObject.mesh.rotation.clone(),
        };
        setCommandStatus(`Copied ${SHAPE_LABELS[selectedObject.name] || selectedObject.name}`);
    }

    // Cut: Ctrl+X
    if ((e.ctrlKey || e.metaKey) && e.key === 'x' && selectedObject) {
        e.preventDefault();
        clipboardObject = {
            name: selectedObject.name,
            position: selectedObject.mesh.position.clone(),
            scale: selectedObject.mesh.scale.x,
            rotation: selectedObject.mesh.rotation.clone(),
        };
        const name = selectedObject.name;
        removeSceneObject(selectedObject);
        setCommandStatus(`Cut ${SHAPE_LABELS[name] || name}`);
    }

    // Paste: Ctrl+V
    if ((e.ctrlKey || e.metaKey) && e.key === 'v' && clipboardObject) {
        e.preventDefault();
        const offset = new THREE.Vector3(0.3, 0.3, 0);
        const newPos = clipboardObject.position.clone().add(offset);
        const worldPos = newPos.clone().applyMatrix4(mesh.matrixWorld);
        const entry = addSceneObject(clipboardObject.name, worldPos, clipboardObject.scale, clipboardObject.rotation);
        // Override position to local since addSceneObject converts from world
        entry.mesh.position.copy(newPos);
        selectObject(entry);
        setCommandStatus(`Pasted ${SHAPE_LABELS[clipboardObject.name] || clipboardObject.name}`);
    }

    if (e.key === 'Escape') {
        if (addShapeMode) { exitAddShapeMode(); }
        deselectObject();
        setCommandStatus('Ready');
    }

    // Delete selected object
    if ((e.key === 'Delete' || e.key === 'Backspace') && selectedObject) {
        e.preventDefault();
        const name = selectedObject.name;
        removeSceneObject(selectedObject);
        setCommandStatus(`Deleted ${SHAPE_LABELS[name] || name}`);
    }

    // Rotate add-shape preview with keyboard
    if (addShapeMode && addShapePreviewMesh) {
        const step = Math.PI / 12; // 15 degrees per press
        let changed = false;
        if (e.key === 'q' || e.key === 'Q') { addShapeRotation.z += step; changed = true; }
        if (e.key === 'e' || e.key === 'E') { addShapeRotation.z -= step; changed = true; }
        if (e.key === 'r' || e.key === 'R' || e.key === 'ArrowUp') { addShapeRotation.x += step; changed = true; }
        if (e.key === 'f' || e.key === 'F' || e.key === 'ArrowDown') { addShapeRotation.x -= step; changed = true; }
        if (e.key === 'ArrowLeft') { addShapeRotation.y += step; changed = true; }
        if (e.key === 'ArrowRight') { addShapeRotation.y -= step; changed = true; }
        if (changed) {
            e.preventDefault();
            updateAddShapePreview(addShapePosition, addShapeScale);
            const deg = (r) => Math.round(r * 180 / Math.PI);
            setCommandStatus(`Rotate: X${deg(addShapeRotation.x)}° Y${deg(addShapeRotation.y)}° Z${deg(addShapeRotation.z)}° — Q/E roll, R/F tilt, ←→ spin`);
        }
    }
});

// Texture selector — applies to selected object or main mesh
document.getElementById('texture-select')?.addEventListener('change', e => {
    const t = TEXTURES[e.target.value];
    if (!t) return;
    if (selectedObject) {
        Object.assign(selectedObject.mesh.material, t);
        selectedObject.mesh.material.needsUpdate = true;
    } else {
        applyTexture(e.target.value);
    }
});

// Model color wheel — applies to selected object or main mesh
document.getElementById('model-color-wheel')?.addEventListener('input', e => {
    if (selectedObject) {
        selectedObject.mesh.material.color.set(e.target.value);
        selectedObject.mesh.material.needsUpdate = true;
    } else {
        material.color.set(e.target.value);
        material.needsUpdate = true;
    }
});

// Abstract color wheel
document.getElementById('abstract-color-wheel')?.addEventListener('input', e => {
    material.color.set(e.target.value);
    material.needsUpdate = true;
    if (currentAppMode === 'abstract') applyBlobGradient();
});

// Sketch color wheel
document.getElementById('sketch-color-wheel')?.addEventListener('input', e => {
    sketchColor = e.target.value;
});

// Sketch tool buttons
document.querySelectorAll('.draw-tool-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.draw-tool-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        sketchTool = btn.dataset.drawtool;
    });
});

// Gradient panels
document.getElementById('gradient-btn')?.addEventListener('click', () => {
    document.getElementById('gradient-panel').style.display = 'block';
});
document.getElementById('abstract-gradient-btn')?.addEventListener('click', () => {
    document.getElementById('gradient-panel').style.display = 'block';
});
document.getElementById('grad-close')?.addEventListener('click', () => {
    document.getElementById('gradient-panel').style.display = 'none';
});
document.getElementById('grad-apply')?.addEventListener('click', () => {
    // Apply gradient as vertex colors
    if (!mesh) return;
    const pos = mesh.geometry.attributes.position;
    const c1 = new THREE.Color(document.getElementById('grad-color1').value);
    const c2 = new THREE.Color(document.getElementById('grad-color2').value);
    const c3 = new THREE.Color(document.getElementById('grad-color3').value);
    const type = document.getElementById('gradient-type').value;
    const colors = new Float32Array(pos.count * 3);
    let yMin = Infinity, yMax = -Infinity;
    for (let i = 0; i < pos.count; i++) {
        const y = pos.getY(i);
        if (y < yMin) yMin = y;
        if (y > yMax) yMax = y;
    }
    const range = Math.max(yMax - yMin, 0.01);
    for (let i = 0; i < pos.count; i++) {
        let t;
        if (type === 'radial') {
            const x = pos.getX(i), z = pos.getZ(i);
            t = Math.min(1, Math.sqrt(x * x + z * z) / 2);
        } else if (type === 'angular') {
            const x = pos.getX(i), z = pos.getZ(i);
            t = (Math.atan2(z, x) + Math.PI) / (2 * Math.PI);
        } else {
            t = (pos.getY(i) - yMin) / range;
        }
        const c = new THREE.Color();
        if (t < 0.5) c.lerpColors(c1, c2, t * 2);
        else c.lerpColors(c2, c3, (t - 0.5) * 2);
        colors[i * 3] = c.r;
        colors[i * 3 + 1] = c.g;
        colors[i * 3 + 2] = c.b;
    }
    mesh.geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    material.vertexColors = true;
    material.needsUpdate = true;
    document.getElementById('gradient-panel').style.display = 'none';
    setCommandStatus('Gradient applied');
});

// Sketch export
document.getElementById('sketch-export-btn')?.addEventListener('click', () => {
    document.getElementById('sketch-export-panel').style.display = 'block';
});
document.getElementById('export-cancel')?.addEventListener('click', () => {
    document.getElementById('sketch-export-panel').style.display = 'none';
});
document.getElementById('export-confirm')?.addEventListener('click', () => {
    const template = document.getElementById('export-template').value;
    exportSketchSVG(template);
    document.getElementById('sketch-export-panel').style.display = 'none';
});

// Check mesh button
document.querySelector('[data-tool="check"]')?.addEventListener('click', checkMesh);

// --- Loading overlay helpers ---
function showLoading(text) {
    const el = document.getElementById('loading-overlay');
    if (el) { el.style.display = 'flex'; el.querySelector('.loading-text').textContent = text || 'Loading...'; }
}
function hideLoading() {
    const el = document.getElementById('loading-overlay');
    if (el) el.style.display = 'none';
}

// --- Error toast ---
function showError(msg, duration = 4000) {
    const el = document.getElementById('error-toast');
    if (!el) return;
    el.textContent = msg;
    el.style.display = 'block';
    clearTimeout(el._timer);
    el._timer = setTimeout(() => { el.style.display = 'none'; }, duration);
}

// Hand tracking toggle (with loading overlay)
const handsBtn = document.getElementById('hands-toggle-btn');
if (handsBtn) {
    handsBtn.addEventListener('click', async () => {
        if (!handTrackingActive) {
            handsBtn.textContent = '⏳ Loading...';
            showLoading('Loading hand tracking model...');
            await initHandTracking();
            hideLoading();
            if (handTrackingActive) {
                handsBtn.textContent = '✋ Hands Off';
                handsBtn.classList.add('active');
            } else {
                handsBtn.textContent = '✋ Hands On';
                showError('Camera unavailable — check permissions');
            }
        } else {
            handTrackingActive = false;
            stopCamera();
            handsBtn.textContent = '✋ Hands On';
            handsBtn.classList.remove('active');
            setCommandStatus('Hand tracking stopped');
        }
    });
}

// Debug toggle
const debugBtn = document.getElementById('debug-toggle-btn');
if (debugBtn) {
    debugBtn.addEventListener('click', () => {
        isDebug = !isDebug;
        debugBtn.textContent = isDebug ? '🛠 Debug On' : '🛠 Debug Off';
        debugBtn.classList.toggle('active', isDebug);
        document.getElementById('debug-overlay').style.display = isDebug ? 'block' : 'none';
    });
}

// --- Pattern Wrap wiring ---
const patternBtn = document.getElementById('pattern-wrap-btn');
const patternPanel = document.getElementById('pattern-wrap-panel');
if (patternBtn && patternPanel) {
    patternBtn.addEventListener('click', () => {
        patternPanel.style.display = patternPanel.style.display === 'none' ? 'block' : 'none';
    });
    document.getElementById('pattern-count')?.addEventListener('input', e => {
        document.getElementById('pattern-count-val').textContent = e.target.value;
    });
    document.getElementById('pattern-scale')?.addEventListener('input', e => {
        document.getElementById('pattern-scale-val').textContent = (e.target.value / 100).toFixed(2);
    });
    document.getElementById('pattern-apply')?.addEventListener('click', () => {
        const shape = document.getElementById('pattern-shape').value;
        const mode = document.getElementById('pattern-mode').value;
        const count = parseInt(document.getElementById('pattern-count').value);
        const scale = parseInt(document.getElementById('pattern-scale').value) / 100;
        patternWrap(shape, scale, count, mode);
        patternPanel.style.display = 'none';
    });
    document.getElementById('pattern-close')?.addEventListener('click', () => {
        patternPanel.style.display = 'none';
    });
}

// --- Tuning Panel wiring ---
const tunePanel = document.getElementById('tune-panel');
const tuneBtn = document.getElementById('tune-toggle-btn');
if (tuneBtn && tunePanel) {
    tuneBtn.addEventListener('click', () => {
        tunePanel.style.display = tunePanel.style.display === 'none' ? 'block' : 'none';
    });
}
// Visual tuning
document.getElementById('tune-accent')?.addEventListener('input', e => {
    document.documentElement.style.setProperty('--accent', e.target.value);
});
document.getElementById('tune-bg')?.addEventListener('input', e => {
    document.documentElement.style.setProperty('--bg-color', e.target.value);
});
document.getElementById('tune-grid')?.addEventListener('input', e => {
    const v = e.target.value / 100;
    document.getElementById('tune-grid-val').textContent = v.toFixed(2);
    grid.material.opacity = v;
});
document.getElementById('tune-shadow')?.addEventListener('input', e => {
    const v = e.target.value / 100;
    document.getElementById('tune-shadow-val').textContent = v.toFixed(2);
    const toolbar = document.getElementById('splice-toolbar');
    if (toolbar) toolbar.style.boxShadow = `0 14px 28px rgba(0,0,0,${v})`;
});
// Gesture tuning
document.getElementById('tune-rot')?.addEventListener('input', e => {
    const v = e.target.value / 10;
    document.getElementById('tune-rot-val').textContent = v.toFixed(1);
    ROT_SENSITIVITY = v;
});
document.getElementById('tune-damp')?.addEventListener('input', e => {
    const v = e.target.value / 100;
    document.getElementById('tune-damp-val').textContent = v.toFixed(2);
    ROT_DAMPING = v;
});
document.getElementById('tune-pinch')?.addEventListener('input', e => {
    const v = e.target.value / 10;
    document.getElementById('tune-pinch-val').textContent = v.toFixed(1);
    TRANSLATE_SENSITIVITY = v;
});
// Sculpting tuning
document.getElementById('tune-brush')?.addEventListener('input', e => {
    const v = e.target.value / 100;
    document.getElementById('tune-brush-val').textContent = v.toFixed(2);
    brushRadius = v;
});
document.getElementById('tune-strength')?.addEventListener('input', e => {
    const v = e.target.value / 100;
    document.getElementById('tune-strength-val').textContent = v.toFixed(2);
    sculptStrength = v;
});
document.getElementById('tune-displace')?.addEventListener('input', e => {
    const v = e.target.value / 100;
    document.getElementById('tune-displace-val').textContent = v.toFixed(2);
    MAX_VERTEX_DISPLACEMENT = v;
});
// Animation tuning
let abstractBreatheSpeed = 1.0;
document.getElementById('tune-breathe')?.addEventListener('input', e => {
    const v = e.target.value / 10;
    document.getElementById('tune-breathe-val').textContent = v.toFixed(1);
    abstractBreatheSpeed = v;
});
document.getElementById('tune-reduced-motion')?.addEventListener('change', e => {
    if (e.target.checked) {
        smoothRotVelX = 0; smoothRotVelY = 0;
        ROT_DAMPING = 0;
    } else {
        ROT_DAMPING = parseFloat(document.getElementById('tune-damp')?.value || 92) / 100;
    }
});
document.getElementById('tune-reset')?.addEventListener('click', () => {
    ROT_SENSITIVITY = 8.0; ROT_DAMPING = 0.92; TRANSLATE_SENSITIVITY = 5.0;
    brushRadius = 0.6; sculptStrength = 0.04; MAX_VERTEX_DISPLACEMENT = 0.15;
    abstractBreatheSpeed = 1.0;
    document.documentElement.style.setProperty('--accent', '#7b8b6f');
    document.documentElement.style.setProperty('--bg-color', '#f8eedb');
    grid.material.opacity = 0.3;
    // Reset slider positions
    const resets = {
        'tune-rot': 80, 'tune-damp': 92, 'tune-pinch': 50,
        'tune-brush': 60, 'tune-strength': 4, 'tune-displace': 15,
        'tune-breathe': 10, 'tune-grid': 30, 'tune-shadow': 15,
    };
    for (const [id, val] of Object.entries(resets)) {
        const el = document.getElementById(id);
        if (el) el.value = val;
    }
    document.getElementById('tune-accent').value = '#7b8b6f';
    document.getElementById('tune-bg').value = '#f8eedb';
    setCommandStatus('Tuning reset to defaults');
});
document.getElementById('tune-close')?.addEventListener('click', () => {
    tunePanel.style.display = 'none';
});

// ============================================================
// 12b. ANIMATE MODE — draw a path, object follows it
// ============================================================
let animPath = [];           // array of {x, y} screen points
let animPath3D = [];         // converted to 3D world positions
let animIsDrawing = false;
let animIsPlaying = false;
let animProgress = 0;        // 0 to 1 along the path
let animSpeed = 5;
let animLoop = true;
let animPathCtx = null;
let animOriginalPos = null;  // mesh position before animation
let animPathLine = null;     // THREE.Line showing the 3D path

function initAnimPathCanvas() {
    const canvas = document.getElementById('anim-path-canvas');
    if (!canvas) return;
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    animPathCtx = canvas.getContext('2d');
    animPathCtx.clearRect(0, 0, canvas.width, canvas.height);
}

function drawAnimPath() {
    if (!animPathCtx || animPath.length < 2) return;
    const canvas = document.getElementById('anim-path-canvas');
    animPathCtx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw the path line
    animPathCtx.strokeStyle = '#ffdd57';
    animPathCtx.lineWidth = 3;
    animPathCtx.lineJoin = 'round';
    animPathCtx.lineCap = 'round';
    animPathCtx.setLineDash([8, 4]);
    animPathCtx.beginPath();
    animPathCtx.moveTo(animPath[0].x, animPath[0].y);
    for (let i = 1; i < animPath.length; i++) {
        animPathCtx.lineTo(animPath[i].x, animPath[i].y);
    }
    animPathCtx.stroke();
    animPathCtx.setLineDash([]);

    // Draw dots at intervals
    animPathCtx.fillStyle = '#ffdd57';
    for (let i = 0; i < animPath.length; i += Math.max(1, Math.floor(animPath.length / 20))) {
        animPathCtx.beginPath();
        animPathCtx.arc(animPath[i].x, animPath[i].y, 4, 0, Math.PI * 2);
        animPathCtx.fill();
    }

    // Draw current position indicator
    if (animIsPlaying && animPath3D.length > 1) {
        const idx = Math.min(Math.floor(animProgress * (animPath.length - 1)), animPath.length - 1);
        const pt = animPath[idx];
        animPathCtx.fillStyle = '#ff6b6b';
        animPathCtx.beginPath();
        animPathCtx.arc(pt.x, pt.y, 8, 0, Math.PI * 2);
        animPathCtx.fill();
    }
}

function convertPathTo3D() {
    animPath3D = [];
    for (const pt of animPath) {
        const ndc = new THREE.Vector2(
            (pt.x / innerWidth) * 2 - 1,
            -(pt.y / innerHeight) * 2 + 1
        );
        const pos3D = new THREE.Vector3(ndc.x * 3, ndc.y * 3, 0);
        animPath3D.push(pos3D);
    }
    // Simplify: keep every Nth point for smoother motion
    if (animPath3D.length > 100) {
        const step = Math.floor(animPath3D.length / 100);
        animPath3D = animPath3D.filter((_, i) => i % step === 0);
        animPath = animPath.filter((_, i) => i % step === 0);
    }
}

function startAnimPlayback() {
    if (animPath3D.length < 2) return;
    animIsPlaying = true;
    animProgress = 0;
    animOriginalPos = mesh.position.clone();
    setCommandStatus('🎬 Playing animation...');
}

function stopAnimPlayback() {
    animIsPlaying = false;
    if (animOriginalPos && mesh) {
        mesh.position.copy(animOriginalPos);
    }
    setCommandStatus('Animation stopped');
}

function clearAnimPath() {
    animPath = [];
    animPath3D = [];
    animIsPlaying = false;
    animProgress = 0;
    if (animOriginalPos && mesh) mesh.position.copy(animOriginalPos);
    if (animPathCtx) {
        const canvas = document.getElementById('anim-path-canvas');
        animPathCtx.clearRect(0, 0, canvas.width, canvas.height);
    }
    if (animPathLine) { scene.remove(animPathLine); animPathLine = null; }
    setCommandStatus('Path cleared');
}

function updateAnimation() {
    if (!animIsPlaying || animPath3D.length < 2 || !mesh) return;

    animProgress += (animSpeed / 1000);

    if (animProgress >= 1) {
        if (animLoop) {
            animProgress = 0;
        } else {
            animProgress = 1;
            animIsPlaying = false;
            setCommandStatus('Animation finished');
        }
    }

    // Interpolate position along path
    const totalPts = animPath3D.length;
    const exactIdx = animProgress * (totalPts - 1);
    const idx = Math.floor(exactIdx);
    const frac = exactIdx - idx;
    const nextIdx = Math.min(idx + 1, totalPts - 1);

    const p1 = animPath3D[idx];
    const p2 = animPath3D[nextIdx];
    mesh.position.set(
        p1.x + (p2.x - p1.x) * frac,
        p1.y + (p2.y - p1.y) * frac,
        p1.z + (p2.z - p1.z) * frac
    );

    drawAnimPath();
}

// Animate path canvas events
const animCanvas = document.getElementById('anim-path-canvas');
if (animCanvas) {
    animCanvas.addEventListener('mousedown', e => {
        if (currentAppMode !== 'animate') return;
        animIsDrawing = true;
        animPath = [{ x: e.clientX, y: e.clientY }];
        if (animIsPlaying) stopAnimPlayback();
    });
    animCanvas.addEventListener('mousemove', e => {
        if (!animIsDrawing || currentAppMode !== 'animate') return;
        animPath.push({ x: e.clientX, y: e.clientY });
        drawAnimPath();
    });
    animCanvas.addEventListener('mouseup', () => {
        if (!animIsDrawing) return;
        animIsDrawing = false;
        convertPathTo3D();
        setCommandStatus(`Path drawn: ${animPath3D.length} points — press Play ▶`);
    });
    // Touch support
    animCanvas.addEventListener('touchstart', e => {
        if (currentAppMode !== 'animate') return;
        e.preventDefault();
        animIsDrawing = true;
        animPath = [{ x: e.touches[0].clientX, y: e.touches[0].clientY }];
    }, { passive: false });
    animCanvas.addEventListener('touchmove', e => {
        if (!animIsDrawing || currentAppMode !== 'animate') return;
        e.preventDefault();
        animPath.push({ x: e.touches[0].clientX, y: e.touches[0].clientY });
        drawAnimPath();
    }, { passive: false });
    animCanvas.addEventListener('touchend', () => {
        animIsDrawing = false;
        convertPathTo3D();
    });
}

// Animate toolbar buttons
document.getElementById('anim-play')?.addEventListener('click', () => {
    convertPathTo3D();
    startAnimPlayback();
});
document.getElementById('anim-pause')?.addEventListener('click', stopAnimPlayback);
document.getElementById('anim-clear')?.addEventListener('click', clearAnimPath);
document.getElementById('anim-speed')?.addEventListener('input', e => {
    animSpeed = parseInt(e.target.value);
});
document.getElementById('anim-loop')?.addEventListener('change', e => {
    animLoop = e.target.checked;
});

// ============================================================
// 13. RENDER LOOP
// ============================================================
function animate() {
    requestAnimationFrame(animate);

    // Hand detection
    detectHands();

    // Apply smooth rotation momentum (runs every frame for fluid motion)
    if (mesh && (Math.abs(smoothRotVelX) > 0.0001 || Math.abs(smoothRotVelY) > 0.0001)) {
        mesh.rotation.y += smoothRotVelY;
        mesh.rotation.x += smoothRotVelX;
        // Damp velocity — keeps spinning with inertia when hand stops
        if (currentGesture !== 'rotate') {
            smoothRotVelX *= ROT_DAMPING;
            smoothRotVelY *= ROT_DAMPING;
        }
    }

    // Abstract breathing animation
    if (currentAppMode === 'abstract' && abstractAnimating) {
        abstractAnimate();
    }

    // Animate mode: move object along drawn path
    if (currentAppMode === 'animate') {
        updateAnimation();
    }

    // Carve-in: continuous push inward while mouse is held still on surface
    if (carveHoldActive && carvePoint && carveNormal && mesh) {
        extrudeAt(carvePoint, carveNormal, 0.008);
        setCommandStatus('Carving in — hold to deepen, release to stop');
    }

    renderer.render(scene, camera);
}

animate();
setCommandStatus('Ready');

// Start in gallery view — hide model elements
document.getElementById('model-actions')?.style.setProperty('display', 'none');
document.getElementById('canvas-container')?.style.setProperty('display', 'none');
document.getElementById('splice-toolbar-wrap')?.style.setProperty('display', 'none');
document.getElementById('instructions-guide')?.style.setProperty('display', 'none');
const galleryViewInit = document.getElementById('gallery-view');
if (galleryViewInit) { galleryViewInit.style.display = 'block'; }
renderGallery();

// ============================================================
// ============================================================
// 14. GALLERY — save/load projects
// ============================================================
function getGalleryProjects() {
    try { return JSON.parse(localStorage.getItem('whimsical_gallery') || '[]'); }
    catch { return []; }
}

function saveGalleryProjects(projects) {
    localStorage.setItem('whimsical_gallery', JSON.stringify(projects));
}

function buildProjectData(name) {
    renderer.render(scene, camera);
    const thumbnail = renderer.domElement.toDataURL('image/jpeg', 0.6);
    const pos = mesh.geometry.attributes.position;
    const mainMeshData = pos.count > 0 ? Array.from(pos.array) : [];
    const objectsData = sceneObjects.map(entry => ({
        name: entry.name,
        position: entry.mesh.position.toArray(),
        scale: entry.mesh.scale.toArray(),
        rotation: [entry.mesh.rotation.x, entry.mesh.rotation.y, entry.mesh.rotation.z],
        color: entry.mesh.material.color.getHex(),
    }));
    return {
        id: Date.now(),
        name,
        date: new Date().toLocaleDateString(),
        thumbnail,
        shape: currentShape,
        positions: mainMeshData,
        color: material.color.getHex(),
        scale: mesh.scale.toArray(),
        position: mesh.position.toArray(),
        rotation: [mesh.rotation.x, mesh.rotation.y, mesh.rotation.z],
        sceneObjects: objectsData,
        type: '3d',
    };
}

// Save — overwrites current project, or creates new if none loaded
function saveProject() {
    if (currentProjectId) {
        const projects = getGalleryProjects();
        const idx = projects.findIndex(p => p.id === currentProjectId);
        if (idx !== -1) {
            const project = buildProjectData(projects[idx].name);
            project.id = currentProjectId;
            projects[idx] = project;
            try {
                saveGalleryProjects(projects);
                setCommandStatus(`Saved "${project.name}"!`);
            } catch (e) {
                project.thumbnail = '';
                saveGalleryProjects(projects);
                setCommandStatus(`Saved "${project.name}" (no thumbnail)`);
            }
            return;
        }
    }
    // No current project — fall through to Save As
    saveAsProject();
}

// Save As — always creates a new project
function saveAsProject() {
    const name = prompt('Project name:', `Project ${getGalleryProjects().length + 1}`);
    if (!name) return;

    const project = buildProjectData(name);
    const projects = getGalleryProjects();
    projects.unshift(project);

    try {
        saveGalleryProjects(projects);
        currentProjectId = project.id;
        setCommandStatus(`Saved "${name}" to gallery!`);
    } catch (e) {
        project.thumbnail = '';
        try {
            saveGalleryProjects(projects);
            currentProjectId = project.id;
            setCommandStatus(`Saved "${name}" (no thumbnail — storage full)`);
        } catch (e2) {
            setCommandStatus('Save failed — storage full. Delete old projects.');
            projects.shift();
        }
    }
}

// Keep old function for gallery button compatibility
function saveCurrentProject() { saveAsProject(); }

// Sketch save — captures canvas as image
function saveSketchProject() {
    if (currentProjectId) {
        const projects = getGalleryProjects();
        const idx = projects.findIndex(p => p.id === currentProjectId);
        if (idx !== -1) {
            const canvas = document.getElementById('sketch-canvas');
            const thumbnail = canvas ? canvas.toDataURL('image/jpeg', 0.6) : '';
            projects[idx].thumbnail = thumbnail;
            projects[idx].sketchData = thumbnail;
            projects[idx].date = new Date().toLocaleDateString();
            saveGalleryProjects(projects);
            setCommandStatus(`Saved "${projects[idx].name}"!`);
            return;
        }
    }
    saveSketchAsProject();
}

function saveSketchAsProject() {
    const name = prompt('Sketch name:', `Sketch ${getGalleryProjects().length + 1}`);
    if (!name) return;
    const canvas = document.getElementById('sketch-canvas');
    const thumbnail = canvas ? canvas.toDataURL('image/jpeg', 0.6) : '';
    const project = {
        id: Date.now(),
        name,
        date: new Date().toLocaleDateString(),
        thumbnail,
        type: 'sketch',
        sketchData: thumbnail,
    };
    const projects = getGalleryProjects();
    projects.unshift(project);
    try {
        saveGalleryProjects(projects);
        currentProjectId = project.id;
        setCommandStatus(`Saved sketch "${name}" to gallery!`);
    } catch (e) {
        project.thumbnail = '';
        project.sketchData = '';
        try {
            saveGalleryProjects(projects);
            currentProjectId = project.id;
            setCommandStatus(`Saved "${name}" (no image — storage full)`);
        } catch (e2) {
            setCommandStatus('Save failed — storage full.');
            projects.shift();
        }
    }
}

document.getElementById('sketch-save-btn')?.addEventListener('click', saveSketchProject);
document.getElementById('sketch-save-as-btn')?.addEventListener('click', saveSketchAsProject);

function loadProject(project) {
    // Clear existing scene objects
    for (const entry of [...sceneObjects]) {
        removeSceneObject(entry);
    }
    deselectObject();

    // Load main mesh
    if (project.positions && project.positions.length > 0) {
        const g = new THREE.BufferGeometry();
        g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(project.positions), 3));
        g.computeVertexNormals();
        if (mesh) { scene.remove(mesh); mesh.geometry.dispose(); }
        mesh = new THREE.Mesh(g, material);
        refreshWireframe();
        scene.add(mesh);
    } else {
        // Empty project — reset to empty mesh
        if (mesh) { scene.remove(mesh); mesh.geometry.dispose(); }
        const emptyGeo = new THREE.BufferGeometry();
        emptyGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(0), 3));
        mesh = new THREE.Mesh(emptyGeo, material);
        scene.add(mesh);
    }

    material.color.setHex(project.color || 0xa8c3e6);
    material.vertexColors = false;
    material.needsUpdate = true;
    if (project.scale) mesh.scale.fromArray(project.scale);
    if (project.position) mesh.position.fromArray(project.position);
    if (project.rotation) mesh.rotation.set(project.rotation[0], project.rotation[1], project.rotation[2]);
    currentShape = project.shape || 'sphere';

    // Restore scene objects
    if (project.sceneObjects && project.sceneObjects.length > 0) {
        for (const objData of project.sceneObjects) {
            // Create object directly as child of mesh with local position
            const geo = SHAPES[objData.name]().toNonIndexed();
            geo.computeVertexNormals();
            const mat = material.clone();
            if (objData.color) mat.color.setHex(objData.color);
            const obj = new THREE.Mesh(geo, mat);
            obj.position.fromArray(objData.position);
            obj.scale.fromArray(objData.scale);
            if (objData.rotation) obj.rotation.set(objData.rotation[0], objData.rotation[1], objData.rotation[2]);
            mesh.add(obj);
            const wire = new THREE.LineSegments(
                new THREE.EdgesGeometry(geo, 15),
                new THREE.LineBasicMaterial({ color: 0x222222, transparent: true, opacity: 0.12 })
            );
            obj.add(wire);
            sceneObjects.push({ mesh: obj, name: objData.name, wireframe: wire });
        }
    }

    currentProjectId = project.id;
    applyAppMode('model');
    setCommandStatus(`Loaded "${project.name}"`);
}

function deleteProject(id) {
    const projects = getGalleryProjects().filter(p => p.id !== id);
    saveGalleryProjects(projects);
    renderGallery();
}

function renderGallery() {
    const grid = document.getElementById('gallery-grid');
    const empty = document.getElementById('gallery-empty');
    if (!grid) return;

    const projects = getGalleryProjects();

    if (projects.length === 0) {
        grid.innerHTML = '';
        if (empty) empty.style.display = 'block';
        return;
    }
    if (empty) empty.style.display = 'none';

    grid.innerHTML = projects.map(p => `
        <div class="gallery-card" data-id="${p.id}">
            <div class="gallery-card-thumb">
                ${p.thumbnail ? `<img src="${p.thumbnail}" alt="${p.name}">` : '<span class="placeholder">🧊</span>'}
            </div>
            <div class="gallery-card-info">
                <div class="gallery-card-name">${p.name}</div>
                <div class="gallery-card-date">${p.date}</div>
                <div class="gallery-card-actions">
                    <button data-action="load" data-id="${p.id}">📂 Open</button>
                    <button data-action="delete" data-id="${p.id}">🗑 Delete</button>
                </div>
            </div>
        </div>
    `).join('');

    // Wire up actions
    grid.querySelectorAll('[data-action="load"]').forEach(btn => {
        btn.addEventListener('click', e => {
            e.stopPropagation();
            const proj = projects.find(p => p.id === parseInt(btn.dataset.id));
            if (proj) loadProject(proj);
        });
    });
    grid.querySelectorAll('[data-action="delete"]').forEach(btn => {
        btn.addEventListener('click', e => {
            e.stopPropagation();
            if (confirm('Delete this project?')) deleteProject(parseInt(btn.dataset.id));
        });
    });
    grid.querySelectorAll('.gallery-card').forEach(card => {
        card.addEventListener('click', () => {
            const proj = projects.find(p => p.id === parseInt(card.dataset.id));
            if (proj) loadProject(proj);
        });
    });
}

document.getElementById('gallery-save-btn')?.addEventListener('click', saveCurrentProject);

// ============================================================
// 15. BOBA — AI Chat Agent (Gemini-powered)
// ============================================================
const catDialogue = document.getElementById('cat-dialogue');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const chatHistory = document.getElementById('chat-history');
const chatHistoryContainer = document.getElementById('chat-history-container');
const toggleHistoryBtn = document.getElementById('toggle-history-btn');

// Wink animation
let winkInterval = setInterval(() => {
    const openEye = document.getElementById('right-eye-open');
    const winkEye = document.getElementById('right-eye-wink');
    if (openEye && winkEye) {
        openEye.style.display = 'none';
        winkEye.style.display = 'block';
        setTimeout(() => { openEye.style.display = ''; winkEye.style.display = 'none'; }, 200);
    }
}, 4000 + Math.random() * 3000);

function bobaReply(text) {
    if (catDialogue) catDialogue.textContent = text;
}

function addToHistory(who, text) {
    if (!chatHistory) return;
    const li = document.createElement('li');
    li.className = who === 'user' ? 'history-user' : 'history-cat';
    li.textContent = who === 'user' ? `You: ${text}` : `🐾 ${text}`;
    chatHistory.appendChild(li);
    if (chatHistoryContainer) chatHistoryContainer.scrollTop = chatHistoryContainer.scrollHeight;
}

// --- AI Integration (OpenAI + Gemini) ---
let aiApiKey = null;
let aiProvider = null; // 'openai' or 'gemini'
let aiChatHistory = [];

const GEMINI_MODELS = [
    'gemini-1.5-flash-8b',
    'gemini-1.5-flash',
    'gemini-2.0-flash',
];

const BOBA_SYSTEM_PROMPT = `You are Boba, a cute cat AI assistant inside a 3D modeling studio called Whimsical Studio. You help users sculpt, model, and design 3D objects using hand gestures and mouse.

You have the personality of a friendly, playful cat. Use occasional cat puns and emojis (🐾 ✨ 🎨) but keep responses concise (1-3 sentences max).

You can execute studio commands by including a JSON action block in your response. Format: [[ACTION:{"cmd":"command_name","args":{}}]]

Available commands:
- {"cmd":"color","args":{"hex":"#ff4444"}} — Change object color (any hex)
- {"cmd":"shape","args":{"name":"sphere"}} — Switch shape (sphere, cube, cylinder, cone, torus, pyramid, icosahedron, dodecahedron, torusknot, helix)
- {"cmd":"texture","args":{"name":"clay"}} — Apply texture (matte, glossy, metallic, wireframe, clay, glass)
- {"cmd":"export_stl"} — Export model as STL file
- {"cmd":"undo"} — Undo last action
- {"cmd":"reset"} — Reset to default shape
- {"cmd":"wireframe_toggle"} — Toggle wireframe view
- {"cmd":"mode","args":{"name":"model"}} — Switch app mode (model, abstract, sketch)
- {"cmd":"add_shape","args":{"name":"sphere"}} — Enter add-shape mode with specified shape
- {"cmd":"scale","args":{"factor":1.5}} — Scale object by factor
- {"cmd":"rotate","args":{"x":0,"y":45,"z":0}} — Rotate object (degrees)
- {"cmd":"merge"} — Merge/bridge gaps between shapes so they connect seamlessly
- {"cmd":"fix_edges","args":{"passes":3}} — Smooth and fix all edges for clean curves
- {"cmd":"pattern","args":{"shape":"sphere","count":8,"scale":0.2,"mode":"all"}} — Pattern wrap: duplicate shape around object (mode: ring/vertical/all)

Context about the current workspace:
- The app has 3 modes: 3D Model, Abstract, and Sketch
- Hand gestures: open palm=rotate, pinch=move/extrude, cupped=mold, fist=freeze
- Activation: hold open palm steady ~1s to activate gesture control

If the user asks something unrelated to 3D modeling, still answer helpfully but stay in character as Boba the cat.
If you want to execute a command, include the action block AND a friendly text response.`;

async function initAI() {
    let apiKey = sessionStorage.getItem('boba_api_key');
    if (apiKey && !apiKey.trim()) { sessionStorage.removeItem('boba_api_key'); apiKey = null; }
    if (!apiKey) {
        apiKey = prompt('🐾 Boba needs an AI key!\n\nSupported:\n• OpenAI: sk-... (platform.openai.com/api-keys)\n• Gemini: AIza... (aistudio.google.com/apikey)\n\nPaste key (or Cancel for offline):');
        if (!apiKey || !apiKey.trim()) { bobaReply("Offline mode~ Type 'connect' to add key! 🐾"); return false; }
        apiKey = apiKey.trim();
    }
    const provider = apiKey.startsWith('sk-') ? 'openai' : apiKey.startsWith('AIza') ? 'gemini' : null;
    if (!provider) { bobaReply("Key not recognized~ Start with 'sk-' (OpenAI) or 'AIza' (Gemini) 🐾"); return false; }

    // Don't waste API calls on validation — just save and test on first real message
    aiApiKey = apiKey;
    aiProvider = provider;
    aiChatHistory = [];
    sessionStorage.setItem('boba_api_key', apiKey);
    bobaReply(`Boba is ready on ${provider === 'openai' ? 'OpenAI' : 'Gemini'}! Ask me anything~ ✨🐾`);
    return true;
}

async function callAI(userMessage) {
    const ctx = `[Current: shape=${currentShape}, mode=${currentAppMode}, color=#${material.color.getHexString()}]`;
    if (aiProvider === 'openai') {
        aiChatHistory.push({ role: 'user', content: ctx + '\n' + userMessage });
        const r = await fetch('https://api.openai.com/v1/chat/completions', {
            method: 'POST', headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${aiApiKey}` },
            body: JSON.stringify({ model: 'gpt-4o-mini', messages: [{ role: 'system', content: BOBA_SYSTEM_PROMPT }, ...aiChatHistory], max_tokens: 300 }),
        });
        if (!r.ok) { const e = await r.json().catch(() => ({})); aiChatHistory.pop(); throw new Error(e.error?.message || `HTTP ${r.status}`); }
        const d = await r.json(); const text = d.choices?.[0]?.message?.content || '';
        aiChatHistory.push({ role: 'assistant', content: text });
        if (aiChatHistory.length > 40) aiChatHistory = aiChatHistory.slice(-40);
        return text;
    } else {
        aiChatHistory.push({ role: 'user', parts: [{ text: ctx + '\n' + userMessage }] });
        for (const m of GEMINI_MODELS) {
            const r = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${m}:generateContent?key=${aiApiKey}`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ system_instruction: { parts: [{ text: BOBA_SYSTEM_PROMPT }] }, contents: aiChatHistory }),
            });
            if (r.ok) { const d = await r.json(); const text = d.candidates?.[0]?.content?.parts?.[0]?.text || '';
                aiChatHistory.push({ role: 'model', parts: [{ text }] });
                if (aiChatHistory.length > 40) aiChatHistory = aiChatHistory.slice(-40); return text; }
            // 404 = model not found, 429 = rate limit — skip to next model
            if (r.status === 404 || r.status === 429) continue;
            const e = await r.json().catch(() => ({})); const msg = e.error?.message || '';
            if (msg.includes('quota') || msg.includes('rate') || msg.includes('not found')) continue;
            aiChatHistory.pop(); throw new Error(msg || `HTTP ${r.status}`);
        }
        aiChatHistory.pop(); throw new Error('Gemini rate limit — wait ~30s and try again');
    }
}

// Execute a studio command from Gemini's response
function executeBobaAction(action) {
    try {
        const { cmd, args } = action;
        switch (cmd) {
            case 'color':
                material.color.set(args.hex);
                material.needsUpdate = true;
                break;
            case 'shape':
                if (SHAPES[args.name]) { saveUndo(); setShape(args.name); }
                break;
            case 'texture':
                applyTexture(args.name);
                break;
            case 'export_stl':
                exportSTL();
                break;
            case 'undo':
                undo();
                break;
            case 'reset':
                saveUndo(); setShape(currentShape);
                break;
            case 'wireframe_toggle':
                material.wireframe = !material.wireframe;
                material.needsUpdate = true;
                break;
            case 'mode':
                applyAppMode(args.name);
                break;
            case 'add_shape':
                if (SHAPES[args.name]) enterAddShapeMode(args.name);
                break;
            case 'scale':
                saveUndo();
                mesh.scale.multiplyScalar(args.factor || 1);
                break;
            case 'rotate':
                saveUndo();
                if (args.x) mesh.rotation.x += (args.x * Math.PI) / 180;
                if (args.y) mesh.rotation.y += (args.y * Math.PI) / 180;
                if (args.z) mesh.rotation.z += (args.z * Math.PI) / 180;
                break;
            case 'merge':
                mergeShapes();
                break;
            case 'smooth':
            case 'fix_edges':
                smoothAllEdges(args?.passes || 3);
                break;
            case 'pattern':
                patternWrap(args.shape || 'sphere', args.scale || 0.2, args.count || 8, args.mode || 'all');
                break;
            default:
                console.warn('Unknown Boba action:', cmd);
        }
    } catch (e) {
        console.error('Boba action failed:', e);
    }
}

// Parse action blocks from Gemini response
function parseBobaResponse(text) {
    const actionRegex = /\[\[ACTION:(.*?)\]\]/g;
    const actions = [];
    let match;
    while ((match = actionRegex.exec(text)) !== null) {
        try { actions.push(JSON.parse(match[1])); } catch (e) { /* skip bad JSON */ }
    }
    const cleanText = text.replace(/\[\[ACTION:.*?\]\]/g, '').trim();
    return { text: cleanText, actions };
}

// Fallback offline command parser (same as old Boba)
function processOfflineCommand(input) {
    const msg = input.toLowerCase().trim();

    // Color — names and hex codes
    const colorMatch = msg.match(/\b(red|blue|green|yellow|orange|purple|pink|white|black|gold|silver|cyan|teal|lime|coral|navy|maroon|olive|salmon|turquoise|violet|indigo|crimson|magenta)\b/);
    const COLORS = { red:'#ff4444', blue:'#4488ff', green:'#44cc66', yellow:'#ffdd44', orange:'#ff8833', purple:'#9944ff', pink:'#ff66aa', white:'#ffffff', black:'#222222', gold:'#ffd700', silver:'#c0c0c0', cyan:'#00cccc', teal:'#008888', lime:'#32cd32', coral:'#ff7f50', navy:'#000080', maroon:'#800000', olive:'#808000', salmon:'#fa8072', turquoise:'#40e0d0', violet:'#ee82ee', indigo:'#4b0082', crimson:'#dc143c', magenta:'#ff00ff' };
    if (colorMatch && COLORS[colorMatch[1]]) {
        material.color.set(COLORS[colorMatch[1]]); material.needsUpdate = true;
        return `Changed color to ${colorMatch[1]}! 🎨`;
    }
    const hexMatch = msg.match(/#([0-9a-f]{6}|[0-9a-f]{3})\b/i);
    if (hexMatch) { material.color.set('#' + hexMatch[1]); material.needsUpdate = true; return `Set color to #${hexMatch[1]}! 🎨`; }

    // Shape — switch or add
    const shapeMatch = msg.match(/\b(sphere|cube|cylinder|cone|torus|pyramid|icosahedron|dodecahedron|helix|torusknot)\b/);
    if (shapeMatch && SHAPES[shapeMatch[1]]) {
        if (msg.includes('add') || msg.includes('attach') || msg.includes('place')) {
            enterAddShapeMode(shapeMatch[1]);
            return `Add shape mode: ${SHAPE_LABELS[shapeMatch[1]]}! Click to place 🐾`;
        }
        saveUndo(); setShape(shapeMatch[1]);
        return `Switched to ${SHAPE_LABELS[shapeMatch[1]]}! ✨`;
    }

    // Texture
    const texMatch = msg.match(/\b(matte|glossy|metallic|wireframe|clay|glass)\b/);
    if (texMatch) { applyTexture(texMatch[1]); return `Applied ${texMatch[1]} texture! 🐾`; }

    // Scale
    const scaleNum = msg.match(/scale\s*(?:to|by)?\s*(\d+\.?\d*)/);
    if (scaleNum) { saveUndo(); mesh.scale.multiplyScalar(parseFloat(scaleNum[1])); return `Scaled by ${scaleNum[1]}x! 🐾`; }
    if (msg.match(/\b(bigger|larger|scale up|grow|enlarge)\b/)) { saveUndo(); mesh.scale.multiplyScalar(1.3); return 'Made it bigger! 🐾'; }
    if (msg.match(/\b(smaller|shrink|scale down|tiny|reduce)\b/)) { saveUndo(); mesh.scale.multiplyScalar(0.7); return 'Made it smaller! 🐾'; }

    // Rotate
    const rotNum = msg.match(/rotate\s*(?:by)?\s*(\d+)/);
    if (rotNum) { saveUndo(); mesh.rotation.y += (parseInt(rotNum[1]) * Math.PI) / 180; return `Rotated ${rotNum[1]}°! 🐾`; }
    if (msg.match(/\b(rotate|spin|turn)\b/)) { saveUndo(); mesh.rotation.y += Math.PI / 4; return 'Rotated 45°! 🐾'; }

    // Modes
    if (msg.match(/\b(abstract|blob)\b/)) { applyAppMode('abstract'); return 'Abstract mode! 🫧'; }
    if (msg.match(/\b(sketch|draw)\b/)) { applyAppMode('sketch'); return 'Sketch mode! ✏️'; }
    if (msg.match(/\bmodel\b/) && !msg.includes('remodel')) { applyAppMode('model'); return '3D Model mode! 🧊'; }

    // Actions
    if (msg.match(/\b(export|stl|save)\b/)) { exportSTL(); return 'Exported STL! 💾'; }
    if (msg.includes('undo')) { undo(); return 'Undone! ↩'; }
    if (msg.match(/\b(reset|start over|clear)\b/)) { saveUndo(); setShape(currentShape); return 'Reset! ✧'; }
    if (msg.match(/\b(wireframe|wire)\b/)) { material.wireframe = !material.wireframe; material.needsUpdate = true; return `Wireframe ${material.wireframe ? 'on' : 'off'}! 🐾`; }
    if (msg.match(/\b(mirror|flip)\b/)) { mirrorMesh('x'); return 'Mirrored! 🪞'; }

    // Fix edges / smooth all
    if (msg.match(/\b(fix edge|smooth edge|clean ?up|fix it|smooth all|smooth out|soften|round edge|curve edge|polish)\b/i)) {
        smoothAllEdges(3);
        return 'Edges smoothed and cleaned up! Purrfect curves~ ✨🐾';
    }

    // Merge shapes
    if (msg.match(/\b(merge|connect|bridge|join|stitch|fill gap|no gap)\b/)) {
        mergeShapes();
        return 'Merged shapes together! No more gaps~ 🔗';
    }

    // Pattern wrap
    const patternMatch = msg.match(/pattern\s+(ring|vertical|all)\s+(\d+)\s*(sphere|cube|star|heart|berry|cone|torus)?/);
    if (patternMatch || msg.includes('pattern') || msg.includes('wrap')) {
        const mode = patternMatch?.[1] || 'all';
        const count = parseInt(patternMatch?.[2]) || 8;
        const shape = patternMatch?.[3] || 'sphere';
        patternWrap(shape, 0.2, count, mode);
        return `Pattern wrap: ${count}× ${shape} (${mode})! ⊛`;
    }

    // Help
    if (msg.match(/\b(help|commands?|what can)\b/)) {
        return '🐾 Try: colors (red, #ff0000), shapes (cube, sphere), textures (glossy, glass), bigger/smaller, rotate, mirror, wireframe, export, add sphere, abstract mode, sketch mode, undo, reset. "connect" for AI mode!';
    }

    // Greetings
    if (msg.match(/\b(hi|hello|hey|meow|sup)\b/)) { return 'Meow~ Tell me what to sculpt! 🐾'; }
    if (msg.match(/\b(thanks|thank you|thx)\b/)) { return "Paw-some! Happy to help~ 🐾"; }

    return null;
}

async function processBobaCommand(input) {
    addToHistory('user', input);
    const msg = input.toLowerCase().trim();

    // Special commands: connect/reconnect AI
    if (msg === 'connect' || msg === 'api key' || msg === 'reconnect') {
        sessionStorage.removeItem('boba_api_key');
        aiApiKey = null;
        aiProvider = null;
        aiChatHistory = [];
        bobaReply('Reconnecting... 🐾');
        await initAI();
        return;
    }

    if (aiApiKey) {
        // Try offline command first — instant, no API call needed for simple stuff
        const offlineResult = processOfflineCommand(input);
        if (offlineResult) {
            bobaReply(offlineResult);
            addToHistory('cat', offlineResult);
            return;
        }
        // Complex request — use AI
        bobaReply('Thinking... 🐾');
        try {
            const responseText = await callAI(input);
            const { text, actions } = parseBobaResponse(responseText);
            for (const action of actions) executeBobaAction(action);
            const reply = text || 'Done! 🐾';
            bobaReply(reply);
            addToHistory('cat', reply);
        } catch (e) {
            console.error('AI error:', e);
            const reply = e.message.includes('rate') || e.message.includes('quota')
                ? 'Rate limited~ Wait 30s and try again, or use offline commands (type "help")! 🐾'
                : 'Meow, glitch~ ' + e.message + ' 🐾';
            bobaReply(reply);
            addToHistory('cat', reply);
        }
    } else {
        const reply = processOfflineCommand(input) || "Offline mode~ Type 'help' for commands or 'connect' for AI! 🐱";
        bobaReply(reply);
        addToHistory('cat', reply);
    }
}

// Initialize AI on load
initAI();

if (sendBtn && chatInput) {
    sendBtn.addEventListener('click', () => {
        if (chatInput.value.trim()) {
            processBobaCommand(chatInput.value);
            chatInput.value = '';
        }
    });
    chatInput.addEventListener('keydown', e => {
        if (e.key === 'Enter' && chatInput.value.trim()) {
            processBobaCommand(chatInput.value);
            chatInput.value = '';
        }
    });
}

if (toggleHistoryBtn && chatHistoryContainer) {
    toggleHistoryBtn.addEventListener('click', () => {
        chatHistoryContainer.style.display = chatHistoryContainer.style.display === 'none' ? 'block' : 'none';
    });
}
