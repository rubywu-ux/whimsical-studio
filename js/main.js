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
const SHAPES = {
    sphere:   () => new THREE.SphereGeometry(1, 32, 24),
    cube:     () => new THREE.BoxGeometry(1.5, 1.5, 1.5, 10, 10, 10),
    cylinder: () => new THREE.CylinderGeometry(0.8, 0.8, 2, 32, 10),
    cone:     () => new THREE.ConeGeometry(1, 2, 32, 10),
    torus:    () => new THREE.TorusGeometry(0.8, 0.35, 20, 40),
};

let currentShape = 'sphere';
let mesh = null, wireframe = null;
const material = new THREE.MeshStandardMaterial({ color: 0xa8c3e6, roughness: 0.35, metalness: 0.1, flatShading: false, side: THREE.DoubleSide });

function setShape(name) {
    if (mesh) { scene.remove(mesh); mesh.geometry.dispose(); }
    const geo = SHAPES[name]().toNonIndexed();
    geo.computeVertexNormals();
    mesh = new THREE.Mesh(geo, material);
    refreshWireframe();
    scene.add(mesh);
    currentShape = name;
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
    ng.translate(ox, oy, oz); ng.scale(scale, scale, scale);
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
    refreshWireframe(); ng.dispose(); ni.dispose();
}

setShape('sphere');

// Shape Palette UI
const palette = document.createElement('div');
palette.id = 'shape-palette';
palette.style.cssText = 'position:absolute;top:80px;left:20px;z-index:15;display:flex;flex-direction:column;gap:5px;';
palette.innerHTML = '<div style="font-weight:600;font-size:12px;color:#2b332b;padding-left:4px;">📐 Shapes</div>';
const emojis = { sphere:'🔴', cube:'🟦', cylinder:'🫙', cone:'🔺', torus:'🍩' };
for (const name of Object.keys(SHAPES)) {
    const row = document.createElement('div');
    row.style.cssText = 'display:flex;gap:3px;';
    const btn = document.createElement('button');
    btn.textContent = `${emojis[name]} ${name[0].toUpperCase()+name.slice(1)}`;
    btn.className = 'pal-btn';
    btn.style.cssText = 'flex:1;background:rgba(248,238,219,0.95);border:1.5px solid #7b8b6f;border-radius:8px;padding:5px 8px;cursor:pointer;font-size:12px;font-family:inherit;color:#2b332b;text-align:left;';
    btn.addEventListener('click', () => { setShape(name); document.querySelectorAll('.pal-btn').forEach(b=>b.style.borderWidth='1.5px'); btn.style.borderWidth='3px'; });
    if (name==='sphere') btn.style.borderWidth='3px';
    const add = document.createElement('button');
    add.textContent = '➕'; add.title = `Add ${name}`;
    add.style.cssText = 'background:rgba(248,238,219,0.95);border:1.5px solid #7b8b6f;border-radius:8px;padding:5px 7px;cursor:pointer;font-size:13px;';
    add.addEventListener('click', () => startPlacement(name));
    row.appendChild(btn); row.appendChild(add); palette.appendChild(row);
}
document.body.appendChild(palette);

// Placement mode — click to set position, drag to scale, release to commit
let placementShape = null;
let placementPreview = null;  // THREE.Mesh preview
let placementPos = null;      // local position {x,y,z}
let placementNormal = null;   // local normal
let placementStartY = 0;      // mouseY at click start
let placementDragging = false;

const placementLabel = document.createElement('div');
placementLabel.style.cssText = 'display:none;position:absolute;bottom:180px;left:50%;transform:translateX(-50%);z-index:20;background:rgba(248,238,219,0.95);border:1.5px solid #7b8b6f;border-radius:8px;padding:8px 16px;font-size:13px;font-family:inherit;color:#2b332b;pointer-events:none;';
document.body.appendChild(placementLabel);

const previewMat = new THREE.MeshStandardMaterial({ color: 0xffdd57, transparent: true, opacity: 0.4, side: THREE.DoubleSide });

function startPlacement(n) {
    placementShape = n;
    document.body.style.cursor = 'crosshair';
    placementLabel.textContent = `Click & drag on mesh to place ${n} — drag to resize (Esc cancel)`;
    placementLabel.style.display = 'block';
}

function cancelPlacement() {
    placementShape = null;
    placementDragging = false;
    placementPos = null;
    if (placementPreview) { scene.remove(placementPreview); placementPreview.geometry.dispose(); placementPreview = null; }
    document.body.style.cursor = '';
    placementLabel.style.display = 'none';
}

function updatePreviewScale(mouseY) {
    if (!placementPreview) return;
    const delta = (placementStartY - mouseY) * 0.005;
    const s = Math.max(0.1, Math.min(2.0, 0.3 + delta));
    placementPreview.scale.setScalar(s);
}

function commitPlacement() {
    if (!placementPreview || !placementPos) { cancelPlacement(); return; }
    saveUndo();
    const s = placementPreview.scale.x;
    addShapeToMesh(placementShape, placementPos.x, placementPos.y, placementPos.z, s);
    cancelPlacement();
}

window.addEventListener('keydown', e => {
    if (e.key === 'Escape') cancelPlacement();
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
toolPanel.style.cssText = 'position:absolute;top:320px;left:20px;z-index:15;background:rgba(248,238,219,0.95);border:1.5px solid #7b8b6f;border-radius:10px;padding:10px 12px;font-size:12px;font-family:inherit;color:#2b332b;width:155px;';
toolPanel.innerHTML = `
    <div style="font-weight:600;margin-bottom:6px;">🛠 Sculpt Tools</div>
    <div id="tool-btns" style="display:flex;flex-wrap:wrap;gap:3px;margin-bottom:8px;"></div>
    <div style="font-weight:600;margin-bottom:4px;">🖌 Size</div>
    <input type="range" id="br" min="0.1" max="1.5" step="0.05" value="0.4" style="width:100%;">
    <div style="display:flex;justify-content:space-between;font-size:10px;"><span>Small</span><span id="brv">0.40</span><span>Big</span></div>
    <div style="font-weight:600;margin:6px 0 4px;">💪 Strength</div>
    <input type="range" id="sr" min="0.01" max="0.12" step="0.005" value="0.04" style="width:100%;">
    <div style="display:flex;justify-content:space-between;font-size:10px;"><span>Soft</span><span id="srv">0.040</span><span>Hard</span></div>
    <div style="margin-top:8px;display:flex;gap:4px;flex-wrap:wrap;">
        <button id="sym-btn" style="flex:1;padding:4px;border:1.5px solid #7b8b6f;border-radius:6px;background:#fff;cursor:pointer;font-size:11px;">🪞 Sym</button>
        <button id="undo-btn" style="flex:1;padding:4px;border:1.5px solid #7b8b6f;border-radius:6px;background:#fff;cursor:pointer;font-size:11px;">↩ Undo</button>
        <button id="export-btn" style="flex:1;padding:4px;border:1.5px solid #7b8b6f;border-radius:6px;background:#fff;cursor:pointer;font-size:11px;">💾 STL</button>
    </div>
`;
document.body.appendChild(toolPanel);

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

const tbc = document.getElementById('tool-btns');
for (const t of TOOLS) {
    const b = document.createElement('button');
    b.textContent = t.em; b.title = t.tip; b.dataset.tool = t.id;
    b.style.cssText = 'padding:5px 8px;border:1.5px solid #7b8b6f;border-radius:6px;background:#fff;cursor:pointer;font-size:15px;transition:all 0.1s;';
    b.addEventListener('click', () => { activeTool=t.id; tbc.querySelectorAll('button').forEach(x=>x.style.background='#fff'); b.style.background='#7b8b6f'; });
    if (t.id==='pull') b.style.background='#7b8b6f';
    tbc.appendChild(b);
}

document.getElementById('br').addEventListener('input', e => { brushRadius=parseFloat(e.target.value); document.getElementById('brv').textContent=brushRadius.toFixed(2); });
document.getElementById('sr').addEventListener('input', e => { sculptStrength=parseFloat(e.target.value); document.getElementById('srv').textContent=sculptStrength.toFixed(3); });

const symBtn = document.getElementById('sym-btn');
symBtn.addEventListener('click', () => { symmetryX=!symmetryX; symBtn.style.background=symmetryX?'#7b8b6f':'#fff'; symBtn.style.color=symmetryX?'#fff':'#2b332b'; });

// Undo
const undoStack = [];
function saveUndo() { const p=mesh.geometry.attributes.position; undoStack.push(new Float32Array(p.array)); if(undoStack.length>20)undoStack.shift(); }
function undo() { if(!undoStack.length)return; mesh.geometry.attributes.position.array.set(undoStack.pop()); mesh.geometry.attributes.position.needsUpdate=true; mesh.geometry.computeVertexNormals(); }
document.getElementById('undo-btn').addEventListener('click', undo);
window.addEventListener('keydown', e => { if((e.ctrlKey||e.metaKey)&&e.key==='z'){e.preventDefault();undo();} });

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

// Sculpt functions — reusable temps
const _v = new THREE.Vector3(), _lp = new THREE.Vector3(), _ln = new THREE.Vector3(), _n = new THREE.Vector3();

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
    switch(activeTool) {
        case 'pull': sculptAt(hit.point, hit.face.normal, sculptStrength); break;
        case 'push': sculptAt(hit.point, hit.face.normal, -sculptStrength); break;
        case 'smooth': smoothAt(hit.point); break;
        case 'flatten': flattenAt(hit.point, hit.face.normal); break;
        case 'inflate': inflateAt(hit.point, sculptStrength); break;
        case 'grab': grabAt(hit.point, dx||0, dy||0); break;
        case 'crease': creaseAt(hit.point, hit.face.normal, sculptStrength); break;
        case 'pinch': pinchAt(hit.point); break;
    }
    if (symmetryX) {
        const inv=mesh.matrixWorld.clone().invert();
        const lp=hit.point.clone().applyMatrix4(inv); lp.x=-lp.x;
        const mp=lp.applyMatrix4(mesh.matrixWorld);
        const mn=hit.face.normal.clone(); mn.x=-mn.x;
        raycaster.set(mp.clone().addScaledVector(mn,0.5), mn.clone().negate());
        const mh=raycaster.intersectObject(mesh);
        if(mh.length) {
            const m=mh[0];
            switch(activeTool) {
                case 'pull': sculptAt(m.point,m.face.normal,sculptStrength); break;
                case 'push': sculptAt(m.point,m.face.normal,-sculptStrength); break;
                case 'smooth': smoothAt(m.point); break;
                case 'flatten': flattenAt(m.point,m.face.normal); break;
                case 'inflate': inflateAt(m.point,sculptStrength); break;
                case 'grab': grabAt(m.point,dx?-dx:0,dy||0); break;
                case 'crease': creaseAt(m.point,m.face.normal,sculptStrength); break;
                case 'pinch': pinchAt(m.point); break;
            }
        }
    }
}

// ============================================================
// 4. MEDIAPIPE
// ============================================================
const video = document.getElementById('webcam');
let handLandmarker, lastVideoTime = -1;
const debugOverlay=document.getElementById('debug-overlay'), debugCanvas=document.getElementById('debug-canvas');
const debugCtx=debugCanvas.getContext('2d'), debugText=document.getElementById('debug-gesture-text');
const drawUtils=new DrawingUtils(debugCtx), debugBtn=document.getElementById('debug-toggle-btn');
let isDebug=false;
debugBtn.addEventListener('click',()=>{isDebug=!isDebug;debugBtn.textContent=isDebug?'🛠 Debug On':'🛠 Debug Off';debugBtn.classList.toggle('active',isDebug);debugOverlay.style.display=isDebug?'block':'none';if(isDebug){debugCanvas.width=innerWidth;debugCanvas.height=innerHeight;}else debugCtx.clearRect(0,0,debugCanvas.width,debugCanvas.height);});

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

    // Left half stays in scene as a grey ghost — clickable later
    const leftMat = material.clone();
    leftMat.color.setHex(0xd0d0d0);
    leftMat.opacity = 0.6;
    leftMat.transparent = true;
    splitLeft.material = leftMat;

    splitActive = false;
    splitLeft = null;
    splitRight = null;
    splitLeft = null;
    splitRight = null;
}

// ============================================================
// 7. RENDER LOOP — hands for rotate/scale/translate/split
// ============================================================
let lastPalm=[null,null], last2HDist=null, rotVX=0, rotVY=0;

function loop() {
    requestAnimationFrame(loop);
    let interacting=false, holding=false;

    if(handLandmarker && video.currentTime!==lastVideoTime) {
        lastVideoTime=video.currentTime;
        const det=handLandmarker.detectForVideo(video,performance.now());

        // Debug
        if(isDebug){debugCanvas.width=innerWidth;debugCanvas.height=innerHeight;debugCtx.clearRect(0,0,debugCanvas.width,debugCanvas.height);debugCtx.save();debugCtx.translate(debugCanvas.width,0);debugCtx.scale(-1,1);debugCtx.globalAlpha=0.35;debugCtx.drawImage(video,0,0,debugCanvas.width,debugCanvas.height);debugCtx.restore();debugCtx.globalAlpha=1;
        if(det.landmarks.length){let lb='';det.landmarks.forEach((h,i)=>{const m=h.map(l=>({...l,x:1-l.x}));drawUtils.drawConnectors(m,HandLandmarker.HAND_CONNECTIONS,{color:i?'#e66767':'#7b8b6f',lineWidth:3});drawUtils.drawLandmarks(m,{color:i?'#fffaec':'#f8eedb',fillColor:i?'#e66767':'#7b8b6f',radius:4});if(isFist(h))lb+='✊ Freeze  ';else if(isPinching(h))lb+='🤏 Move/Duplicate  ';else if(isOpenPalm(h))lb+='🤚 Rotate  ';else lb+='❓  ';});debugText.textContent=lb.trim();}else debugText.textContent='No hands';}

        if(det.landmarks.length) {
            interacting=true;

            // Two hands
            if(det.landmarks.length===2) {
                const h1=det.landmarks[0],h2=det.landmarks[1];
                const d=Math.hypot(h1[9].x-h2[9].x,h1[9].y-h2[9].y);

                // Both open → SCALE
                if(isOpenPalm(h1)&&isOpenPalm(h2)&&last2HDist!==null){
                    const dd=last2HDist-d;
                    if(Math.abs(dd)>0.005){let s=Math.max(0.2,Math.min(5,mesh.scale.x-dd*0.8));mesh.scale.set(s,s,s);rotVX=0;rotVY=0;}
                }
                // Both pinch → TRANSLATE or SPLIT
                if(isPinching(h1)&&isPinching(h2)){
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
                        const v=new THREE.Vector3(-(px*2-1),-(py*2-1),0.5).unproject(camera);
                        v.sub(camera.position).normalize();
                        const t=(mesh.position.z-camera.position.z)/v.z;
                        mesh.position.lerp(camera.position.clone().add(v.multiplyScalar(t)),0.2);
                        rotVX=0;rotVY=0;
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

                if(isFist(hand)){holding=true;lastPalm[idx]=null;return;}
                if(det.landmarks.length===2)return;

                // Single pinch → TRANSLATE
                if(isPinching(hand)){
                    const px=(hand[4].x+hand[8].x)/2,py=(hand[4].y+hand[8].y)/2;
                    const ndc=new THREE.Vector2(-(px*2-1),-(py*2-1));
                    const v=new THREE.Vector3(ndc.x,ndc.y,0.5).unproject(camera);
                    v.sub(camera.position).normalize();
                    const t=(mesh.position.z-camera.position.z)/v.z;
                    mesh.position.lerp(camera.position.clone().add(v.multiplyScalar(t)),0.15);
                    lastPalm[idx]=null;rotVX=0;rotVY=0;
                }
                // Open palm → ROTATE
                else if(isOpenPalm(hand)){
                    if(lastPalm[idx]){const dx=palm.x-lastPalm[idx].x,dy=palm.y-lastPalm[idx].y;rotVX-=dx*5;rotVY+=dy*5;}
                    lastPalm[idx]=palm;
                } else lastPalm[idx]=null;
            });
        } else lastPalm=[null,null];
    }

    if(holding){rotVX=0;rotVY=0;}
    mesh.rotation.y+=rotVX;mesh.rotation.x+=rotVY;
    rotVX*=0.85;rotVY*=0.85;
    if(!interacting&&Math.abs(rotVX)<0.01)mesh.rotation.y+=0.002;
    const tz=Math.max(4,mesh.scale.x*3);camera.position.z+=(tz-camera.position.z)*0.08;
    renderer.render(scene,camera);
}

// ============================================================
// 7. MOUSE SCULPTING
// ============================================================
let mDown=false, mBtn=-1, mLast={x:0,y:0}, mOnMesh=false, mUndoSaved=false;
window.addEventListener('contextmenu',e=>e.preventDefault());

window.addEventListener('mousedown',e=>{
    // Placement mode: click on mesh to start placing
    if(placementShape&&e.button===0){
        const nx=(e.clientX/innerWidth)*2-1,ny=-(e.clientY/innerHeight)*2+1;
        raycaster.setFromCamera(new THREE.Vector2(nx,ny),camera);
        const h=raycaster.intersectObject(mesh);
        if(h.length){
            const inv=mesh.matrixWorld.clone().invert();
            const lp=h[0].point.clone().applyMatrix4(inv);
            const ln=h[0].face.normal.clone().transformDirection(inv).normalize();
            placementPos = { x:lp.x+ln.x*0.2, y:lp.y+ln.y*0.2, z:lp.z+ln.z*0.2 };
            placementNormal = ln;
            placementStartY = e.clientY;
            placementDragging = true;
            // Create preview
            const previewGeo = SHAPES[placementShape]();
            placementPreview = new THREE.Mesh(previewGeo, previewMat);
            placementPreview.position.copy(h[0].point).addScaledVector(h[0].face.normal, 0.2);
            placementPreview.scale.setScalar(0.3);
            scene.add(placementPreview);
            placementLabel.textContent = `Drag up/down to resize — release to place`;
        }
        return;
    }
    mDown=true;mBtn=e.button;mUndoSaved=false;mLast={x:e.clientX,y:e.clientY};
    const nx=(e.clientX/innerWidth)*2-1,ny=-(e.clientY/innerHeight)*2+1;
    raycaster.setFromCamera(new THREE.Vector2(nx,ny),camera);
    mOnMesh=raycaster.intersectObject(mesh).length>0;
    if(mOnMesh&&mBtn!==1){saveUndo();mUndoSaved=true;}
});

window.addEventListener('mousemove',e=>{
    // Placement drag: resize preview
    if(placementDragging&&placementPreview){
        updatePreviewScale(e.clientY);
        return;
    }
    const nx=(e.clientX/innerWidth)*2-1,ny=-(e.clientY/innerHeight)*2+1;
    raycaster.setFromCamera(new THREE.Vector2(nx,ny),camera);
    const hits=raycaster.intersectObject(mesh);
    if(!mDown){if(hits.length)showBrush(hits[0].point,hits[0].face.normal);else hideBrush();return;}
    const dx=e.clientX-mLast.x,dy=e.clientY-mLast.y;mLast={x:e.clientX,y:e.clientY};
    if(!mOnMesh||mBtn===1){rotVX+=dx*0.003;rotVY+=dy*0.003;}
    else if(hits.length){
        showBrush(hits[0].point,hits[0].face.normal);
        if(e.shiftKey){smoothAt(hits[0].point);}
        else{const saved=activeTool;if(mBtn===2){if(activeTool==='pull')activeTool='push';else if(activeTool==='push')activeTool='pull';}applyTool(hits[0],dx,dy);activeTool=saved;}
    }
});

window.addEventListener('mouseup',()=>{
    if(placementDragging){ commitPlacement(); return; }
    mDown=false;mBtn=-1;mOnMesh=false;mUndoSaved=false;
});
window.addEventListener('wheel',e=>{brushRadius=Math.max(0.1,Math.min(1.5,brushRadius-e.deltaY*0.001));document.getElementById('br').value=brushRadius;document.getElementById('brv').textContent=brushRadius.toFixed(2);});

initHands();

// ============================================================
// 8. CAT AI (Boba)
// ============================================================
const chatInput=document.getElementById('chat-input'),sendBtn=document.getElementById('send-btn'),micBtn=document.getElementById('mic-btn');
const catDialogue=document.getElementById('cat-dialogue'),chatHistoryContainer=document.getElementById('chat-history-container');
const chatHistory=document.getElementById('chat-history'),toggleHistoryBtn=document.getElementById('toggle-history-btn');
const bgMusic=document.getElementById('bg-music'),musicToggleBtn=document.getElementById('music-toggle-btn');
let isMusicPlaying=false;bgMusic.volume=0.2;
musicToggleBtn.addEventListener('click',()=>{if(isMusicPlaying){bgMusic.pause();musicToggleBtn.textContent='🔇 Music Off';}else{bgMusic.play().catch(()=>{});musicToggleBtn.textContent='🎵 Music On';}isMusicPlaying=!isMusicPlaying;});
let isHistoryOpen=false;

function catCmd(cmd){
    const lc=cmd.toLowerCase();let resp='',done=false;
    for(const n of Object.keys(SHAPES)){if(lc.includes(n)){setShape(n);resp=`Switched to ${n}! 🎨`;done=true;break;}}
    if(!done&&/\b(reset|new|restart|clear)\b/i.test(lc)){resp="Fresh! 🆕";setShape(currentShape);done=true;}
    else if(!done&&/\b(bigger|larger|grow|expand)\b/i.test(lc)){resp="Growing!";mesh.scale.multiplyScalar(1.3);done=true;}
    else if(!done&&/\b(smaller|tiny|shrink)\b/i.test(lc)){resp="Shrinking!";mesh.scale.multiplyScalar(0.7);done=true;}
    else if(!done&&/\b(red|pink)\b/i.test(lc)){material.color.setHex(0xe66767);resp="Red! 🍓";done=true;}
    else if(!done&&/\b(blue|azure)\b/i.test(lc)){material.color.setHex(0xa8c3e6);resp="Blue! 🫐";done=true;}
    else if(!done&&/\b(green|sage)\b/i.test(lc)){material.color.setHex(0x9cb89c);resp="Green! 🌿";done=true;}
    else if(!done&&/\b(yellow|gold)\b/i.test(lc)){material.color.setHex(0xf4d03f);resp="Yellow! ☀️";done=true;}
    else if(!done&&/\b(white|cream)\b/i.test(lc)){material.color.setHex(0xfffaec);resp="White! 🥛";done=true;}
    else if(!done&&/\b(dark|black)\b/i.test(lc)){material.color.setHex(0x333333);resp="Dark! 🐈‍⬛";done=true;}
    else if(!done&&/\b(metal|shiny)\b/i.test(lc)){material.metalness=0.8;material.roughness=0.1;material.needsUpdate=true;resp="Shiny! ✨";done=true;}
    else if(!done&&/\b(matte|clay)\b/i.test(lc)){material.metalness=0.1;material.roughness=0.8;material.needsUpdate=true;resp="Matte! 🧱";done=true;}
    else if(!done&&/\b(glass|transparent)\b/i.test(lc)){material.transparent=true;material.opacity=0.5;material.metalness=1;material.roughness=0;material.needsUpdate=true;resp="Glass! 🧊";done=true;}
    else if(!done&&/\b(wireframe|lines)\b/i.test(lc)){material.wireframe=!material.wireframe;material.needsUpdate=true;resp="Wireframe! 📐";done=true;}
    else if(!done&&/\b(export|stl|print)\b/i.test(lc)){document.getElementById('export-btn').click();resp="Exporting STL! 💾";done=true;}
    else if(!done&&(/\b(help|how)\b/i.test(lc)||cmd.length>25)){resp="Mouse: left=pull, right=push, shift=smooth. Hands: palm=rotate, pinch=move, two palms=scale, two fists apart=duplicate. Try shapes, colors, 'export stl'!";done=true;}
    if(!done)resp="Meow? Try shapes (sphere,cube,torus), colors, 'export stl', or 'reset'!";
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
