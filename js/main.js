import * as THREE from 'three';
import { HandLandmarker, FilesetResolver, DrawingUtils } from '@mediapipe/tasks-vision';

// --- 1. SET UP THREE.JS SCENE ---
const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();

// Studio/CAD Environment Lighting
const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
scene.add(ambientLight);
const directionalLight = new THREE.DirectionalLight(0xffffff, 1.0);
directionalLight.position.set(5, 10, 7);
scene.add(directionalLight);

// Add a grid helper for CAD scale reference, matched to sage green aesthetic
const gridHelper = new THREE.GridHelper(20, 20, 0x7b8b6f, 0x7b8b6f);
gridHelper.position.y = -0.75;
gridHelper.material.opacity = 0.4;
gridHelper.material.transparent = true;
scene.add(gridHelper);

// Camera
const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.z = 8;
camera.position.y = 0;

// Renderer setup
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight);
container.appendChild(renderer.domElement);

// The Core Object (A Box Primitive)
// Using primitive shapes is much preferred for CAD/Rhino style modeling.
const geometry = new THREE.BoxGeometry(1.5, 1.5, 1.5); 

const material = new THREE.MeshStandardMaterial({ 
    color: 0xa8c3e6, // Soft CAD blueprint blue
    roughness: 0.3, 
    metalness: 0.1,
    flatShading: true
});
const objectMesh = new THREE.Mesh(geometry, material);
objectMesh.position.y = 0;

// Enhance the CAD look with an Edge/Wireframe Highlight
const edges = new THREE.EdgesGeometry(geometry);
const line = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0x222222, linewidth: 2 }));
objectMesh.add(line);
scene.add(objectMesh);

// We need a raycaster to project hand coordinates into the 3D space
const raycaster = new THREE.Raycaster();

window.addEventListener('resize', () => {
    renderer.setSize(window.innerWidth, window.innerHeight);
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    if (isDebugMode) {
        debugCanvas.width = window.innerWidth;
        debugCanvas.height = window.innerHeight;
    }
});

// --- 2. SET UP MEDIAPIPE HAND TRACKING ---
const video = document.getElementById('webcam');
let handLandmarker;
let lastVideoTime = -1;

// Debug Mode Overlays
const debugOverlay = document.getElementById('debug-overlay');
const debugCanvas = document.getElementById('debug-canvas');
const debugCtx = debugCanvas.getContext('2d');
const debugGestureText = document.getElementById('debug-gesture-text');
const drawingUtils = new DrawingUtils(debugCtx);
const debugToggleBtn = document.getElementById('debug-toggle-btn');
let isDebugMode = false;

debugToggleBtn.addEventListener('click', () => {
    isDebugMode = !isDebugMode;
    if (isDebugMode) {
        debugToggleBtn.textContent = '🛠 Debug On';
        debugToggleBtn.classList.add('active');
        debugOverlay.style.display = 'block';
        debugCanvas.width = window.innerWidth;
        debugCanvas.height = window.innerHeight;
    } else {
        debugToggleBtn.textContent = '🛠 Debug Off';
        debugToggleBtn.classList.remove('active');
        debugOverlay.style.display = 'none';
        debugCtx.clearRect(0, 0, debugCanvas.width, debugCanvas.height);
    }
});

async function initHandTracking() {
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
    );
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 2 // Allow 2 hands for scale/translate operations
    });

    navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
        .then((stream) => {
            video.srcObject = stream;
            video.addEventListener("loadeddata", renderLoop);
        })
        .catch((err) => console.error("Camera error:", err));
}

// --- 3. GEOMETRY MANIPULATION LOGIC ---
// Pinch check (translating/pulling)
function isPinching(landmarks) {
    const thumbTip = landmarks[4];
    const indexTip = landmarks[8];
    const distance = Math.hypot(thumbTip.x - indexTip.x, thumbTip.y - indexTip.y, thumbTip.z - indexTip.z);
    return distance < 0.06;
}

// Flat hand (rotating/scaling)
function isOpenPalm(landmarks) {
    const wrist = landmarks[0];
    const tips = [8, 12, 16, 20];
    const pips = [6, 10, 14, 18];
    let extendedCount = 0;
    for(let i=0; i<4; i++) {
        const tip = landmarks[tips[i]];
        const pip = landmarks[pips[i]];
        if (Math.hypot(tip.x - wrist.x, tip.y - wrist.y) > Math.hypot(pip.x - wrist.x, pip.y - wrist.y)) {
            extendedCount++;
        }
    }
    return extendedCount >= 3;
}

// Closed fist (holding/anchor)
function isClosedFist(landmarks) {
    const wrist = landmarks[0];
    const tips = [8, 12, 16, 20];
    const pips = [6, 10, 14, 18];
    let curledCount = 0;
    for(let i=0; i<4; i++) {
        const tip = landmarks[tips[i]];
        const pip = landmarks[pips[i]];
        if (Math.hypot(tip.x - wrist.x, tip.y - wrist.y) < Math.hypot(pip.x - wrist.x, pip.y - wrist.y)) {
            curledCount++;
        }
    }
    return curledCount >= 3;
}

// Hover Check
function isHandOnObject(landmarks) {
    const palmCenter = landmarks[9];
    const ndcX = -(palmCenter.x * 2 - 1); 
    const ndcY = -(palmCenter.y * 2 - 1);
    raycaster.setFromCamera(new THREE.Vector2(ndcX, ndcY), camera);
    return raycaster.intersectObject(objectMesh).length > 0;
}

// Map Pinch to 3D Space (Translation / Drag)
function translateObject(landmarks) {
    const pinchX = (landmarks[4].x + landmarks[8].x) / 2;
    const pinchY = (landmarks[4].y + landmarks[8].y) / 2;

    const ndcX = -(pinchX * 2 - 1); 
    const ndcY = -(pinchY * 2 - 1);

    const vec = new THREE.Vector3(ndcX, ndcY, 0.5);
    vec.unproject(camera);
    vec.sub(camera.position).normalize();
    const distance = (objectMesh.position.z - camera.position.z) / vec.z;
    const targetPos = camera.position.clone().add(vec.multiplyScalar(distance));
    
    // Smooth snapping / grid drag feel
    objectMesh.position.lerp(targetPos, 0.2);
}

// Pull individual vertices around for manual vertex manipulation (Rhino/Blender Edit Mode)
function pullVertex(landmarks) {
    const pinchX = (landmarks[4].x + landmarks[8].x) / 2;
    const pinchY = (landmarks[4].y + landmarks[8].y) / 2;

    const ndcX = -(pinchX * 2 - 1); 
    const ndcY = -(pinchY * 2 - 1);

    raycaster.setFromCamera(new THREE.Vector2(ndcX, ndcY), camera);
    const intersects = raycaster.intersectObject(objectMesh);

    if (intersects.length > 0) {
        const hitPoint = intersects[0].point;
        const positions = objectMesh.geometry.attributes.position;
        const vNum = positions.count;
        let modified = false;

        const pullRadius = 0.5; // Hard CAD falloff radius
        const vertex = new THREE.Vector3();

        for (let i = 0; i < vNum; i++) {
            vertex.fromBufferAttribute(positions, i);
            vertex.applyMatrix4(objectMesh.matrixWorld);
            
            const dist = vertex.distanceTo(hitPoint);
            // Snap logic: pull the nearest vertices as a cohesive block
            if (dist < pullRadius) {
                const pullDirection = raycaster.ray.direction.clone().multiplyScalar('-1');
                vertex.addScaledVector(pullDirection, 0.03); // Extrude outward

                vertex.applyMatrix4(objectMesh.matrixWorld.clone().invert());
                positions.setXYZ(i, vertex.x, vertex.y, vertex.z);
                modified = true;
            }
        }
        
        if (modified) {
            positions.needsUpdate = true;
            objectMesh.geometry.computeVertexNormals();
            
            // Re-draw grid lines so the stroke updates around the new extrusion
            objectMesh.remove(line);
            const newEdges = new THREE.EdgesGeometry(objectMesh.geometry);
            line.geometry.dispose();
            line.geometry = newEdges;
            objectMesh.add(line);
        }
    }
}

// Scale Object Globally (Two Hands, Open, Push/Pulling)
function scaleObjectGlobal(distanceDelta) {
    const scaleSpeed = 0.8;
    let currentScale = objectMesh.scale.x; 
    let newScale = currentScale - (distanceDelta * scaleSpeed);
    newScale = Math.max(0.2, Math.min(newScale, 5.0)); // Cap dimensions
    objectMesh.scale.set(newScale, newScale, newScale);
}

// --- 4. RENDER LOOP ---
let lastHandPositions = [null, null];
let lastTwoHandDistance = null;
let rotationVelocityX = 0;
let rotationVelocityY = 0;

function renderLoop() {
    requestAnimationFrame(renderLoop);

    let isInteracting = false;
    let isHolding = false;

    if (handLandmarker && video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;
        const detections = handLandmarker.detectForVideo(video, performance.now());
        
        if (detections.landmarks.length > 0) {
            isInteracting = true;

            // Two-hand geometric scaling
            if (detections.landmarks.length === 2) {
                const hand1 = detections.landmarks[0];
                const hand2 = detections.landmarks[1];
                const center1 = hand1[9]; // knuckle
                const center2 = hand2[9]; // knuckle
                
                const currentTwoHandDist = Math.hypot(center2.x - center1.x, center2.y - center1.y);
                
                // Both hands open, pull apart/push together to SCALE
                if (isOpenPalm(hand1) && isOpenPalm(hand2) && lastTwoHandDistance !== null) {
                    const distanceDelta = lastTwoHandDistance - currentTwoHandDist;
                    if (Math.abs(distanceDelta) > 0.005) {
                        scaleObjectGlobal(distanceDelta);
                        rotationVelocityX = 0;
                        rotationVelocityY = 0;
                    }
                }
                
                // Both hands pinching = global translation (DRAG)
                if (isPinching(hand1) && isPinching(hand2)) {
                    translateObject(hand1);
                    rotationVelocityX = 0;
                    rotationVelocityY = 0;
                }

                lastTwoHandDistance = currentTwoHandDist;
            } else {
                lastTwoHandDistance = null;
            }

            // Single Hand Interactions
            detections.landmarks.forEach((hand, index) => {
                if (index > 1) return;
                const palmCenter = hand[9];
                
                if (isClosedFist(hand)) {
                    isHolding = true;
                    lastHandPositions[index] = null;
                    return; 
                }

                // If two hands are active, ignore detailed single-hand sculpt logic
                if (detections.landmarks.length === 2) return;
                
                // Pinch to Extrude/Sculpt geometry faces (Edit Mode)
                if (isPinching(hand)) {
                    if (isHandOnObject(hand)) {
                        pullVertex(hand);
                    }
                    lastHandPositions[index] = null; 
                    if (!isHolding) {
                        rotationVelocityX = 0;
                        rotationVelocityY = 0;
                    }
                } 
                // Open Palm to Rotate Canvas (Orbit Mode)
                // Relaxed the condition so any non-fist, non-pinching hand allows rotation. 
                // This prevents rotation from magically stopping mid-swipe if a finger curls too much.
                else if (!isClosedFist(hand)) {
                    if (lastHandPositions[index]) {
                        const deltaX = palmCenter.x - lastHandPositions[index].x;
                        const deltaY = palmCenter.y - lastHandPositions[index].y;
                        
                        // Removed the !isHandOnObject restriction so swiping OVER the shape 
                        // doesn't suddenly stutter the canvas rotation
                        rotationVelocityX -= deltaX * 14.0;
                        rotationVelocityY += deltaY * 14.0;
                    }
                    lastHandPositions[index] = palmCenter;
                } else {
                    lastHandPositions[index] = null;
                }
            });
        } else {
            lastHandPositions = [null, null];
        }
    }

    if (isHolding) {
        rotationVelocityX = 0;
        rotationVelocityY = 0;
    }

    // Apply rotation velocity
    objectMesh.rotation.y += rotationVelocityX;
    objectMesh.rotation.x += rotationVelocityY;

    // Apply friction to rotation
    rotationVelocityX *= 0.90;
    rotationVelocityY *= 0.90;

    // Static Idle orbit when no hands
    if (!isInteracting && Math.abs(rotationVelocityX) < 0.01) {
        objectMesh.rotation.y += 0.002;
    }

    // Focus Camera dynamic scale
    const targetZ = Math.max(5.0, objectMesh.scale.x * 4.0);
    camera.position.z += (targetZ - camera.position.z) * 0.08;

    renderer.render(scene, camera);
}

// --- 5. MOUSE INTERFACE (For Fine Details & Fallback) ---
let isMouseDown = false;
let isMouseOnObject = false;
let lastMousePos = { x: 0, y: 0 };

window.addEventListener('mousedown', (e) => {
    isMouseDown = true;
    lastMousePos = { x: e.clientX, y: e.clientY };

    const ndcX = (e.clientX / window.innerWidth) * 2 - 1;
    const ndcY = -(e.clientY / window.innerHeight) * 2 + 1;
    raycaster.setFromCamera(new THREE.Vector2(ndcX, ndcY), camera);
    const intersects = raycaster.intersectObject(objectMesh);
    isMouseOnObject = intersects.length > 0;
});

window.addEventListener('mousemove', (e) => {
    if (!isMouseDown) return;

    const deltaX = e.clientX - lastMousePos.x;
    const deltaY = e.clientY - lastMousePos.y;
    lastMousePos = { x: e.clientX, y: e.clientY };

    if (!isMouseOnObject) {
        // Rotate Canvas (Empty space drag)
        rotationVelocityX += deltaX * 0.002;
        rotationVelocityY += deltaY * 0.002;
    } else {
        // Fine details / Stretch on item (Drag on object)
        const ndcX = (e.clientX / window.innerWidth) * 2 - 1;
        const ndcY = -(e.clientY / window.innerHeight) * 2 + 1;
        raycaster.setFromCamera(new THREE.Vector2(ndcX, ndcY), camera);
        const intersects = raycaster.intersectObject(objectMesh);
        if (intersects.length > 0) {
            const hitPoint = intersects[0].point;
            const positions = objectMesh.geometry.attributes.position;
            const vNum = positions.count;
            let modified = false;
            
            // A much smaller radius for precise, small mouse curves
            const pullRadius = 0.25; 
            const vertex = new THREE.Vector3();

            for (let i = 0; i < vNum; i++) {
                vertex.fromBufferAttribute(positions, i);
                vertex.applyMatrix4(objectMesh.matrixWorld);
                if (vertex.distanceTo(hitPoint) < pullRadius) {
                    const pullDir = raycaster.ray.direction.clone().multiplyScalar('-1');
                    // Much weaker pull force for fine detailing
                    vertex.addScaledVector(pullDir, 0.015); 
                    vertex.applyMatrix4(objectMesh.matrixWorld.clone().invert());
                    positions.setXYZ(i, vertex.x, vertex.y, vertex.z);
                    modified = true;
                }
            }
            
            if (modified) {
                positions.needsUpdate = true;
                objectMesh.geometry.computeVertexNormals();
                
                // Redraw wireframe
                objectMesh.remove(line);
                const newEdges = new THREE.EdgesGeometry(objectMesh.geometry);
                line.geometry.dispose();
                line.geometry = newEdges;
                objectMesh.add(line);
            }
        }
    }
});

window.addEventListener('mouseup', () => {
    isMouseDown = false;
    isMouseOnObject = false;
});

initHandTracking();

// --- 6. CAT AI AGENT LOGIC (Voice & Text) ---
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const micBtn = document.getElementById('mic-btn');
const catDialogue = document.getElementById('cat-dialogue');
const chatHistoryContainer = document.getElementById('chat-history-container');
const chatHistory = document.getElementById('chat-history');
const toggleHistoryBtn = document.getElementById('toggle-history-btn');

// --- NEW: Audio Control Logic ---
const bgMusic = document.getElementById('bg-music');
const musicToggleBtn = document.getElementById('music-toggle-btn');
let isMusicPlaying = false;

// Set background music volume completely low to not be distracting
bgMusic.volume = 0.2; 

musicToggleBtn.addEventListener('click', () => {
    if (isMusicPlaying) {
        bgMusic.pause();
        musicToggleBtn.textContent = '🔇 Music Off';
    } else {
        bgMusic.play().catch(e => console.warn("Audio play blocked by browser:", e));
        musicToggleBtn.textContent = '🎵 Music On';
    }
    isMusicPlaying = !isMusicPlaying;
});

let isHistoryOpen = false;

// Process design adjustments based on NLP keywords
function catProcessCommand(command) {
    const lowerCmd = command.toLowerCase();
    let response = "Meow? I don't quite understand, but I adjusted something anyway!";
    let actionTaken = false;

    // Check for exact keywords to prevent false negatives
    const wantsSmoothing = /\b(smooth|soften|curve|round)\b/i.test(lowerCmd);
    const wantsSharpening = /\b(sharp|harden|block|flat|crisp)\b/i.test(lowerCmd);
    const wantsScaleUp = /\b(bigger|larger|scale up|grow|expand)\b/i.test(lowerCmd);
    const wantsScaleDown = /\b(smaller|tiny|scale down|shrink)\b/i.test(lowerCmd);
    const wantsRefine = /\b(refine|fix|details|snap|straighten)\b/i.test(lowerCmd);

    // Colors
    const wantsColorRed = /\b(red|crimson|ruby|pink)\b/i.test(lowerCmd);
    const wantsColorBlue = /\b(blue|azure|sapphire)\b/i.test(lowerCmd);
    const wantsColorGreen = /\b(green|sage|emerald)\b/i.test(lowerCmd);
    const wantsColorYellow = /\b(yellow|gold|sun)\b/i.test(lowerCmd);
    const wantsColorDark = /\b(black|dark|charcoal|grey)\b/i.test(lowerCmd);
    const wantsColorWhite = /\b(white|cream|snow|light)\b/i.test(lowerCmd);

    // Materials & Textures
    const wantsMetal = /\b(metal|shiny|chrome|silver|glossy|reflective)\b/i.test(lowerCmd);
    const wantsMatte = /\b(matte|clay|rough|plastic|dull)\b/i.test(lowerCmd);
    const wantsGlass = /\b(glass|transparent|clear|see through)\b/i.test(lowerCmd);
    const wantsWireframe = /\b(wireframe|lines|mesh only)\b/i.test(lowerCmd);

    // Conversational & Advanced CAD Help
    const wantsPrintInfo = /\b(print|stl|obj|export|3d print)\b/i.test(lowerCmd);
    const wantsHelp = /\b(help|how|what|why|tell me|explain)\b/i.test(lowerCmd);

    // Build the responses based on the string checking via word boundary Regex
    if (wantsSmoothing) {
        response = "Purrr... Smoothing out those rough, hard geometric edges for you.";
        objectMesh.material.flatShading = false;
        objectMesh.material.needsUpdate = true;
        objectMesh.geometry.computeVertexNormals();
        actionTaken = true;
    } 
    else if (wantsSharpening) {
        response = "Hiss! Sharpening those edges right up into a hard block.";
        objectMesh.material.flatShading = true;
        objectMesh.material.needsUpdate = true;
        const positions = objectMesh.geometry.attributes.position;
        objectMesh.geometry.computeVertexNormals();
        positions.needsUpdate = true;
        actionTaken = true;
    } 
    else if (wantsScaleUp) {
        response = "Mew! Stretching your canvas object outward!";
        objectMesh.scale.multiplyScalar(1.3);
        actionTaken = true;
    } 
    else if (wantsScaleDown) {
        response = "Squeezing it down for tiny details.";
        objectMesh.scale.multiplyScalar(0.7);
        actionTaken = true;
    }
    else if (wantsRefine) {
        response = "Let me fine-tune that... (Locking rotation into a clean alignment!)";
        objectMesh.rotation.set(0, 0, 0); // Snap rotation
        actionTaken = true;
    }
    // --- NEW: COLOR LOGIC ---
    else if (wantsColorRed) {
        response = "Painting it a lovely crimson red! 🍓";
        objectMesh.material.color.setHex(0xe66767);
        actionTaken = true;
    }
    else if (wantsColorBlue) {
        response = "Switching to a calming blueprint blue! 🫐";
        objectMesh.material.color.setHex(0xa8c3e6);
        actionTaken = true;
    }
    else if (wantsColorGreen) {
        response = "Ah, a natural Studio Ghibli sage green! 🌿";
        objectMesh.material.color.setHex(0x9cb89c);
        actionTaken = true;
    }
    else if (wantsColorYellow) {
        response = "Brightening it up with sunny yellow! ☀️";
        objectMesh.material.color.setHex(0xf4d03f);
        actionTaken = true;
    }
    else if (wantsColorDark) {
        response = "Going into dark mode with charcoal! 🐈‍⬛";
        objectMesh.material.color.setHex(0x333333);
        actionTaken = true;
    }
    else if (wantsColorWhite) {
        response = "Purrrfect! A pure cream white clay! 🥛";
        objectMesh.material.color.setHex(0xfffaec);
        actionTaken = true;
    }
    // --- NEW: MATERIAL & TEXTURE LOGIC ---
    else if (wantsMetal) {
        response = "Making it shiny! I've increased the metalness and gloss. ✨";
        objectMesh.material.metalness = 0.8;
        objectMesh.material.roughness = 0.1;
        objectMesh.material.transparent = false;
        objectMesh.material.wireframe = false;
        objectMesh.material.needsUpdate = true;
        actionTaken = true;
    }
    else if (wantsMatte) {
        response = "Going back to a standard rough, matte clay texture. 🧱";
        objectMesh.material.metalness = 0.1;
        objectMesh.material.roughness = 0.8;
        objectMesh.material.transparent = false;
        objectMesh.material.wireframe = false;
        objectMesh.material.needsUpdate = true;
        actionTaken = true;
    }
    else if (wantsGlass) {
        response = "Meow! It's see-through now! Adjusting opacity for a glass look. 🧊";
        objectMesh.material.transparent = true;
        objectMesh.material.opacity = 0.5;
        objectMesh.material.roughness = 0.0;
        objectMesh.material.metalness = 1.0;
        objectMesh.material.wireframe = false;
        objectMesh.material.needsUpdate = true;
        actionTaken = true;
    }
    else if (wantsWireframe) {
        response = "Showing the wireframe skeleton! Great for technical debugging. 📐";
        objectMesh.material.wireframe = true;
        objectMesh.material.transparent = false;
        objectMesh.material.needsUpdate = true;
        actionTaken = true;
    }
    // --- NEW: CONVERSATIONAL AGENT FALLBACKS ---
    else if (wantsPrintInfo) {
        response = "To 3D print this, you'll need an exported STL or OBJ file. For now, you can shape the geometry, and the final exporter will process these vertices directly to your slicer software!";
        actionTaken = true;
    }
    else if (wantsHelp || command.length > 25) {
        // Broad intelligent-sounding fallback if they ask a question or type a long sentence
        response = "As your AI Assistant, I can help you modify this shape! Try asking me to change the 'material to glass', 'color to sage green', 'smooth the edges', or 'make it metal'. Hand tracking also lets you grab and scale!";
        actionTaken = true;
    }

    if (!actionTaken) {
        // Did not match anything specific
        response = "Meow... I don't quite catch that. You can ask me to change materials (metal, glass, clay), colors, or reshape the geometry entirely!";
    }

    // Playful update transition
    catDialogue.style.opacity = 0;
    setTimeout(() => {
        catDialogue.style.opacity = 1;
        catDialogue.innerHTML = `<em>"${response}"</em>`;
        addHistoryLog(command, response);
    }, 200);
}

// Add history entries
function addHistoryLog(userText, botText) {
    const userLi = document.createElement("li");
    userLi.className = "history-user";
    userLi.textContent = `You: "${userText}"`;

    const botLi = document.createElement("li");
    botLi.className = "history-cat";
    botLi.textContent = `Cat: "${botText}"`;

    chatHistory.appendChild(userLi);
    chatHistory.appendChild(botLi);

    // Auto scroll bottom
    chatHistoryContainer.scrollTop = chatHistoryContainer.scrollHeight;
}

// Toggle History Window
toggleHistoryBtn.addEventListener('click', () => {
    isHistoryOpen = !isHistoryOpen;
    chatHistoryContainer.style.display = isHistoryOpen ? 'block' : 'none';
});

// Handle text input commands
sendBtn.addEventListener('click', () => {
    if (chatInput.value.trim()) {
        catProcessCommand(chatInput.value);
        chatInput.value = '';
    }
});
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && chatInput.value.trim()) {
        catProcessCommand(chatInput.value);
        chatInput.value = '';
    }
});

// Boba Interaction & Chat Bubble Toggle
const catCharacterNode = document.getElementById('cat-character');
const chatInterfaceNode = document.getElementById('chat-interface');
const rightEyeOpen = document.getElementById('right-eye-open');
const rightEyeWink = document.getElementById('right-eye-wink');

let isChatVisible = true;
catCharacterNode.addEventListener('click', () => {
    // Toggle Chat visibility
    isChatVisible = !isChatVisible;
    chatInterfaceNode.style.display = isChatVisible ? 'flex' : 'none';

    // 1. Wink Animation
    rightEyeOpen.style.display = 'none';
    rightEyeWink.style.display = 'block';
    
    // Play text update if Chat is open
    if (isChatVisible) {
        catDialogue.style.opacity = 0;
        setTimeout(() => {
            catDialogue.innerHTML = `<em>"Meow! ✨"</em>`;
            catDialogue.style.opacity = 1;
        }, 150);
    }
    
    setTimeout(() => {
        rightEyeOpen.style.display = 'block';
        rightEyeWink.style.display = 'none';
    }, 450); // 450ms wink duration
});

// Setup Web Speech API for Voice Interaction
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

if (SpeechRecognition) {
    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.lang = 'en-US';

    recognition.onstart = () => {
        micBtn.classList.add('recording');
        catDialogue.innerHTML = '<em>Listening... (Purr)</em>';
    };

    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        chatInput.value = transcript; // Fill input box with dictation
        catProcessCommand(transcript);
        chatInput.value = ''; // clear immediately after processing
    };

    recognition.onerror = (event) => {
        micBtn.classList.remove('recording');
        console.error("Speech recognition error:", event.error);
        if(event.error === 'not-allowed') {
            catDialogue.innerHTML = "<em>I can't hear you! Microphone access was denied.</em>";
        }
    };

    recognition.onend = () => {
        micBtn.classList.remove('recording');
    };

    // Hold or click mouse to speak
    micBtn.addEventListener('click', () => {
        recognition.start();
    });
} else {
    // Hide mic button if browser doesn't support Web Speech API
    micBtn.style.display = 'none';
    catDialogue.textContent = "Voice commands not supported in this browser. Use text!";
}
