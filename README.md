# Whimsical Clay Studio

A browser-based 3D sculpting app with hand gesture control, inspired by Studio Ghibli aesthetics. Built for prototyping 3D-printable objects like cute characters, figurines, and everyday items.

## Tech Stack
- **Three.js** — 3D rendering (via CDN import map)
- **MediaPipe Hands** — real-time hand tracking via webcam
- **Vanilla JS** — no build tools, no frameworks
- **HTML/CSS** — single-page app

## File Structure
```
index.html          — Layout, SVG cat companion (Boba), debug overlay, gesture guide
css/style.css       — Ghibli cream/sage theme, UI component styles
js/main.js          — All logic (~600 lines), organized into numbered sections:
                       1. Scene setup (Three.js, lighting, grid, camera, renderer)
                       2. Shapes & Mesh (5 primitives, shape switching, add-shape merge)
                       3. Sculpt Tools (8 Blender-style brushes)
                       4. MediaPipe hand tracking setup
                       5. Gesture detection functions
                       6. Split system (two-pinch-apart splits mesh in half)
                       7. Render loop (hand gestures for rotate/scale/translate/split)
                       8. Mouse sculpting (drag-based, all 8 tools)
                       9. Cat AI chat agent (Boba — NLP commands for colors, materials, shapes)
```

## Architecture Decisions

### Interaction Model (Mouse = Sculpt, Hands = Manipulate)
Hand tracking is imprecise for detail work, so:
- **Mouse** handles all sculpting (pull, push, smooth, flatten, inflate, grab, crease, pinch)
- **Hand gestures** handle spatial manipulation only (rotate, scale, translate, split)

### Geometry
- All shapes use `.toNonIndexed()` BufferGeometry so each triangle owns its vertices
- Subdivision counts kept moderate (32×24 sphere) for performance
- Sculpt functions use squared-distance checks and reusable temp vectors to avoid GC

### Sculpt Tools (Section 3 of main.js)
| Tool | Function | Description |
|------|----------|-------------|
| Pull | `sculptAt(+)` | Move vertices outward along face normal |
| Push | `sculptAt(-)` | Move vertices inward |
| Smooth | `smoothAt()` | Average vertices within brush radius |
| Flatten | `flattenAt()` | Project vertices onto tangent plane |
| Inflate | `inflateAt()` | Push along each vertex's own normal |
| Grab | `grabAt()` | Move chunk of vertices with mouse drag |
| Crease | `creaseAt()` | Create sharp ridge by pulling toward center line |
| Pinch | `pinchAt()` | Pull vertices toward brush center |

All tools use cosine falloff and support X-axis symmetry mirror.

### Hand Gestures (Section 5-7)
| Gesture | Detection | Action |
|---------|-----------|--------|
| Open palm | 3+ fingers extended, thumb spread | Rotate object |
| Fist | 4 fingers curled | Freeze (stop all motion) |
| Pinch | Thumb+index tips < 0.05, joints < 0.12 | Translate/move object |
| Two open palms | Both hands open | Scale (spread=bigger, close=smaller) |
| Two pinches apart | Both pinching, spreading | Split mesh in half along X |

### Key Features
- **Shape palette**: Sphere, Cube, Cylinder, Cone, Torus — click to start fresh, ➕ to add onto mesh
- **Add shape placement**: Click ➕, click mesh surface, drag to resize preview, release to commit
- **Brush controls**: Size slider (0.1–1.5), Strength slider (0.01–0.12), scroll wheel for size
- **Symmetry**: 🪞 toggle mirrors sculpting across X axis
- **Undo**: Up to 20 steps, Ctrl+Z or button
- **Split**: Press X key or two-pinch-apart gesture — splits mesh into left/right halves
- **STL Export**: 💾 button generates binary STL for 3D printing
- **Boba (Cat AI)**: Text/voice chat for changing shapes, colors, materials, or exporting
- **Debug overlay**: Shows webcam feed + hand landmarks + gesture labels

### CSS Theme
- Background: `#f8eedb` (Ghibli cream)
- Accent: `#7b8b6f` (sage green)
- Grid: 60px sage squares
- Floor: gingham pattern at bottom 15vh
- All UI panels: cream background, sage borders, 10-12px border-radius

### Known Considerations
- `bg-music.mp3` is referenced but not in the repo
- Webcam required for hand tracking (falls back to mouse-only gracefully)
- Performance: keep vertex counts under ~5000 per shape for smooth sculpting
- Non-indexed geometry means high vertex counts — avoid adding too many shapes

## Running
```bash
npx serve . -l 8000
```
Open the forwarded port URL in your browser. Allow camera access for hand tracking.

## Keyboard Shortcuts
| Key | Action |
|-----|--------|
| Ctrl+Z | Undo |
| X | Split mesh in half |
| Tab | (reserved, was selection mode — currently unused) |
| Esc | Cancel shape placement |
| Scroll wheel | Adjust brush size |
