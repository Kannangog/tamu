# Artist's Hand v2: Proof of Capability Notes

Evaluated 4 scripted pieces produced purely programmatically (NO ML) via the v2 Hand layer.

---

### 1. Bicycle Ink Study (`bicycle_ink_study.png` / `.svg`)
- **What the hand did well**:
  - Light-directed line-weight variation gives the frame tangible physical mass: undersides of down tube and chainstays are thick and dark, while sun-lit top tubes stay delicate.
  - Directional hatching ground shadow creates a firm contact plane, removing the "floating on white" look of v1 doodles.
  - Spokes and concentric rim ellipses render sharply with clean Catmull-Rom interpolation.
- **What it still can't do**:
  - Dynamic line occlusion: in real drawing, overlapping foreground tubes automatically occlude background spokes without needing explicit layer masking or clipping paths.
  - Mechanical chain tension: chain links are rendered as straight lines rather than realistic hanging catenary curves.

---

### 2. Pencil Still Life (`pencil_still_life.png` / `.svg`)
- **What the hand did well**:
  - Authentic graphite tooth texture: the pressure-modulated paper grain noise faithfully reproduces the look of 2B/4B graphite on cold-press paper.
  - Faithful depiction of light physics: specular core shadows, reflected bounce light on the underside of the sphere, and projected cast shadows on the table plane.
  - Searching underdrawing pencil passes create the feel of an artist constructing forms.
- **What it still can't do**:
  - Curved surface hatching: hatching lines on the sphere are linear rather than wrapping around the 3D spherical surface contours (cross-contour hatching).
  - Smudging / blending stump: real pencil drawings utilize physical tortillon blending for smooth gradients between core shadow and penumbra.

---

### 3. Color Landscape (`color_landscape.png` / `.svg`)
- **What the hand did well**:
  - Smooth atmospheric depth perspective: background mountain ridges naturally desaturate and shift toward the sky's warm apricot-tinted horizon, creating immediate depth cues.
  - Multi-tier layer blending: linear gradient sky + radial sun glow composited cleanly over mountain silhouettes and foreground pine trees.
  - Rule-of-Thirds composition: sun and tree clusters anchored along power points.
- **What it still can't do**:
  - Wet-on-wet watercolor blooming: washes have drying edges, but lack pigment migration physics when two wet washes touch.
  - Atmospheric haze scattering: no Rayleigh scattering model for true distance aerial haze.

---

### 4. Charcoal Portrait Study (`charcoal_portrait.png` / `.svg`)
- **What the hand did well**:
  - Loomis construction proportion guides: cranial sphere, brow, nose, and chin divisions provide a structured geometric foundation.
  - Rough powdery charcoal dispersion: high friction edges and soft charcoal halo around dark hair masses.
  - Bold value distribution: high-contrast chiaroscuro with distinct light and shadow planes.
- **What it still can't do**:
  - Anatomy / organic muscle blending: portrait faces require subtle anatomical transitions around zygomatic arches and nasolabial folds that simple polygons and 2D hatching approximate coarsely.
  - Hair strand flow: hair masses are rendered with sweeping broad strokes rather than tapering hair clusters.

---

### Overall Verdict on the Hand
The Hand layer successfully breaks free from the 64x64 doodle prison. It can produce high-resolution, layered, expressive, textured artwork with line weight variation, gradients, and hatching that would take an experienced human artist significant time to execute. It provides the expressivity that a future generative model can learn to orchestrate.
