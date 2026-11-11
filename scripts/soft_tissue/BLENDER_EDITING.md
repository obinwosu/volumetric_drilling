# Editing the ear's skin and cartilage in Blender

The placement viewer builds the ear. Blender is where you shape it:

- **Skin** (`ct_ear`): mark which part is the auricle, and cut the CT ear down to the auricle's
  skin.
- **Cartilage** (`cartilage`): cut the incisura, trim and model the framework.

**`Ctrl+S` in Blender sends both back.** The viewer notices within a couple of seconds, loads
them and rebuilds the ear. There is nothing to export or load by hand.

Install Blender once: `sudo snap install blender --classic`

## The viewer's panel

| Section | What it is for |
|---|---|
| 1 Place the ear | Position, orientation, landmarks, auto-place. Place the ear before editing in Blender: Blender edits the ear where it sits. |
| 2 Build the ear | **Build ear** opens the meatus, joins the ear to the scalp skin, lines the canal and builds the cartilage. **Undo (raw ear)** goes back. |
| 3 Edit in Blender | **Edit skin**, **Edit cartilage**, and **use the Blender edits** (untick it for the original CT ear). Under **More**: the cartilage plate thickness, **Load edits now**, **Rebuild cartilage**. |
| 4 Skin envelope | Layer thicknesses, the thinner envelope around the ear, temporalis, **Rebuild envelope**. |
| 5 Show | What is drawn, skin opacity, the layer cutaway. |
| Older tools | The `ear.obj` workflow and the drawing tools from before Blender. Not needed for the CT ear. |

**Save** (bottom) writes the placement and every mesh.

## Editing the skin

1. Viewer: **3 Edit in Blender → Edit skin**. Blender opens with the ear selected, in Weight
   Paint. The bone, the envelope's skin and the EAC are shown as locked references.
2. Paint the auricle into the **auricle** group: weight 1, shown in red. To do it in one go:
   1. Press `Tab` for Edit Mode, then `Alt+Z` for X-ray so the lasso goes through the whole
      ear.
   2. Lasso the auricle.
   3. Properties → green triangle tab → Vertex Groups → **auricle** → **Assign**.
3. Cut away what is not auricle: Edit Mode, X-ray on, lasso, then `X` → Vertices. Or leave it
   unpainted.
4. `Ctrl+S`.

In the viewer:
- An **open sheet** (cut down to the auricle's skin) is used as it is. Its base edge is carried
  smoothly down onto the scalp skin.
- A **closed CT ear** is trimmed instead: unpainted tissue is pressed under the envelope.
- Either way, the cartilage is built inside the red only.

## Editing the cartilage

1. Viewer: **3 Edit in Blender → Edit cartilage**. Blender opens on the blue `cartilage`
   object, with the skin hidden. If Blender is already open, press FIVRS tab (`N`) →
   **Show cartilage** there.
2. Cut and model it:
   - **Lasso Trim (freehand, stays closed — the easiest):**
     1. Mode menu → **Sculpt Mode**.
     2. Left toolbar → press and hold the trim tool → **Lasso Trim**. Use **Polyline Trim**
        for straight segments: click points, double-click to finish.
     3. Header: Trim Mode **Difference**, Orientation **View**.
     4. Rotate to the view the cut should go along, then drag a loop.

     Everything inside the loop is cut away through the whole thickness, and the cut faces
     are closed for you. `Ctrl+Z` undoes it.
   - **Delete, then bridge:** Edit Mode, `Alt+Z`, lasso, `X` → Vertices. A cut through a solid
     leaves two open edges, one on the front skin and one on the back. Select → Select All by
     Trait → Non Manifold, then Edge → **Bridge Edge Loops** to join them with a side wall.
     (`F` / `Alt+F` would only cap each one flat.)
   - **Boolean with a cylinder or cube:**
     1. In Object Mode, Add → Mesh → Cylinder or Cube.
     2. Place it over the notch: move `G`, rotate `R`, scale `S`.
     3. Select the cartilage → Properties → wrench tab → Add Modifier → Generate →
        **Boolean** → Difference → Object: the cylinder.
     4. Hide the cylinder.

     The cut stays editable, and `Ctrl+S` sends it with the cut applied.
   - **Delete:** Edit Mode, `Alt+Z`, lasso, `X` → Vertices. The cartilage is a solid, so close
     the cut afterwards: Select → Select All by Trait → Non Manifold, then `Alt+F`.
   - Sculpt Mode (Smooth, Flatten, Grab) works too.
   - Where the skin is open underneath the auricle (its top and back), the viewer closes it with
     skin stretched smooth from the skin's base edge down to where the cartilage reaches the
     envelope (the postauricular crease). Cartilage that would come out through that skin is cut
     back under it. The cleft between the auricle and the head stays open, with the envelope's
     skin over the fascia.
3. `Ctrl+S`. From then on the viewer shows your cartilage instead of building one.
   **More → Rebuild cartilage** sets your edit aside (as `.ply.bak`) and builds it again from
   the skin.

## Moving around in Blender

| Action | Mouse / keys |
|---|---|
| Orbit | middle-mouse drag (or drag the axis ball, top right) |
| Pan | Shift + middle-mouse (or the hand icon) |
| Zoom | scroll (or the magnifier icon) |
| Frame the selection | numpad `.` |
| See through (X-ray) | `Alt+Z` |
| Hide or show an object | FIVRS tab (`N`) checkboxes, or the eye icon in the Outliner |
| Undo / redo | `Ctrl+Z` / `Ctrl+Shift+Z` |

Without a middle mouse button: Edit → Preferences → Input → **Emulate 3 Button Mouse**, then
Alt + left-drag orbits.

## Files

Everything for Blender is in
`resources/soft_tissue/blender/volume_256_RT143_ct_right_ear/`:

| File | What it is |
|---|---|
| `ct_ear.blend` | your Blender scene |
| `ct_ear_edited.ply`, `cartilage_edited.ply` | written on every `Ctrl+S` |
| `cartilage.ply` | the built cartilage the viewer hands over |

The viewer keeps its copies in the ear's own frame (`resources/soft_tissue/auricle/*_edited.ply`),
so they follow the ear's placement.

**Start the Blender scene over** (for example after moving the ear): close Blender, delete
`ct_ear.blend`, then press **Edit skin**.

Logs: `blender.log` in that folder, and `~/.cache/fivrs-placement-viewer/viewer.log`.
