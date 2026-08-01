# ResPlan Floor-Plan Extruder

The ResPlan extruder converts the vector geometry in `ResPlan.pkl` into
watertight, metric 3D meshes. It provides:

- A batch-first command-line exporter for OBJ and GLB.
- A local Streamlit web viewer with 2D and interactive 3D previews.
- Floors, adjustable walls, optional ceilings, door modes, and window modes.
- Per-plan metadata plus a batch manifest recording warnings and failures.

Models use metres, the Z axis points upward, and the finished floor surface is
at Z=0. The source dataset's normalized wall depth establishes XY scale using
a fixed 0.20 m reference. The `wall-thickness` setting then resizes wall bands
independently, so changing thickness does not resize the floor plan or alter
wall height.

## Installation

Run these commands from the folder containing `ResPlan.pkl`:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

`requirements.txt` is the pinned CLI/viewer runtime used for deployment. The
larger notebook, graph, and ML toolchain is intentionally separate in
`requirements-dataset.txt` and is not required by the extruder.
NetworkX is part of the minimal runtime because `ResPlan.pkl` contains
serialized NetworkX graph objects, even though extrusion itself only consumes
the plan geometry. SciPy is also required by Trimesh's colour-preserving GLB
export path.

The editable installation provides `resplan-extrude` and `resplan-viewer`.
Some Windows Python installations do not place their `Scripts` directory on
`PATH`. Every example below therefore also has a reliable `python -m`
equivalent.

Confirm the exporter is available:

```powershell
python -m resplan_extruder.cli --help
```

Only load pickle files from a trusted source. Python pickle files can execute
code while loading.

## Command-line exporter

### Quick examples

Export two IDs to OBJ using the defaults:

```powershell
python -m resplan_extruder.cli --ids 14433,14926 --output exports
```

Export 25 test plans to both OBJ and GLB, including ceilings:

```powershell
python -m resplan_extruder.cli `
  --split test `
  --limit 25 `
  --format both `
  --ceiling `
  --output exports
```

Create 3.2 m walls, leave doors open to the ceiling, and replace window
openings with solid wall:

```powershell
python -m resplan_extruder.cli `
  --ids 14433 `
  --wall-height 3.2 `
  --door-mode full-height `
  --window-mode solid `
  --format both `
  --output exports
```

Create a robotics-safe perimeter and narrow two internal door openings:

```powershell
python -m resplan_extruder.cli `
  --ids 14433 `
  --close-boundary-doors `
  --restricted-door-count 2 `
  --restricted-door-mode width `
  --restricted-door-width 0.35 `
  --restricted-door-seed 17 `
  --output exports
```

Create a reproducible irregular variant with diagonal/rounded corners, bowed
walls, and plan-5142-style edge noise:

```powershell
python -m resplan_extruder.cli `
  --ids 14433 `
  --diagonal-corner-percent 20 `
  --rounded-corner-percent 15 `
  --curved-wall-percent 20 `
  --noisy-wall-percent 25 `
  --geometry-seed 73 `
  --output exports
```

Export the full dataset while recording and skipping malformed plans:

```powershell
python -m resplan_extruder.cli `
  --all `
  --format glb `
  --on-error skip `
  --output exports
```

If the installed scripts are on `PATH`, the shorter form is equivalent:

```powershell
resplan-extrude --ids 14433,14926 --output exports
```

### Plan selection options

Exactly one of `--ids`, `--split`, or `--all` is required.

| Option | Default | Description |
|---|---:|---|
| `--data PATH` | `ResPlan.pkl` | Trusted dataset pickle to load. |
| `--splits PATH` | `split.json` | Split definitions used with `--split`. |
| `--ids ID,ID,...` | — | Comma-separated public dataset IDs. These are not list indexes. |
| `--split NAME` | — | Select a canonical split such as `train`, `val`, `test`, or `augmented`. |
| `--all` | off | Select every plan in dataset order. |
| `--limit N` | unlimited | Keep only the first `N` plans from the selection. Useful for smoke tests. |

Examples:

```powershell
python -m resplan_extruder.cli --ids 14433,14926
python -m resplan_extruder.cli --split val --limit 10
python -m resplan_extruder.cli --all --limit 100
```

### Output and failure options

| Option | Default | Description |
|---|---:|---|
| `--output PATH` | `exports` | Root output directory. |
| `--format obj\|glb\|both` | `obj` | Model format or formats written for each plan. |
| `--on-error skip\|fail` | `skip` | Continue after a plan failure or stop at the first failure. |

OBJ is a simple, Z-up mesh with semantic object/group names. GLB retains
separate semantic components and their preview colors in one binary file.

With `--on-error skip`, a multi-plan batch continues and records each failed
plan. A failed single-plan request still returns a non-zero exit status. With
`--on-error fail`, processing stops immediately and returns a non-zero status.

### Scale, floor, walls, and ceiling

| Option | Default | Description |
|---|---:|---|
| `--wall-thickness METRES` | `0.20` | Physical wall-band thickness. It does not change plan scale or wall height. |
| `--wall-height METRES` | `2.70` | Global wall height for the selected plans. |
| `--floor-thickness METRES` | `0.20` | Slab thickness below Z=0. The floor is always generated. |
| `--ceiling` | off | Add a ceiling slab above the walls. |
| `--ceiling-thickness METRES` | `0.15` | Ceiling thickness; used only with `--ceiling`. |
| `--exclude-balcony` | off | Exclude balcony polygons from the floor footprint. |
| `--no-center` | off | Preserve the scaled source XY position instead of centering the footprint at the origin. |

The floor occupies `-floor_thickness <= Z <= 0`. Walls begin at Z=0. When
enabled, the ceiling begins at `wall_height` and does not cover balconies.
The slab footprint is built from the labeled living, bedroom, bathroom,
kitchen, storage, and stair spaces together with walls and openings. The
dataset's `inner` layer is used only as a fallback when no labeled room
geometry is available, because some records contain unrelated or oversized
`inner` masks.

XY metre scale is always derived by treating the supplied normalized
`wall_depth` as 0.20 m. Wall thickness is applied afterward: connected wall
bands are expanded or contracted around their existing footprint. Door and
window gaps are temporarily restored before resizing the complete wall band,
then cut back out at the requested thickness. This keeps thin walls joined at
junctions and flush with lintels, sills, and headers. Very extreme thickness
changes can still merge or remove narrow source artifacts; any such case is
recorded as a warning in metadata.

Wall height is global per command. The dataset does not contain per-wall height
attributes, so variable heights for individual wall segments require a separate
override format or classification rule.

### Door options

| Option | Default | Description |
|---|---:|---|
| `--door-mode lintel\|full-height` | `lintel` | Retain wall above doors or leave door footprints open to wall height. |
| `--door-height METRES` | `2.10` | Top of the door void in `lintel` mode. |
| `--close-boundary-doors` | off | Fill home entrances and exterior/balcony door footprints with full-height wall. |
| `--restricted-door-count N` | `0` | Restrict up to `N` non-boundary interior doors. |
| `--restricted-door-mode height\|width\|both` | `height` | Lower clearance, narrow the opening, or apply both changes. |
| `--restricted-door-height METRES` | `1.00` | Clearance beneath the lintel in `height` or `both` mode. |
| `--restricted-door-width METRES` | `0.40` | Centred opening width in `width` or `both` mode. |
| `--restricted-door-seed N` | `0` | Reproducible seed used to choose which eligible interior doors are restricted. |

Door mode behavior:

- `lintel`: the door is open from Z=0 to `door_height`; wall continues from
  `door_height` to `wall_height`.
- `full-height`: the complete door footprint remains empty from Z=0 to
  `wall_height`; `--door-height` is not used.

Both interior `door` polygons and `front_door` polygons follow the selected
mode. In `lintel` mode, validation requires:

```text
0 < door_height < wall_height
```

#### Robotics door controls

`--close-boundary-doors` reconstructs full-height wall in:

- Every explicitly labeled `front_door` footprint.
- Every ordinary door footprint lying on the enclosed-plan boundary, which
  includes doors connecting the home to supplied balcony geometry.

This prevents the generated floor from retaining a traversable opening at the
home or balcony perimeter. The floor footprint itself is unchanged; use
`--exclude-balcony` separately when balcony slab geometry should also be
removed.

Restricted doors can become horizontally narrower, vertically lower, or both.
For example, keep a centred 0.35 m opening in three selected doorways:

```powershell
python -m resplan_extruder.cli `
  --ids 14433 `
  --restricted-door-count 3 `
  --restricted-door-mode width `
  --restricted-door-width 0.35 `
  --restricted-door-seed 17
```

The width restriction restores full-height wall symmetrically at both ends of
the original door footprint. It does not scale or move the plan, and it works
with either normal door mode. If the requested width is not smaller than an
individual source opening, that opening is left unchanged and metadata records
a warning.

Use both restrictions together when desired:

```powershell
python -m resplan_extruder.cli `
  --ids 14433 `
  --restricted-door-count 3 `
  --restricted-door-mode both `
  --restricted-door-width 0.35 `
  --restricted-door-height 0.60 `
  --restricted-door-seed 17
```

The exporter considers only non-boundary interior doors. It starts from a
stable spatial ordering and uses `restricted-door-seed` together with the plan
ID to sample `N` doors. The same plan, seed, and options always produce the
same selection; changing the seed resamples a reproducible subset. With a
small number of eligible doors, two different seeds can occasionally select
the same subset.
Boundary and front doors are excluded from this selection. If a plan has fewer
eligible doors than requested, the count is clamped and a warning is written
to metadata.

In `height` or `both` mode, restricted doors override the normal door mode:
even when ordinary doors use `full-height`, selected restricted doors retain
wall from `restricted_door_height` to `wall_height`. Set the restricted height
below the robot's collision height. A height of `0` fills the selected opening
vertically. In width-only mode, `full-height` doors remain vertically open.

For the enabled restriction dimensions, validation requires:

```text
0 <= restricted_door_height < wall_height
restricted_door_width > 0
```

The mode, target/effective widths, height, seed, eligible/selected/closed door
counts, and modified doors' metric XY centers are recorded under
`door_treatments` in each plan's `metadata.json`. This makes robotics
experiments repeatable and lets downstream code identify the modified
openings.

### Window options

| Option | Default | Description |
|---|---:|---|
| `--window-mode opening\|solid` | `opening` | Create window openings or replace their footprints with full-height wall. |
| `--window-sill-height METRES` | `0.90` | Bottom of the void in `opening` mode. |
| `--window-head-height METRES` | `2.10` | Top of the void in `opening` mode. |

Window mode behavior:

- `opening`: wall occupies the window footprint below the sill and above the
  head, leaving the interval between them open.
- `solid`: the window footprint is filled from Z=0 to `wall_height`;
  sill/head values are not used.

In `opening` mode, validation requires:

```text
0 <= window_sill_height < window_head_height < wall_height
```

The exporter creates structural openings only. It does not invent window
frames, glass, door leaves, or swing directions.

### Seeded geometry variations

All geometry effects are disabled by default. Percentages sample eligible wall
corners or wall-face segments across both the exterior shell and partitions.
Areas around doors and windows are excluded so openings remain aligned.

| Option | Default | Description |
|---|---:|---|
| `--diagonal-corner-percent P` | `0` | Replace `P%` of eligible corners with a straight diagonal. |
| `--diagonal-corner-size METRES` | `0.30` | Distance cut back along both sides of a diagonal corner. |
| `--rounded-corner-percent P` | `0` | Round `P%` of remaining eligible corners. |
| `--rounded-corner-radius METRES` | `0.30` | Requested corner radius, capped to fit short edges. |
| `--curved-wall-percent P` | `0` | Bow `P%` of eligible wall-face segments. |
| `--curved-wall-amplitude METRES` | `0.15` | Maximum perpendicular depth of each curve. |
| `--noisy-wall-percent P` | `0` | Subdivide and jitter `P%` of eligible wall-face segments. |
| `--noisy-wall-amplitude METRES` | `0.08` | Maximum perpendicular jitter. |
| `--geometry-seed N` | `0` | Shared reproducible seed for every geometry effect. |

Corner percentages form one mutually exclusive mix, and curved/noisy wall
percentages form another. When a group's total exceeds 100%, its values are
treated as relative shares at 100% coverage (for example, curve=100 and
noise=100 produces an approximately 50/50 mix, never both effects on one wall).
The two faces of a physical wall are paired and moved together. Effects on a
wall beside a changed corner begin after a straight shoulder and ease in with
zero slope, avoiding diagonal wedges where modifications meet.

At 100%, every eligible paired corner or wall span receives the selected
effect. Adjacent spans are allowed because their displacement eases to zero at
shared endpoints. Door/window intervals remain straight while deformation
tapers in on either side. The applied count can still be lower than the raw
polygon count because the two wall faces must be paired and extremely short
features are excluded. Requested corner sizes are capped locally rather than
discarding short-but-usable corners. Invalid intermediate polygons are repaired
before extrusion. The slab and optional ceiling follow changes at the building
perimeter, while internal wall changes leave the continuous floor intact.

The sidecar's `geometry_variations` object records the seed and eligible/applied
count for every effect. The same plan, options, and seed reproduce the same
mesh; change only `--geometry-seed` to sample another layout.

### Combined option examples

Keep normal windows but make every door a full-height opening:

```powershell
python -m resplan_extruder.cli `
  --ids 14433 `
  --door-mode full-height `
  --window-mode opening
```

Keep door lintels but remove all windows:

```powershell
python -m resplan_extruder.cli `
  --ids 14433 `
  --door-mode lintel `
  --door-height 2.2 `
  --window-mode solid
```

Create taller walls and windows:

```powershell
python -m resplan_extruder.cli `
  --ids 14433 `
  --wall-height 3.4 `
  --door-height 2.3 `
  --window-sill-height 1.0 `
  --window-head-height 2.5
```

Create a lower, fully open shell without validation conflicts from unused
door/window heights:

```powershell
python -m resplan_extruder.cli `
  --ids 14433 `
  --wall-height 1.5 `
  --door-mode full-height `
  --window-mode solid
```

Create a sealed perimeter while leaving two deliberately low internal
passages:

```powershell
python -m resplan_extruder.cli `
  --ids 14433 `
  --close-boundary-doors `
  --door-mode full-height `
  --restricted-door-count 2 `
  --restricted-door-mode height `
  --restricted-door-height 0.5 `
  --restricted-door-seed 17
```

## Output layout and metadata

Each plan receives its own directory:

```text
exports/
├── manifest.json
├── 14433/
│   ├── plan.obj
│   ├── plan.glb       # when GLB was requested
│   └── metadata.json
└── 14926/
    ├── plan.obj
    └── metadata.json
```

`metadata.json` records:

- Plan ID and success/failure state.
- Effective options and XY scale factor.
- The fixed 0.20 m plan-scale wall reference, separately from the requested
  physical wall thickness.
- Normal, restricted, and closed-boundary door counts, modified-door centers,
  and target/effective restricted opening widths.
- Geometry-variation seed and eligible/applied counts for each effect.
- Source wall depth and source center.
- Metric model dimensions.
- Vertex/face counts and watertight status per component.
- Repairs, skipped geometry, and other warnings.
- Paths to generated artifacts.

The root `manifest.json` records the complete batch selection, options,
timestamps, successes, failures, warnings, and artifact paths.

## Local web viewer

Launch the interactive viewer from the repository root:

```powershell
python -m streamlit run streamlit_app.py
```

If the installed script is on `PATH`, this is equivalent:

```powershell
resplan-viewer
```

Streamlit prints the local URL, normally:

```text
http://localhost:8501
```

Use a different port:

```powershell
python -m streamlit run streamlit_app.py --server.port 8502
```

To listen on all interfaces for a LAN or container deployment:

```powershell
python -m streamlit run streamlit_app.py `
  --server.address 0.0.0.0 `
  --server.port 8501
```

Do not expose the development server directly to an untrusted network without
appropriate authentication, TLS, and network controls.

### Viewer controls

The plan selector remains visible at the top. IDs are sorted numerically, its
drop-down has a persistent draggable scrollbar, and **Previous**/**Next**
buttons provide stepwise navigation. Less common controls are grouped into
collapsible sidebar panels:

- **Dataset** changes the local pickle path and, when that file is absent, the
  verified download URL.
- **Structure** contains wall/floor dimensions, balcony floor, and ceiling.
- **Doors and windows** contains opening treatment and heights.
- **Robotics access** contains boundary closure and restricted-door controls.
- **Geometry Lab** contains the two corner shares, two wall shares, their strengths,
  the shared seed, and **Randomize geometry**.
- **Preview** controls ceiling visibility without changing downloads.

The main view displays the original 2D vector layers beside an orbitable 3D
model. Buttons download the current model as OBJ or GLB.

Door height is disabled when doors are open to wall height. Window sill and
head controls are disabled when windows are replaced with solid wall. The
restriction-method selector offers **Narrow width**, **Lower height**, and
**Width and height**. Dimension controls that do not apply to the selected
method are disabled. The restricted-seed field and **Randomize restricted
doors** button choose a different reproducible subset. If every eligible door
is already restricted, the viewer explains that changing the seed cannot alter
the selection. Geometry strength fields are disabled while their percentage is
zero, and its
randomizer is disabled until at least one effect is active. The downloaded
model always uses the settings currently selected in the viewer.

## Python API

The same engine can be used without either frontend:

```python
from resplan_extruder import (
    ExtrusionOptions,
    export_plan,
    extrude_plan,
    load_dataset,
)

plans = load_dataset("ResPlan.pkl")
plan = next(plan for plan in plans if plan["id"] == 14433)

options = ExtrusionOptions(
    wall_height=3.2,
    ceiling=True,
    door_mode="full-height",
    close_boundary_doors=True,
    restricted_door_count=2,
    restricted_door_mode="both",
    restricted_door_width=0.35,
    restricted_door_height=0.65,
    restricted_door_seed=17,
    window_mode="solid",
    diagonal_corner_percent=20,
    noisy_wall_percent=25,
    geometry_seed=73,
)
result = extrude_plan(plan, options)
artifacts = export_plan(result, "exports", formats="both")
print(artifacts)
```

`ExtrusionOptions` supports the same geometry settings documented for the CLI.
`extrude_plan` returns semantic meshes and metadata without writing files.
`export_plan` writes the chosen formats and sidecar.

## Web deployment

The current viewer is a dynamic Python/Streamlit application. GitHub Pages only
hosts static sites and cannot run it directly.

Two deployment approaches are suitable:

1. Connect a GitHub repository to Streamlit Community Cloud. It installs
   `requirements.txt`, runs the viewer, and redeploys when the repository
   changes. GitHub Actions is not required for this route.
2. Add a Docker image and use GitHub Actions to test, build, and deploy it to a
   container host such as Cloud Run, Azure Web Apps, AWS, or another service
   that can run Streamlit.

`ResPlan.pkl` is approximately 300 MB, above GitHub's normal 100 MiB per-file
limit. It therefore remains excluded by `.gitignore`. The viewer can download
the file at startup, verify its SHA-256, and cache it in the Streamlit runtime.

### Streamlit Community Cloud with a GitHub Release

The simplest public deployment is a GitHub Release asset. Release assets are
separate from the Git repository's 100 MiB file limit.

1. In the GitHub repository, create a release and attach the trusted local
   `ResPlan.pkl` file. Keep the asset name exactly `ResPlan.pkl`.
2. Push this code and deploy `streamlit_app.py` in Streamlit
   Community Cloud. `requirements.txt` contains Python 3.14-compatible binary
   dependencies.
3. On first startup, the viewer uses this stable latest-release URL:

   ```text
   https://github.com/AlfonsoLRz/ResPlan-viewer/releases/latest/download/ResPlan.pkl
   ```

   It verifies the download against the built-in SHA-256:

   ```text
   2a73179cf11e6066384400494683072eb1648ed56ae000750d0e9f3fa499c570
   ```

Do not publish the dataset until its redistribution license has been checked.
The code license and dataset license are separate concerns.

### Custom storage or Kaggle

Set these values in the Streamlit app's **Settings > Secrets** to override the
default release asset without changing code:

```toml
RESPLAN_DATA_URL = "https://example.org/path/ResPlan.pkl"
RESPLAN_DATA_SHA256 = "2a73179cf11e6066384400494683072eb1648ed56ae000750d0e9f3fa499c570"
```

The URL must return the pickle bytes directly. A normal Kaggle dataset page or
login redirect is not sufficient. A private Kaggle download would require an
authenticated downloader and Kaggle credentials stored as Streamlit secrets;
for a public app, a public GitHub Release asset or object-storage URL is much
simpler and avoids distributing account credentials.

The cached file is ephemeral: Streamlit may download it again after an app
restart or migration. Existing local dataset files are never overwritten, and
a failed or mismatched download is removed before unpickling.

Useful references:

- [Streamlit Community Cloud deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [GitHub Actions deployments](https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/control-deployments)
- [GitHub large-file limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files)
- [GitHub release assets](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)

## Troubleshooting

### Command not found

Use the module form if `resplan-extrude` or `resplan-viewer` is not on `PATH`:

```powershell
python -m resplan_extruder.cli --help
python -m streamlit run streamlit_app.py
```

### Dataset not found

Run commands from the repository root or provide an explicit path:

```powershell
python -m resplan_extruder.cli `
  --data C:\datasets\ResPlan.pkl `
  --ids 14433
```

In the viewer, change **Dataset path** in the sidebar.

### Invalid height combination

For `lintel` doors and `opening` windows, keep the configured heights below the
wall height. If those structures are not needed, use `--door-mode full-height`
or `--window-mode solid`.

When height restriction is enabled, its height must be below wall height. A
width restriction must be greater than zero. The count is safely clamped when
a plan has fewer eligible interior doors.

### Geometry warnings

The exporter repairs common invalid Shapely polygons and microscopic
non-manifold pinch points. Review `metadata.json` or `manifest.json` for the
exact repair. A skipped or failed plan is not silently treated as successful.
