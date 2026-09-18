# Kawasaki C151 exterior assets

Two real-scale, exterior-only carriage assets for Three.js / React Three Fiber.

| File | Triangles | File size | Dimensions (L × W × H) |
|---|---:|---:|---|
| [Cab GLB](assets/c151-cab.glb) | 5,818 | 137,404 bytes | 23 × 3.2 × 3.7 m |
| [Intermediate GLB](assets/c151-intermediate.glb) | 5,414 | 132,476 bytes | 23 × 3.2 × 3.7 m |

Each GLB embeds one 1024×1024 PNG albedo and uses exactly four glTF metallic–roughness materials: `MRT_Body`, `MRT_Glass`, `MRT_Metal`, `MRT_Light`. Draco compression is required. The intermediate retains the unused `MRT_Light` material to preserve the four-material contract. Glass uses standard alpha blending. No extra material extensions, cameras, scene lights, interiors, logos, pantographs or roof cables are exported.

## View locally

Run `start-preview.cmd`, or run this command in the extracted folder:

```console
node scripts/serve.mjs
```

Open **http://127.0.0.1:43151/preview/**. Node.js is the only requirement for the preview: the Three.js runtime and Draco decoder are bundled under `vendor`, so no package installation or CDN is needed. Cab, intermediate, six-car assembly, door opening, wireframe and camera views are available. **Save PNG** writes the current canvas to `reports/renders/` when using the included server.

Track, floor and lighting belong to the preview only. The destination display is deliberately blank and ready for your own texture artwork.

## Coordinates and assembly

- Metres, Y up, +Z forward; root rotation identity and scale one.
- Origin: rear coupling face, track centre, rail level `[0,0,0]`.
- Closed model bounds: X = −1.6…1.6, Y = 0…3.7, Z = 0…23. Draco introduces less than 0.1 mm rounding in the measured bounds.
- Bogie pivots: rear `[0,0.43,4]`, front `[0,0.43,19]`; rotate about Y.
- Both roof air-conditioning units fit inside the 3.7 m maximum height.

A six-car consist spans 138 m. For a train occupying Z=0…138, use these placements:

| Slot | Asset | Root Z | Yaw |
|---|---|---:|---:|
| 1 | Cab facing −Z | 23 | π |
| 2 | Intermediate | 23 | 0 |
| 3 | Intermediate | 46 | 0 |
| 4 | Intermediate | 69 | 0 |
| 5 | Intermediate | 92 | 0 |
| 6 | Cab facing +Z | 115 | 0 |

The reversed cab extends 23 m behind its rear-coupler origin. This is why its root coincides with the first intermediate's root at their joined coupling faces. Ordinary car placement uses a 23 m pitch; no hidden mesh offset is needed. The instance yaw is applied by your app, not baked into the exported root.

## Runtime node contract

`Body_Shell`, `Bogie_Front`, `Bogie_Rear`, `Glass_Windows`, `Glass_Windscreen`, `Roof_AC_1`, `Roof_AC_2`, and `Door_L1`…`Door_L4` / `Door_R1`…`Door_R4` are exact node names. `Headlight_L` and `Headlight_R` appear only in the cab asset and use emissive `MRT_Light`. `Glass_Windscreen` is an empty compatibility node in the intermediate.

L is negative X and R is positive X, looking forward; doors are numbered rear to front at Z=3.7, 8.7, 13.7 and 18.7. Each named door node identifies one bi-parting **pair**, with two mesh children such as `Door_L1_Leaf_A` and `Door_L1_Leaf_B`. Both leaves pivot at their own outer longitudinal edges. Door glass is part of the moving leaf and uses `MRT_Glass`; stationary side glass is in `Glass_Windows`.

All mesh-node rotations are identity and scales are one. Two intentional non-mesh coordinate frames, `DoorTrack_L` and `DoorTrack_R`, are rotated 90° about Y so that their local X points along carriage −Z. This reconciles local-X door animation with a Z-forward carriage. Do not apply or remove these runtime frames. Each pair controller and each leaf has identity rotation/scale within that frame.

Animate each leaf's `position.x` from its glTF extras (`userData` in Three.js):

```js
const { closedPosition, slideSign, stroke } = leaf.userData;
leaf.position.x = closedPosition[0] + slideSign * stroke * openFraction;
```

`stroke` is 0.77 m and `openFraction` is 0…1. The pair node is a stable grouping handle; moving only that group would move both leaves together. Use the included [Three.js guide](examples/README.md) or [React Three Fiber component](examples/C151Train.tsx) for opposite leaf motion, independent bogie steering and cloned carriage instances.

## Livery and UV editing

The red band is texture paint; there is no band geometry. Side panels and door skins share the same continuous planar UV projection, so a texture edit keeps the livery aligned on closed doors. The side strips are separate, non-overlapping atlas regions:

| Atlas region | Pixel rectangle (top-left image coordinates) |
|---|---|
| Left body side | X 16…1008, Y 16…168 |
| Right body side | X 16…1008, Y 188…340 |
| Roof | X 16…1008, Y 358…470 |
| Cab/front end | X 16…256, Y 488…728 |
| Rear/intermediate end | X 272…512, Y 488…728 |
| AC tops | X 540…992, Y 492…714 |
| Material color swatches | Y 936…1012 |

Along each side strip, U increases rear-to-front; image Y increases downward. The red stripe occupies rows 95…118 on the left and 267…290 on the right, corresponding to world Y=1.69…2.035 m. Exact paint colors: white `#F2F4F7`, red `#D5202B`, skirt `#59606B`.

Editable PNG and SVG atlases are supplied beside each GLB. To embed an edited 1024×1024 PNG without regenerating the geometry, install the build dependencies once with `npm ci`, then run:

```console
node scripts/retexture.mjs cab path/to/edited.png
```

This writes `assets/c151-cab-retextured.glb`, leaving the original untouched. Use `intermediate` for the other carriage. The SVG files are optional authoring sources; only PNG data is embedded in each delivered GLB.

## Rebuild and verification

```console
npm ci
npm run build
npm run validate
```

`scripts/build.mjs` contains the complete procedural geometry and atlas source. Regenerating replaces the original assets and atlases. `reports/validation.json` records independent checks on the actual decoded compressed files: dimensions, triangle counts, node hierarchy, local door pivots, affine side UV strips, embedded PNG colors/resolution, material names, normals/winding, end chamfers, AC sidewalls, and exact six-car coupling joins.

Both files report **zero Khronos glTF validator errors and warnings**. The validator itself does not decode Draco; the separate Draco decoding and geometric checks cover the compressed meshes. The six-car scene totals 33,292 triangles and all five coupling gaps are zero. Browser visual checks covered both carriages, front/side views, open doors and the assembled consist. The R3F component is provided as integration source; the standalone Three.js viewer was exercised directly.

Reference photos supplied in the request were used for visual proportions only. No photo pixels are embedded and the source photographs are not redistributed. The model follows the requested full-width blank display, two-piece windscreen and clean livery where those differ from photographed fleet variants.

Bundled preview software: Three.js (MIT; `vendor/three/LICENSE`) and Draco (Apache 2.0; `vendor/draco/`).
