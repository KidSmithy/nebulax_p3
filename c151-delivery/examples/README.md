# Browser integration

The two binary glTF assets use metres, Y up and Z forward. Their root transform is the rear coupling face at rail level. Each model occupies X = −1.6…1.6, Y = 0…3.7, Z = 0…23. GLB materials use the standard metallic–roughness model and the PNG albedo is embedded. Draco decoding is required.

## Three.js

```js
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';

const draco = new DRACOLoader().setDecoderPath('/draco/');
const loader = new GLTFLoader().setDRACOLoader(draco);
const { scene: car } = await loader.loadAsync('/assets/c151-cab.glb');
scene.add(car); // rail level at Y=0, rear coupler at Z=0
```

Copy `three/examples/jsm/libs/draco/gltf/` decoder files into the app's public `/draco/` directory. The preview uses the decoder directly from bundled `vendor/three` files.

## Doors

`Door_L1…Door_L4` and `Door_R1…Door_R4` identify the eight bi-parting door pairs. Each contains two independently moving leaves, named for example `Door_L1_Leaf_A` and `Door_L1_Leaf_B`. The fixed track frames map local X to the length of the carriage while keeping the door leaves' rotations and scales applied. Animate the leaf children, so the two halves move in opposite directions:

```js
function setDoors(car, fraction) {
  const t = Math.min(1, Math.max(0, fraction));
  car.traverse(node => {
    const { closedPosition, slideSign } = node.userData;
    if (Array.isArray(closedPosition) && typeof slideSign === 'number') {
      node.position.x = closedPosition[0] + slideSign * 0.77 * t;
    }
  });
}
```

The glTF extras become `userData` in Three.js. The visible door glass travels with its leaf. A leaf's `closedPosition` is in its immediate parent's local space. The pair nodes retain the outer-edge pivots.

## Six-car placement

| Car | Asset | Root Z | Root yaw |
|---|---|---:|---:|
| 1, outward cab | Cab | 23 | π |
| 2 | Intermediate | 23 | 0 |
| 3 | Intermediate | 46 | 0 |
| 4 | Intermediate | 69 | 0 |
| 5 | Intermediate | 92 | 0 |
| 6, outward cab | Cab | 115 | 0 |

The consist spans Z = 0…138. All ordinary joins use a 23 m pitch. The first car's root is at Z=23 because turning around the rear-coupler origin makes that car extend toward −Z. The application applies that assembly rotation to an instance; the exported root stays unrotated.

Clone each loaded scene before adding multiple instances. `scene.clone(true)` gives each carriage independent node transforms while sharing its geometry and materials. Search required node names within the individual car instance; every instance intentionally has the same names.

## React Three Fiber

`C151Train.tsx` exports `C151Car`, `C151Consist`, and a minimal canvas example. It requires React, `three`, `@react-three/fiber`, and the usual TypeScript types. Place the GLBs under your app's `public/assets/` and the Draco decoder under `public/draco/`. Mount the example in an element with a nonzero width and height. The `doorOpen` prop accepts 0…1; each frame eases the leaves to the requested opening. Optional `bogieFrontYaw` and `bogieRearYaw` props steer the two bogies around their own pivots.

The eight door-pair names and two bogie names are stable runtime handles. `Glass_Windows` and `Glass_Windscreen` use `MRT_Glass`; windshield and headlight nodes apply to the driving car. The source export has no cameras, scene lights, interior, pantograph, logos or operator names.
