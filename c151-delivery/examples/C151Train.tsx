import { Suspense, useEffect, useMemo } from 'react';
import { Canvas, useFrame, useLoader } from '@react-three/fiber';
import { MathUtils, Object3D } from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/examples/jsm/loaders/DRACOLoader.js';

// Copy assets/*.glb to your application's public/assets directory.
// Copy node_modules/three/examples/jsm/libs/draco/gltf/* to public/draco/.
const modelURLs = {
  cab: '/assets/c151-cab.glb',
  intermediate: '/assets/c151-intermediate.glb',
};

type CarProps = {
  variant: 'cab' | 'intermediate';
  position?: [number, number, number];
  yaw?: number;
  /** 0 = closed, 1 = fully open. Each leaf moves 0.77 m. */
  doorOpen?: number;
  /** Optional independent bogie steering, in radians. */
  bogieFrontYaw?: number;
  bogieRearYaw?: number;
};

type DoorLeaf = { node: Object3D; closedX: number; slideSign: number };

/** The eight Door_L/R1…4 nodes identify pairs; their two leaves slide apart. */
export function getDoorLeaves(root: Object3D): DoorLeaf[] {
  const leaves: DoorLeaf[] = [];
  root.traverse((node) => {
    const { closedPosition, slideSign } = node.userData;
    if (Array.isArray(closedPosition) && typeof slideSign === 'number') {
      leaves.push({ node, closedX: closedPosition[0], slideSign });
    }
  });
  return leaves;
}

export function C151Car({
  variant,
  position = [0, 0, 0],
  yaw = 0,
  doorOpen = 0,
  bogieFrontYaw = 0,
  bogieRearYaw = 0,
}: CarProps) {
  const gltf = useLoader(GLTFLoader, modelURLs[variant], (loader) => {
    const draco = new DRACOLoader();
    draco.setDecoderPath('/draco/');
    loader.setDRACOLoader(draco);
  });

  // Clone transforms for every instance. Geometry/materials stay shared on the GPU.
  const car = useMemo(() => gltf.scene.clone(true), [gltf.scene]);
  const doors = useMemo(() => getDoorLeaves(car), [car]);
  const bogies = useMemo(() => ({
    front: car.getObjectByName('Bogie_Front'),
    rear: car.getObjectByName('Bogie_Rear'),
  }), [car]);

  useEffect(() => {
    car.traverse((node) => {
      node.castShadow = true;
      node.receiveShadow = true;
    });
  }, [car]);

  useFrame((_, delta) => {
    const amount = MathUtils.clamp(doorOpen, 0, 1);
    for (const { node, closedX, slideSign } of doors) {
      const target = closedX + slideSign * 0.77 * amount;
      node.position.x = MathUtils.damp(node.position.x, target, 10, delta);
    }
    if (bogies.front) bogies.front.rotation.y = bogieFrontYaw;
    if (bogies.rear) bogies.rear.rotation.y = bogieRearYaw;
  });

  return <group position={position} rotation={[0, yaw, 0]}>
    <primitive object={car} dispose={null} />
  </group>;
}

/** Six-car train spanning global Z = 0…138 m, with both cabs facing outward. */
export function C151Consist({ doorOpen = 0 }: { doorOpen?: number }) {
  return <group name="C151_Consist">
    <C151Car variant="cab" position={[0, 0, 23]} yaw={Math.PI} doorOpen={doorOpen} />
    {[23, 46, 69, 92].map((z) =>
      <C151Car key={z} variant="intermediate" position={[0, 0, z]} doorOpen={doorOpen} />
    )}
    <C151Car variant="cab" position={[0, 0, 115]} doorOpen={doorOpen} />
  </group>;
}

/** Minimal working canvas; add your own OrbitControls, track and environment. */
export default function C151Example() {
  return <Canvas shadows camera={{ position: [28, 12, 39], fov: 35, near: 0.1, far: 500 }}
    onCreated={({ camera }) => camera.lookAt(0, 1.5, 11.5)}>
    <color attach="background" args={['#e7e9e9']} />
    <hemisphereLight args={['#f8fcff', '#677279', 2.1]} />
    <directionalLight position={[12, 24, 28]} intensity={3} castShadow />
    <Suspense fallback={null}><C151Car variant="cab" /></Suspense>
  </Canvas>;
}
