import React, { useEffect, useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { useGLTF } from '@react-three/drei';
import * as THREE from 'three';

useGLTF.setDecoderPath('/draco/');

const MODEL_URLS = {
  cab: '/models/c151-cab.glb',
  intermediate: '/models/c151-intermediate.glb',
} as const;

export type C151Variant = keyof typeof MODEL_URLS;

interface DoorLeaf {
  node: THREE.Object3D;
  name: string;
  closedX: number;
  slideSign: number;
}

/** Every Door_*_Leaf_A/B node carries closedPosition/slideSign baked in from the glTF extras. */
function getDoorLeaves(root: THREE.Object3D): DoorLeaf[] {
  const leaves: DoorLeaf[] = [];
  root.traverse((node) => {
    const { closedPosition, slideSign } = node.userData || {};
    if (Array.isArray(closedPosition) && typeof slideSign === 'number') {
      leaves.push({ node, name: node.name, closedX: closedPosition[0], slideSign });
    }
  });
  return leaves;
}

/** Car-local (not world) positions of the nodes the HUD hooks interactivity onto. */
export interface C151Anchors {
  doorR3: [number, number, number] | null;
  doorL3: [number, number, number] | null;
  bogieFront: [number, number, number] | null;
  bogieRear: [number, number, number] | null;
  acUnit1: [number, number, number] | null;
  acUnit2: [number, number, number] | null;
}

function resolveAnchors(car: THREE.Object3D): C151Anchors {
  car.updateMatrixWorld(true);
  // Measured relative to the car root, so the result is the same whether or not the
  // parent group (position/yaw of this car in the consist) has been matrix-updated yet.
  const toCarLocal = new THREE.Matrix4().copy(car.matrixWorld).invert();
  const at = (name: string): [number, number, number] | null => {
    const node = car.getObjectByName(name);
    if (!node) return null;
    const v = new THREE.Vector3().setFromMatrixPosition(node.matrixWorld).applyMatrix4(toCarLocal);
    return [v.x, v.y, v.z];
  };
  return {
    doorR3: at('Door_R3'),
    doorL3: at('Door_L3'),
    bogieFront: at('Bogie_Front'),
    bogieRear: at('Bogie_Rear'),
    acUnit1: at('Roof_AC_1'),
    acUnit2: at('Roof_AC_2'),
  };
}

export interface C151CarProps {
  variant: C151Variant;
  position?: [number, number, number];
  yaw?: number;
  /** Every door pair cycles together, driven by the monitored door's own telemetry. */
  doorCycleState?: string;
  doorAnomalyScore?: number;
  /** 0..1 stress colour driven onto Bogie_Front only; omit to leave its stock material. */
  bogieStressIntensity?: number;
  bogieXrayOpacity?: number;
  /** See-through hull: only Body_Shell is swapped, glass/doors/bogies stay normal. */
  xrayMode?: boolean;
  /** Fires once after load with car-local anchor positions for hotspots/particles. */
  onReady?: (anchors: C151Anchors) => void;
}

const xrayMaterial = new THREE.MeshStandardMaterial({
  color: '#0284c7',
  transparent: true,
  opacity: 0.2,
  wireframe: true,
});

const ghostMaterial = new THREE.MeshBasicMaterial({
  color: '#009645',
  wireframe: true,
  transparent: true,
  opacity: 0.45,
});

/** The Door_R3 / Door_L3 pair is the one the HUD tracks; its ghost shows the healthy baseline. */
const MONITORED_DOOR_PREFIXES = ['Door_R3_Leaf_', 'Door_L3_Leaf_'];

const feaUniforms = () => ({
  u_stress_intensity: { value: 0.2 },
  u_hotspot_coords: { value: new THREE.Vector3(0, 0.15, 0) },
  u_time: { value: 0 },
  u_xray_opacity: { value: 1.0 },
});

const feaVertexShader = `
  varying vec3 vPosition;
  varying vec3 vNormal;
  void main() {
    vPosition = position;
    vNormal = normalize(normalMatrix * normal);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const feaFragmentShader = `
  uniform float u_stress_intensity;
  uniform vec3 u_hotspot_coords;
  uniform float u_time;
  uniform float u_xray_opacity;
  varying vec3 vPosition;
  varying vec3 vNormal;

  vec3 getStressColor(float val) {
    vec3 cNavy = vec3(0.118, 0.227, 0.541);
    vec3 cEmerald = vec3(0.020, 0.588, 0.412);
    vec3 cAmber = vec3(0.851, 0.467, 0.024);
    vec3 cCrimson = vec3(0.882, 0.114, 0.282);
    if (val < 0.33) return mix(cNavy, cEmerald, val / 0.33);
    if (val < 0.66) return mix(cEmerald, cAmber, (val - 0.33) / 0.33);
    return mix(cAmber, cCrimson, (val - 0.66) / 0.34);
  }

  void main() {
    float dist = length(vPosition - u_hotspot_coords);
    float hotspot = exp(-dist * 1.6) * u_stress_intensity;
    float totalStress = clamp(0.12 + hotspot * 1.4 + 0.04 * sin(u_time * 8.0), 0.0, 1.0);
    vec3 baseColor = getStressColor(totalStress);
    vec3 lightDir = normalize(vec3(1.0, 2.0, 1.5));
    float diff = max(dot(vNormal, lightDir), 0.25);
    gl_FragColor = vec4(baseColor * (diff * 0.7 + 0.4), u_xray_opacity);
  }
`;

export const C151Car: React.FC<C151CarProps> = ({
  variant,
  position = [0, 0, 0],
  yaw = 0,
  doorCycleState = 'CLOSED_LOCKED',
  doorAnomalyScore = 0,
  bogieStressIntensity,
  bogieXrayOpacity = 1.0,
  xrayMode = false,
  onReady,
}) => {
  const { scene } = useGLTF(MODEL_URLS[variant]) as unknown as { scene: THREE.Group };
  const car = useMemo(() => scene.clone(true), [scene]);
  const doors = useMemo(() => getDoorLeaves(car), [car]);
  const ghostLeaves = useRef<DoorLeaf[]>([]);
  const animProgress = useRef(0);
  const nominalProgress = useRef(0);
  const bodyShellOriginalMaterials = useRef<Map<THREE.Mesh, THREE.Material | THREE.Material[]>>(new Map());
  const feaMaterial = useMemo(
    () =>
      new THREE.ShaderMaterial({
        uniforms: feaUniforms(),
        vertexShader: feaVertexShader,
        fragmentShader: feaFragmentShader,
        transparent: true,
      }),
    []
  );
  const readyFired = useRef(false);

  useEffect(() => {
    car.traverse((node) => {
      node.castShadow = true;
      node.receiveShadow = true;
    });

    if (bogieStressIntensity !== undefined) {
      const bogieFront = car.getObjectByName('Bogie_Front') as THREE.Mesh | undefined;
      if (bogieFront && bogieFront.isMesh) bogieFront.material = feaMaterial;
    }

    const bodyShell = car.getObjectByName('Body_Shell');
    if (bodyShell) {
      bodyShell.traverse((n) => {
        if ((n as THREE.Mesh).isMesh) {
          bodyShellOriginalMaterials.current.set(n as THREE.Mesh, (n as THREE.Mesh).material);
        }
      });
    }

    // Ghost baseline: clone the monitored door's leaves into the same parent frame,
    // recolour them as translucent wireframe, and drive them at the healthy (nominal) fraction.
    const ghosts: DoorLeaf[] = [];
    for (const leaf of doors) {
      if (!MONITORED_DOOR_PREFIXES.some((p) => leaf.name.startsWith(p))) continue;
      if (!leaf.node.parent) continue;
      const clone = leaf.node.clone(true);
      clone.traverse((n) => {
        if ((n as THREE.Mesh).isMesh) (n as THREE.Mesh).material = ghostMaterial;
      });
      clone.visible = false;
      leaf.node.parent.add(clone);
      ghosts.push({ node: clone, name: `${leaf.name}_ghost`, closedX: leaf.closedX, slideSign: leaf.slideSign });
    }
    ghostLeaves.current = ghosts;

    if (!readyFired.current) {
      readyFired.current = true;
      onReady?.(resolveAnchors(car));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [car]);

  useEffect(() => {
    for (const [mesh, original] of bodyShellOriginalMaterials.current) {
      mesh.material = xrayMode ? xrayMaterial : original;
    }
  }, [car, xrayMode]);

  useFrame((state, delta) => {
    const target = doorCycleState === 'OPENING' || doorCycleState === 'DWELL_OPEN' ? 1 : 0;

    // Nominal (healthy) speed, used for the ghost baseline.
    nominalProgress.current = THREE.MathUtils.lerp(nominalProgress.current, target, delta * 2.0);

    // Actual speed, slowed by friction when anomalous.
    const frictionFactor =
      doorAnomalyScore > 0.5 ? Math.max(0.4, 1.0 - doorAnomalyScore * 0.6) : 1.0;
    animProgress.current = THREE.MathUtils.lerp(animProgress.current, target, delta * 2.0 * frictionFactor);

    let jitter = 0;
    if (doorAnomalyScore > 0.5 && animProgress.current > 0.05 && animProgress.current < 0.95) {
      jitter = Math.sin(state.clock.elapsedTime * 35.0) * 0.015 * doorAnomalyScore;
    }
    const actualFraction = THREE.MathUtils.clamp(animProgress.current + jitter, 0, 1);

    for (const { node, closedX, slideSign } of doors) {
      node.position.x = closedX + slideSign * 0.77 * actualFraction;
    }

    const isAnomalous = doorAnomalyScore > 0.4;
    for (const { node, closedX, slideSign } of ghostLeaves.current) {
      node.visible = isAnomalous;
      node.position.x = closedX + slideSign * 0.77 * nominalProgress.current;
    }

    if (bogieStressIntensity !== undefined) {
      feaMaterial.uniforms.u_stress_intensity.value = bogieStressIntensity;
      feaMaterial.uniforms.u_time.value = state.clock.elapsedTime;
      feaMaterial.uniforms.u_xray_opacity.value = bogieXrayOpacity;
    }
  });

  return (
    <group position={position} rotation={[0, yaw, 0]}>
      <primitive object={car} dispose={null} />
    </group>
  );
};

useGLTF.preload(MODEL_URLS.cab);
useGLTF.preload(MODEL_URLS.intermediate);
