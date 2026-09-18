import React, { useRef, useEffect } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera } from '@react-three/drei';
import * as THREE from 'three';
import { useTwinStore } from '../../store/useTwinStore';
import { TrainAssembly } from './TrainAssembly';
import { TrackCorridor } from './TrackCorridor';

const CameraController: React.FC = () => {
  const cameraMode = useTwinStore((state) => state.cameraMode);
  const selectedSubsystem = useTwinStore((state) => state.selectedSubsystem);
  const conductorState = useTwinStore((state) => state.conductorState);
  const controlsRef = useRef<any>(null);
  const aspect = useThree((s) => s.size.width / s.size.height);

  const isTransitioning = useRef(false);
  const isUserInteracting = useRef(false);
  const targetPos = useRef(new THREE.Vector3(15.0, 6.0, 13.0));
  const targetLookAt = useRef(new THREE.Vector3(0, 1.2, 0));

  // Real C151 scale: 23m-long car recentred on the origin (see TrainAssembly's
  // CAR_OFFSET), so Bogie_Front/Rear and the roof AC units sit at these world Z's.
  // Trigger camera transition ONLY when cameraMode or selectedSubsystem explicitly changes
  useEffect(() => {
    const pos = new THREE.Vector3(15.0, 6.0, 13.0);
    const look = new THREE.Vector3(0, 1.2, 0);

    if (cameraMode === 'macro') {
      // All 8 cars (world Z -57.5..126.5) from a 36-degree side elevation.
      // Distance is solved from the aspect ratio so the ~184 m train fills the
      // ~65% of the width the left rail leaves free; the look target is nudged
      // past the train's midpoint so it sits in that free region.
      const halfWidth = 145;
      const dist = Math.min(260, Math.max(110, halfWidth / (Math.tan(THREE.MathUtils.degToRad(22.5)) * aspect)));
      look.set(0, 0, 70.0);
      pos.set(dist * Math.cos(Math.PI / 5), dist * Math.sin(Math.PI / 5), 70.0);
    } else if (cameraMode === 'meso') {
      pos.set(15.0, 6.0, 13.0);
      look.set(0, 1.5, 2.2);
    } else if (cameraMode === 'micro') {
      // Anchored to the real C151 model's resolved node positions (see TrainAssembly).
      if (selectedSubsystem === 'door') {
        pos.set(5.1, 2.1, 5.4);
        look.set(1.59, 1.1, 2.95);
      } else if (selectedSubsystem === 'shm') {
        pos.set(4.6, 1.4, 10.5);
        look.set(0, 0.43, 7.5);
      } else if (selectedSubsystem === 'acv') {
        pos.set(3.5, 5.8, -3.0);
        look.set(0, 3.49, -4.95);
      } else {
        pos.set(4.5, 1.8, 3.0);
        look.set(0, 1.2, 0);
      }
    }

    // The Conductor popup docks bottom-left; pan the whole camera+target pair
    // right by the same amount so the framing shifts without re-angling,
    // and the car never sits behind the popup.
    if (conductorState === 'popup') {
      pos.x -= 2.5;
      look.x -= 2.5;
    }

    targetPos.current.copy(pos);
    targetLookAt.current.copy(look);
    isTransitioning.current = true;
  }, [cameraMode, selectedSubsystem, conductorState, aspect]);

  // Listen to user interaction on OrbitControls
  useEffect(() => {
    const controls = controlsRef.current;
    if (!controls) return;

    const onStart = () => {
      // User is touching/dragging camera -> immediately surrender control to user!
      isUserInteracting.current = true;
      isTransitioning.current = false;
    };
    const onEnd = () => {
      isUserInteracting.current = false;
    };

    controls.addEventListener('start', onStart);
    controls.addEventListener('end', onEnd);
    return () => {
      controls.removeEventListener('start', onStart);
      controls.removeEventListener('end', onEnd);
    };
  }, []);

  useFrame((state, delta) => {
    // Fog rides with the camera so zooming out never whites the train away.
    if (state.scene.fog instanceof THREE.Fog) {
      const d = state.camera.position.length();
      state.scene.fog.near = Math.max(45, d + 60);
      state.scene.fog.far = Math.max(130, d + 220);
    }

    if (!isTransitioning.current || isUserInteracting.current) return;

    const posDist = state.camera.position.distanceTo(targetPos.current);
    const lookDist = controlsRef.current ? controlsRef.current.target.distanceTo(targetLookAt.current) : 0;

    // Arrived at target: complete transition and release camera to user
    if (posDist < 0.05 && lookDist < 0.05) {
      isTransitioning.current = false;
      return;
    }

    const t = Math.min(delta * 4.0, 1.0);
    state.camera.position.lerp(targetPos.current, t);
    if (controlsRef.current) {
      controlsRef.current.target.lerp(targetLookAt.current, t);
      controlsRef.current.update();
    }
  });

  return (
    <OrbitControls
      ref={controlsRef}
      enableDamping
      dampingFactor={0.08}
      maxPolarAngle={Math.PI / 2 - 0.01}
      minDistance={1.5}
      maxDistance={260.0}
    />
  );
};

export const TwinCanvas: React.FC = () => {
  const setActiveInspection = useTwinStore((state) => state.setActiveInspection);

  return (
    <div className="w-full h-full relative">
      <Canvas dpr={[1, 1.5]} onPointerMissed={() => setActiveInspection(null)}>
        <PerspectiveCamera makeDefault position={[15.0, 6.0, 13.0]} fov={45} />
        <CameraController />

        {/* Light Station/Depot Inspection Lighting */}
        <ambientLight intensity={1.1} color="#ffffff" />
        {/* Primary inspection illumination facing train */}
        <directionalLight position={[14, 16, 14]} intensity={1.4} color="#ffffff" />
        {/* Soft fill from ceiling/side */}
        <directionalLight position={[-10, 14, -8]} intensity={0.7} color="#e2e8f0" />
        {/* Bogie & track inspection soft fill */}
        <pointLight position={[0, 0.6, 7.5]} intensity={1.2} color="#f87171" distance={16} />

        {/* Crisp Light Atmospheric Fog */}
        <fog attach="fog" args={['#f1f5f9', 45, 130]} />

        {/* 3D Scene Objects */}
        <group position={[0, 0, 0]}>
          <TrainAssembly />
          <TrackCorridor />
        </group>
      </Canvas>
    </div>
  );
};
