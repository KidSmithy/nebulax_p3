import React, { useRef, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera } from '@react-three/drei';
import * as THREE from 'three';
import { useTwinStore } from '../../store/useTwinStore';
import { TrainAssembly } from './TrainAssembly';
import { TrackCorridor } from './TrackCorridor';

const CameraController: React.FC = () => {
  const cameraMode = useTwinStore((state) => state.cameraMode);
  const selectedSubsystem = useTwinStore((state) => state.selectedSubsystem);
  const controlsRef = useRef<any>(null);

  const isTransitioning = useRef(false);
  const isUserInteracting = useRef(false);
  const targetPos = useRef(new THREE.Vector3(7.5, 4.0, 8.5));
  const targetLookAt = useRef(new THREE.Vector3(0, 0.8, 0));

  // Trigger camera transition ONLY when cameraMode or selectedSubsystem explicitly changes
  useEffect(() => {
    const pos = new THREE.Vector3(7.5, 4.0, 8.5);
    const look = new THREE.Vector3(0, 0.8, 0);

    if (cameraMode === 'macro') {
      pos.set(16.0, 11.0, 20.0);
      look.set(0, 0, 0);
    } else if (cameraMode === 'meso') {
      pos.set(8.5, 3.6, 7.5);
      look.set(0, 0.8, 1.5);
    } else if (cameraMode === 'micro') {
      if (selectedSubsystem === 'door') {
        pos.set(3.2, 1.2, 3.2);
        look.set(1.26, 0.8, 1.8);
      } else if (selectedSubsystem === 'shm') {
        pos.set(3.5, 0.4, 5.5);
        look.set(0, 0.1, 3.8);
      } else if (selectedSubsystem === 'acv') {
        pos.set(2.8, 3.4, 1.5);
        look.set(0, 2.0, 0);
      } else {
        pos.set(3.2, 0.8, 1.5);
        look.set(0, 0.5, 0);
      }
    }

    targetPos.current.copy(pos);
    targetLookAt.current.copy(look);
    isTransitioning.current = true;
  }, [cameraMode, selectedSubsystem]);

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
      maxDistance={60.0}
    />
  );
};

export const TwinCanvas: React.FC = () => {
  const setActiveInspection = useTwinStore((state) => state.setActiveInspection);

  return (
    <div className="w-full h-full relative">
      <Canvas dpr={[1, 1.5]} onPointerMissed={() => setActiveInspection(null)}>
        <PerspectiveCamera makeDefault position={[8.5, 3.6, 7.5]} fov={45} />
        <CameraController />

        {/* Light Station/Depot Inspection Lighting */}
        <ambientLight intensity={1.1} color="#ffffff" />
        {/* Primary inspection illumination facing train */}
        <directionalLight position={[8, 9, 8]} intensity={1.4} color="#ffffff" />
        {/* Soft fill from ceiling/side */}
        <directionalLight position={[-6, 8, -4]} intensity={0.7} color="#e2e8f0" />
        {/* Bogie & track inspection soft fill */}
        <pointLight position={[0, 0.4, 3.8]} intensity={1.2} color="#f87171" distance={9} />

        {/* Crisp Light Atmospheric Fog */}
        <fog attach="fog" args={['#f1f5f9', 30, 85]} />

        {/* 3D Scene Objects */}
        <group position={[0, 0, 0]}>
          <TrainAssembly />
          <TrackCorridor />
        </group>
      </Canvas>
    </div>
  );
};
