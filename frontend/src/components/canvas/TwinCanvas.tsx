import React, { useRef, useEffect } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera } from '@react-three/drei';
import * as THREE from 'three';
import { useTwinStore } from '../../store/useTwinStore';
import { TrainAssembly } from './TrainAssembly';
import { TrackCorridor } from './TrackCorridor';
import { carShiftZ } from './consist';
import { LINES } from '../../lib/lines';

type Pose = { pos: THREE.Vector3; look: THREE.Vector3 };

/** Close-up on the selected component, authored against Car 3. */
function microPose(subsystem: string): Pose {
  if (subsystem === 'door') return { pos: new THREE.Vector3(5.1, 2.1, 5.4), look: new THREE.Vector3(1.59, 1.1, 2.95) };
  if (subsystem === 'shm') return { pos: new THREE.Vector3(4.6, 1.4, 10.5), look: new THREE.Vector3(0, 0.43, 7.5) };
  if (subsystem === 'acv') return { pos: new THREE.Vector3(3.5, 5.8, -3.0), look: new THREE.Vector3(0, 3.49, -4.95) };
  return { pos: new THREE.Vector3(4.5, 1.8, 3.0), look: new THREE.Vector3(0, 1.2, 0) };
}

/** One whole car, authored against Car 3. */
const MESO_POSE: Pose = { pos: new THREE.Vector3(15.0, 6.0, 13.0), look: new THREE.Vector3(0, 1.5, 2.2) };

/**
 * All 8 cars (world Z -57.5..126.5) from a 36-degree side elevation. Distance
 * is solved from the aspect ratio so the ~184 m train fills the ~65% of the
 * width the left rail leaves free; the look target is nudged past the
 * train's midpoint so it sits in that free region.
 */
function macroPose(aspect: number): Pose {
  const halfWidth = 145;
  const dist = Math.min(260, Math.max(110, halfWidth / (Math.tan(THREE.MathUtils.degToRad(22.5)) * aspect)));
  return {
    pos: new THREE.Vector3(dist * Math.cos(Math.PI / 5), dist * Math.sin(Math.PI / 5), 70.0),
    look: new THREE.Vector3(0, 0, 70.0),
  };
}

/**
 * The camera is locked to a single dolly axis driven by `cameraZoom`
 * (0 = close on the component, 0.5 = one car, 1 = all 8 cars): no orbiting or
 * panning, just zooming in and out along keyframed poses.
 */
const CameraController: React.FC = () => {
  const cameraZoom = useTwinStore((state) => state.cameraZoom);
  const selectedSubsystem = useTwinStore((state) => state.selectedSubsystem);
  const monitoredCar = useTwinStore((state) => state.monitoredCar);
  const conductorState = useTwinStore((state) => state.conductorState);
  const controlsRef = useRef<any>(null);
  const aspect = useThree((s) => s.size.width / s.size.height);

  const targetPos = useRef(new THREE.Vector3(15.0, 6.0, 13.0));
  const targetLookAt = useRef(new THREE.Vector3(0, 1.2, 0));

  useEffect(() => {
    const shift = carShiftZ(monitoredCar);
    const micro = microPose(selectedSubsystem);
    const meso = MESO_POSE;
    const macro = macroPose(aspect);

    const pos = new THREE.Vector3();
    const look = new THREE.Vector3();
    if (cameraZoom <= 0.5) {
      const t = cameraZoom / 0.5;
      pos.copy(micro.pos).lerp(meso.pos, t);
      look.copy(micro.look).lerp(meso.look, t);
      pos.z += shift;
      look.z += shift;
    } else {
      const t = (cameraZoom - 0.5) / 0.5;
      const mesoPos = meso.pos.clone().setZ(meso.pos.z + shift);
      const mesoLook = meso.look.clone().setZ(meso.look.z + shift);
      pos.copy(mesoPos).lerp(macro.pos, t);
      look.copy(mesoLook).lerp(macro.look, t);
    }

    // The Conductor popup docks bottom-left; pan the camera+target pair right
    // by the same amount so the framing shifts without re-angling.
    if (conductorState === 'popup') {
      pos.x -= 2.5;
      look.x -= 2.5;
    }

    targetPos.current.copy(pos);
    targetLookAt.current.copy(look);
  }, [cameraZoom, selectedSubsystem, monitoredCar, conductorState, aspect]);

  useFrame((state, delta) => {
    // Fog rides with the camera so zooming out never whites the train away.
    if (state.scene.fog instanceof THREE.Fog) {
      const d = state.camera.position.length();
      state.scene.fog.near = Math.max(45, d + 60);
      state.scene.fog.far = Math.max(130, d + 220);
    }

    const t = Math.min(delta * 4.0, 1.0);
    state.camera.position.lerp(targetPos.current, t);
    if (controlsRef.current) {
      controlsRef.current.target.lerp(targetLookAt.current, t);
      controlsRef.current.update();
    }
  });

  return <OrbitControls ref={controlsRef} enableRotate={false} enablePan={false} enableZoom={false} />;
};

export const TwinCanvas: React.FC = () => {
  const setActiveInspection = useTwinStore((state) => state.setActiveInspection);
  const setCameraZoom = useTwinStore((state) => state.setCameraZoom);
  const activeLine = useTwinStore((state) => state.activeLine);

  return (
    <div
      className="w-full h-full relative"
      onWheel={(e) => setCameraZoom(useTwinStore.getState().cameraZoom + e.deltaY * 0.0006)}
    >
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
        <pointLight position={[0, 0.6, 7.5]} intensity={1.2} color={LINES[activeLine].light} distance={16} />

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
