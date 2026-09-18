import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { useTwinStore } from '../../../store/useTwinStore';
import { GhostMesh } from './GhostMesh';

interface DoorAssemblyProps {
  position?: [number, number, number];
  rotation?: [number, number, number];
}

export const DoorAssembly: React.FC<DoorAssemblyProps> = ({
  position = [1.26, 0.1, 1.2],
  rotation = [0, 0, 0],
}) => {
  const leftLeafRef = useRef<THREE.Group>(null);
  const rightLeafRef = useRef<THREE.Group>(null);

  const currentFrame = useTwinStore((state) => state.currentFrame);
  const doorTelemetry = currentFrame?.subsystems?.door;

  // Track animation progress (0.0 to 1.0)
  const animProgress = useRef(0.0);
  const nominalProgress = useRef(0.0);

  useFrame((state, delta) => {
    if (!doorTelemetry) return;

    const { cycle_state, anomaly_score, ghost_deviation_mm } = doorTelemetry;

    let target = 0.0;
    if (cycle_state === 'OPENING' || cycle_state === 'DWELL_OPEN') {
      target = 1.0;
    } else if (cycle_state === 'CLOSING' || cycle_state === 'CLOSED_LOCKED') {
      target = 0.0;
    }

    // Nominal speed
    nominalProgress.current = THREE.MathUtils.lerp(nominalProgress.current, target, delta * 2.0);

    // Actual speed (delayed by friction if anomalous)
    const frictionFactor = anomaly_score > 0.5 ? Math.max(0.4, 1.0 - anomaly_score * 0.6) : 1.0;
    animProgress.current = THREE.MathUtils.lerp(animProgress.current, target, delta * 2.0 * frictionFactor);

    // Add mechanical jitter if anomalous
    let jitter = 0.0;
    if (anomaly_score > 0.5 && animProgress.current > 0.05 && animProgress.current < 0.95) {
      jitter = (Math.sin(state.clock.elapsedTime * 35.0) * 0.015) * anomaly_score;
    }

    const stroke = 0.50; // Max lateral stroke in meters
    const currentSlide = (animProgress.current * stroke) + jitter;

    if (leftLeafRef.current) {
      leftLeafRef.current.position.x = -0.28 - currentSlide;
    }
    if (rightLeafRef.current) {
      rightLeafRef.current.position.x = 0.28 + currentSlide;
    }
  });

  const anomalyScore = doorTelemetry?.anomaly_score || 0.0;
  const isAnomalous = anomalyScore > 0.4;
  const stroke = 0.50;
  const nominalSlide = nominalProgress.current * stroke;

  return (
    <group position={position} rotation={rotation}>
      {/* Door frame surround / header box */}
      <mesh position={[0, 0.96, 0]}>
        <boxGeometry args={[1.28, 0.12, 0.09]} />
        <meshStandardMaterial color="#1e293b" metalness={0.8} roughness={0.3} />
      </mesh>
      {/* SMRT Header Warning Indicator Strip */}
      <mesh position={[0, 0.98, 0.05]}>
        <boxGeometry args={[0.5, 0.03, 0.01]} />
        <meshStandardMaterial
          color={isAnomalous ? '#ED1C24' : '#009645'}
          emissive={isAnomalous ? '#ED1C24' : '#009645'}
          emissiveIntensity={1.2}
        />
      </mesh>

      {/* Left Frame Post */}
      <mesh position={[-0.62, 0, 0]}>
        <boxGeometry args={[0.08, 1.95, 0.08]} />
        <meshStandardMaterial color="#1e293b" metalness={0.8} roughness={0.3} />
      </mesh>
      {/* Right Frame Post */}
      <mesh position={[0.62, 0, 0]}>
        <boxGeometry args={[0.08, 1.95, 0.08]} />
        <meshStandardMaterial color="#1e293b" metalness={0.8} roughness={0.3} />
      </mesh>

      {/* --- Left Door Leaf --- */}
      <group ref={leftLeafRef as any} position={[-0.28, 0, 0]}>
        {/* Leaf Stile Frame Outer */}
        <mesh position={[0, 0, 0]}>
          <boxGeometry args={[0.54, 1.84, 0.04]} />
          <meshStandardMaterial
            color={isAnomalous ? '#dc2626' : '#1e293b'}
            metalness={0.8}
            roughness={0.25}
          />
        </mesh>

        {/* Large Transparent Safety Glass Pane */}
        <mesh position={[0, 0.22, 0.005]}>
          <boxGeometry args={[0.44, 1.15, 0.035]} />
          <meshPhysicalMaterial
            color="#e2e8f0"
            transmission={0.82}
            opacity={0.85}
            transparent
            roughness={0.08}
            ior={1.5}
          />
        </mesh>

        {/* Lower Stainless Steel Kickplate */}
        <mesh position={[0, -0.62, 0.01]}>
          <boxGeometry args={[0.46, 0.52, 0.025]} />
          <meshStandardMaterial color="#94a3b8" metalness={0.85} roughness={0.2} />
        </mesh>

        {/* Middle Eye-Level Chevron Warning Band (Singapore MRT Standard) */}
        <mesh position={[0, 0.12, 0.02]}>
          <boxGeometry args={[0.44, 0.18, 0.01]} />
          <meshStandardMaterial
            color="#ffffff"
            transparent
            opacity={0.45}
            roughness={0.9}
          />
        </mesh>

        {/* Vertical Rubber Safety Edge */}
        <mesh position={[0.26, 0, 0]}>
          <boxGeometry args={[0.02, 1.84, 0.045]} />
          <meshStandardMaterial color="#090d16" roughness={0.95} />
        </mesh>
      </group>

      {/* --- Right Door Leaf --- */}
      <group ref={rightLeafRef as any} position={[0.28, 0, 0]}>
        {/* Leaf Stile Frame Outer */}
        <mesh position={[0, 0, 0]}>
          <boxGeometry args={[0.54, 1.84, 0.04]} />
          <meshStandardMaterial
            color={isAnomalous ? '#dc2626' : '#1e293b'}
            metalness={0.8}
            roughness={0.25}
          />
        </mesh>

        {/* Large Transparent Safety Glass Pane */}
        <mesh position={[0, 0.22, 0.005]}>
          <boxGeometry args={[0.44, 1.15, 0.035]} />
          <meshPhysicalMaterial
            color="#e2e8f0"
            transmission={0.82}
            opacity={0.85}
            transparent
            roughness={0.08}
            ior={1.5}
          />
        </mesh>

        {/* Lower Stainless Steel Kickplate */}
        <mesh position={[0, -0.62, 0.01]}>
          <boxGeometry args={[0.46, 0.52, 0.025]} />
          <meshStandardMaterial color="#94a3b8" metalness={0.85} roughness={0.2} />
        </mesh>

        {/* Middle Eye-Level Chevron Warning Band (Singapore MRT Standard) */}
        <mesh position={[0, 0.12, 0.02]}>
          <boxGeometry args={[0.44, 0.18, 0.01]} />
          <meshStandardMaterial
            color="#ffffff"
            transparent
            opacity={0.45}
            roughness={0.9}
          />
        </mesh>

        {/* Vertical Rubber Safety Edge */}
        <mesh position={[-0.26, 0, 0]}>
          <boxGeometry args={[0.02, 1.84, 0.045]} />
          <meshStandardMaterial color="#090d16" roughness={0.95} />
        </mesh>
      </group>

      {/* Kinematic Ghost Mesh Baseline */}
      <GhostMesh
        nominalOpenDist={nominalSlide}
        visible={isAnomalous}
      />
    </group>
  );
};
