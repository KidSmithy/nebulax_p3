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
  const leftLeafRef = useRef<THREE.Mesh>(null);
  const rightLeafRef = useRef<THREE.Mesh>(null);

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
      {/* Door frame surround */}
      <mesh position={[0, 0.95, 0]}>
        <boxGeometry args={[1.25, 0.08, 0.08]} />
        <meshStandardMaterial color="#1e293b" metalness={0.8} roughness={0.3} />
      </mesh>
      <mesh position={[-0.60, 0, 0]}>
        <boxGeometry args={[0.08, 1.9, 0.08]} />
        <meshStandardMaterial color="#1e293b" metalness={0.8} roughness={0.3} />
      </mesh>
      <mesh position={[0.60, 0, 0]}>
        <boxGeometry args={[0.08, 1.9, 0.08]} />
        <meshStandardMaterial color="#1e293b" metalness={0.8} roughness={0.3} />
      </mesh>

      {/* Left Leaf */}
      <mesh ref={leftLeafRef} position={[-0.28, 0, 0]}>
        <boxGeometry args={[0.54, 1.82, 0.04]} />
        <meshStandardMaterial
          color={isAnomalous ? '#dc2626' : '#94a3b8'}
          metalness={0.7}
          roughness={0.3}
        />
        {/* Glass window */}
        <mesh position={[0, 0.25, 0.01]}>
          <boxGeometry args={[0.34, 0.70, 0.03]} />
          <meshPhysicalMaterial
            color="#0284c7"
            transmission={0.7}
            opacity={0.8}
            transparent
            roughness={0.1}
          />
        </mesh>
      </mesh>

      {/* Right Leaf */}
      <mesh ref={rightLeafRef} position={[0.28, 0, 0]}>
        <boxGeometry args={[0.54, 1.82, 0.04]} />
        <meshStandardMaterial
          color={isAnomalous ? '#dc2626' : '#94a3b8'}
          metalness={0.7}
          roughness={0.3}
        />
        {/* Glass window */}
        <mesh position={[0, 0.25, 0.01]}>
          <boxGeometry args={[0.34, 0.70, 0.03]} />
          <meshPhysicalMaterial
            color="#0284c7"
            transmission={0.7}
            opacity={0.8}
            transparent
            roughness={0.1}
          />
        </mesh>
      </mesh>

      {/* Kinematic Ghost Mesh Baseline */}
      <GhostMesh
        nominalOpenDist={nominalSlide}
        visible={isAnomalous}
      />
    </group>
  );
};
