import React, { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { useTwinStore } from '../../../store/useTwinStore';

interface AirflowParticlesProps {
  position?: [number, number, number];
  count?: number;
}

export const AirflowParticles: React.FC<AirflowParticlesProps> = ({
  position = [0, 1.45, 0],
  count = 200,
}) => {
  const pointsRef = useRef<THREE.Points>(null);
  const currentFrame = useTwinStore((state) => state.currentFrame);
  const acv = currentFrame?.subsystems?.acv;

  const compressorPower = acv?.compressor_power_kw || 4.5;
  const anomalyScore = acv?.anomaly_score || 0.1;

  // Particle positions, initial velocities, and lifetime
  const [positions, colors, velocities] = useMemo(() => {
    const pos = new Float32Array(count * 3);
    const col = new Float32Array(count * 3);
    const vel = new Float32Array(count * 3);

    const amber = new THREE.Color('#f59e0b');
    const cyan = new THREE.Color('#06b6d4');
    const crimson = new THREE.Color('#ef4444');

    for (let i = 0; i < count; i++) {
      // Bounded box around rooftop AC unit: X [-0.6, 0.6], Y [0, 0.4], Z [-1.2, 1.2]
      const t = i / count;
      pos[i * 3] = (Math.random() - 0.5) * 1.2;
      pos[i * 3 + 1] = Math.random() * 0.4;
      pos[i * 3 + 2] = (Math.random() - 0.5) * 2.4;

      vel[i * 3] = (Math.random() - 0.5) * 0.05;
      vel[i * 3 + 1] = 0.2 + Math.random() * 0.3; // Upward buoyant flow
      vel[i * 3 + 2] = -0.4 - Math.random() * 0.6; // Stream backwards along train slipstream

      // Gradient: intake warm amber -> supply cool cyan (or crimson if degraded)
      const targetColor = anomalyScore > 0.4 ? crimson : cyan;
      const c = amber.clone().lerp(targetColor, t);
      col[i * 3] = c.r;
      col[i * 3 + 1] = c.g;
      col[i * 3 + 2] = c.b;
    }

    return [pos, col, vel];
  }, [count, anomalyScore]);

  useFrame((_, delta) => {
    if (!pointsRef.current) return;
    const geo = pointsRef.current.geometry;
    const posAttr = geo.attributes.position;
    const posArr = posAttr.array as Float32Array;

    const flowSpeed = (compressorPower / 4.5) * 1.2;

    for (let i = 0; i < count; i++) {
      const idx = i * 3;
      posArr[idx] += velocities[idx] * delta * flowSpeed;
      posArr[idx + 1] += velocities[idx + 1] * delta * flowSpeed;
      posArr[idx + 2] += velocities[idx + 2] * delta * flowSpeed;

      // Reset particles when they flow out of stream bounds
      if (posArr[idx + 2] < -2.2 || posArr[idx + 1] > 1.2) {
        posArr[idx] = (Math.random() - 0.5) * 1.0;
        posArr[idx + 1] = 0.05;
        posArr[idx + 2] = 1.0 + Math.random() * 0.3;
      }
    }
    posAttr.needsUpdate = true;
  });

  return (
    <group position={position}>
      <points ref={pointsRef}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[positions, 3]}
          />
          <bufferAttribute
            attach="attributes-color"
            args={[colors, 3]}
          />
        </bufferGeometry>
        <pointsMaterial
          size={0.06}
          vertexColors
          transparent
          opacity={0.75}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
        />
      </points>
    </group>
  );
};
