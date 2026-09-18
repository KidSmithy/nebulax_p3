import React, { useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { useTwinStore } from '../../store/useTwinStore';
import { DoorAssembly } from './kinematics/DoorAssembly';
import { AirflowParticles } from './particles/AirflowParticles';
import { FEAStressShaderMaterial } from './shaders/FEAStressShader';
import { InspectionTag } from './InspectionTag';

export const TrainAssembly: React.FC = () => {
  const xrayMode = useTwinStore((state) => state.xrayMode);
  const currentFrame = useTwinStore((state) => state.currentFrame);
  const setActiveInspection = useTwinStore((state) => state.setActiveInspection);

  const shm = currentFrame?.subsystems?.shm;
  const stressIntensity = shm?.anomaly_score || 0.2;

  // Custom Shader Material for Bogie FEA Stress
  const feaMaterial = useMemo(() => {
    return new THREE.ShaderMaterial({
      uniforms: THREE.UniformsUtils.clone(FEAStressShaderMaterial.uniforms),
      vertexShader: FEAStressShaderMaterial.vertexShader,
      fragmentShader: FEAStressShaderMaterial.fragmentShader,
      transparent: true,
    });
  }, []);

  useFrame((state) => {
    if (feaMaterial) {
      feaMaterial.uniforms.u_stress_intensity.value = stressIntensity;
      feaMaterial.uniforms.u_time.value = state.clock.elapsedTime;
      feaMaterial.uniforms.u_xray_opacity.value = xrayMode ? 0.85 : 1.0;
    }
  });

  return (
    <group position={[0, 0.75, 0]}>
      {/* --- SMRT / LTA CAR BODY (Glossy Pearl White Body) --- */}
      <mesh position={[0, 0.55, 0]}>
        <boxGeometry args={[2.5, 2.1, 12.0]} />
        <meshStandardMaterial
          color={xrayMode ? '#0284c7' : '#f8fafc'}
          metalness={0.15}
          roughness={0.2}
          transparent={xrayMode}
          opacity={xrayMode ? 0.20 : 1.0}
          wireframe={xrayMode}
        />
      </mesh>

      {/* Curved Roof Deck Crown */}
      <mesh position={[0, 1.62, 0]}>
        <boxGeometry args={[2.3, 0.12, 11.9]} />
        <meshStandardMaterial color="#e2e8f0" metalness={0.2} roughness={0.3} />
      </mesh>

      {/* --- SMRT / LTA DUAL LIVERY STRIPES (From Uploaded R151 Photo) --- */}
      {!xrayMode && (
        <>
          {/* Top SMRT Red Accent Stripe (Runs across upper door/window frame) */}
          <mesh position={[1.265, 1.35, 0]}>
            <boxGeometry args={[0.015, 0.08, 11.9]} />
            <meshStandardMaterial color="#ED1C24" roughness={0.3} />
          </mesh>
          <mesh position={[-1.265, 1.35, 0]}>
            <boxGeometry args={[0.015, 0.08, 11.9]} />
            <meshStandardMaterial color="#ED1C24" roughness={0.3} />
          </mesh>

          {/* Lower LTA Lush Green Beltline Stripe (Runs below window ribbon) */}
          <mesh position={[1.265, 0.28, 0]}>
            <boxGeometry args={[0.015, 0.18, 11.9]} />
            <meshStandardMaterial color="#009645" roughness={0.3} />
          </mesh>
          <mesh position={[-1.265, 0.28, 0]}>
            <boxGeometry args={[0.015, 0.18, 11.9]} />
            <meshStandardMaterial color="#009645" roughness={0.3} />
          </mesh>

          {/* SMRT Circular Chevron Emblem (Beside front passenger door) */}
          <mesh position={[1.268, 0.55, 4.2]} rotation={[0, 0, Math.PI / 2]}>
            <cylinderGeometry args={[0.16, 0.16, 0.01, 24]} />
            <meshStandardMaterial color="#ED1C24" roughness={0.3} />
          </mesh>
        </>
      )}

      {/* Continuous Tinted Window Band */}
      <mesh position={[1.26, 0.85, 0]}>
        <boxGeometry args={[0.02, 0.85, 11.8]} />
        <meshStandardMaterial
          color="#080c14"
          metalness={0.9}
          roughness={0.1}
          transparent
          opacity={0.92}
        />
      </mesh>
      <mesh position={[-1.26, 0.85, 0]}>
        <boxGeometry args={[0.02, 0.85, 11.8]} />
        <meshStandardMaterial
          color="#080c14"
          metalness={0.9}
          roughness={0.1}
          transparent
          opacity={0.92}
        />
      </mesh>

      {/* Undercarriage Equipment Skirts (Center bay) */}
      <mesh position={[1.22, -0.32, 0]}>
        <boxGeometry args={[0.04, 0.35, 6.8]} />
        <meshStandardMaterial color="#475569" metalness={0.6} roughness={0.4} />
      </mesh>
      <mesh position={[-1.22, -0.32, 0]}>
        <boxGeometry args={[0.04, 0.35, 6.8]} />
        <meshStandardMaterial color="#475569" metalness={0.6} roughness={0.4} />
      </mesh>

      {/* --- AERODYNAMIC CAB NOSE (Modern Curved R151 Design) --- */}
      <group position={[0, 0, 6.0]}>
        {/* Sloping Pearl White Cab Nose Shell */}
        <mesh position={[0, 0.5, 0.6]} rotation={[0.22, 0, 0]}>
          <boxGeometry args={[2.46, 2.0, 1.4]} />
          <meshStandardMaterial
            color={xrayMode ? '#0284c7' : '#f8fafc'}
            metalness={0.2}
            roughness={0.2}
            transparent={xrayMode}
            opacity={xrayMode ? 0.25 : 1.0}
          />
        </mesh>
        {/* Forward Tapered Bumper Skirt */}
        <mesh position={[0, -0.22, 1.3]} rotation={[0.1, 0, 0]}>
          <boxGeometry args={[2.4, 0.6, 0.6]} />
          <meshStandardMaterial color="#e2e8f0" metalness={0.3} roughness={0.3} />
        </mesh>

        {/* Wraparound Black Glass Windscreen Canopy */}
        <mesh position={[0, 0.72, 1.15]} rotation={[0.38, 0, 0]}>
          <boxGeometry args={[2.15, 1.15, 0.08]} />
          <meshStandardMaterial
            color="#050810"
            metalness={0.95}
            roughness={0.05}
          />
        </mesh>

        {/* LTA Header Ribbon on Top Windscreen */}
        <mesh position={[0, 1.18, 0.95]} rotation={[0.38, 0, 0]}>
          <boxGeometry args={[1.4, 0.12, 0.09]} />
          <meshStandardMaterial
            color="#0f172a"
            emissive="#38bdf8"
            emissiveIntensity={0.2}
          />
        </mesh>

        {/* Slanted High-Intensity LED Headlights (Left & Right Cheeks) */}
        {/* Left Headlight */}
        <mesh position={[0.88, 0.08, 1.35]} rotation={[0.2, -0.3, 0.4]}>
          <boxGeometry args={[0.35, 0.1, 0.04]} />
          <meshStandardMaterial
            color="#ffffff"
            emissive="#ffffff"
            emissiveIntensity={2.5}
          />
        </mesh>
        {/* Right Headlight */}
        <mesh position={[-0.88, 0.08, 1.35]} rotation={[0.2, 0.3, -0.4]}>
          <boxGeometry args={[0.35, 0.1, 0.04]} />
          <meshStandardMaterial
            color="#ffffff"
            emissive="#ffffff"
            emissiveIntensity={2.5}
          />
        </mesh>

        {/* Center Automatic Scharfenberg Coupler */}
        <mesh position={[0, -0.32, 1.6]}>
          <boxGeometry args={[0.38, 0.22, 0.5]} />
          <meshStandardMaterial color="#334155" metalness={0.8} roughness={0.3} />
        </mesh>
        <mesh position={[0, -0.32, 1.88]} rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[0.06, 0.06, 0.2, 12]} />
          <meshStandardMaterial color="#1e293b" metalness={0.9} roughness={0.2} />
        </mesh>

        {/* Anti-Climber Corrugated Crash Buffers (Left & Right) */}
        {[-0.85, 0.85].map((xBuf, bIdx) => (
          <mesh key={bIdx} position={[xBuf, -0.32, 1.45]}>
            <boxGeometry args={[0.42, 0.24, 0.12]} />
            <meshStandardMaterial color="#1e293b" metalness={0.7} roughness={0.4} />
          </mesh>
        ))}
      </group>

      {/* --- PASSENGER DOOR 3R (Subsystem Kinematics & Clickable Tag) --- */}
      <group
        position={[1.26, 0.35, 1.8]}
        onClick={(e) => {
          e.stopPropagation();
          setActiveInspection('door');
        }}
        onPointerOver={(e) => {
          e.stopPropagation();
          document.body.style.cursor = 'pointer';
        }}
        onPointerOut={() => {
          document.body.style.cursor = 'auto';
        }}
      >
        <DoorAssembly position={[0, 0, 0]} />
        <InspectionTag type="door" position={[0.1, 0.6, 0]} beaconLabel="Door 3R System" />
      </group>

      {/* Opposing Door 3L */}
      <DoorAssembly position={[-1.26, 0.35, 1.8]} rotation={[0, Math.PI, 0]} />

      {/* --- ROOFTOP ACV CLIMATE PACK 1 (Clickable Tag) --- */}
      <group
        position={[0, 1.75, 0]}
        onClick={(e) => {
          e.stopPropagation();
          setActiveInspection('acv');
        }}
        onPointerOver={(e) => {
          e.stopPropagation();
          document.body.style.cursor = 'pointer';
        }}
        onPointerOut={() => {
          document.body.style.cursor = 'auto';
        }}
      >
        <mesh>
          <boxGeometry args={[1.8, 0.4, 3.2]} />
          <meshStandardMaterial
            color="#334155"
            metalness={0.6}
            roughness={0.4}
          />
        </mesh>
        {/* Exhaust condenser fans */}
        <mesh position={[0, 0.22, -0.8]} rotation={[-Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[0.45, 0.45, 0.08, 16]} />
          <meshStandardMaterial color="#0f172a" metalness={0.8} />
        </mesh>
        <mesh position={[0, 0.22, 0.8]} rotation={[-Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[0.45, 0.45, 0.08, 16]} />
          <meshStandardMaterial color="#0f172a" metalness={0.8} />
        </mesh>

        {/* Dynamic Airflow Particles */}
        <AirflowParticles position={[0, 0.25, 0]} count={120} />

        {/* Interactive ACV Stats Inspection Tag */}
        <InspectionTag type="acv" position={[0, 0.45, 0]} beaconLabel="ACV Climate Pack" />
      </group>

      {/* --- BOGIE ASSEMBLIES --- */}
      {/* Front Bogie (Equipped with FEA Stress Shader & Clickable Tag) */}
      <group
        position={[0, -0.45, 3.8]}
        onClick={(e) => {
          e.stopPropagation();
          setActiveInspection('shm');
        }}
        onPointerOver={(e) => {
          e.stopPropagation();
          document.body.style.cursor = 'pointer';
        }}
        onPointerOut={() => {
          document.body.style.cursor = 'auto';
        }}
      >
        <mesh material={feaMaterial} position={[0, 0.15, 0]}>
          <boxGeometry args={[2.1, 0.25, 2.8]} />
        </mesh>
        {/* Bolster Rib Hotspot Marker */}
        {stressIntensity > 0.6 && (
          <mesh position={[0.7, 0.28, 0.4]}>
            <sphereGeometry args={[0.12, 16, 16]} />
            <meshBasicMaterial color="#ED1C24" wireframe />
          </mesh>
        )}

        {/* Axles & Wheels */}
        {[-1.0, 1.0].map((zOffset, idx) => (
          <group key={idx} position={[0, 0, zOffset]}>
            <mesh rotation={[0, 0, Math.PI / 2]}>
              <cylinderGeometry args={[0.07, 0.07, 2.3, 12]} />
              <meshStandardMaterial color="#475569" metalness={0.9} roughness={0.2} />
            </mesh>
            <mesh position={[-1.05, 0, 0]} rotation={[0, 0, Math.PI / 2]}>
              <cylinderGeometry args={[0.42, 0.42, 0.12, 24]} />
              <meshStandardMaterial color="#0f172a" metalness={0.9} roughness={0.1} />
            </mesh>
            <mesh position={[1.05, 0, 0]} rotation={[0, 0, Math.PI / 2]}>
              <cylinderGeometry args={[0.42, 0.42, 0.12, 24]} />
              <meshStandardMaterial color="#0f172a" metalness={0.9} roughness={0.1} />
            </mesh>
          </group>
        ))}

        {/* Interactive Bogie SHM Stats Inspection Tag */}
        <InspectionTag type="shm" position={[0.8, 0.4, 0]} beaconLabel="Bogie SHM" />
      </group>

      {/* Rear Bogie */}
      <group position={[0, -0.45, -3.8]}>
        <mesh position={[0, 0.15, 0]}>
          <boxGeometry args={[2.1, 0.25, 2.8]} />
          <meshStandardMaterial color="#334155" metalness={0.7} roughness={0.3} />
        </mesh>
        {[-1.0, 1.0].map((zOffset, idx) => (
          <group key={idx} position={[0, 0, zOffset]}>
            <mesh rotation={[0, 0, Math.PI / 2]}>
              <cylinderGeometry args={[0.07, 0.07, 2.3, 12]} />
              <meshStandardMaterial color="#475569" metalness={0.9} roughness={0.2} />
            </mesh>
            <mesh position={[-1.05, 0, 0]} rotation={[0, 0, Math.PI / 2]}>
              <cylinderGeometry args={[0.42, 0.42, 0.12, 24]} />
              <meshStandardMaterial color="#0f172a" metalness={0.9} roughness={0.1} />
            </mesh>
            <mesh position={[1.05, 0, 0]} rotation={[0, 0, Math.PI / 2]}>
              <cylinderGeometry args={[0.42, 0.42, 0.12, 24]} />
              <meshStandardMaterial color="#0f172a" metalness={0.9} roughness={0.1} />
            </mesh>
          </group>
        ))}
      </group>
    </group>
  );
};
