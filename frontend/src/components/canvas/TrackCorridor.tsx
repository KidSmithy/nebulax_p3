import React, { useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { useTwinStore } from '../../store/useTwinStore';
import { CorrugationRibbonShaderMaterial } from './shaders/CorrugationRibbonShader';
import { InspectionTag } from './InspectionTag';

export const TrackCorridor: React.FC = () => {
  const currentFrame = useTwinStore((state) => state.currentFrame);
  const setActiveInspection = useTwinStore((state) => state.setActiveInspection);
  const railData = currentFrame?.subsystems?.rail_corrugation;
  const trainSpeed = currentFrame?.train_speed_kmh || 68.0;

  const severity = railData?.severity_score || 0.15;
  const depthMicrons = railData?.depth_microns || 12.0;

  const railShaderMat = useMemo(() => {
    return new THREE.ShaderMaterial({
      uniforms: THREE.UniformsUtils.clone(CorrugationRibbonShaderMaterial.uniforms),
      vertexShader: CorrugationRibbonShaderMaterial.vertexShader,
      fragmentShader: CorrugationRibbonShaderMaterial.fragmentShader,
    });
  }, []);

  useFrame((state) => {
    if (railShaderMat) {
      railShaderMat.uniforms.u_depth_microns.value = depthMicrons;
      railShaderMat.uniforms.u_severity.value = severity;
      railShaderMat.uniforms.u_time.value = state.clock.elapsedTime;
      railShaderMat.uniforms.u_speed.value = trainSpeed / 30.0;
    }
  });

  // Underground track geometry parameters
  const tieCount = 44;
  const tieSpacing = 1.1;
  const corridorLength = tieCount * tieSpacing;

  return (
    <group position={[0, -0.15, 0]}>
      {/* ========================================================= */}
      {/* 1. UNDERGROUND BORED TUNNEL CUTAWAY VAULT                 */}
      {/* Curved precast segmental lining behind and above train    */}
      {/* Fully open to the camera (+X side) for 100% visibility   */}
      {/* ========================================================= */}
      <group position={[0, 0, 0]}>
        {/* Main Rear Curved Wall */}
        <mesh position={[-3.3, 1.3, 0]}>
          <boxGeometry args={[0.2, 2.6, corridorLength]} />
          <meshStandardMaterial color="#1a2333" roughness={0.9} metalness={0.1} />
        </mesh>

        {/* Upper Wall Incline */}
        <mesh position={[-2.5, 2.8, 0]} rotation={[0, 0, -Math.PI / 6]}>
          <boxGeometry args={[0.2, 1.4, corridorLength]} />
          <meshStandardMaterial color="#172030" roughness={0.9} metalness={0.1} />
        </mesh>

        {/* Ceiling Vault Arch */}
        <mesh position={[-1.2, 3.6, 0]} rotation={[0, 0, -Math.PI / 3]}>
          <boxGeometry args={[0.2, 1.8, corridorLength]} />
          <meshStandardMaterial color="#141c2b" roughness={0.9} metalness={0.1} />
        </mesh>

        {/* Tunnel Roof Crown (Overhead utility spine) */}
        <mesh position={[0.2, 4.0, 0]}>
          <boxGeometry args={[1.6, 0.16, corridorLength]} />
          <meshStandardMaterial color="#101725" roughness={0.85} metalness={0.2} />
        </mesh>

        {/* Precast Segmental Lining Arch Rings (Every 3.6m along Z) */}
        {[-21.6, -18.0, -14.4, -10.8, -7.2, -3.6, 0, 3.6, 7.2, 10.8, 14.4, 18.0, 21.6].map((zPos, idx) => (
          <group key={idx} position={[0, 0, zPos]}>
            <mesh position={[-3.25, 1.3, 0]}>
              <boxGeometry args={[0.3, 2.62, 0.28]} />
              <meshStandardMaterial color="#0c1322" roughness={0.92} />
            </mesh>
            <mesh position={[-2.45, 2.8, 0]} rotation={[0, 0, -Math.PI / 6]}>
              <boxGeometry args={[0.3, 1.42, 0.28]} />
              <meshStandardMaterial color="#0c1322" roughness={0.92} />
            </mesh>
            <mesh position={[-1.15, 3.6, 0]} rotation={[0, 0, -Math.PI / 3]}>
              <boxGeometry args={[0.3, 1.82, 0.28]} />
              <meshStandardMaterial color="#0c1322" roughness={0.92} />
            </mesh>
            <mesh position={[0.2, 4.02, 0]}>
              <boxGeometry args={[1.62, 0.2, 0.28]} />
              <meshStandardMaterial color="#0c1322" roughness={0.92} />
            </mesh>
          </group>
        ))}

        {/* Tunnel Service LED Fixtures (Mounted along upper crown) */}
        {[-14.4, -3.6, 7.2, 18.0].map((zPos, idx) => (
          <group key={idx} position={[-0.4, 3.9, zPos]}>
            {/* Lamp fixture housing */}
            <mesh>
              <boxGeometry args={[0.18, 0.08, 1.6]} />
              <meshStandardMaterial
                color="#f8fafc"
                emissive="#f8fafc"
                emissiveIntensity={1.4}
              />
            </mesh>
            {/* Spot luminaire casting downward glow on train roof */}
            <pointLight
              color="#e2e8f0"
              intensity={0.6}
              distance={8}
              decay={2}
            />
          </group>
        ))}

        {/* Wall Cable Conduits (Signaling & Traction feeders) */}
        {[-0.2, 0.1, 0.4].map((yOffset, cIdx) => (
          <mesh
            key={cIdx}
            position={[-3.18, 1.2 + yOffset, 0]}
            rotation={[Math.PI / 2, 0, 0]}
          >
            <cylinderGeometry args={[0.035, 0.035, corridorLength, 8]} />
            <meshStandardMaterial color="#090d16" roughness={0.6} metalness={0.7} />
          </mesh>
        ))}

        {/* SMRT Tunnel Chainage ID Sign */}
        <group position={[-3.16, 2.0, 1.8]} rotation={[0, Math.PI / 2, 0]}>
          <mesh>
            <boxGeometry args={[1.3, 0.4, 0.04]} />
            <meshStandardMaterial color="#0b1120" roughness={0.4} />
          </mesh>
          {/* NSL Red Tag */}
          <mesh position={[-0.45, 0, 0.03]}>
            <boxGeometry args={[0.22, 0.24, 0.02]} />
            <meshStandardMaterial color="#ED1C24" roughness={0.3} />
          </mesh>
        </group>
      </group>

      {/* ========================================================= */}
      {/* 2. UNDERGROUND CONCRETE TRACK SLAB (BALLASTLESS)          */}
      {/* Plinth trackform as standard in Singapore MRT tunnels     */}
      {/* ========================================================= */}
      {/* Primary Invert Track Slab */}
      <mesh position={[-0.3, -0.22, 0]}>
        <boxGeometry args={[5.8, 0.28, corridorLength]} />
        <meshStandardMaterial color="#131b28" roughness={0.92} />
      </mesh>

      {/* Center Drainage Trough */}
      <mesh position={[0, -0.1, 0]}>
        <boxGeometry args={[0.42, 0.06, corridorLength]} />
        <meshStandardMaterial color="#070a10" roughness={0.98} />
      </mesh>

      {/* Emergency Evacuation / Maintenance Walkway Platform */}
      <mesh position={[-2.3, 0.06, 0]}>
        <boxGeometry args={[1.1, 0.32, corridorLength]} />
        <meshStandardMaterial color="#1e293b" roughness={0.8} />
      </mesh>
      {/* Yellow Walkway Safety Border */}
      <mesh position={[-1.74, 0.23, 0]}>
        <boxGeometry args={[0.06, 0.02, corridorLength]} />
        <meshStandardMaterial color="#eab308" roughness={0.4} />
      </mesh>

      {/* Concrete Rail Plinth Blocks under Rails */}
      {Array.from({ length: tieCount }).map((_, i) => {
        const z = (i - tieCount / 2) * tieSpacing;
        return (
          <group key={i} position={[0, 0, z]}>
            {/* Left Rail Plinth */}
            <mesh position={[-1.05, -0.04, 0]}>
              <boxGeometry args={[0.42, 0.12, 0.32]} />
              <meshStandardMaterial color="#334155" roughness={0.85} />
            </mesh>
            {/* Right Rail Plinth */}
            <mesh position={[1.05, -0.04, 0]}>
              <boxGeometry args={[0.42, 0.12, 0.32]} />
              <meshStandardMaterial color="#334155" roughness={0.85} />
            </mesh>
          </group>
        );
      })}

      {/* ========================================================= */}
      {/* 3. RUNNING RAILS (With Dynamic Corrugation Shader)        */}
      {/* ========================================================= */}
      {/* Left Steel Rail Ribbon */}
      <mesh
        material={railShaderMat}
        position={[-1.05, 0.08, 0]}
        onClick={(e) => {
          e.stopPropagation();
          setActiveInspection('rail');
        }}
      >
        <boxGeometry args={[0.09, 0.15, corridorLength, 1, 1, 120]} />
      </mesh>

      {/* Right Steel Rail Ribbon */}
      <mesh
        material={railShaderMat}
        position={[1.05, 0.08, 0]}
        onClick={(e) => {
          e.stopPropagation();
          setActiveInspection('rail');
        }}
      >
        <boxGeometry args={[0.09, 0.15, corridorLength, 1, 1, 120]} />
      </mesh>

      {/* ========================================================= */}
      {/* 4. SMRT 750V DC THIRD RAIL SYSTEM (Conductor Rail)        */}
      {/* Bottom-contact conductor rail with protective cover board */}
      {/* ========================================================= */}
      <group position={[1.52, 0.16, 0]}>
        {/* Steel/Aluminum Conductor Rail Bar */}
        <mesh position={[0, 0, 0]}>
          <boxGeometry args={[0.07, 0.1, corridorLength]} />
          <meshStandardMaterial color="#64748b" metalness={0.9} roughness={0.2} />
        </mesh>

        {/* Fiberglass Protective Shroud / Coverboard */}
        <mesh position={[0, 0.08, 0]}>
          <boxGeometry args={[0.16, 0.06, corridorLength]} />
          <meshStandardMaterial color="#1e293b" roughness={0.6} />
        </mesh>
        {/* High Voltage Safety Yellow Top Stripe */}
        <mesh position={[0, 0.115, 0]}>
          <boxGeometry args={[0.1, 0.015, corridorLength]} />
          <meshStandardMaterial color="#eab308" roughness={0.4} />
        </mesh>

        {/* Third Rail Insulator Support Brackets (every 3.3m) */}
        {Array.from({ length: Math.floor(corridorLength / 3.3) }).map((_, bIdx) => {
          const bZ = (bIdx - Math.floor(corridorLength / 6.6)) * 3.3;
          return (
            <mesh key={bIdx} position={[0, -0.15, bZ]}>
              <cylinderGeometry args={[0.04, 0.06, 0.22, 10]} />
              <meshStandardMaterial color="#94a3b8" metalness={0.3} roughness={0.5} />
            </mesh>
          );
        })}
      </group>

      {/* ========================================================= */}
      {/* 5. INTERACTIVE RAIL CORRUGATION INSPECTION BEACON          */}
      {/* ========================================================= */}
      <group
        position={[1.9, 0.45, 2.0]}
        onClick={(e) => {
          e.stopPropagation();
          setActiveInspection('rail');
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
          <boxGeometry args={[0.08, 0.6, 0.35]} />
          <meshStandardMaterial color="#ED1C24" metalness={0.5} roughness={0.3} />
        </mesh>
        <InspectionTag type="rail" position={[0, 0.4, 0]} beaconLabel="Track Corrugation" />
      </group>
    </group>
  );
};
