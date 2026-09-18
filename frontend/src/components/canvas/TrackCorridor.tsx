import React, { useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { useTwinStore } from '../../store/useTwinStore';
import { CorrugationRibbonShaderMaterial } from './shaders/CorrugationRibbonShader';
import { InspectionTag } from './InspectionTag';
import { classifyMetric } from '../../lib/metricGlossary';

export const TrackCorridor: React.FC = () => {
  const currentFrame = useTwinStore((state) => state.currentFrame);
  const setActiveInspection = useTwinStore((state) => state.setActiveInspection);
  const railData = currentFrame?.subsystems?.rail_corrugation;
  const trainSpeed = currentFrame?.train_speed_kmh || 68.0;
  const railStatus =
    currentFrame?.plain_status?.['rail_corrugation.severity_score'] ??
    classifyMetric('rail_corrugation.severity_score', railData?.severity_score);

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
  // Long enough for the 8-car consist (world Z -57.5..126.5); the corridor is
  // centred on TRACK_Z so the monitored car / rail beacon stay where they were.
  const tieCount = 180;
  const tieSpacing = 1.1;
  const corridorLength = tieCount * tieSpacing;
  const TRACK_Z = 34;

  return (
    <group position={[0, -0.15, TRACK_Z]}>
      {/* ========================================================= */}
      {/* 1. CONCRETE TRACKBED / DEPOT SLAB                         */}
      {/* Clean open trackform suitable for studio / inspection     */}
      {/* ========================================================= */}
      {/* Primary Invert Track Slab */}
      <mesh position={[-0.3, -0.22, 0]}>
        <boxGeometry args={[5.8, 0.28, corridorLength]} />
        <meshStandardMaterial color="#e2e8f0" roughness={0.85} />
      </mesh>

      {/* Center Drainage Trough */}
      <mesh position={[0, -0.1, 0]}>
        <boxGeometry args={[0.42, 0.06, corridorLength]} />
        <meshStandardMaterial color="#cbd5e1" roughness={0.9} />
      </mesh>

      {/* Emergency Evacuation / Maintenance Walkway Platform */}
      <mesh position={[-2.3, 0.06, 0]}>
        <boxGeometry args={[1.1, 0.32, corridorLength]} />
        <meshStandardMaterial color="#d1d5db" roughness={0.8} />
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
              <meshStandardMaterial color="#94a3b8" roughness={0.8} />
            </mesh>
            {/* Right Rail Plinth */}
            <mesh position={[1.05, -0.04, 0]}>
              <boxGeometry args={[0.42, 0.12, 0.32]} />
              <meshStandardMaterial color="#94a3b8" roughness={0.8} />
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
        <boxGeometry args={[0.09, 0.15, corridorLength, 1, 1, 480]} />
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
        <boxGeometry args={[0.09, 0.15, corridorLength, 1, 1, 480]} />
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
        position={[1.9, 0.45, -1.8 - TRACK_Z]}
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
        <InspectionTag type="rail" position={[0, 0.4, 0]} beaconLabel="Track Corrugation" status={railStatus} />
      </group>
    </group>
  );
};
