import React from 'react';

interface GhostMeshProps {
  nominalOpenDist: number; // 0.0 (closed) to 0.55 (fully open)
  visible: boolean;
}

export const GhostMesh: React.FC<GhostMeshProps> = ({ nominalOpenDist, visible }) => {
  if (!visible) return null;

  return (
    <group position={[0, 0, 0]}>
      {/* Left Leaf Ghost (Nominal Baseline wireframe) */}
      <mesh position={[-0.28 - nominalOpenDist, 0, 0]}>
        <boxGeometry args={[0.54, 1.82, 0.04]} />
        <meshBasicMaterial
          color="#009645"
          wireframe
          transparent
          opacity={0.45}
        />
      </mesh>

      {/* Right Leaf Ghost */}
      <mesh position={[0.28 + nominalOpenDist, 0, 0]}>
        <boxGeometry args={[0.54, 1.82, 0.04]} />
        <meshBasicMaterial
          color="#009645"
          wireframe
          transparent
          opacity={0.45}
        />
      </mesh>
    </group>
  );
};
