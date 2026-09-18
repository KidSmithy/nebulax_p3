import React, { useState } from 'react';
import { useTwinStore } from '../../store/useTwinStore';
import { C151Car, C151Anchors } from './c151/C151Car';
import { AirflowParticles } from './particles/AirflowParticles';
import { InspectionTag } from './InspectionTag';
import { classifyMetric } from '../../lib/metricGlossary';

/**
 * Recentres the 23m-long real-scale car so its midpoint sits near the scene
 * origin, matching where the old placeholder car and its tuned camera/track
 * geometry expected the train to be.
 */
const CAR_OFFSET: [number, number, number] = [0, 0, -11.5];

export const TrainAssembly: React.FC = () => {
  const xrayMode = useTwinStore((state) => state.xrayMode);
  const currentFrame = useTwinStore((state) => state.currentFrame);
  const setActiveInspection = useTwinStore((state) => state.setActiveInspection);

  const door = currentFrame?.subsystems?.door;
  const acv = currentFrame?.subsystems?.acv;
  const shm = currentFrame?.subsystems?.shm;
  const stressIntensity = shm?.anomaly_score || 0.2;
  const status = currentFrame?.plain_status ?? {};

  const doorStatus = status['door.anomaly_score'] ?? classifyMetric('door.anomaly_score', door?.anomaly_score);
  const acvStatus = status['acv.efficiency_rating'] ?? classifyMetric('acv.efficiency_rating', acv?.efficiency_rating);
  const shmStatus = status['shm.vibration_rms_g'] ?? classifyMetric('shm.vibration_rms_g', shm?.vibration_rms_g);

  const [anchors, setAnchors] = useState<C151Anchors | null>(null);

  return (
    <group position={CAR_OFFSET}>
      <C151Car
        variant="intermediate"
        doorCycleState={door?.cycle_state}
        doorAnomalyScore={door?.anomaly_score ?? 0}
        bogieStressIntensity={stressIntensity}
        bogieXrayOpacity={xrayMode ? 0.85 : 1.0}
        xrayMode={xrayMode}
        onReady={setAnchors}
      />

      {/* --- Interactive hotspots, anchored to the real model's node positions --- */}
      {anchors?.doorR3 && (
        <group
          position={anchors.doorR3}
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
          <InspectionTag type="door" position={[0.3, 1.1, 0]} beaconLabel="Door 3R System" status={doorStatus} />
        </group>
      )}

      {(anchors?.acUnit1 || anchors?.acUnit2) && (
        <group
          position={anchors.acUnit1 ?? anchors.acUnit2!}
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
          <AirflowParticles position={[0, 0.3, 0]} count={120} />
          <InspectionTag type="acv" position={[0, 0.55, 0]} beaconLabel="ACV Climate Pack" status={acvStatus} />
        </group>
      )}

      {anchors?.bogieFront && (
        <group
          position={anchors.bogieFront}
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
          <InspectionTag type="shm" position={[0.9, 0.5, 0]} beaconLabel="Bogie SHM" status={shmStatus} />
        </group>
      )}
    </group>
  );
};
