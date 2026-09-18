import React, { useState } from 'react';
import { Html } from '@react-three/drei';
import { useTwinStore } from '../../store/useTwinStore';
import { CAR_PITCH, REFERENCE_CAR } from './consist';
import { C151Car, C151Anchors } from './c151/C151Car';
import { AirflowParticles } from './particles/AirflowParticles';
import { InspectionTag } from './InspectionTag';
import { classifyMetric } from '../../lib/metricGlossary';

/**
 * Eight-car consist (cab + 6 intermediates + cab, 23 m pitch, per
 * c151/MODEL_CONTRACT.md). Car 3 keeps the exact world position the single car
 * used to have, so the tuned cameras and hotspot anchors are authored against
 * it; the monitored car (any of the 8) just shifts those by whole car pitches.
 */
const CAR_COUNT = 8;
const CAR_OFFSET: [number, number, number] = [0, 0, -11.5 - (REFERENCE_CAR - 1) * CAR_PITCH];

const CARS = Array.from({ length: CAR_COUNT }, (_, i) => {
  if (i === 0) return { variant: 'cab' as const, z: CAR_PITCH, yaw: Math.PI };
  if (i === CAR_COUNT - 1) return { variant: 'cab' as const, z: i * CAR_PITCH, yaw: 0 };
  return { variant: 'intermediate' as const, z: i * CAR_PITCH, yaw: 0 };
});

const STATUS_RING: Record<string, string> = {
  GOOD: '#0f766e',
  WATCH: '#B45309',
  ACTION_NEEDED: '#ED1C24',
  UNKNOWN: '#94a3b8',
};

/** Car numbers over the roofs in the full-train view; the monitored car is filled and status-ringed. */
const CarMarkers: React.FC<{ status: string; monitoredIndex: number }> = ({ status, monitoredIndex }) => (
  <>
    {CARS.map((car, i) => {
      const monitored = i === monitoredIndex;
      const ring = STATUS_RING[status] ?? STATUS_RING.UNKNOWN;
      return (
        <Html
          key={i}
          position={[0, 4.6, i * CAR_PITCH + CAR_PITCH / 2]}
          center
          zIndexRange={[10, 0]}
          style={{ pointerEvents: 'none' }}
        >
          <div
            className={`w-6 h-6 rounded-full flex items-center justify-center text-[11px] font-bold border-2 ${
              monitored ? 'bg-ink-900 text-white' : 'bg-white text-slate-600 border-slate-300'
            }`}
            style={monitored ? { borderColor: ring } : undefined}
          >
            {i + 1}
          </div>
        </Html>
      );
    })}
  </>
);

export const TrainAssembly: React.FC = () => {
  const fullTrainView = useTwinStore((state) => state.cameraMode) === 'macro';
  const monitoredCar = useTwinStore((state) => state.monitoredCar);
  const livery = useTwinStore((state) => state.activeLine) === 'EWL' ? 'green' : 'red';
  const monitoredIndex = monitoredCar - 1;
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
  // Every car reports its own anchors; the cab lacks some nodes, so keep the first
  // non-null value per node instead of whichever car happened to load first.
  const mergeAnchors = (a: C151Anchors) =>
    setAnchors((prev) => {
      if (!prev) return a;
      const merged = { ...prev };
      (Object.keys(a) as (keyof C151Anchors)[]).forEach((k) => {
        merged[k] = prev[k] ?? a[k];
      });
      return merged;
    });

  // On the reversed cab the viewer-facing door is the car's left one.
  const doorAnchor = CARS[monitoredIndex].yaw ? anchors?.doorL3 : anchors?.doorR3;

  return (
    <group position={CAR_OFFSET}>
      {CARS.map((car, i) =>
        i === monitoredIndex ? (
          <C151Car
            key={i}
            variant={car.variant}
            position={[0, 0, car.z]}
            yaw={car.yaw}
            livery={livery}
            doorCycleState={door?.cycle_state}
            doorAnomalyScore={door?.anomaly_score ?? 0}
            bogieStressIntensity={stressIntensity}
            bogieXrayOpacity={xrayMode ? 0.85 : 1.0}
            xrayMode={xrayMode}
            onReady={mergeAnchors}
          />
        ) : (
          <C151Car
            key={i}
            variant={car.variant}
            position={[0, 0, car.z]}
            yaw={car.yaw}
            livery={livery}
            xrayMode={xrayMode}
            onReady={mergeAnchors}
          />
        )
      )}

      {fullTrainView && <CarMarkers status={acvStatus} monitoredIndex={monitoredIndex} />}

      {/* --- Interactive hotspots, anchored to the real model's node positions.
          Anchors are car-local, so lift them onto the monitored car's slot
          (and turn them with it, for the reversed cab). --- */}
      <group position={[0, 0, CARS[monitoredIndex].z]} rotation={[0, CARS[monitoredIndex].yaw, 0]}>
      {doorAnchor && (
        <group
          position={doorAnchor}
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
          <InspectionTag type="door" position={[0.3, 1.1, 0]} beaconLabel="Door 3R System" />
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
          <InspectionTag type="acv" position={[0, 0.55, 0]} beaconLabel="ACV Climate Pack" />
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
          <InspectionTag type="shm" position={[0.9, 0.5, 0]} beaconLabel="Bogie SHM" />
        </group>
      )}
      </group>
    </group>
  );
};
