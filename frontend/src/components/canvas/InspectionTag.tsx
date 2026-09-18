import React, { useEffect, useState } from 'react';
import { useFrame } from '@react-three/fiber';
import { Html, Line } from '@react-three/drei';
import {
  Activity,
  DoorOpen,
  ExternalLink,
  FileText,
  GitCommit,
  Lightbulb,
  Wind,
  Wrench,
  X,
} from 'lucide-react';
import { useTwinStore } from '../../store/useTwinStore';
import { carShiftZ } from './consist';
import { Finding, Line as LineId } from '../../types/telemetry';
import { STATUS_SHORT, STATUS_STYLES } from '../../lib/metricGlossary';
import { ACTIONS } from '../hud/WhatIfPanel';

type IssueType = 'door' | 'acv' | 'shm' | 'rail';

interface InspectionTagProps {
  type: IssueType;
  position: [number, number, number];
  /** Card title. The bubble itself carries no text. */
  beaconLabel: string;
}

/** World-space offset from the anchor to where the bubble itself floats. */
const BUBBLE_OFFSET: [number, number, number] = [0.36, 0.5, 0.12];

/**
 * One fixed colour per issue type, so a bubble says *which* subsystem it is
 * without a label. Deliberately none of red / amber / green - those stay
 * reserved for severity (the status pill and the pulse).
 */
const ISSUE_STYLE: Record<IssueType, { color: string; Icon: React.FC<{ className?: string }> }> = {
  door: { color: '#2563eb', Icon: DoorOpen },
  acv: { color: '#0891b2', Icon: Wind },
  shm: { color: '#7c3aed', Icon: Activity },
  rail: { color: '#c026d3', Icon: GitCommit },
};

const PING_COLOR: Record<Finding['status'], string> = {
  GOOD: 'bg-slate-400',
  WATCH: 'bg-status-watch',
  ACTION_NEEDED: 'bg-status-fault',
  UNKNOWN: 'bg-slate-400',
};

interface IssueCopy {
  explanation: string;
  suggestion: string;
}

/** AI-written card copy for one subsystem; the backend caches, so re-opening is cheap. */
function useIssueCopy(type: IssueType, enabled: boolean, refetchKey: string, line: LineId, car: number) {
  const [copy, setCopy] = useState<IssueCopy | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    setFailed(false);
    fetch('/api/ai/issue', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ subsystem: type, line, car }),
    })
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error(String(res.status)))))
      .then((data) => {
        if (!cancelled) setCopy({ explanation: data.explanation, suggestion: data.suggestion });
      })
      .catch(() => {
        if (!cancelled) setFailed(true);
      });
    return () => {
      cancelled = true;
    };
  }, [type, enabled, refetchKey, line, car]);

  return { copy, failed };
}

export const InspectionTag: React.FC<InspectionTagProps> = ({ type, position, beaconLabel }) => {
  const activeInspection = useTwinStore((state) => state.activeInspection);
  const setActiveInspection = useTwinStore((state) => state.setActiveInspection);
  const setSelectedSubsystem = useTwinStore((state) => state.setSelectedSubsystem);
  const setRightDrawerTab = useTwinStore((state) => state.setRightDrawerTab);
  const finding = useTwinStore((state) => state.currentFrame?.uploads_by_line?.[state.activeLine]?.[type]);
  const activeLine = useTwinStore((state) => state.activeLine);
  const monitoredCarNo = useTwinStore((state) => state.monitoredCar);
  const resolveFinding = useTwinStore((state) => state.resolveFinding);
  const isHudVisible = useTwinStore((state) => state.isHudVisible);
  // Bubbles are sized for close inspection; zoomed far out they stack into an
  // unreadable pile, so they step aside (measured from the monitored car).
  const monitoredCar = useTwinStore((state) => state.monitoredCar);
  const [farAway, setFarAway] = useState(false);
  useFrame(({ camera }) => {
    const { x, y, z } = camera.position;
    const far = Math.hypot(x, y, z - carShiftZ(monitoredCar)) > 70;
    if (far !== farAway) setFarAway(far);
  });

  const isActive = activeInspection === type;
  const { copy, failed } = useIssueCopy(
    type,
    isActive && Boolean(finding),
    finding?.file_name ?? '',
    activeLine,
    monitoredCarNo
  );

  // A bubble exists only for an unresolved uploaded finding: it appears when
  // the upload lands and stays until Resolve, which clears it and logs it.
  if (!finding || !isHudVisible || farAway) return null;
  // If another inspection is open, hide this bubble so it doesn't clutter or overlap.
  if (activeInspection && !isActive) return null;

  const { color, Icon } = ISSUE_STYLE[type];
  const status = finding.status;
  const statusStyles = STATUS_STYLES[status];
  const repair = ACTIONS.find((a) => a.action === finding.recommended_action);
  // If the AI copy can't load (e.g. backend without the endpoint), fall back to
  // the model's own output rather than an error.
  const explanation = copy?.explanation ?? (failed ? finding.conductor_summary : null);
  const suggestion =
    copy?.suggestion ?? (failed ? repair?.plainDescription ?? 'No action needed.' : null);

  const toggle = (e: React.MouseEvent) => {
    e.stopPropagation();
    setActiveInspection(isActive ? null : type);
  };

  const handleResolve = (e: React.MouseEvent) => {
    e.stopPropagation();
    resolveFinding(type);
    setActiveInspection(null);
  };

  const handleOpenCockpit = (e: React.MouseEvent) => {
    e.stopPropagation();
    setSelectedSubsystem(type);
    setRightDrawerTab('telemetry');
    useTwinStore.setState({ isRightDrawerOpen: true });
  };

  return (
    <group position={position}>
      {/* Ringed anchor dot on the part itself, plus a 1px leader line up to the bubble. */}
      <mesh>
        <ringGeometry args={[0.034, 0.055, 24]} />
        <meshBasicMaterial color={color} toneMapped={false} />
      </mesh>
      <mesh>
        <circleGeometry args={[0.022, 16]} />
        <meshBasicMaterial color={color} toneMapped={false} />
      </mesh>
      <Line points={[[0, 0, 0], BUBBLE_OFFSET]} color="#dee0d8" lineWidth={1} />

      <group position={BUBBLE_OFFSET}>
        <Html center style={{ zIndex: isActive ? 1000 : 10 }}>
          {!isActive ? (
            <button
              onClick={toggle}
              aria-label={`${beaconLabel} - unresolved finding`}
              title={beaconLabel}
              className="relative flex items-center justify-center w-11 h-11 rounded-full text-white border-[3px] border-white cursor-pointer select-none glass-panel-floating hover:scale-110 transition-transform duration-150"
              style={{ backgroundColor: color }}
            >
              <span
                className={`absolute inset-0 rounded-full animate-ping opacity-50 ${PING_COLOR[status]}`}
                aria-hidden="true"
              />
              <Icon className="relative w-5 h-5" />
            </button>
          ) : (
            <div
              onClick={(e) => e.stopPropagation()}
              className="w-80 glass-panel-floating p-4 rounded text-slate-800 font-sans pointer-events-auto select-none relative z-50"
            >
              {/* Header */}
              <div className="flex items-center justify-between border-b border-slate-100 pb-2.5 mb-3.5">
                <div className="flex items-center space-x-2">
                  <span
                    className="flex items-center justify-center w-6 h-6 rounded-full text-white shrink-0"
                    style={{ backgroundColor: color }}
                  >
                    <Icon className="w-3.5 h-3.5" />
                  </span>
                  <span className="font-bold text-sm text-slate-900">{beaconLabel}</span>
                </div>
                <div className="flex items-center space-x-1.5">
                  <span
                    className={`px-2 py-0.5 rounded-full text-label font-semibold border ${statusStyles.bg} ${statusStyles.text} ${statusStyles.border}`}
                  >
                    {STATUS_SHORT[status]}
                  </span>
                  <button
                    onClick={toggle}
                    aria-label="Close"
                    className="p-1 rounded text-slate-400 hover:text-slate-800 hover:bg-slate-100 transition-colors"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              </div>

              <CardSection icon={<FileText className="w-3.5 h-3.5" />} title="EXPLANATION">
                {explanation ?? <Skeleton />}
              </CardSection>

              <div className="mt-4">
                <CardSection icon={<Lightbulb className="w-3.5 h-3.5" />} title="SUGGESTION">
                  {suggestion ?? <Skeleton />}
                </CardSection>
              </div>

              <button
                onClick={handleResolve}
                className="w-full mt-5 flex items-center justify-center gap-2 bg-ink-900 text-white text-sm font-semibold py-2.5 rounded hover:bg-ink-700 transition-colors duration-150"
              >
                <Wrench className="w-4 h-4" />
                Resolve
              </button>

              {/* Quiet link to the full charts - not a solid button */}
              <button
                onClick={handleOpenCockpit}
                className="w-full mt-3 text-left font-mono text-label font-semibold text-ink-700 hover:text-ink-900 flex items-center justify-between transition-colors duration-150"
              >
                <span>Open cockpit charts</span>
                <ExternalLink className="w-3 h-3" />
              </button>
            </div>
          )}
        </Html>
      </group>
    </group>
  );
};

const CardSection: React.FC<{ icon: React.ReactNode; title: string; children: React.ReactNode }> = ({
  icon,
  title,
  children,
}) => (
  <div className="space-y-1.5">
    <div className="flex items-center gap-1.5 text-[10px] font-bold text-slate-400 tracking-[0.14em]">
      {icon}
      {title}
    </div>
    <div className="text-[13px] text-slate-700 leading-relaxed">{children}</div>
  </div>
);

const Skeleton: React.FC = () => (
  <div className="space-y-2 animate-pulse" aria-label="Loading">
    <div className="h-2.5 rounded bg-slate-100 w-full" />
    <div className="h-2.5 rounded bg-slate-100 w-4/5" />
  </div>
);
