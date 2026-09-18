import React from 'react';
import { Html, Line } from '@react-three/drei';
import { CheckCircle2, FileText, Lightbulb, Wrench } from 'lucide-react';
import { useTwinStore } from '../../store/useTwinStore';
import { SubsystemSelection } from '../../types/telemetry';
import { ACTIONS } from '../hud/WhatIfPanel';

const STATUS_DOT_COLOR: Record<string, string> = {
  GOOD: '#0f766e',
  WATCH: '#B45309',
  ACTION_NEEDED: '#ED1C24',
  UNKNOWN: '#94a3b8',
};

const PILL_OFFSET: [number, number, number] = [-0.32, 0.58, -0.12];

interface FindingBubbleProps {
  subsystem: Extract<SubsystemSelection, 'door' | 'acv' | 'shm' | 'rail'>;
  position: [number, number, number];
  label: string;
}

/**
 * One bubble per active (unresolved) uploaded finding. Same light-annotation
 * language as InspectionTag (white pill, hairline, leader line, ringed
 * anchor dot) but the card it opens is Explanation/Suggestion/Resolve, not a
 * metric readout - a different job, so a separate component rather than
 * overloading InspectionTag's existing click-through.
 */
export const FindingBubble: React.FC<FindingBubbleProps> = ({ subsystem, position, label }) => {
  const finding = useTwinStore((s) => s.currentFrame?.latest_upload_results?.[subsystem]);
  const isHudVisible = useTwinStore((s) => s.isHudVisible);
  const [open, setOpen] = React.useState(false);
  const resolveFinding = useTwinStore((s) => s.resolveFinding);

  if (!finding || !isHudVisible) return null;

  const dotColor = STATUS_DOT_COLOR[finding.status] ?? STATUS_DOT_COLOR.UNKNOWN;
  const repair = ACTIONS.find((a) => a.action === finding.recommended_action);

  return (
    <group position={position}>
      <mesh>
        <ringGeometry args={[0.028, 0.045, 24]} />
        <meshBasicMaterial color={dotColor} toneMapped={false} />
      </mesh>
      <mesh>
        <circleGeometry args={[0.018, 16]} />
        <meshBasicMaterial color={dotColor} toneMapped={false} />
      </mesh>
      <Line points={[[0, 0, 0], PILL_OFFSET]} color="#dee0d8" lineWidth={1} />

      <group position={PILL_OFFSET}>
        <Html center style={{ zIndex: open ? 1000 : 10 }}>
          {!open ? (
            <button
              onClick={(e) => {
                e.stopPropagation();
                setOpen(true);
              }}
              className="flex items-center space-x-1.5 px-2 py-1 rounded-full bg-white border border-slate-200 text-ink-900 hover:border-ink-500 transition-colors duration-150 cursor-pointer select-none glass-panel-floating"
            >
              <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ backgroundColor: dotColor }} aria-hidden="true" />
              <span className="text-label font-mono font-semibold whitespace-nowrap">{label}: new finding</span>
            </button>
          ) : (
            <div
              onClick={(e) => e.stopPropagation()}
              className="w-72 glass-panel-floating p-3 rounded text-slate-800 pointer-events-auto select-none animate-in fade-in zoom-in-95 duration-150"
            >
              <div className="flex items-center justify-between border-b border-slate-100 pb-1.5 mb-2">
                <div className="flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ backgroundColor: dotColor }} />
                  <span className="font-bold text-label text-slate-900">{label}</span>
                </div>
                <span className="text-label text-slate-400 font-mono truncate max-w-[100px]">
                  {finding.file_name}
                </span>
              </div>

              <div className="space-y-1">
                <div className="flex items-center gap-1 text-[9px] font-bold text-slate-400 tracking-[0.14em]">
                  <FileText className="w-3 h-3" />
                  EXPLANATION
                </div>
                <p className="text-[11px] text-slate-700 leading-relaxed">{finding.conductor_summary}</p>
              </div>

              {repair && (
                <div className="space-y-1 mt-2.5">
                  <div className="flex items-center gap-1 text-[9px] font-bold text-slate-400 tracking-[0.14em]">
                    <Lightbulb className="w-3 h-3" />
                    SUGGESTION
                  </div>
                  <p className="text-[11px] text-slate-700 leading-relaxed">{repair.plainDescription}</p>
                </div>
              )}

              <button
                onClick={() => {
                  resolveFinding(subsystem);
                  setOpen(false);
                }}
                className="w-full mt-3 flex items-center justify-center gap-1.5 bg-ink-900 text-white text-xs font-semibold py-2 rounded hover:bg-ink-700 transition-colors duration-150"
              >
                {repair ? <Wrench className="w-3.5 h-3.5" /> : <CheckCircle2 className="w-3.5 h-3.5" />}
                Resolve
              </button>
            </div>
          )}
        </Html>
      </group>
    </group>
  );
};
