import React, { useEffect, useRef, useState } from 'react';
import {
  AlertTriangle,
  ChevronDown,
  Eye,
  GraduationCap,
  Layout,
  Radio,
  ShieldCheck,
  SlidersHorizontal,
  Train,
  Wrench,
} from 'lucide-react';
import { useTwinStore } from '../../store/useTwinStore';
import { CameraPreset, MetricStatus } from '../../types/telemetry';
import { InfoTip } from './InfoTip';
import { METRIC_GLOSSARY, STATUS_STYLES, classifyMetric } from '../../lib/metricGlossary';

const CAMERA_COPY: Record<CameraPreset, { plain: string; tip: string }> = {
  macro: { plain: 'Whole line', tip: 'Pull back to see the train moving along the corridor.' },
  meso: { plain: 'Whole train', tip: 'Look at the carriage as a whole.' },
  micro: { plain: 'Close-up', tip: 'Zoom in on the selected component.' },
};

const SUBSYSTEM_HEADER_LABELS: Record<string, string> = {
  door: 'Passenger doors',
  acv: 'Air-conditioning',
  shm: 'Wheels & frame',
  rail_corrugation: 'Track condition',
};

const VERDICT_RANK: Record<MetricStatus, number> = {
  ACTION_NEEDED: 3,
  WATCH: 2,
  GOOD: 1,
  UNKNOWN: 0,
};

/** The single most urgent subsystem right now, so the header can name it directly. */
function worstSubsystem(
  status: Record<string, MetricStatus> | undefined
): { label: string; verdict: MetricStatus } | null {
  if (!status) return null;
  let best: { label: string; verdict: MetricStatus; rank: number } | null = null;
  for (const [path, verdict] of Object.entries(status)) {
    const label = SUBSYSTEM_HEADER_LABELS[path.split('.')[0]];
    if (!label) continue;
    const rank = VERDICT_RANK[verdict] ?? 0;
    if (!best || rank > best.rank) best = { label, verdict, rank };
  }
  return best && best.rank > VERDICT_RANK.GOOD ? { label: best.label, verdict: best.verdict } : null;
}

export const CockpitHeader: React.FC = () => {
  const currentFrame = useTwinStore((state) => state.currentFrame);
  const isConnected = useTwinStore((state) => state.isConnected);
  const xrayMode = useTwinStore((state) => state.xrayMode);
  const toggleXray = useTwinStore((state) => state.toggleXray);
  const cameraMode = useTwinStore((state) => state.cameraMode);
  const setCameraMode = useTwinStore((state) => state.setCameraMode);
  const isHudVisible = useTwinStore((state) => state.isHudVisible);
  const toggleHud = useTwinStore((state) => state.toggleHud);
  const uiMode = useTwinStore((state) => state.uiMode);
  const toggleUiMode = useTwinStore((state) => state.toggleUiMode);
  const activeInterventions = useTwinStore((state) => state.activeInterventions);

  const beginner = uiMode === 'beginner';
  const healthIndex = currentFrame ? currentFrame.fleet_health_index : 0.85;
  const verdict = classifyMetric('fleet_health_index', healthIndex);
  const styles = STATUS_STYLES[verdict];
  const healthDef = METRIC_GLOSSARY.fleet_health_index;

  const healthWord =
    verdict === 'GOOD' ? 'Healthy' : verdict === 'WATCH' ? 'Monitor' : 'Needs work';

  const repairCount = Object.values(activeInterventions).filter(Boolean).length;
  const worst = worstSubsystem(currentFrame?.plain_status);

  const [viewMenuOpen, setViewMenuOpen] = useState(false);
  const viewMenuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!viewMenuOpen) return;
    const onDocClick = (e: MouseEvent) => {
      if (viewMenuRef.current && !viewMenuRef.current.contains(e.target as Node)) {
        setViewMenuOpen(false);
      }
    };
    const onEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setViewMenuOpen(false);
    };
    document.addEventListener('mousedown', onDocClick);
    document.addEventListener('keydown', onEsc);
    return () => {
      document.removeEventListener('mousedown', onDocClick);
      document.removeEventListener('keydown', onEsc);
    };
  }, [viewMenuOpen]);

  const viewMenuHasActiveToggle = xrayMode || !isHudVisible;

  return (
    <header className="absolute top-4 left-4 right-4 z-30 flex items-start justify-between pointer-events-none gap-2">
      {/* Left branding and fleet info */}
      <div className="flex items-center space-x-3 pointer-events-auto">
        <div className="glass-panel-glow px-3.5 py-2 rounded-xl flex items-center space-x-3">
          <div className="w-8 h-8 rounded-lg bg-red-600 border border-red-500/60 flex items-center justify-center text-white shadow-md shadow-red-600/40">
            <Train className="w-4 h-4 stroke-[2.5]" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-xs font-black tracking-wider text-slate-900 uppercase flex items-center space-x-1.5">
                <span className="text-red-600 font-extrabold">SMRT</span>
                <span>•</span>
                <span>LTA Digital Twin</span>
              </h1>
              <span className="text-[9px] bg-red-50 text-red-600 font-mono px-1 py-0.5 rounded border border-red-200 font-bold">
                NSL
              </span>
            </div>
            <p className="text-[10px] text-slate-500 font-mono">
              C151B-SET-402 • CAR 3 • NORTH-SOUTH LINE
            </p>
          </div>
        </div>

        {/* Real-time telemetry badges */}
        <div className="glass-panel px-3 py-1.5 rounded-xl flex items-center space-x-4 text-xs font-mono">
          <div>
            <div className="text-[9px] text-slate-500 uppercase flex items-center gap-1">
              {beginner ? 'Position' : 'Chainage'}
              <InfoTip
                title={METRIC_GLOSSARY.track_chainage_km.plainName}
                whatItIs={METRIC_GLOSSARY.track_chainage_km.whatItIs}
                whyItMatters={METRIC_GLOSSARY.track_chainage_km.whyItMatters}
                analogy={METRIC_GLOSSARY.track_chainage_km.analogy}
              />
            </div>
            <div className="text-slate-900 font-semibold text-xs">
              KP {currentFrame ? currentFrame.track_chainage_km.toFixed(3) : '12.500'}
            </div>
          </div>
          <div className="h-5 w-[1px] bg-slate-200" />
          <div>
            <div className="text-[9px] text-slate-500 uppercase">Speed</div>
            <div className="text-red-600 font-semibold text-xs">
              {currentFrame ? currentFrame.train_speed_kmh.toFixed(1) : '68.4'}{' '}
              <span className="text-[9px] text-slate-400">km/h</span>
            </div>
          </div>
          <div className="h-5 w-[1px] bg-slate-200" />
          <div>
            <div className="text-[9px] text-slate-500 uppercase">Time</div>
            <div className="text-slate-600 text-xs">
              {currentFrame ? currentFrame.timestamp.slice(11, 19) : '10:00:00'}
            </div>
          </div>
        </div>
      </div>

      {/* Right controls */}
      <div className="flex items-center space-x-2 pointer-events-auto">
        {/* Simulated repairs indicator */}
        {repairCount > 0 && (
          <div className="glass-panel border border-emerald-300 bg-emerald-50/80 px-2.5 py-1.5 rounded-xl flex items-center space-x-1.5">
            <Wrench className="w-3.5 h-3.5 text-emerald-700" />
            <div>
              <div className="text-[8px] uppercase tracking-wider font-mono text-emerald-700 opacity-80">
                Simulated
              </div>
              <div className="text-[11px] font-mono font-bold leading-none text-emerald-800">
                {repairCount} repair{repairCount > 1 ? 's' : ''}
              </div>
            </div>
          </div>
        )}

        {/* Health index with plain-language verdict */}
        <div
          className={`glass-panel border px-3 py-1.5 rounded-xl flex items-center space-x-2 ${styles.border} ${styles.bg}`}
        >
          {verdict === 'GOOD' ? (
            <ShieldCheck className={`w-4 h-4 ${styles.text}`} />
          ) : (
            <AlertTriangle className={`w-4 h-4 ${styles.text}`} />
          )}
          <div>
            <div
              className={`text-[8px] uppercase tracking-wider font-mono opacity-80 flex items-center gap-1 ${styles.text}`}
            >
              {beginner ? healthWord : 'Health'}
              <InfoTip
                title={healthDef.plainName}
                whatItIs={healthDef.whatItIs}
                whyItMatters={healthDef.whyItMatters}
                analogy={healthDef.analogy}
                range="Healthy at or above 80%"
                side="right"
              />
            </div>
            <div className={`text-sm font-mono font-bold leading-none ${styles.text}`}>
              {(healthIndex * 100).toFixed(0)}%
            </div>
            {worst && (
              <div className={`text-[9px] font-semibold leading-tight mt-0.5 ${styles.text}`}>
                {worst.label}
              </div>
            )}
          </div>
        </div>

        {/* Camera presets */}
        <div className="glass-panel p-1 rounded-xl flex items-center space-x-1 text-xs font-mono">
          {(['macro', 'meso', 'micro'] as CameraPreset[]).map((preset) => (
            <button
              key={preset}
              onClick={() => setCameraMode(preset)}
              title={CAMERA_COPY[preset].tip}
              className={`px-2 py-1 rounded-lg uppercase tracking-wide text-[10px] transition-all font-bold ${
                cameraMode === preset
                  ? 'bg-red-600 text-white shadow-md shadow-red-600/30'
                  : 'text-slate-500 hover:text-slate-800 hover:bg-slate-100'
              }`}
            >
              {beginner ? CAMERA_COPY[preset].plain : preset}
            </button>
          ))}
        </div>

        {/* View options: display mode, x-ray, zen - grouped so they read as one
            "settings" affordance instead of three same-weight header pills. */}
        <div className="relative" ref={viewMenuRef}>
          <button
            onClick={() => setViewMenuOpen((o) => !o)}
            aria-expanded={viewMenuOpen}
            className={`glass-panel px-2.5 py-1.5 rounded-xl flex items-center space-x-1.5 text-[11px] font-medium transition-all border ${
              viewMenuOpen
                ? 'bg-slate-100 text-slate-800 border-slate-300'
                : 'text-slate-600 border-transparent hover:bg-slate-100'
            }`}
            title="Display, x-ray and panel visibility options"
          >
            <SlidersHorizontal className="w-3.5 h-3.5" />
            <span>VIEW</span>
            {viewMenuHasActiveToggle && (
              <span className="w-1.5 h-1.5 rounded-full bg-red-500" aria-hidden="true" />
            )}
            <ChevronDown className={`w-3 h-3 transition-transform ${viewMenuOpen ? 'rotate-180' : ''}`} />
          </button>

          {viewMenuOpen && (
            <div className="absolute right-0 top-full mt-1.5 w-52 glass-panel p-1.5 rounded-xl shadow-xl flex flex-col space-y-1 z-40">
              <button
                onClick={toggleUiMode}
                aria-pressed={beginner}
                className={`px-2.5 py-1.5 rounded-lg flex items-center space-x-1.5 text-[11px] font-medium transition-all ${
                  beginner ? 'bg-sky-50 text-sky-700' : 'text-slate-600 hover:bg-slate-100'
                }`}
                title={
                  beginner
                    ? 'Beginner mode: plain-language labels. Click for engineering terms.'
                    : 'Expert mode: engineering terms. Click for plain language.'
                }
              >
                <GraduationCap className="w-3.5 h-3.5" />
                <span>{beginner ? 'SIMPLE labels' : 'EXPERT labels'}</span>
              </button>

              <button
                onClick={toggleXray}
                aria-pressed={xrayMode}
                className={`px-2.5 py-1.5 rounded-lg flex items-center space-x-1.5 text-[11px] font-medium transition-all ${
                  xrayMode ? 'bg-red-50 text-red-700' : 'text-slate-600 hover:bg-slate-100'
                }`}
                title="See through the carriage shell to the equipment inside"
              >
                <Eye className="w-3.5 h-3.5" />
                <span>{beginner ? 'SEE INSIDE' : 'X-RAY'}</span>
              </button>

              <button
                onClick={toggleHud}
                aria-pressed={!isHudVisible}
                className={`px-2.5 py-1.5 rounded-lg flex items-center space-x-1.5 text-[11px] font-medium transition-all ${
                  !isHudVisible ? 'bg-amber-50 text-amber-700' : 'text-slate-600 hover:bg-slate-100'
                }`}
                title={isHudVisible ? 'Hide all panels' : 'Show all panels'}
              >
                <Layout className="w-3.5 h-3.5" />
                <span>{isHudVisible ? 'ZEN mode' : 'Show panels'}</span>
              </button>
            </div>
          )}
        </div>

        {/* Connection status */}
        <div className="glass-panel px-2.5 py-1.5 rounded-xl flex items-center space-x-1.5 text-xs font-mono">
          <Radio className={`w-3 h-3 ${isConnected ? 'text-emerald-500 animate-pulse' : 'text-red-500'}`} />
          <span className={`text-[10px] ${isConnected ? 'text-emerald-600' : 'text-red-600'}`}>
            {isConnected ? 'LIVE' : 'OFFLINE'}
          </span>
        </div>
      </div>
    </header>
  );
};
