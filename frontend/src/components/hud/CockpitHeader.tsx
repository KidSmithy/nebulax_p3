import React from 'react';
import { Activity, Eye, Radio, ShieldCheck, AlertTriangle, Layout, Train } from 'lucide-react';
import { useTwinStore } from '../../store/useTwinStore';
import { CameraPreset } from '../../types/telemetry';

export const CockpitHeader: React.FC = () => {
  const currentFrame = useTwinStore((state) => state.currentFrame);
  const isConnected = useTwinStore((state) => state.isConnected);
  const xrayMode = useTwinStore((state) => state.xrayMode);
  const toggleXray = useTwinStore((state) => state.toggleXray);
  const cameraMode = useTwinStore((state) => state.cameraMode);
  const setCameraMode = useTwinStore((state) => state.setCameraMode);
  const isHudVisible = useTwinStore((state) => state.isHudVisible);
  const toggleHud = useTwinStore((state) => state.toggleHud);

  const healthIndex = currentFrame ? currentFrame.fleet_health_index : 0.85;
  const isHealthy = healthIndex >= 0.80;
  const isWarning = healthIndex >= 0.60 && healthIndex < 0.80;

  const healthColor = isHealthy
    ? 'text-emerald-400 border-emerald-500/40 bg-emerald-500/10'
    : isWarning
    ? 'text-amber-400 border-amber-500/40 bg-amber-500/10'
    : 'text-red-500 border-red-500/40 bg-red-500/10';

  return (
    <header className="absolute top-4 left-4 right-4 z-30 flex items-center justify-between pointer-events-none">
      {/* Left branding and fleet info */}
      <div className="flex items-center space-x-3 pointer-events-auto">
        <div className="glass-panel-glow px-3.5 py-2 rounded-xl flex items-center space-x-3">
          {/* SMRT Iconic Red Brand Icon */}
          <div className="w-8 h-8 rounded-lg bg-red-600 border border-red-500/60 flex items-center justify-center text-white shadow-md shadow-red-600/40">
            <Train className="w-4 h-4 stroke-[2.5]" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-xs font-black tracking-wider text-white uppercase flex items-center space-x-1.5">
                <span className="text-red-500 font-extrabold">SMRT</span>
                <span>•</span>
                <span>LTA Digital Twin</span>
              </h1>
              <span className="text-[9px] bg-red-500/20 text-red-400 font-mono px-1 py-0.5 rounded border border-red-500/40 font-bold">
                NSL
              </span>
            </div>
            <p className="text-[10px] text-slate-400 font-mono">
              C151B-SET-402 • CAR 3 • NORTH-SOUTH LINE
            </p>
          </div>
        </div>

        {/* Real-time telemetry badges */}
        <div className="glass-panel px-3 py-1.5 rounded-xl flex items-center space-x-4 text-xs font-mono">
          <div>
            <div className="text-[9px] text-slate-400 uppercase">Chainage</div>
            <div className="text-slate-100 font-semibold text-xs">
              KP {currentFrame ? currentFrame.track_chainage_km.toFixed(3) : '12.500'}
            </div>
          </div>
          <div className="h-5 w-[1px] bg-slate-700/60" />
          <div>
            <div className="text-[9px] text-slate-400 uppercase">Speed</div>
            <div className="text-red-400 font-semibold text-xs">
              {currentFrame ? currentFrame.train_speed_kmh.toFixed(1) : '68.4'} <span className="text-[9px] text-slate-400">km/h</span>
            </div>
          </div>
          <div className="h-5 w-[1px] bg-slate-700/60" />
          <div>
            <div className="text-[9px] text-slate-400 uppercase">SGT Time</div>
            <div className="text-slate-300 text-xs">
              {currentFrame ? currentFrame.timestamp.slice(11, 19) : '10:00:00'}
            </div>
          </div>
        </div>
      </div>

      {/* Right controls: Health index, X-Ray, Camera presets, HUD Zen toggle */}
      <div className="flex items-center space-x-2.5 pointer-events-auto">
        {/* Fleet Health Index Meter */}
        <div className={`glass-panel border px-3 py-1.5 rounded-xl flex items-center space-x-2 ${healthColor}`}>
          {isHealthy ? <ShieldCheck className="w-4 h-4" /> : <AlertTriangle className="w-4 h-4" />}
          <div>
            <div className="text-[8px] uppercase tracking-wider font-mono opacity-80">Health</div>
            <div className="text-sm font-mono font-bold leading-none">
              {(healthIndex * 100).toFixed(0)}%
            </div>
          </div>
        </div>

        {/* X-Ray Mode Toggle */}
        <button
          onClick={toggleXray}
          className={`glass-panel px-2.5 py-1.5 rounded-xl flex items-center space-x-1.5 text-xs font-medium transition-all ${
            xrayMode
              ? 'bg-red-600/30 text-red-300 border-red-500/60 shadow-md shadow-red-600/20'
              : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
          }`}
          title="Toggle Car Body X-Ray Inspection"
        >
          <Eye className="w-3.5 h-3.5" />
          <span className="text-[11px]">X-RAY</span>
        </button>

        {/* Camera Presets Selector */}
        <div className="glass-panel p-1 rounded-xl flex items-center space-x-1 text-xs font-mono">
          {(['macro', 'meso', 'micro'] as CameraPreset[]).map((preset) => (
            <button
              key={preset}
              onClick={() => setCameraMode(preset)}
              className={`px-2.5 py-1 rounded-lg uppercase tracking-wider text-[11px] transition-all font-bold ${
                cameraMode === preset
                  ? 'bg-red-600 text-white shadow-md shadow-red-600/40'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
              }`}
            >
              {preset}
            </button>
          ))}
        </div>

        {/* Zen View / HUD Visibility Toggle */}
        <button
          onClick={toggleHud}
          className={`glass-panel px-2.5 py-1.5 rounded-xl flex items-center space-x-1.5 text-[11px] font-medium transition-all ${
            !isHudVisible
              ? 'bg-amber-500/20 text-amber-300 border-amber-500/50'
              : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
          }`}
          title={isHudVisible ? 'Hide Overlays (Zen Mode)' : 'Show All Overlays'}
        >
          <Layout className="w-3.5 h-3.5" />
          <span>{isHudVisible ? 'ZEN' : 'HUD'}</span>
        </button>

        {/* Connection status */}
        <div className="glass-panel px-2.5 py-1.5 rounded-xl flex items-center space-x-1.5 text-xs font-mono">
          <Radio className={`w-3 h-3 ${isConnected ? 'text-emerald-400 animate-pulse' : 'text-red-500'}`} />
          <span className={`text-[10px] ${isConnected ? 'text-emerald-400' : 'text-red-400'}`}>
            {isConnected ? 'LIVE' : 'OFFLINE'}
          </span>
        </div>
      </div>
    </header>
  );
};
