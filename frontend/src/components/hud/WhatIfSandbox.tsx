import React from 'react';
import { Wrench, Sparkles, Check, RotateCcw, AlertOctagon } from 'lucide-react';
import { useTwinStore } from '../../store/useTwinStore';

export const WhatIfSandbox: React.FC = () => {
  const activeInterventions = useTwinStore((state) => state.activeInterventions);
  const triggerWhatIf = useTwinStore((state) => state.triggerWhatIf);
  const currentFrame = useTwinStore((state) => state.currentFrame);

  const interventions = [
    {
      action: 'ACTION_GRIND_RAIL',
      title: 'Simulate Rail Grinding',
      description: 'Autonomous track grinding across corrugated KP chainage',
      targetSubsystem: 'Track & Bogie SHM',
      gain: '-2.1g RMS vibration • +45% weld life',
    },
    {
      action: 'ACTION_LUBRICATE_DOOR',
      title: 'Lubricate Door Guides',
      description: 'Clean guide-rails & apply low-viscosity PTFE lubricant',
      targetSubsystem: 'Door 3R Kinematics',
      gain: '-3.6A peak current • 0mm ghost lag',
    },
    {
      action: 'ACTION_REPLACE_FILTER',
      title: 'Replace ACV Air Filter',
      description: 'Fresh filter pack & refrigerant charge replenishment',
      targetSubsystem: 'ACV Climate Pack',
      gain: '+2.2°C thermal delta • -0.75kW power',
    },
    {
      action: 'ACTION_INSPECT_BEARING',
      title: 'Inspect Front Bearing',
      description: 'Ultrasonic flaw evaluation & high-pressure re-greasing',
      targetSubsystem: 'Axle-Box Bearing',
      gain: 'Defect risk normalized to 0.05',
    },
  ];

  return (
    <div className="absolute bottom-4 right-4 z-20 w-84 pointer-events-none">
      <div className="glass-panel p-4 rounded-2xl pointer-events-auto border-white/10">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-2">
            <Wrench className="w-4 h-4 text-cyan-400" />
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-100">
              Counterfactual Sandbox
            </h3>
          </div>
          <span className="text-[10px] font-mono text-cyan-400 flex items-center space-x-1">
            <Sparkles className="w-3 h-3" />
            <span>WHAT-IF ENGINE</span>
          </span>
        </div>

        {/* Action Intervention Toggles */}
        <div className="space-y-2">
          {interventions.map((item) => {
            const isActive = !!(activeInterventions as any)[item.action];

            return (
              <div
                key={item.action}
                className={`p-2.5 rounded-xl border transition-all ${
                  isActive
                    ? 'bg-cyan-950/40 border-cyan-500/50 shadow-md shadow-cyan-500/10'
                    : 'bg-dark-850/60 border-white/5 hover:border-white/15'
                }`}
              >
                <div className="flex items-start justify-between">
                  <div>
                    <div className="text-xs font-semibold text-slate-100 flex items-center space-x-1.5">
                      <span>{item.title}</span>
                    </div>
                    <div className="text-[10px] text-slate-400 mt-0.5 leading-snug">
                      {item.description}
                    </div>
                  </div>

                  <button
                    onClick={() => triggerWhatIf(item.action, !isActive)}
                    className={`text-[10px] font-mono font-bold px-2.5 py-1 rounded-lg transition-all flex items-center space-x-1 ${
                      isActive
                        ? 'bg-cyan-500 text-slate-950 shadow-sm shadow-cyan-500/50'
                        : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                    }`}
                  >
                    {isActive ? <Check className="w-3 h-3 stroke-[3]" /> : null}
                    <span>{isActive ? 'APPLIED' : 'APPLY'}</span>
                  </button>
                </div>

                {/* Impact Delta callout */}
                <div className="mt-1.5 pt-1.5 border-t border-white/5 text-[9px] font-mono flex items-center justify-between">
                  <span className="text-slate-500 uppercase">{item.targetSubsystem}</span>
                  <span className={isActive ? 'text-emerald-400 font-semibold' : 'text-slate-400'}>
                    {item.gain}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
