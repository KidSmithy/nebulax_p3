import React, { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import {
  Activity,
  Wrench,
  Sparkles,
  Check,
  ChevronRight,
  ChevronLeft,
  TrendingUp,
  BarChart2
} from 'lucide-react';
import { useTwinStore } from '../../store/useTwinStore';

export const RightDrawer: React.FC = () => {
  const isRightDrawerOpen = useTwinStore((state) => state.isRightDrawerOpen);
  const toggleRightDrawer = useTwinStore((state) => state.toggleRightDrawer);
  const rightDrawerTab = useTwinStore((state) => state.rightDrawerTab);
  const setRightDrawerTab = useTwinStore((state) => state.setRightDrawerTab);

  const currentFrame = useTwinStore((state) => state.currentFrame);
  const activeInterventions = useTwinStore((state) => state.activeInterventions);
  const triggerWhatIf = useTwinStore((state) => state.triggerWhatIf);

  const subsystems = currentFrame?.subsystems;
  const door = subsystems?.door;
  const acv = subsystems?.acv;
  const shm = subsystems?.shm;
  const rail = subsystems?.rail_corrugation;

  // Chart 1: Door Motor Current (SMRT Red vs LTA Lush Green Nominal)
  const doorChartOption = useMemo(() => {
    const rawWaveform = door?.waveform_window || [2.1, 4.3, 7.8, 11.85, 11.2, 5.0, 1.2];
    const nominalWaveform = [1.8, 3.8, 6.5, 8.2, 7.8, 4.2, 1.0];

    return {
      backgroundColor: 'transparent',
      grid: { top: 20, right: 10, bottom: 20, left: 30 },
      tooltip: { trigger: 'axis' },
      xAxis: {
        type: 'category',
        data: ['0.0s', '0.5s', '1.0s', '1.5s', '2.0s', '2.5s', '3.0s'],
        axisLine: { lineStyle: { color: '#374151' } },
        axisLabel: { color: '#9ca3af', fontSize: 9, fontFamily: 'monospace' },
      },
      yAxis: {
        type: 'value',
        name: 'A',
        nameTextStyle: { color: '#6b7280', fontSize: 9 },
        splitLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } },
        axisLabel: { color: '#9ca3af', fontSize: 9, fontFamily: 'monospace' },
      },
      series: [
        {
          name: 'Active Current',
          type: 'line',
          smooth: true,
          data: rawWaveform,
          lineStyle: { color: '#ED1C24', width: 2.2 },
          areaStyle: {
            color: 'rgba(237, 28, 36, 0.22)',
          },
        },
        {
          name: 'LTA Nominal Envelope',
          type: 'line',
          smooth: true,
          data: nominalWaveform,
          lineStyle: { color: '#009645', width: 1.5, type: 'dashed' },
        },
      ],
    };
  }, [door]);

  // Chart 2: Bogie FFT Spectrum (LTA Lush Green -> Circle Line Amber -> SMRT Red)
  const fftChartOption = useMemo(() => {
    const spectrum = shm?.fft_spectrum || [
      { freq_hz: 25.0, amp: 0.42 },
      { freq_hz: 142.5, amp: 2.89 },
      { freq_hz: 300.0, amp: 0.15 },
    ];

    return {
      backgroundColor: 'transparent',
      grid: { top: 20, right: 10, bottom: 20, left: 30 },
      tooltip: { trigger: 'item' },
      xAxis: {
        type: 'category',
        data: spectrum.map((s) => `${s.freq_hz}Hz`),
        axisLine: { lineStyle: { color: '#374151' } },
        axisLabel: { color: '#9ca3af', fontSize: 8, fontFamily: 'monospace' },
      },
      yAxis: {
        type: 'value',
        name: 'g',
        nameTextStyle: { color: '#6b7280', fontSize: 9 },
        splitLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } },
        axisLabel: { color: '#9ca3af', fontSize: 9, fontFamily: 'monospace' },
      },
      series: [
        {
          name: 'FFT',
          type: 'bar',
          data: spectrum.map((s) => s.amp),
          itemStyle: {
            color: (params: any) => {
              const val = params.value;
              if (val > 2.0) return '#ED1C24'; // SMRT Red (Alert)
              if (val > 1.0) return '#FF9E1B'; // Circle Line Amber (Warn)
              return '#009645'; // LTA Lush Green (Nominal)
            },
            borderRadius: [2, 2, 0, 0],
          },
        },
      ],
    };
  }, [shm]);

  const interventions = [
    {
      action: 'ACTION_GRIND_RAIL',
      title: 'Simulate Rail Grinding',
      description: 'LTA continuous track grinding across corrugated KP chainage',
      gain: '-2.1g RMS vibration • +45% weld life',
    },
    {
      action: 'ACTION_LUBRICATE_DOOR',
      title: 'Lubricate Door Guides',
      description: 'Clean guide-rails & apply low-viscosity PTFE lubricant',
      gain: '-3.6A peak current • 0mm ghost lag',
    },
    {
      action: 'ACTION_REPLACE_FILTER',
      title: 'Replace ACV Air Filter',
      description: 'Fresh filter pack & refrigerant charge replenishment',
      gain: '+2.2°C thermal delta • -0.75kW power',
    },
    {
      action: 'ACTION_INSPECT_BEARING',
      title: 'Inspect Front Bearing',
      description: 'Ultrasonic flaw evaluation & high-pressure re-greasing',
      gain: 'Defect risk normalized to 0.05',
    },
  ];

  return (
    <aside className="absolute top-20 right-4 z-20 transition-all duration-300 pointer-events-none">
      {/* Collapsed state pill trigger */}
      {!isRightDrawerOpen ? (
        <button
          onClick={toggleRightDrawer}
          className="glass-panel-glow px-3 py-2 rounded-xl pointer-events-auto flex items-center space-x-2 text-xs font-mono text-red-400 hover:text-white transition-all shadow-xl border-red-500/40"
        >
          <BarChart2 className="w-4 h-4 text-red-500" />
          <span>EXPAND COCKPIT</span>
          <ChevronLeft className="w-3.5 h-3.5" />
        </button>
      ) : (
        /* Expanded Unified Right Dock */
        <div className="glass-panel p-3.5 rounded-2xl pointer-events-auto border-white/10 w-80 shadow-2xl flex flex-col space-y-3">
          {/* Header with Tabs & Collapse button */}
          <div className="flex items-center justify-between border-b border-white/5 pb-2">
            <div className="flex items-center space-x-1 bg-dark-850/90 p-0.5 rounded-lg border border-white/5 font-mono text-[11px]">
              <button
                onClick={() => setRightDrawerTab('telemetry')}
                className={`px-2.5 py-1 rounded-md flex items-center space-x-1.5 transition-all ${
                  rightDrawerTab === 'telemetry'
                    ? 'bg-red-600 text-white font-bold shadow-md shadow-red-600/30'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Activity className="w-3.5 h-3.5" />
                <span>TELEMETRY</span>
              </button>
              <button
                onClick={() => setRightDrawerTab('whatif')}
                className={`px-2.5 py-1 rounded-md flex items-center space-x-1.5 transition-all ${
                  rightDrawerTab === 'whatif'
                    ? 'bg-red-600 text-white font-bold shadow-md shadow-red-600/30'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Wrench className="w-3.5 h-3.5" />
                <span>WHAT-IF</span>
              </button>
            </div>

            <button
              onClick={toggleRightDrawer}
              className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800/80 transition-colors"
              title="Collapse panel"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>

          {/* TAB 1: TELEMETRY & WAVEFORMS */}
          {rightDrawerTab === 'telemetry' && (
            <div className="flex flex-col space-y-3">
              {/* Quick domain status matrix */}
              <div className="grid grid-cols-2 gap-2 font-mono text-xs">
                <div className="bg-dark-850/80 p-2 rounded-lg border border-white/5">
                  <div className="text-[9px] text-slate-400 uppercase">Door Current</div>
                  <div className="text-slate-100 font-semibold text-xs mt-0.5">
                    {door ? door.motor_current_amps.toFixed(2) : '8.20'} A
                  </div>
                  <div className="text-[8px] text-slate-500">
                    Nom: 8.2A
                  </div>
                </div>

                <div className="bg-dark-850/80 p-2 rounded-lg border border-white/5">
                  <div className="text-[9px] text-slate-400 uppercase">Bogie RMS</div>
                  <div className={`font-semibold text-xs mt-0.5 ${(shm?.vibration_rms_g || 0) > 3.0 ? 'text-red-400' : 'text-slate-100'}`}>
                    {shm ? shm.vibration_rms_g.toFixed(2) : '1.80'} g
                  </div>
                  <div className="text-[8px] text-slate-500">
                    Peak: {shm ? shm.peak_frequency_hz : 95.0}Hz
                  </div>
                </div>

                <div className="bg-dark-850/80 p-2 rounded-lg border border-white/5">
                  <div className="text-[9px] text-slate-400 uppercase">ACV Delta-T</div>
                  <div className="text-sky-400 font-semibold text-xs mt-0.5">
                    {acv ? acv.delta_temp_c.toFixed(1) : '7.3'} °C
                  </div>
                  <div className="text-[8px] text-slate-500">
                    COP: {acv ? acv.efficiency_rating : 0.85}
                  </div>
                </div>

                <div className="bg-dark-850/80 p-2 rounded-lg border border-white/5">
                  <div className="text-[9px] text-slate-400 uppercase">Corrugation</div>
                  <div className={`font-semibold text-xs mt-0.5 ${(rail?.depth_microns || 0) > 30.0 ? 'text-amber-400' : 'text-slate-100'}`}>
                    {rail ? rail.depth_microns.toFixed(1) : '12.0'} μm
                  </div>
                  <div className="text-[8px] text-slate-500 truncate">
                    {rail ? rail.wavelength_class : 'NOMINAL'}
                  </div>
                </div>
              </div>

              {/* Physical Cross-Correlation Callout */}
              <div className="p-2 rounded-lg bg-red-950/30 border border-red-500/30 text-[10px] flex items-start space-x-2">
                <TrendingUp className="w-3.5 h-3.5 text-red-400 shrink-0 mt-0.5" />
                <div className="text-slate-300 leading-snug">
                  <strong className="text-red-400">Physics Coupling:</strong> KP {rail?.kp_start.toFixed(3)} corrugation induces shock loads into bogie weld {shm?.critical_weld_node}.
                </div>
              </div>

              {/* Chart 1: Door Motor Current */}
              <div className="bg-dark-850/60 p-2.5 rounded-xl border border-white/5">
                <div className="flex items-center justify-between mb-1 text-[10px] font-mono font-bold text-slate-300 uppercase">
                  <span>Door Motor Current</span>
                  <span className="text-red-400 font-bold">{door?.cycle_state}</span>
                </div>
                <div className="h-28 w-full">
                  <ReactECharts option={doorChartOption} style={{ height: '100%', width: '100%' }} notMerge={true} />
                </div>
              </div>

              {/* Chart 2: Bogie FFT Spectrum */}
              <div className="bg-dark-850/60 p-2.5 rounded-xl border border-white/5">
                <div className="flex items-center justify-between mb-1 text-[10px] font-mono font-bold text-slate-300 uppercase">
                  <span>Axle-Box FFT Vibration</span>
                  <span className="text-amber-400">{shm?.peak_frequency_hz} Hz</span>
                </div>
                <div className="h-28 w-full">
                  <ReactECharts option={fftChartOption} style={{ height: '100%', width: '100%' }} notMerge={true} />
                </div>
              </div>
            </div>
          )}

          {/* TAB 2: WHAT-IF COUNTERFACTUAL SANDBOX */}
          {rightDrawerTab === 'whatif' && (
            <div className="flex flex-col space-y-2">
              <div className="flex items-center justify-between text-[11px] font-mono text-slate-400 px-1">
                <span>LTA Actions</span>
                <span className="text-red-400 flex items-center space-x-1">
                  <Sparkles className="w-3 h-3" />
                  <span>PREDICTIVE</span>
                </span>
              </div>

              {interventions.map((item) => {
                const isActive = !!(activeInterventions as any)[item.action];

                return (
                  <div
                    key={item.action}
                    className={`p-2 rounded-xl border transition-all ${
                      isActive
                        ? 'bg-red-950/40 border-red-500/60 shadow-sm shadow-red-600/20'
                        : 'bg-dark-850/60 border-white/5 hover:border-white/15'
                    }`}
                  >
                    <div className="flex items-start justify-between">
                      <div>
                        <div className="text-xs font-semibold text-slate-100">
                          {item.title}
                        </div>
                        <div className="text-[10px] text-slate-400 mt-0.5 leading-snug">
                          {item.description}
                        </div>
                      </div>

                      <button
                        onClick={() => triggerWhatIf(item.action, !isActive)}
                        className={`text-[9px] font-mono font-bold px-2 py-0.5 rounded-md transition-all flex items-center space-x-1 shrink-0 ml-2 ${
                          isActive
                            ? 'bg-red-600 text-white shadow-sm shadow-red-600/40'
                            : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                        }`}
                      >
                        {isActive && <Check className="w-2.5 h-2.5 stroke-[3]" />}
                        <span>{isActive ? 'APPLIED' : 'APPLY'}</span>
                      </button>
                    </div>

                    <div className="mt-1 pt-1 border-t border-white/5 text-[9px] font-mono text-emerald-400">
                      {item.gain}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </aside>
  );
};
