import React, { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import { AlertCircle, CheckCircle2, TrendingUp, Cpu } from 'lucide-react';
import { useTwinStore } from '../../store/useTwinStore';

export const TelemetryMonitor: React.FC = () => {
  const currentFrame = useTwinStore((state) => state.currentFrame);
  const selectedSubsystem = useTwinStore((state) => state.selectedSubsystem);

  const subsystems = currentFrame?.subsystems;
  const door = subsystems?.door;
  const acv = subsystems?.acv;
  const shm = subsystems?.shm;
  const rail = subsystems?.rail_corrugation;

  // Chart 1: Door Motor Current Profile
  const doorChartOption = useMemo(() => {
    const rawWaveform = door?.waveform_window || [2.1, 4.3, 7.8, 11.85, 11.2, 5.0, 1.2];
    const nominalWaveform = [1.8, 3.8, 6.5, 8.2, 7.8, 4.2, 1.0];

    return {
      backgroundColor: 'transparent',
      grid: { top: 25, right: 15, bottom: 20, left: 35 },
      tooltip: { trigger: 'axis' },
      xAxis: {
        type: 'category',
        data: ['0.0s', '0.5s', '1.0s', '1.5s', '2.0s', '2.5s', '3.0s'],
        axisLine: { lineStyle: { color: '#334155' } },
        axisLabel: { color: '#94a3b8', fontSize: 10, fontFamily: 'monospace' },
      },
      yAxis: {
        type: 'value',
        name: 'Amps',
        nameTextStyle: { color: '#64748b', fontSize: 9 },
        splitLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } },
        axisLabel: { color: '#94a3b8', fontSize: 10, fontFamily: 'monospace' },
      },
      series: [
        {
          name: 'Active Current',
          type: 'line',
          smooth: true,
          data: rawWaveform,
          lineStyle: { color: (door?.anomaly_score || 0) > 0.5 ? '#f43f5e' : '#06b6d4', width: 2.5 },
          areaStyle: {
            color: (door?.anomaly_score || 0) > 0.5 ? 'rgba(244, 63, 94, 0.2)' : 'rgba(6, 182, 212, 0.2)',
          },
        },
        {
          name: 'Nominal Envelope',
          type: 'line',
          smooth: true,
          data: nominalWaveform,
          lineStyle: { color: '#10b981', width: 1.5, type: 'dashed' },
        },
      ],
    };
  }, [door]);

  // Chart 2: Bogie FFT Vibration Spectrum
  const fftChartOption = useMemo(() => {
    const spectrum = shm?.fft_spectrum || [
      { freq_hz: 25.0, amp: 0.42 },
      { freq_hz: 142.5, amp: 2.89 },
      { freq_hz: 300.0, amp: 0.15 },
    ];

    return {
      backgroundColor: 'transparent',
      grid: { top: 25, right: 15, bottom: 20, left: 35 },
      tooltip: { trigger: 'item' },
      xAxis: {
        type: 'category',
        data: spectrum.map((s) => `${s.freq_hz}Hz`),
        axisLine: { lineStyle: { color: '#334155' } },
        axisLabel: { color: '#94a3b8', fontSize: 9, fontFamily: 'monospace' },
      },
      yAxis: {
        type: 'value',
        name: 'Amplitude (g)',
        nameTextStyle: { color: '#64748b', fontSize: 9 },
        splitLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } },
        axisLabel: { color: '#94a3b8', fontSize: 10, fontFamily: 'monospace' },
      },
      series: [
        {
          name: 'FFT Amplitude',
          type: 'bar',
          data: spectrum.map((s) => s.amp),
          itemStyle: {
            color: (params: any) => {
              const val = params.value;
              if (val > 2.0) return '#f43f5e';
              if (val > 1.0) return '#f59e0b';
              return '#06b6d4';
            },
            borderRadius: [3, 3, 0, 0],
          },
        },
      ],
    };
  }, [shm]);

  return (
    <aside className="absolute top-20 right-4 z-20 w-84 pointer-events-none flex flex-col space-y-3">
      {/* 1. Subsystem Diagnostics Summary Card */}
      <div className="glass-panel p-4 rounded-xl pointer-events-auto border-white/10">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center space-x-2">
            <Cpu className="w-4 h-4 text-cyan-400" />
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
              {selectedSubsystem === 'door'
                ? 'Door Kinematic Analysis'
                : selectedSubsystem === 'acv'
                ? 'ACV Thermodynamics'
                : selectedSubsystem === 'shm'
                ? 'Bogie Structural Fatigue'
                : selectedSubsystem === 'rail'
                ? 'Track Roughness Index'
                : 'Cross-Subsystem Correlation'}
            </h3>
          </div>
          <span className="text-[10px] font-mono text-slate-400">100ms TICK</span>
        </div>

        {/* Selected domain quick stats */}
        <div className="grid grid-cols-2 gap-2 font-mono text-xs mt-3">
          <div className="bg-dark-850/80 p-2 rounded-lg border border-white/5">
            <div className="text-[10px] text-slate-400 uppercase">Door Current</div>
            <div className="text-slate-100 font-semibold text-sm">
              {door ? door.motor_current_amps.toFixed(2) : '8.20'} A
            </div>
            <div className="text-[9px] text-slate-400 mt-0.5">
              Nominal: {door ? door.nominal_current_amps : 8.2}A
            </div>
          </div>

          <div className="bg-dark-850/80 p-2 rounded-lg border border-white/5">
            <div className="text-[10px] text-slate-400 uppercase">Bogie RMS</div>
            <div className={`font-semibold text-sm ${(shm?.vibration_rms_g || 0) > 3.0 ? 'text-rose-400' : 'text-slate-100'}`}>
              {shm ? shm.vibration_rms_g.toFixed(2) : '1.80'} g
            </div>
            <div className="text-[9px] text-slate-400 mt-0.5">
              Peak: {shm ? shm.peak_frequency_hz : 95.0}Hz
            </div>
          </div>

          <div className="bg-dark-850/80 p-2 rounded-lg border border-white/5">
            <div className="text-[10px] text-slate-400 uppercase">ACV Delta-T</div>
            <div className="text-cyan-400 font-semibold text-sm">
              {acv ? acv.delta_temp_c.toFixed(1) : '7.3'} °C
            </div>
            <div className="text-[9px] text-slate-400 mt-0.5">
              COP: {acv ? acv.efficiency_rating : 0.85}
            </div>
          </div>

          <div className="bg-dark-850/80 p-2 rounded-lg border border-white/5">
            <div className="text-[10px] text-slate-400 uppercase">Corrugation</div>
            <div className={`font-semibold text-sm ${(rail?.depth_microns || 0) > 30.0 ? 'text-amber-400' : 'text-slate-100'}`}>
              {rail ? rail.depth_microns.toFixed(1) : '12.0'} μm
            </div>
            <div className="text-[9px] text-slate-400 mt-0.5 truncate">
              {rail ? rail.wavelength_class : 'NOMINAL'}
            </div>
          </div>
        </div>

        {/* Physical Cross-Correlation Callout */}
        <div className="mt-3 p-2.5 rounded-lg bg-cyan-950/30 border border-cyan-500/20 text-[11px] flex items-start space-x-2">
          <TrendingUp className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
          <div className="text-slate-300 leading-relaxed">
            <strong className="text-cyan-300">Physics Coupling:</strong> Corrugation at{' '}
            <span className="font-mono text-white">KP {rail?.kp_start.toFixed(3)}</span> induces high-frequency
            dynamic load into wheelset, driving bogie weld stress at{' '}
            <span className="font-mono text-white">{shm?.critical_weld_node}</span>.
          </div>
        </div>
      </div>

      {/* 2. Synchronized EChart: Door Motor Current Waveform */}
      <div className="glass-panel p-3 rounded-xl pointer-events-auto border-white/10">
        <div className="flex items-center justify-between mb-1 px-1">
          <span className="text-[11px] font-bold text-slate-200 uppercase tracking-wide">
            Door Motor Current Profile
          </span>
          <span className="text-[10px] font-mono text-cyan-400">
            {door?.fault_type === 'NONE' ? 'NOMINAL' : door?.fault_type}
          </span>
        </div>
        <div className="h-36 w-full">
          <ReactECharts option={doorChartOption} style={{ height: '100%', width: '100%' }} notMerge={true} />
        </div>
      </div>

      {/* 3. Synchronized EChart: Bogie FFT Spectrum */}
      <div className="glass-panel p-3 rounded-xl pointer-events-auto border-white/10">
        <div className="flex items-center justify-between mb-1 px-1">
          <span className="text-[11px] font-bold text-slate-200 uppercase tracking-wide">
            Axle-Box FFT Vibration Spectrum
          </span>
          <span className="text-[10px] font-mono text-amber-400">
            PEAK {shm?.peak_frequency_hz} Hz
          </span>
        </div>
        <div className="h-36 w-full">
          <ReactECharts option={fftChartOption} style={{ height: '100%', width: '100%' }} notMerge={true} />
        </div>
      </div>
    </aside>
  );
};
