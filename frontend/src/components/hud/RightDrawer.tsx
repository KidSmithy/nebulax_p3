import React, { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import {
  Activity,
  BarChart2,
  BrainCircuit,
  ChevronLeft,
  ChevronRight,
  Link2,
  Wrench,
} from 'lucide-react';
import { useTwinStore } from '../../store/useTwinStore';
import { MetricReadout } from './MetricReadout';
import { InfoTip } from './InfoTip';
import { AIInsightPanel } from './AIInsightPanel';
import { WhatIfPanel } from './WhatIfPanel';
import { explainTerm, plainTerm } from '../../lib/metricGlossary';

export const RightDrawer: React.FC = () => {
  const isRightDrawerOpen = useTwinStore((state) => state.isRightDrawerOpen);
  const toggleRightDrawer = useTwinStore((state) => state.toggleRightDrawer);
  const rightDrawerTab = useTwinStore((state) => state.rightDrawerTab);
  const setRightDrawerTab = useTwinStore((state) => state.setRightDrawerTab);
  const currentFrame = useTwinStore((state) => state.currentFrame);
  const beginner = useTwinStore((state) => state.uiMode) === 'beginner';

  const subsystems = currentFrame?.subsystems;
  const door = subsystems?.door;
  const acv = subsystems?.acv;
  const shm = subsystems?.shm;
  const rail = subsystems?.rail_corrugation;
  const status = currentFrame?.plain_status ?? {};

  // Chart 1: Door Motor Current (measured vs healthy envelope)
  const doorChartOption = useMemo(() => {
    const rawWaveform = door?.waveform_window || [];
    const peakNominal = door?.nominal_current_amps ?? 9.4;
    const shape = [0.22, 0.48, 0.88, 1.0, 0.94, 0.72, 0.48, 0.3, 0.15, 0.05];
    const nominalWaveform = shape.map((s) => +(s * peakNominal).toFixed(2));

    return {
      backgroundColor: 'transparent',
      grid: { top: 22, right: 8, bottom: 18, left: 28 },
      tooltip: { trigger: 'axis' },
      legend: {
        show: true,
        top: 0,
        right: 0,
        itemWidth: 8,
        itemHeight: 6,
        textStyle: { fontSize: 8, color: '#64748b' },
        data: ['Measured', 'Healthy door'],
      },
      xAxis: {
        type: 'category',
        data: rawWaveform.map((_, i) => `${(i * 0.34).toFixed(1)}s`),
        axisLine: { lineStyle: { color: '#cbd5e1' } },
        axisLabel: { color: '#94a3b8', fontSize: 8, fontFamily: 'monospace' },
      },
      yAxis: {
        type: 'value',
        name: 'amps',
        nameTextStyle: { color: '#94a3b8', fontSize: 8 },
        splitLine: { lineStyle: { color: 'rgba(0,0,0,0.06)' } },
        axisLabel: { color: '#94a3b8', fontSize: 8, fontFamily: 'monospace' },
      },
      series: [
        {
          name: 'Measured',
          type: 'line',
          smooth: true,
          symbol: 'none',
          data: rawWaveform,
          lineStyle: { color: '#ED1C24', width: 2.2 },
          areaStyle: { color: 'rgba(237, 28, 36, 0.14)' },
        },
        {
          name: 'Healthy door',
          type: 'line',
          smooth: true,
          symbol: 'none',
          data: nominalWaveform,
          lineStyle: { color: '#009645', width: 1.5, type: 'dashed' },
        },
      ],
    };
  }, [door]);

  // Chart 2: Bogie FFT Spectrum
  const fftChartOption = useMemo(() => {
    const spectrum = shm?.fft_spectrum || [];
    return {
      backgroundColor: 'transparent',
      grid: { top: 16, right: 8, bottom: 18, left: 28 },
      tooltip: {
        trigger: 'item',
        formatter: (p: any) =>
          `${p.name}: ${p.value} g<br/><span style="font-size:10px;color:#64748b">` +
          `${p.value > 2.0 ? 'Strong shaking at this pitch' : p.value > 1.0 ? 'Moderate' : 'Background level'}</span>`,
      },
      xAxis: {
        type: 'category',
        data: spectrum.map((s) => `${s.freq_hz}Hz`),
        axisLine: { lineStyle: { color: '#cbd5e1' } },
        axisLabel: { color: '#94a3b8', fontSize: 7, fontFamily: 'monospace', interval: 1 },
      },
      yAxis: {
        type: 'value',
        name: 'g',
        nameTextStyle: { color: '#94a3b8', fontSize: 8 },
        splitLine: { lineStyle: { color: 'rgba(0,0,0,0.06)' } },
        axisLabel: { color: '#94a3b8', fontSize: 8, fontFamily: 'monospace' },
      },
      series: [
        {
          name: 'Vibration',
          type: 'bar',
          data: spectrum.map((s) => s.amp),
          itemStyle: {
            color: (params: any) => {
              const val = params.value;
              if (val > 2.0) return '#ED1C24';
              if (val > 1.0) return '#FF9E1B';
              return '#009645';
            },
            borderRadius: [2, 2, 0, 0],
          },
        },
      ],
    };
  }, [shm]);

  const doorFault = explainTerm(door?.fault_type);
  const acvFault = explainTerm(acv?.fault_type);
  const railClass = explainTerm(rail?.wavelength_class);
  const weldNode = explainTerm(shm?.critical_weld_node);

  const tabs = [
    { id: 'telemetry' as const, label: beginner ? 'READINGS' : 'TELEMETRY', icon: Activity },
    { id: 'ai' as const, label: 'AI', icon: BrainCircuit },
    { id: 'whatif' as const, label: beginner ? 'REPAIRS' : 'WHAT-IF', icon: Wrench },
  ];

  return (
    <aside className="absolute top-20 right-4 bottom-20 z-20 transition-all duration-300 pointer-events-none flex flex-col items-end">
      {!isRightDrawerOpen ? (
        <button
          onClick={toggleRightDrawer}
          className="glass-panel-glow px-3 py-2 rounded-xl pointer-events-auto flex items-center space-x-2 text-xs font-mono text-red-600 hover:text-red-800 transition-all shadow-xl"
        >
          <BarChart2 className="w-4 h-4 text-red-600" />
          <span>{beginner ? 'OPEN DETAILS' : 'EXPAND COCKPIT'}</span>
          <ChevronLeft className="w-3.5 h-3.5" />
        </button>
      ) : (
        <div className="glass-panel p-3 rounded-2xl pointer-events-auto w-[21rem] shadow-2xl flex flex-col space-y-2.5 max-h-full">
          {/* Tabs */}
          <div className="flex items-center justify-between border-b border-slate-200 pb-2 shrink-0">
            <div className="flex items-center space-x-1 bg-slate-100 p-0.5 rounded-lg border border-slate-200 font-mono text-[10px]">
              {tabs.map((t) => {
                const Icon = t.icon;
                return (
                  <button
                    key={t.id}
                    onClick={() => setRightDrawerTab(t.id)}
                    className={`px-2 py-1 rounded-md flex items-center space-x-1 transition-all ${
                      rightDrawerTab === t.id
                        ? 'bg-red-600 text-white font-bold shadow-sm'
                        : 'text-slate-500 hover:text-slate-800'
                    }`}
                  >
                    <Icon className="w-3 h-3" />
                    <span>{t.label}</span>
                  </button>
                );
              })}
            </div>

            <button
              onClick={toggleRightDrawer}
              className="p-1 rounded-lg text-slate-400 hover:text-slate-800 hover:bg-slate-100 transition-colors"
              title="Collapse panel"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>

          <div className="overflow-y-auto custom-scrollbar pr-0.5 -mr-0.5">
            {/* ============================ READINGS ======================= */}
            {rightDrawerTab === 'telemetry' && (
              <div className="flex flex-col space-y-2.5">
                {beginner && (
                  <p className="text-[10px] text-slate-500 leading-relaxed">
                    Four systems are monitored. Hover the{' '}
                    <span className="font-bold text-slate-700">?</span> beside any reading to see
                    what it means and whether it is healthy.
                  </p>
                )}

                {/* Doors */}
                <section className="space-y-1.5">
                  <h3 className="text-[9.5px] font-bold text-slate-500 uppercase tracking-wide flex items-center gap-1">
                    {beginner ? 'Passenger doors' : 'Door subsystem'}
                    <span className="font-mono text-slate-400 normal-case">
                      {plainTerm(door?.cycle_state)}
                    </span>
                  </h3>
                  <div className="grid grid-cols-2 gap-1.5">
                    <MetricReadout
                      path="door.motor_current_amps"
                      value={door?.motor_current_amps}
                      status={status['door.motor_current_amps']}
                      beginner={beginner}
                      footnote={
                        door ? `healthy: ${door.nominal_current_amps.toFixed(2)} A` : undefined
                      }
                    />
                    <MetricReadout
                      path="door.anomaly_score"
                      value={door?.anomaly_score}
                      status={status['door.anomaly_score']}
                      beginner={beginner}
                      side="right"
                    />
                    <MetricReadout
                      path="door.transit_time_seconds"
                      value={door?.transit_time_seconds}
                      status={status['door.transit_time_seconds']}
                      beginner={beginner}
                    />
                    <MetricReadout
                      path="door.ghost_deviation_mm"
                      value={door?.ghost_deviation_mm}
                      status={status['door.ghost_deviation_mm']}
                      beginner={beginner}
                      side="right"
                    />
                  </div>
                  {doorFault && door?.fault_type !== 'NONE' && (
                    <div className="p-1.5 rounded-lg bg-amber-50 border border-amber-200 text-[10px] text-amber-900 leading-relaxed">
                      <strong>{doorFault.plainName}:</strong> {doorFault.explanation}
                    </div>
                  )}
                </section>

                {/* Aircon */}
                <section className="space-y-1.5">
                  <h3 className="text-[9.5px] font-bold text-slate-500 uppercase tracking-wide">
                    {beginner ? 'Air-conditioning' : 'ACV climate pack'}
                  </h3>
                  <div className="grid grid-cols-2 gap-1.5">
                    <MetricReadout
                      path="acv.delta_temp_c"
                      value={acv?.delta_temp_c}
                      status={status['acv.delta_temp_c']}
                      beginner={beginner}
                      footnote={
                        acv ? `${acv.return_temp_c}°C in → ${acv.supply_temp_c}°C out` : undefined
                      }
                    />
                    <MetricReadout
                      path="acv.efficiency_rating"
                      value={acv?.efficiency_rating}
                      status={status['acv.efficiency_rating']}
                      beginner={beginner}
                      side="right"
                    />
                    <MetricReadout
                      path="acv.compressor_power_kw"
                      value={acv?.compressor_power_kw}
                      status={status['acv.compressor_power_kw']}
                      beginner={beginner}
                    />
                  </div>
                  {acvFault && acv?.fault_type !== 'NONE' && (
                    <div className="p-1.5 rounded-lg bg-amber-50 border border-amber-200 text-[10px] text-amber-900 leading-relaxed">
                      <strong>{acvFault.plainName}:</strong> {acvFault.explanation}
                    </div>
                  )}
                </section>

                {/* Wheels */}
                <section className="space-y-1.5">
                  <h3 className="text-[9.5px] font-bold text-slate-500 uppercase tracking-wide">
                    {beginner ? 'Wheels & frame' : 'Bogie structural health'}
                  </h3>
                  <div className="grid grid-cols-2 gap-1.5">
                    <MetricReadout
                      path="shm.vibration_rms_g"
                      value={shm?.vibration_rms_g}
                      status={status['shm.vibration_rms_g']}
                      beginner={beginner}
                      footnote={shm ? `pitch: ${shm.peak_frequency_hz} Hz` : undefined}
                    />
                    <MetricReadout
                      path="shm.bearing_defect_prob"
                      value={shm?.bearing_defect_prob}
                      status={status['shm.bearing_defect_prob']}
                      beginner={beginner}
                      side="right"
                    />
                    <MetricReadout
                      path="shm.fatigue_damage_index"
                      value={shm?.fatigue_damage_index}
                      status={status['shm.fatigue_damage_index']}
                      beginner={beginner}
                    />
                    <div className="rounded-lg border border-slate-200 bg-slate-50 p-1.5">
                      <div className="flex items-center gap-1">
                        <span className="text-[9.5px] font-semibold text-slate-600 uppercase tracking-wide">
                          {beginner ? 'Most stressed joint' : 'Critical weld node'}
                        </span>
                        {weldNode && (
                          <InfoTip
                            title={weldNode.plainName}
                            whatItIs={weldNode.explanation}
                            whyItMatters="Cracks start at the most heavily loaded weld, so this is where inspectors look first."
                            side="right"
                          />
                        )}
                      </div>
                      <div className="mt-0.5 text-[10px] font-bold text-slate-700 leading-tight">
                        {beginner ? weldNode?.plainName ?? '—' : shm?.critical_weld_node ?? '—'}
                      </div>
                    </div>
                  </div>
                </section>

                {/* Track */}
                <section className="space-y-1.5">
                  <h3 className="text-[9.5px] font-bold text-slate-500 uppercase tracking-wide">
                    {beginner ? 'Track under the train' : 'Rail corrugation'}
                  </h3>
                  <div className="grid grid-cols-2 gap-1.5">
                    <MetricReadout
                      path="rail_corrugation.depth_microns"
                      value={rail?.depth_microns}
                      status={status['rail_corrugation.depth_microns']}
                      beginner={beginner}
                      footnote={rail ? `KP ${rail.kp_start.toFixed(3)}` : undefined}
                    />
                    <MetricReadout
                      path="rail_corrugation.severity_score"
                      value={rail?.severity_score}
                      status={status['rail_corrugation.severity_score']}
                      beginner={beginner}
                      side="right"
                    />
                  </div>
                  {railClass && (
                    <div className="p-1.5 rounded-lg bg-slate-50 border border-slate-200 text-[10px] text-slate-700 leading-relaxed">
                      <strong>{railClass.plainName}:</strong> {railClass.explanation}
                    </div>
                  )}
                </section>

                {/* Physics coupling, explained */}
                <div className="p-2 rounded-lg bg-red-50 border border-red-200 text-[10px] flex items-start gap-1.5">
                  <Link2 className="w-3 h-3 text-red-600 shrink-0 mt-0.5" />
                  <div className="text-slate-700 leading-relaxed">
                    <strong className="text-red-700">How these connect: </strong>
                    {beginner ? (
                      <>
                        the ripples in the rail at KP {rail?.kp_start.toFixed(3)} punch the wheels as
                        the train rolls over them. That shock travels up into the frame, which is why
                        rough track and a shaking bogie always rise together.
                      </>
                    ) : (
                      <>
                        KP {rail?.kp_start.toFixed(3)} corrugation induces shock loads into bogie
                        weld {shm?.critical_weld_node}.
                      </>
                    )}
                  </div>
                </div>

                {/* Charts */}
                <div className="bg-white p-2 rounded-xl border border-slate-200">
                  <div className="flex items-center justify-between mb-1 text-[9.5px] font-bold text-slate-600 uppercase">
                    <span className="flex items-center gap-1">
                      {beginner ? 'Door effort over one cycle' : 'Door motor current'}
                      <InfoTip
                        title="Door effort over one cycle"
                        whatItIs="The red line is the current the motor is actually drawing as the door travels. The dashed green line is what a healthy door would draw."
                        whyItMatters="A red line sitting above the green one means the door is working harder than it should. The size of the gap is the size of the problem."
                        analogy="Like comparing your heart rate on a hill against your normal resting pace."
                        side="right"
                      />
                    </span>
                    <span className="text-red-600 font-bold normal-case">
                      {plainTerm(door?.cycle_state)}
                    </span>
                  </div>
                  <div className="h-28 w-full">
                    <ReactECharts
                      option={doorChartOption}
                      style={{ height: '100%', width: '100%' }}
                      notMerge
                    />
                  </div>
                </div>

                <div className="bg-white p-2 rounded-xl border border-slate-200">
                  <div className="flex items-center justify-between mb-1 text-[9.5px] font-bold text-slate-600 uppercase">
                    <span className="flex items-center gap-1">
                      {beginner ? 'Vibration by pitch' : 'Axle-box FFT spectrum'}
                      <InfoTip
                        title="Vibration broken down by pitch"
                        whatItIs="Vibration is split into the different frequencies making it up, from low rumbles on the left to high whines on the right. Taller bars mean more shaking at that pitch."
                        whyItMatters="Each fault has its own signature pitch, so the tallest bar tells you which component is complaining rather than just that something is wrong."
                        analogy="Like an equaliser display on a music player showing which notes are loudest."
                        side="right"
                      />
                    </span>
                    <span className="text-amber-600 normal-case">
                      loudest at {shm?.peak_frequency_hz} Hz
                    </span>
                  </div>
                  <div className="h-28 w-full">
                    <ReactECharts
                      option={fftChartOption}
                      style={{ height: '100%', width: '100%' }}
                      notMerge
                    />
                  </div>
                </div>
              </div>
            )}

            {/* ============================== AI ========================== */}
            {rightDrawerTab === 'ai' && <AIInsightPanel />}

            {/* ============================ WHAT-IF ======================= */}
            {rightDrawerTab === 'whatif' && <WhatIfPanel />}
          </div>
        </div>
      )}
    </aside>
  );
};
