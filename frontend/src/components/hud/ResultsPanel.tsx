import React, { useState } from 'react';
import { Activity, ChevronLeft, ChevronRight, DoorOpen, Download, GitCommit, Table2, Wind } from 'lucide-react';
import { useTwinStore } from '../../store/useTwinStore';
import { Finding, Line, MetricStatus, SubmissionCsv } from '../../types/telemetry';
import { downloadSubmission } from '../../lib/downloadCsv';
import { STATUS_SHORT, STATUS_STYLES } from '../../lib/metricGlossary';
import { LINES } from '../../lib/lines';
import { ACTIONS } from './WhatIfPanel';

type Subsystem = 'acv' | 'door' | 'shm' | 'rail';

const ORDER: Subsystem[] = ['acv', 'door', 'shm', 'rail'];

const META: Record<Subsystem, { label: string; tab: string; Icon: React.FC<{ className?: string }>; color: string }> = {
  acv: { label: 'Air conditioning', tab: 'Aircon', Icon: Wind, color: '#0891b2' },
  door: { label: 'Passenger doors', tab: 'Doors', Icon: DoorOpen, color: '#2563eb' },
  shm: { label: 'Structural health (SHM)', tab: 'SHM', Icon: Activity, color: '#7c3aed' },
  rail: { label: 'Rail corrugation', tab: 'Rail', Icon: GitCommit, color: '#c026d3' },
};

/** The one number that best summarises each finding, shown under its tab's title. */
function headline(sub: Subsystem, f: Finding): string {
  switch (sub) {
    case 'acv':
      return f.most_likely_faulty_car ? `Car ${Number(f.most_likely_faulty_car)}` : 'No fault found';
    case 'door':
      return `${f.abnormal_cycles ?? 0} / ${f.total_cycles ?? 0} cycles abnormal`;
    case 'shm':
      return `${f.vibration_rms_g ?? '-'} g RMS`;
    case 'rail':
      return `${f.depth_microns ?? '-'} μm deep`;
  }
}

const StatusPill: React.FC<{ status: MetricStatus }> = ({ status }) => {
  const st = STATUS_STYLES[status] ?? STATUS_STYLES.UNKNOWN;
  return (
    <span className={`px-1.5 py-0.5 rounded-full text-label font-semibold border whitespace-nowrap ${st.bg} ${st.text} ${st.border}`}>
      {STATUS_SHORT[status] ?? status}
    </span>
  );
};

const Th: React.FC<{ children?: React.ReactNode; right?: boolean }> = ({ children, right }) => (
  <th className={`px-2 py-1.5 text-[9px] font-bold tracking-[0.12em] text-slate-400 uppercase ${right ? 'text-right' : 'text-left'}`}>
    {children}
  </th>
);

const Td: React.FC<{ children?: React.ReactNode; right?: boolean; className?: string }> = ({ children, right, className = '' }) => (
  <td className={`px-2 py-1.5 text-[11px] text-slate-700 ${right ? 'text-right font-mono' : ''} ${className}`}>{children}</td>
);

/** Label / value rows for a finding's headline numbers. */
const KeyValues: React.FC<{ rows: Array<[string, React.ReactNode]> }> = ({ rows }) => (
  <table className="w-full border border-slate-200 rounded overflow-hidden">
    <tbody>
      {rows.map(([k, v]) => (
        <tr key={k} className="border-b border-slate-100 last:border-b-0">
          <Td className="text-slate-500">{k}</Td>
          <Td right className="font-semibold text-slate-900">{v}</Td>
        </tr>
      ))}
    </tbody>
  </table>
);

const DataTable: React.FC<{ head: Array<{ label: string; right?: boolean }>; children: React.ReactNode }> = ({ head, children }) => (
  <table className="w-full border border-slate-200">
    <thead className="bg-slate-50 sticky top-0">
      <tr>
        {head.map((h) => (
          <Th key={h.label} right={h.right}>{h.label}</Th>
        ))}
      </tr>
    </thead>
    <tbody className="divide-y divide-slate-100">{children}</tbody>
  </table>
);

const Detail: React.FC<{ sub: Subsystem; f: Finding }> = ({ sub, f }) => {
  const repair = ACTIONS.find((a) => a.action === f.recommended_action);
  const action: [string, React.ReactNode] = ['Recommended repair', repair ? repair.plainTitle : 'None needed'];

  if (sub === 'acv') {
    const cars = f.car_diagnostics ?? [];
    return (
      <div className="space-y-3">
        <KeyValues
          rows={[
            ['Most likely faulty car', f.most_likely_faulty_car ? `Car ${Number(f.most_likely_faulty_car)}` : 'None'],
            ['Localisation strength', f.confidence ?? '-'],
            action,
          ]}
        />
        <DataTable head={[{ label: 'Rank' }, { label: 'Car' }, { label: 'Health idx', right: true }, { label: 'Warmer by', right: true }, { label: 'Persist.', right: true }]}>
          {cars.map((c) => (
            <tr key={c.car} className={c.rank === 1 ? 'bg-slate-50 font-semibold' : ''}>
              <Td>{c.rank}</Td>
              <Td>Car {Number(c.car)}</Td>
              <Td right>{c.health_index.toFixed(2)}</Td>
              <Td right>{c.mean_rel_c >= 0 ? '+' : ''}{c.mean_rel_c.toFixed(2)} °C</Td>
              <Td right>{c.persistence_pct.toFixed(0)}%</Td>
            </tr>
          ))}
        </DataTable>
      </div>
    );
  }

  if (sub === 'door') {
    const segments = f.segments ?? [];
    return (
      <div className="space-y-3">
        <KeyValues
          rows={[
            ['Cycles analysed', f.total_cycles ?? '-'],
            ['Abnormal cycles', `${f.abnormal_cycles ?? 0} (${f.fault_rate_pct ?? 0}%)`],
            ['Mean motor current', `${f.mean_current_rms_a ?? '-'} A`],
            ['Peak motor current', `${f.max_current_peak_a ?? '-'} A`],
            ['Mean cycle time', `${f.mean_duration_s ?? '-'} s`],
            ['Fault class', f.fault_type ?? '-'],
            action,
          ]}
        />
        <DataTable head={[{ label: 'Cycle' }, { label: 'Op' }, { label: 'Time', right: true }, { label: 'RMS', right: true }, { label: 'Peak', right: true }, { label: 'Fault', right: true }]}>
          {segments.map((s) => {
            const abnormal = s.status !== 'Normal';
            return (
              <tr key={s.cycle} className={abnormal ? 'bg-red-50/60' : ''}>
                <Td>{s.cycle}</Td>
                <Td>{s.operation}</Td>
                <Td right>{s.duration_s.toFixed(2)}s</Td>
                <Td right>{s.current_rms_a.toFixed(2)}A</Td>
                <Td right>{s.peak_current_a.toFixed(2)}A</Td>
                <Td right className={abnormal ? 'text-status-fault font-semibold' : ''}>
                  {Math.round(s.fault_probability * 100)}%
                </Td>
              </tr>
            );
          })}
        </DataTable>
        {(f.total_cycles ?? 0) > segments.length && (
          <p className="text-label text-slate-400">Showing the first {segments.length} of {f.total_cycles} cycles.</p>
        )}
      </div>
    );
  }

  if (sub === 'shm') {
    const peaks = [...(f.fft_spectrum ?? [])].filter((p) => p.freq_hz > 0).sort((a, b) => b.amp - a.amp).slice(0, 6);
    return (
      <div className="space-y-3">
        <KeyValues
          rows={[
            ['Vibration RMS', `${f.vibration_rms_g ?? '-'} g`],
            ['Peak acceleration', `${f.abs_peak_g ?? '-'} g`],
            ['Dominant frequency', `${f.dominant_freq_hz ?? '-'} Hz`],
            ['Fatigue damage index', f.fatigue_damage_index ?? '-'],
            ['Bearing defect probability', `${Math.round((f.bearing_defect_prob ?? 0) * 100)}%`],
            ['Most stressed node', f.critical_weld_node ?? '-'],
            action,
          ]}
        />
        {peaks.length > 0 && (
          <DataTable head={[{ label: 'Spectrum peak' }, { label: 'Frequency', right: true }, { label: 'Amplitude', right: true }]}>
            {peaks.map((p) => (
              <tr key={p.freq_hz}>
                <Td>#{peaks.indexOf(p) + 1}</Td>
                <Td right>{p.freq_hz} Hz</Td>
                <Td right>{p.amp.toFixed(3)}</Td>
              </tr>
            ))}
          </DataTable>
        )}
      </div>
    );
  }

  return (
    <KeyValues
      rows={[
        ['Estimated ripple depth', `${f.depth_microns ?? '-'} μm`],
        ['Wavelength class', (f.wavelength_class ?? '-').replace('_', ' ')],
        ['Maintenance urgency', (f.maintenance_urgency ?? '-').replace(/_/g, ' ')],
        ['Grinding priority rank', f.grinding_priority_rank ?? '-'],
        ['Mean vibration (all channels)', `${f.mean_channel_rms ?? '-'} g`],
        action,
      ]}
    />
  );
};

/** Downloads the finding's prediction CSV in the PS3 submission format. */
const DownloadButton: React.FC<{ submission?: SubmissionCsv | null }> = ({ submission }) => {
  if (!submission || submission.rows.length === 0) return null;
  return (
    <button
      onClick={() => downloadSubmission(submission)}
      className="w-full flex items-center justify-center gap-2 border border-slate-300 text-slate-700 text-xs font-semibold py-2 rounded hover:bg-slate-50 hover:text-slate-900 transition-colors duration-150"
    >
      <Download className="w-3.5 h-3.5" />
      Download {submission.filename}
      <span className="font-mono font-normal text-slate-400">
        · {submission.rows.length} {submission.rows.length === 1 ? 'row' : 'rows'}
      </span>
    </button>
  );
};

/**
 * Right-hand results panel for the active line: one tab per unresolved upload,
 * each showing that finding's breakdown and its downloadable prediction CSV. A
 * finding leaves the panel (and its tab goes) as soon as it is resolved. Docks
 * into the layout on wide screens (the scene reflows) and overlays on narrow
 * ones; collapses to a handle on the right edge.
 */
export const ResultsPanel: React.FC = () => {
  const open = useTwinStore((s) => s.resultsPanelOpen);
  const setOpen = useTwinStore((s) => s.setResultsPanelOpen);
  const activeLine: Line = useTwinStore((s) => s.activeLine);
  const findings = useTwinStore((s) => s.currentFrame?.uploads_by_line?.[s.activeLine]);
  const [picked, setPicked] = useState<Subsystem | null>(null);

  const present = ORDER.filter((sub) => findings?.[sub]);
  const selected: Subsystem | null = picked && present.includes(picked) ? picked : present[0] ?? null;
  const finding = selected ? findings![selected]! : null;

  if (!open) {
    return (
      <button
        onClick={() => setOpen(true)}
        title="Show results"
        className="fixed right-0 top-1/2 -translate-y-1/2 z-30 glass-panel-floating rounded-l pl-1.5 pr-2 py-2.5 flex flex-col items-center gap-1.5 text-slate-500 hover:text-ink-900 transition-colors duration-150"
      >
        <ChevronLeft className="w-4 h-4" />
        <Table2 className="w-4 h-4" />
        {present.length > 0 && (
          <span className="text-label font-mono font-bold bg-ink-900 text-white rounded-full px-1.5">{present.length}</span>
        )}
      </button>
    );
  }

  return (
    <aside className="fixed xl:relative top-0 right-0 h-full w-[400px] max-w-full z-40 xl:z-auto bg-white border-l border-slate-200 flex flex-col shrink-0">
      {/* Hide handle: a tab on the panel's left edge, halfway down. */}
      <button
        onClick={() => setOpen(false)}
        title="Hide results"
        aria-label="Hide results"
        className="absolute top-1/2 -left-7 -translate-y-1/2 z-10 w-7 h-14 flex items-center justify-center rounded-l border border-r-0 border-slate-200 bg-white text-slate-500 hover:text-slate-900 hover:bg-slate-50 shadow-sm transition-colors"
      >
        <ChevronRight className="w-5 h-5" />
      </button>

      <div className="flex items-center gap-2 px-3 py-2.5 border-b border-slate-200 shrink-0">
        <Table2 className="w-4 h-4 text-slate-500 shrink-0" />
        <div className="min-w-0">
          <div className="text-sm font-bold text-slate-900 leading-tight">Results</div>
          <div className="text-label text-slate-500 leading-tight flex items-center gap-1">
            <span className={`w-1.5 h-1.5 rounded-full ${LINES[activeLine].dotClass}`} aria-hidden="true" />
            {LINES[activeLine].tab} · unresolved uploads
          </div>
        </div>
      </div>

      {present.length === 0 ? (
        <p className="px-4 py-6 text-xs text-slate-400 leading-relaxed">
          No unresolved results for this line. Upload a test file from the Conductor and its results will appear here.
        </p>
      ) : (
        <>
          <div role="tablist" aria-label="Uploaded data" className="flex border-b border-slate-200 shrink-0">
            {present.map((sub) => {
              const { Icon, color, tab } = META[sub];
              const active = selected === sub;
              const st = STATUS_STYLES[findings![sub]!.status] ?? STATUS_STYLES.UNKNOWN;
              return (
                <button
                  key={sub}
                  role="tab"
                  aria-selected={active}
                  onClick={() => setPicked(sub)}
                  className={`flex-1 min-w-0 flex items-center justify-center gap-1.5 px-2 py-2.5 text-xs font-semibold border-b-2 -mb-px transition-colors duration-150 ${
                    active
                      ? 'border-slate-900 text-slate-900'
                      : 'border-transparent text-slate-500 hover:text-slate-800 hover:bg-slate-50'
                  }`}
                >
                  <span className="w-5 h-5 rounded-full flex items-center justify-center text-white shrink-0" style={{ backgroundColor: color }}>
                    <Icon className="w-3 h-3" />
                  </span>
                  <span className="truncate">{tab}</span>
                  <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${st.bar}`} aria-label={STATUS_SHORT[findings![sub]!.status]} />
                </button>
              );
            })}
          </div>

          {selected && finding && (
            <div role="tabpanel" className="flex-1 min-h-0 overflow-y-auto custom-scrollbar-lg p-3 space-y-3">
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0">
                  <h3 className="text-xs font-bold text-slate-900">{META[selected].label}</h3>
                  <p className="text-label font-mono text-slate-400 truncate">{finding.file_name}</p>
                </div>
                <StatusPill status={finding.status} />
              </div>
              <p className="text-xs font-semibold text-slate-700">{headline(selected, finding)}</p>
              <Detail sub={selected} f={finding} />
              <DownloadButton submission={finding.submission} />
            </div>
          )}
        </>
      )}
    </aside>
  );
};
