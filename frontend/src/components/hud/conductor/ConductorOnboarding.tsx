import React, { useEffect, useRef, useState } from 'react';
import { CheckCircle2, HardHat, Loader2, UploadCloud, X } from 'lucide-react';
import { useTwinStore } from '../../../store/useTwinStore';
import { useMonitoredItems, MonitoredItem } from '../../../lib/useMonitoredItems';
import { STATUS_SHORT, STATUS_STYLES } from '../../../lib/metricGlossary';
import { SubsystemSelection } from '../../../types/telemetry';

type Step = 'subsystem' | 'upload' | 'results';
const STEP_ORDER: Step[] = ['subsystem', 'upload', 'results'];

const ONBOARDING_SUBSYSTEMS: { id: SubsystemSelection; label: string }[] = [
  { id: 'acv', label: 'Air conditioning' },
  { id: 'door', label: 'Passenger doors' },
  { id: 'rail', label: 'Rail corrugation' },
  { id: 'shm', label: 'Structural health monitoring (SHM)' },
];
const labelFor = (id: SubsystemSelection) =>
  ONBOARDING_SUBSYSTEMS.find((o) => o.id === id)?.label ?? id;

const Prompt: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div>
    <div className="text-[9px] font-bold text-slate-400 tracking-[0.18em] mb-1">CONDUCTOR</div>
    <p className="text-xs text-slate-700 leading-relaxed">{children}</p>
  </div>
);

const UserReply: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div className="inline-block bg-slate-100 rounded px-2.5 py-1.5 text-xs text-slate-800 font-medium">
    {children}
  </div>
);

const ResultsCard: React.FC<{ item: MonitoredItem; beginner: boolean; onDone: () => void }> = ({
  item,
  beginner,
  onDone,
}) => {
  const verdict = item.verdict ?? 'UNKNOWN';
  const styles = STATUS_STYLES[verdict];
  const Icon = item.icon;

  return (
    <>
      <Prompt>Here's what it's reading right now.</Prompt>
      <div className="border border-slate-200 p-3">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2 min-w-0">
            <div
              className={`w-8 h-8 flex items-center justify-center shrink-0 ${
                item.verdict ? `${styles.bg} ${styles.text} border ${styles.border}` : 'bg-slate-100 text-slate-500'
              }`}
            >
              <Icon className="w-4 h-4" />
            </div>
            <div className="min-w-0">
              <div className="text-xs font-semibold text-slate-800 truncate">
                {labelFor(item.id)}
              </div>
              <div className="text-label text-slate-500 font-mono truncate">
                {beginner ? item.plainSub : item.technicalSub}
              </div>
            </div>
          </div>
          {item.verdict && (
            <span
              className={`text-label font-bold px-1.5 py-0.5 rounded-full border shrink-0 ${styles.text} ${styles.border} bg-white`}
            >
              {STATUS_SHORT[verdict]}
            </span>
          )}
        </div>
      </div>
      <button
        onClick={onDone}
        className="w-full flex items-center justify-center gap-1.5 bg-ink-900 text-white text-xs font-semibold py-2.5 hover:bg-ink-700 transition-colors duration-150"
      >
        <CheckCircle2 className="w-3.5 h-3.5" />
        Go to the dashboard
      </button>
    </>
  );
};

/**
 * Full-screen takeover shown first: pick a subsystem, "upload" a file (no
 * backend to send it to yet, so this is a scripted beat that hands off to
 * the live feed), then see that subsystem's live reading. Reusable from the
 * composer's "Upload new data" action, which unmounts and remounts this so
 * every run starts clean at step one.
 */
export const ConductorOnboarding: React.FC = () => {
  const setConductorState = useTwinStore((s) => s.setConductorState);
  const setSelectedSubsystem = useTwinStore((s) => s.setSelectedSubsystem);
  const beginner = useTwinStore((s) => s.uiMode) === 'beginner';
  const { orderedItems } = useMonitoredItems();
  const subsystemOptions = ONBOARDING_SUBSYSTEMS.map((o) => orderedItems.find((i) => i.id === o.id)!);

  const [step, setStep] = useState<Step>('subsystem');
  const [subsystem, setSubsystem] = useState<SubsystemSelection | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [processing, setProcessing] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const finish = () => setConductorState('docked');

  useEffect(() => {
    const onEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') finish();
    };
    document.addEventListener('keydown', onEsc);
    return () => document.removeEventListener('keydown', onEsc);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const chooseSubsystem = (id: SubsystemSelection) => {
    setSubsystem(id);
    setSelectedSubsystem(id);
    setStep('upload');
  };

  const runProcessing = (name: string | null) => {
    setFileName(name);
    setProcessing(true);
    window.setTimeout(() => {
      setProcessing(false);
      setStep('results');
    }, 1400);
  };

  const selected = subsystemOptions.find((i) => i.id === subsystem);
  const stepIndex = STEP_ORDER.indexOf(step);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-slate-900/20 backdrop-blur-sm" />

      <div className="relative w-full max-w-md max-h-[85vh] bg-white border border-slate-200 shadow-xl flex flex-col overflow-hidden">
        <div className="flex items-center justify-between px-4 py-3 border-b border-slate-200 shrink-0">
          <div className="flex items-center gap-2 min-w-0">
            <span className="w-7 h-7 rounded-full bg-ink-900 flex items-center justify-center shrink-0">
              <HardHat className="w-3.5 h-3.5 text-white" />
            </span>
            <div className="min-w-0">
              <div className="text-sm font-bold text-slate-900 leading-tight">Conductor</div>
              <div className="text-label text-slate-500 leading-tight truncate">Car 3 setup</div>
            </div>
          </div>
          <div className="flex items-center gap-3 shrink-0">
            <div className="flex items-center gap-1">
              {STEP_ORDER.map((s, i) => (
                <span
                  key={s}
                  className={`w-1.5 h-1.5 rounded-full ${i <= stepIndex ? 'bg-ink-900' : 'bg-slate-200'}`}
                />
              ))}
            </div>
            <button
              onClick={finish}
              className="p-1 rounded text-slate-400 hover:text-slate-800 hover:bg-slate-100 transition-colors"
              title="Skip for now"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        <div className="flex-1 min-h-0 overflow-y-auto custom-scrollbar px-4 py-4 space-y-3">
          <Prompt>Welcome! I'm the Conductor — let's get this car set up.</Prompt>

          {step === 'subsystem' && (
            <>
              <Prompt>Which subsystem do you want to monitor?</Prompt>
              <div className="grid grid-cols-2 gap-2">
                {subsystemOptions.map((item) => {
                  const Icon = item.icon;
                  return (
                    <button
                      key={item.id}
                      onClick={() => chooseSubsystem(item.id)}
                      className="text-left p-3 border border-slate-200 hover:border-ink-500 hover:bg-slate-50 transition-colors duration-150"
                    >
                      <Icon className="w-4 h-4 text-slate-500 mb-1.5" />
                      <div className="text-xs font-semibold text-slate-800">
                        {labelFor(item.id)}
                      </div>
                    </button>
                  );
                })}
              </div>
            </>
          )}

          {step !== 'subsystem' && selected && (
            <UserReply>{labelFor(selected.id)}</UserReply>
          )}

          {step === 'upload' && (
            <>
              <Prompt>
                Upload a data file for {selected ? labelFor(selected.id) : 'this subsystem'}
                , or skip it and I'll keep watching the live feed.
              </Prompt>

              {!processing ? (
                <>
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="w-full flex flex-col items-center justify-center gap-1.5 border border-dashed border-slate-300 hover:border-ink-500 hover:bg-slate-50 transition-colors duration-150 py-6"
                  >
                    <UploadCloud className="w-5 h-5 text-slate-400" />
                    <span className="text-xs font-medium text-slate-600">Click to choose a file</span>
                    <span className="text-label text-slate-400">CSV, JSON, or log export</span>
                  </button>
                  <input
                    ref={fileInputRef}
                    type="file"
                    className="hidden"
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (f) runProcessing(f.name);
                    }}
                  />
                  <button
                    onClick={() => runProcessing(null)}
                    className="w-full text-center text-label font-semibold text-slate-500 hover:text-ink-900 py-1.5 transition-colors duration-150"
                  >
                    Skip — use the live feed
                  </button>
                </>
              ) : (
                <div className="flex items-center gap-2 text-label text-slate-500 py-3">
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  {fileName ? `Reading ${fileName}…` : 'Connecting to the live feed…'}
                </div>
              )}
            </>
          )}

          {step === 'results' && selected && (
            <ResultsCard item={selected} beginner={beginner} onDone={finish} />
          )}
        </div>
      </div>
    </div>
  );
};
