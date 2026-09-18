import React, { useEffect, useRef, useState } from 'react';
import { ArrowRight, CheckCircle2, FileText, HardHat, Loader2, UploadCloud, Wrench, X, AlertCircle } from 'lucide-react';
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

const ModelResultsCard: React.FC<{
  uploadResult: any;
  onDone: () => void;
  onOpenChat: () => void;
}> = ({ uploadResult, onDone, onOpenChat }) => {
  const verdict = (uploadResult.verdict || uploadResult.status || 'GOOD') as 'GOOD' | 'WATCH' | 'ACTION_NEEDED';
  const styles = STATUS_STYLES[verdict] || STATUS_STYLES['GOOD'];
  const triggerWhatIf = useTwinStore((s) => s.triggerWhatIf);
  const [appliedAction, setAppliedAction] = useState(false);

  return (
    <div className="space-y-3">
      <Prompt>Here are the model prediction results from your uploaded file:</Prompt>
      
      <div className="border border-slate-200 bg-white p-3 rounded-sm space-y-2.5 shadow-sm">
        <div className="flex items-center justify-between gap-2 border-b border-slate-100 pb-2">
          <div className="min-w-0 flex items-center gap-1.5">
            <FileText className="w-4 h-4 text-slate-500 shrink-0" />
            <div className="min-w-0">
              <div className="text-xs font-bold text-slate-800 truncate">
                {uploadResult.file_name || 'Uploaded Dataset'}
              </div>
              <div className="text-[10px] text-slate-500 font-mono">
                Subsystem: {String(uploadResult.subsystem).toUpperCase()} · Model Evaluated
              </div>
            </div>
          </div>
          <span className={`text-label font-bold px-2 py-0.5 rounded-full border shrink-0 ${styles.text} ${styles.border} ${styles.bg}`}>
            {STATUS_SHORT[verdict] || verdict}
          </span>
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-2 gap-2 text-xs">
          {uploadResult.subsystem === 'door' && (
            <>
              <div className="bg-slate-50 p-2 rounded border border-slate-100">
                <span className="text-[10px] text-slate-500 block font-medium">Cycles Evaluated</span>
                <span className="font-semibold text-slate-800">{uploadResult.total_cycles} total ({uploadResult.abnormal_cycles} abnormal)</span>
              </div>
              <div className="bg-slate-50 p-2 rounded border border-slate-100">
                <span className="text-[10px] text-slate-500 block font-medium">Fault Rate</span>
                <span className={`font-semibold ${uploadResult.fault_rate_pct > 0 ? 'text-rose-600' : 'text-emerald-600'}`}>
                  {uploadResult.fault_rate_pct}%
                </span>
              </div>
              <div className="bg-slate-50 p-2 rounded border border-slate-100">
                <span className="text-[10px] text-slate-500 block font-medium">Peak Motor Current</span>
                <span className="font-semibold text-slate-800">{uploadResult.max_current_peak_a} A</span>
              </div>
              <div className="bg-slate-50 p-2 rounded border border-slate-100">
                <span className="text-[10px] text-slate-500 block font-medium">Mean Transit Time</span>
                <span className="font-semibold text-slate-800">{uploadResult.mean_duration_s} s</span>
              </div>
            </>
          )}

          {uploadResult.subsystem === 'shm' && (
            <>
              <div className="bg-slate-50 p-2 rounded border border-slate-100">
                <span className="text-[10px] text-slate-500 block font-medium">Vibration RMS</span>
                <span className="font-semibold text-slate-800">{uploadResult.vibration_rms_g} g</span>
              </div>
              <div className="bg-slate-50 p-2 rounded border border-slate-100">
                <span className="text-[10px] text-slate-500 block font-medium">Fatigue Damage Index</span>
                <span className="font-semibold text-slate-800">{uploadResult.fatigue_damage_index}</span>
              </div>
              <div className="bg-slate-50 p-2 rounded border border-slate-100">
                <span className="text-[10px] text-slate-500 block font-medium">Bearing Defect Risk</span>
                <span className={`font-semibold ${uploadResult.bearing_defect_prob > 0.4 ? 'text-amber-600' : 'text-slate-800'}`}>
                  {Math.round((uploadResult.bearing_defect_prob || 0) * 100)}%
                </span>
              </div>
              <div className="bg-slate-50 p-2 rounded border border-slate-100">
                <span className="text-[10px] text-slate-500 block font-medium">Critical Weld Node</span>
                <span className="font-semibold text-slate-800 truncate block">{uploadResult.critical_weld_node}</span>
              </div>
            </>
          )}

          {uploadResult.subsystem === 'acv' && (
            <>
              <div className="bg-slate-50 p-2 rounded border border-slate-100">
                <span className="text-[10px] text-slate-500 block font-medium">Faulty Car Location</span>
                <span className="font-semibold text-slate-800">{uploadResult.most_likely_faulty_car || 'Consist Normal'}</span>
              </div>
              <div className="bg-slate-50 p-2 rounded border border-slate-100">
                <span className="text-[10px] text-slate-500 block font-medium">Detection Confidence</span>
                <span className="font-semibold text-slate-800">{uploadResult.confidence}</span>
              </div>
            </>
          )}

          {uploadResult.subsystem === 'rail' && (
            <>
              <div className="bg-slate-50 p-2 rounded border border-slate-100">
                <span className="text-[10px] text-slate-500 block font-medium">Corrugation Depth</span>
                <span className="font-semibold text-slate-800">{uploadResult.depth_microns} μm</span>
              </div>
              <div className="bg-slate-50 p-2 rounded border border-slate-100">
                <span className="text-[10px] text-slate-500 block font-medium">Wavelength Pattern</span>
                <span className="font-semibold text-slate-800">{uploadResult.wavelength_class}</span>
              </div>
            </>
          )}
        </div>

        {/* Conductor Narrative */}
        <div className="bg-slate-50 border-l-2 border-ink-900 p-2.5 rounded-r">
          <div className="text-[9px] font-bold text-slate-500 uppercase tracking-wider mb-1 flex items-center gap-1">
            <HardHat className="w-3 h-3 text-ink-900" />
            Conductor Diagnosis
          </div>
          <p className="text-[11px] text-slate-700 leading-relaxed">
            {uploadResult.conductor_summary}
          </p>
        </div>
      </div>

      {/* Suggested Intervention Button */}
      {uploadResult.recommended_action && uploadResult.recommended_action !== 'NONE' && (
        <button
          onClick={() => {
            triggerWhatIf(uploadResult.recommended_action, true);
            setAppliedAction(true);
          }}
          disabled={appliedAction}
          className={`w-full flex items-center justify-center gap-1.5 py-2 px-3 text-xs font-semibold rounded border transition-colors ${
            appliedAction 
              ? 'bg-emerald-50 text-emerald-700 border-emerald-300' 
              : 'bg-amber-50 text-amber-900 border-amber-300 hover:bg-amber-100'
          }`}
        >
          <Wrench className="w-3.5 h-3.5" />
          {appliedAction 
            ? 'Intervention applied in Digital Twin' 
            : `Simulate fix: ${uploadResult.recommended_action.replace('ACTION_', '').replace(/_/g, ' ')}`}
        </button>
      )}

      {/* Action buttons */}
      <div className="flex items-center gap-2 pt-1">
        <button
          onClick={onOpenChat}
          className="flex-1 flex items-center justify-center gap-1.5 bg-ink-900 text-white text-xs font-semibold py-2.5 hover:bg-ink-700 transition-colors duration-150 rounded-sm"
        >
          Discuss in Conductor Chat
          <ArrowRight className="w-3.5 h-3.5" />
        </button>
        <button
          onClick={onDone}
          className="px-3 py-2.5 border border-slate-200 text-slate-600 text-xs font-semibold hover:bg-slate-50 transition-colors duration-150 rounded-sm"
        >
          Dashboard
        </button>
      </div>
    </div>
  );
};

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
  const [uploadResult, setUploadResult] = useState<any | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const finish = () => setConductorState('docked');
  const openChat = () => setConductorState('popup');

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
    setUploadResult(null);
    setUploadError(null);
    setStep('upload');
  };

  const handleFileUpload = async (file: File) => {
    if (!subsystem) return;
    setFileName(file.name);
    setProcessing(true);
    setUploadError(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('subsystem', subsystem);

    try {
      const res = await fetch('/api/predict/upload', {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Upload failed with status ${res.status}`);
      }
      const data = await res.json();
      setUploadResult(data);

      // Seed conductor chat with the findings from this upload
      useTwinStore.setState((state) => ({
        aiAnswers: [
          ...state.aiAnswers,
          {
            question: `Diagnose uploaded file (${file.name})`,
            answer: data.conductor_summary,
            source: 'model',
          },
        ].slice(-8),
      }));

      setProcessing(false);
      setStep('results');
    } catch (err: any) {
      console.error('[Upload] Error processing file:', err);
      setUploadError(err.message || 'Error processing model inference on uploaded file.');
      setProcessing(false);
    }
  };

  const skipToLive = () => {
    setFileName(null);
    setUploadResult(null);
    setProcessing(true);
    window.setTimeout(() => {
      setProcessing(false);
      setStep('results');
    }, 800);
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
                Upload a CSV data file for {selected ? labelFor(selected.id) : 'this subsystem'}
                , or skip it and I'll keep watching the live feed.
              </Prompt>

              {uploadError && (
                <div className="flex items-start gap-2 p-2.5 bg-rose-50 border border-rose-200 rounded text-xs text-rose-800">
                  <AlertCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
                  <div>
                    <span className="font-semibold block">Inference Error</span>
                    <span>{uploadError}</span>
                  </div>
                </div>
              )}

              {!processing ? (
                <>
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="w-full flex flex-col items-center justify-center gap-1.5 border border-dashed border-slate-300 hover:border-ink-500 hover:bg-slate-50 transition-colors duration-150 py-6 rounded"
                  >
                    <UploadCloud className="w-6 h-6 text-slate-400" />
                    <span className="text-xs font-medium text-slate-700">Click to choose a file</span>
                    <span className="text-label text-slate-400">CSV dataset (Door, SHM, ACV, Rail)</span>
                  </button>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".csv,.txt,.log"
                    className="hidden"
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (f) handleFileUpload(f);
                    }}
                  />
                  <button
                    onClick={skipToLive}
                    className="w-full text-center text-label font-semibold text-slate-500 hover:text-ink-900 py-1.5 transition-colors duration-150"
                  >
                    Skip — use the live feed
                  </button>
                </>
              ) : (
                <div className="flex flex-col items-center justify-center gap-2 text-label text-slate-500 py-8">
                  <Loader2 className="w-5 h-5 animate-spin text-ink-900" />
                  <span className="text-xs font-medium text-slate-700">
                    {fileName ? `Evaluating ${fileName} with machine learning model…` : 'Connecting to the live feed…'}
                  </span>
                  <span className="text-[10px] text-slate-400">
                    Running feature extraction and fault classification
                  </span>
                </div>
              )}
            </>
          )}

          {step === 'results' && selected && (
            uploadResult ? (
              <ModelResultsCard
                uploadResult={uploadResult}
                onDone={finish}
                onOpenChat={openChat}
              />
            ) : (
              <ResultsCard item={selected} beginner={beginner} onDone={finish} />
            )
          )}
        </div>
      </div>
    </div>
  );
};
