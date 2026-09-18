import React, { useEffect, useRef, useState } from 'react';
import {
  AlertTriangle,
  BrainCircuit,
  CheckCircle2,
  CornerDownLeft,
  Eye,
  Lightbulb,
  Loader2,
  RefreshCw,
  Sparkles,
  Wrench,
} from 'lucide-react';
import { useTwinStore } from '../../store/useTwinStore';
import { MetricStatus } from '../../types/telemetry';
import { STATUS_STYLES } from '../../lib/metricGlossary';

const SEVERITY_ICON: Record<string, React.FC<{ className?: string }>> = {
  GOOD: CheckCircle2,
  WATCH: Eye,
  ACTION_NEEDED: AlertTriangle,
  UNKNOWN: Eye,
};

const SEVERITY_TEXT: Record<string, string> = {
  GOOD: 'Everything normal',
  WATCH: 'Worth keeping an eye on',
  ACTION_NEEDED: 'Needs attention',
  UNKNOWN: 'No verdict',
};

const URGENCY_TEXT: Record<string, string> = {
  NONE: 'No deadline',
  ROUTINE: 'At the next routine check',
  WITHIN_7_DAYS: 'Within 7 days',
  WITHIN_48_HOURS: 'Within 48 hours',
};

const SUBSYSTEM_TEXT: Record<string, string> = {
  door: 'Doors',
  acv: 'Air-conditioning',
  shm: 'Wheels & frame',
  rail: 'Track',
  rail_corrugation: 'Track',
  fleet: 'Whole train',
};

const SUGGESTED_QUESTIONS = [
  'What is the most urgent problem right now?',
  'Explain the vibration reading like I know nothing about trains',
  'Would grinding the rail actually help here?',
  'Which of these costs the most if ignored?',
];

export const AIInsightPanel: React.FC = () => {
  const insight = useTwinStore((s) => s.aiInsight);
  const loading = useTwinStore((s) => s.aiInsightLoading);
  const fetchAiInsight = useTwinStore((s) => s.fetchAiInsight);
  const fetchAiStatus = useTwinStore((s) => s.fetchAiStatus);
  const aiEnabled = useTwinStore((s) => s.aiEnabled);
  const aiModel = useTwinStore((s) => s.aiModel);
  const autoRefresh = useTwinStore((s) => s.aiAutoRefresh);
  const setAutoRefresh = useTwinStore((s) => s.setAiAutoRefresh);
  const answers = useTwinStore((s) => s.aiAnswers);
  const asking = useTwinStore((s) => s.aiAsking);
  const askAi = useTwinStore((s) => s.askAi);

  const [question, setQuestion] = useState('');
  const answersEndRef = useRef<HTMLDivElement>(null);

  // Fetch status once, and an opening insight so the panel is never empty.
  useEffect(() => {
    fetchAiStatus();
    if (!insight) fetchAiInsight();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Auto-refresh is opt-in and deliberately slow: the model call takes several
  // seconds and costs money, so a 10 Hz stream must not drive it.
  useEffect(() => {
    if (!autoRefresh) return;
    const id = setInterval(() => fetchAiInsight(), 20000);
    return () => clearInterval(id);
  }, [autoRefresh, fetchAiInsight]);

  useEffect(() => {
    answersEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [answers.length]);

  const severity = (insight?.severity ?? 'UNKNOWN') as MetricStatus;
  const styles = STATUS_STYLES[severity];
  const SeverityIcon = SEVERITY_ICON[severity] ?? Eye;

  const submit = (q: string) => {
    if (!q.trim()) return;
    askAi(q);
    setQuestion('');
  };

  return (
    <div className="flex flex-col space-y-2.5">
      {/* Panel header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-1.5">
          <BrainCircuit className="w-3.5 h-3.5 text-red-600" />
          <span className="text-[11px] font-bold text-slate-700 uppercase tracking-wide">
            AI Explanation
          </span>
        </div>
        <div className="flex items-center space-x-1">
          <button
            onClick={() => setAutoRefresh(!autoRefresh)}
            className={`text-[9px] font-mono px-1.5 py-0.5 rounded border transition-colors ${
              autoRefresh
                ? 'bg-red-50 text-red-700 border-red-200'
                : 'bg-slate-50 text-slate-500 border-slate-200 hover:text-slate-700'
            }`}
            title="Automatically refresh the explanation every 20 seconds"
          >
            AUTO {autoRefresh ? 'ON' : 'OFF'}
          </button>
          <button
            onClick={() => fetchAiInsight(true)}
            disabled={loading}
            className="p-1 rounded-md text-slate-500 hover:text-red-600 hover:bg-red-50 disabled:opacity-40 transition-colors"
            title="Re-analyse the current readings"
          >
            {loading ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <RefreshCw className="w-3.5 h-3.5" />
            )}
          </button>
        </div>
      </div>

      {/* Model provenance / degradation notice */}
      {aiEnabled === false && (
        <div className="p-2 rounded-lg bg-amber-50 border border-amber-200 text-[10px] text-amber-800 leading-snug">
          The AI assistant is not configured, so these explanations come from built-in rules
          instead of a model. Add <code className="font-mono">OPENAI_API_KEY</code> to{' '}
          <code className="font-mono">backend/.env</code> to enable it.
        </div>
      )}

      {loading && !insight && (
        <div className="p-4 flex flex-col items-center justify-center text-slate-500 space-y-2">
          <Loader2 className="w-5 h-5 animate-spin text-red-500" />
          <span className="text-[10px]">Reading the live telemetry…</span>
        </div>
      )}

      {insight && (
        <>
          {/* Headline verdict */}
          <div className={`p-2.5 rounded-xl border ${styles.border} ${styles.bg}`}>
            <div className="flex items-start space-x-2">
              <SeverityIcon className={`w-4 h-4 shrink-0 mt-0.5 ${styles.text}`} />
              <div className="min-w-0">
                <div className={`text-[8.5px] font-bold uppercase tracking-wider ${styles.text}`}>
                  {SEVERITY_TEXT[severity]}
                </div>
                <div className="text-xs font-bold text-slate-900 leading-snug mt-0.5">
                  {insight.headline}
                </div>
              </div>
            </div>
            <p className="mt-2 text-[10.5px] text-slate-700 leading-relaxed">{insight.summary}</p>
          </div>

          {/* Everyday analogy */}
          {insight.analogy && (
            <div className="p-2 rounded-lg bg-sky-50 border border-sky-200 flex items-start space-x-1.5">
              <Lightbulb className="w-3 h-3 text-sky-600 shrink-0 mt-0.5" />
              <p className="text-[10px] text-sky-900 leading-relaxed italic">{insight.analogy}</p>
            </div>
          )}

          {/* Per-subsystem findings */}
          {insight.findings.length > 0 && (
            <div className="space-y-1.5">
              <div className="text-[9.5px] font-bold text-slate-500 uppercase tracking-wide">
                What each reading means
              </div>
              {insight.findings.map((f, i) => (
                <div
                  key={i}
                  className="p-2 rounded-lg bg-white border border-slate-200 space-y-1"
                >
                  <div className="text-[9px] font-bold text-red-700 uppercase tracking-wide">
                    {SUBSYSTEM_TEXT[f.subsystem] ?? f.subsystem}
                  </div>
                  <p className="text-[10.5px] text-slate-800 leading-relaxed">{f.what_it_means}</p>
                  <p className="text-[10px] text-slate-500 leading-relaxed">
                    <strong className="text-slate-600">Impact: </strong>
                    {f.why_it_matters}
                  </p>
                </div>
              ))}
            </div>
          )}

          {/* Recommendation */}
          <div className="p-2 rounded-lg bg-slate-50 border border-slate-200">
            <div className="flex items-center space-x-1.5">
              <Wrench className="w-3 h-3 text-slate-600" />
              <span className="text-[9px] font-bold text-slate-600 uppercase tracking-wide">
                Recommended next step
              </span>
            </div>
            <p className="mt-1 text-[10.5px] text-slate-800 leading-relaxed font-medium">
              {insight.recommended_action}
            </p>
            <div className="mt-1 text-[9px] font-mono text-slate-500">
              Timeframe: {URGENCY_TEXT[insight.urgency] ?? insight.urgency}
            </div>
          </div>

          <div className="text-[8.5px] text-slate-400 font-mono flex items-center justify-between">
            <span>
              {insight.source === 'openai'
                ? `Generated by ${insight.model ?? aiModel ?? 'model'}`
                : 'Generated by built-in rules'}
              {insight.stale ? ' • cached' : ''}
            </span>
            {insight.degraded_reason && (
              <span className="text-amber-600 truncate max-w-[120px]" title={insight.degraded_reason}>
                degraded
              </span>
            )}
          </div>
        </>
      )}

      {/* ---------------------------------------------------------------- */}
      {/* Ask a question                                                    */}
      {/* ---------------------------------------------------------------- */}
      <div className="pt-2 border-t border-slate-200 space-y-1.5">
        <div className="flex items-center space-x-1.5">
          <Sparkles className="w-3 h-3 text-red-600" />
          <span className="text-[9.5px] font-bold text-slate-600 uppercase tracking-wide">
            Ask about this train
          </span>
        </div>

        {answers.length > 0 && (
          <div className="max-h-48 overflow-y-auto custom-scrollbar space-y-1.5 pr-1">
            {answers.map((a, i) => (
              <div key={i} className="space-y-1">
                <div className="text-[10px] font-semibold text-slate-700 bg-slate-100 rounded-lg px-2 py-1">
                  {a.question}
                </div>
                <p className="text-[10.5px] text-slate-700 leading-relaxed px-2">{a.answer}</p>
              </div>
            ))}
            <div ref={answersEndRef} />
          </div>
        )}

        {answers.length === 0 && (
          <div className="flex flex-wrap gap-1">
            {SUGGESTED_QUESTIONS.map((q) => (
              <button
                key={q}
                onClick={() => submit(q)}
                disabled={asking}
                className="text-[9px] text-left px-1.5 py-1 rounded-md bg-slate-50 border border-slate-200 text-slate-600 hover:border-red-300 hover:text-red-700 disabled:opacity-40 transition-colors"
              >
                {q}
              </button>
            ))}
          </div>
        )}

        <form
          onSubmit={(e) => {
            e.preventDefault();
            submit(question);
          }}
          className="relative"
        >
          <label htmlFor="ai-question" className="sr-only">
            Ask a question about this train
          </label>
          <input
            id="ai-question"
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="e.g. why is the vibration high?"
            maxLength={500}
            className="w-full text-[10.5px] pl-2 pr-7 py-1.5 rounded-lg bg-white border border-slate-300 text-slate-800 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-red-400 focus:border-red-400"
          />
          <button
            type="submit"
            disabled={asking || !question.trim()}
            aria-label="Send question"
            className="absolute right-1 top-1/2 -translate-y-1/2 p-1 rounded-md text-slate-400 hover:text-red-600 disabled:opacity-30 transition-colors"
          >
            {asking ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <CornerDownLeft className="w-3.5 h-3.5" />
            )}
          </button>
        </form>
      </div>
    </div>
  );
};
