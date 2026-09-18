import React, { useEffect, useRef, useState } from 'react';
import { ArrowRight, CornerDownLeft, Loader2, UploadCloud, Wrench } from 'lucide-react';
import { useTwinStore } from '../../../store/useTwinStore';
import { ACTIONS, ActionCopy } from '../WhatIfPanel';
import { InterventionAction } from '../../../types/telemetry';

const REPAIR_META: Partial<Record<InterventionAction, { duration: string; location: string }>> = {
  ACTION_REPLACE_FILTER: { duration: '40 min', location: 'Depot' },
  ACTION_LUBRICATE_DOOR: { duration: '25 min', location: 'Depot' },
  ACTION_GRIND_RAIL: { duration: '2 hr', location: 'Trackside' },
  ACTION_INSPECT_BEARING: { duration: '35 min', location: 'Depot' },
};

const SUGGESTED_QUESTIONS = [
  'What is the most urgent problem right now?',
  'Explain the vibration reading like I know nothing about trains',
  'What should we do about it, and how urgent?',
];

/** Splits a leading intro sentence from a numbered list, if the answer has one. */
function parseAnswer(text: string): { intro: string; steps: string[] | null } {
  const lines = text.split(/\n+/).map((l) => l.trim()).filter(Boolean);
  const stepIdx = lines.findIndex((l) => /^\d+[.)]\s+/.test(l));
  if (stepIdx === -1) return { intro: text, steps: null };
  const steps = lines
    .slice(stepIdx)
    .filter((l) => /^\d+[.)]\s+/.test(l))
    .map((l) => l.replace(/^\d+[.)]\s+/, ''));
  return { intro: lines.slice(0, stepIdx).join(' '), steps };
}

function findRepairMention(text: string): ActionCopy | null {
  const lower = text.toLowerCase();
  return ACTIONS.find((a) => lower.includes(a.plainTitle.toLowerCase())) ?? null;
}

const RepairReference: React.FC<{ action: ActionCopy }> = ({ action }) => {
  const setSelectedSubsystem = useTwinStore((s) => s.setSelectedSubsystem);
  const setRightDrawerTab = useTwinStore((s) => s.setRightDrawerTab);
  const meta = REPAIR_META[action.action];

  return (
    <button
      onClick={() => {
        setRightDrawerTab('whatif');
        useTwinStore.setState({ isRightDrawerOpen: true });
      }}
      className="w-full text-left p-2 rounded bg-white border border-slate-200 relative overflow-hidden hover:border-slate-300 transition-colors duration-150"
    >
      <div className="absolute left-0 top-0 bottom-0 w-1 bg-status-nominal" aria-hidden="true" />
      <div className="pl-2 flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="flex items-center gap-1 text-slate-800 font-semibold text-xs">
            <Wrench className="w-3 h-3 text-slate-500 shrink-0" />
            {action.plainTitle}
          </div>
          {meta && (
            <div className="text-label text-slate-500 font-mono mt-0.5">
              Repairs tab · {meta.duration} · {meta.location}
            </div>
          )}
        </div>
        <ArrowRight className="w-3.5 h-3.5 text-slate-400 shrink-0 mt-0.5" />
      </div>
    </button>
  );
};

interface ConductorChatProps {
  /** Compact = popup (tighter type); false = expanded panel. */
  compact?: boolean;
}

export const ConductorChat: React.FC<ConductorChatProps> = ({ compact = true }) => {
  const answers = useTwinStore((s) => s.aiAnswers);
  const asking = useTwinStore((s) => s.aiAsking);
  const askAi = useTwinStore((s) => s.askAi);
  const setConductorState = useTwinStore((s) => s.setConductorState);
  const [question, setQuestion] = useState('');
  const [confirmRestart, setConfirmRestart] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [answers.length]);

  const submit = (q: string) => {
    if (!q.trim() || asking) return;
    askAi(q);
    setQuestion('');
  };

  return (
    <div className="relative flex flex-col h-full min-h-0">
      {confirmRestart && (
        <div className="absolute inset-0 z-10 bg-white/95 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="w-full max-w-[260px] text-center space-y-3">
            <p className="text-xs text-slate-700 leading-relaxed">
              Restart the setup walkthrough? This takes over the screen to pick a subsystem and upload
              data again.
            </p>
            <div className="flex items-center justify-center gap-2">
              <button
                onClick={() => setConfirmRestart(false)}
                className="px-3 py-1.5 rounded text-label font-semibold text-slate-600 border border-slate-200 hover:bg-slate-50 transition-colors duration-150"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  setConfirmRestart(false);
                  setConductorState('onboarding');
                }}
                className="px-3 py-1.5 rounded text-label font-semibold text-white bg-ink-900 hover:bg-ink-700 transition-colors duration-150"
              >
                Restart
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="flex-1 min-h-0 overflow-y-auto custom-scrollbar px-3 py-2 space-y-2.5">
        {answers.length === 0 && (
          <div className="flex flex-col gap-1.5">
            <p className="text-label text-slate-500 leading-relaxed">
              Ask about anything you see on this car - the Conductor explains and points you to
              the right repair. It doesn't schedule anything itself.
            </p>
            {SUGGESTED_QUESTIONS.map((q) => (
              <button
                key={q}
                onClick={() => submit(q)}
                className="text-left text-label px-2 py-1.5 rounded bg-white border border-slate-200 text-slate-600 hover:border-ink-500 hover:text-ink-900 transition-colors duration-150"
              >
                {q}
              </button>
            ))}
          </div>
        )}

        {answers.map((a, i) => {
          const { intro, steps } = parseAnswer(a.answer);
          const repair = findRepairMention(a.answer);
          return (
            <div key={i} className="space-y-1.5">
              {/* User turn: soft selected card, never a chat bubble */}
              <div className="bg-slate-100 rounded px-2.5 py-1.5 text-xs text-slate-800 font-medium">
                {a.question}
              </div>

              {/* Assistant turn: plain prose under a micro-caps label */}
              <div>
                <div className="text-[9px] font-bold text-slate-400 tracking-[0.18em] mb-1">
                  CONDUCTOR
                </div>
                <p className={`${compact ? 'text-[11px]' : 'text-xs'} text-slate-700 leading-relaxed`}>
                  {intro}
                </p>
                {steps && (
                  <ol className="mt-1.5 space-y-1.5">
                    {steps.map((step, si) => (
                      <li key={si} className="flex items-start gap-1.5">
                        <span className="shrink-0 w-4 h-4 rounded-full bg-slate-200 text-slate-700 text-[9px] font-bold flex items-center justify-center mt-0.5">
                          {si + 1}
                        </span>
                        <span className={`${compact ? 'text-[11px]' : 'text-xs'} text-slate-700 leading-relaxed`}>
                          {step}
                        </span>
                      </li>
                    ))}
                  </ol>
                )}
                {repair && (
                  <div className="mt-1.5">
                    <RepairReference action={repair} />
                  </div>
                )}
              </div>
            </div>
          );
        })}

        {asking && (
          <div className="flex items-center gap-1.5 text-label text-slate-400">
            <Loader2 className="w-3 h-3 animate-spin" />
            The Conductor is reading the live data…
          </div>
        )}
        <div ref={endRef} />
      </div>

      <div className="shrink-0 px-2 pt-2 border-t border-slate-200">
        <button
          onClick={() => setConfirmRestart(true)}
          className="inline-flex items-center gap-1.5 text-label font-semibold px-2 py-1 rounded bg-white border border-slate-200 text-slate-600 hover:border-ink-500 hover:text-ink-900 transition-colors duration-150"
        >
          <UploadCloud className="w-3 h-3" />
          Upload new data
        </button>
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          submit(question);
        }}
        className="relative shrink-0 p-2"
      >
        <label htmlFor="conductor-question" className="sr-only">
          Ask the Conductor about this car
        </label>
        <input
          id="conductor-question"
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask about this car…"
          maxLength={500}
          className="w-full text-[11px] pl-2.5 pr-8 py-2 rounded bg-white border border-slate-200 text-slate-800 placeholder:text-slate-400 focus:outline-none focus:border-ink-500"
        />
        <button
          type="submit"
          disabled={asking || !question.trim()}
          aria-label="Send question"
          className="absolute right-3 top-1/2 -translate-y-1/2 w-6 h-6 rounded-full bg-ink-900 text-white flex items-center justify-center disabled:opacity-30 transition-opacity duration-150"
        >
          {asking ? (
            <Loader2 className="w-3 h-3 animate-spin" />
          ) : (
            <CornerDownLeft className="w-3 h-3" />
          )}
        </button>
      </form>
    </div>
  );
};
