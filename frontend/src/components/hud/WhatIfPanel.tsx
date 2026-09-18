import React from 'react';
import {
  ArrowRight,
  Check,
  Info,
  MapPin,
  RotateCcw,
  TrendingDown,
  TrendingUp,
  Wrench,
} from 'lucide-react';
import { useTwinStore } from '../../store/useTwinStore';
import { InterventionAction } from '../../types/telemetry';
import { plainTerm } from '../../lib/metricGlossary';
import { InfoTip } from './InfoTip';

export interface ActionCopy {
  action: InterventionAction;
  plainTitle: string;
  technicalTitle: string;
  plainDescription: string;
  technicalDescription: string;
  /** Only meaningful in specific places along the track. */
  locationSensitive?: boolean;
}

export const ACTIONS: ActionCopy[] = [
  {
    action: 'ACTION_GRIND_RAIL',
    plainTitle: 'Grind the rail smooth',
    technicalTitle: 'Simulate Rail Grinding',
    plainDescription:
      'A grinding train shaves the rippled surface off the rail, so the wheels stop hammering as they roll.',
    technicalDescription: 'LTA continuous track grinding across corrugated KP chainage',
    locationSensitive: true,
  },
  {
    action: 'ACTION_LUBRICATE_DOOR',
    plainTitle: 'Grease the door tracks',
    technicalTitle: 'Lubricate Door Guides',
    plainDescription:
      'Clean and lubricate the rails the door slides along, so the motor stops fighting friction.',
    technicalDescription: 'Clean guide-rails & apply low-viscosity PTFE lubricant',
  },
  {
    action: 'ACTION_REPLACE_FILTER',
    plainTitle: 'Replace the aircon filter',
    technicalTitle: 'Replace ACV Air Filter',
    plainDescription:
      'Fit a clean filter and top up the refrigerant so cold air can flow freely again.',
    technicalDescription: 'Fresh filter pack & refrigerant charge replenishment',
  },
  {
    action: 'ACTION_INSPECT_BEARING',
    plainTitle: 'Service the wheel bearing',
    technicalTitle: 'Inspect Front Bearing',
    plainDescription:
      'Ultrasonically check the axle bearing for cracks and re-grease it, removing the risk of a seized wheel.',
    technicalDescription: 'Ultrasonic flaw evaluation & high-pressure re-greasing',
  },
];

/** Which counterfactual metrics belong to which action, for per-card display. */
const ACTION_METRIC_PREFIX: Record<InterventionAction, string[]> = {
  ACTION_GRIND_RAIL: ['rail_corrugation.', 'shm.'],
  ACTION_LUBRICATE_DOOR: ['door.'],
  ACTION_REPLACE_FILTER: ['acv.'],
  ACTION_INSPECT_BEARING: ['shm.'],
};

function fmt(m: { before: number; after: number; unit: string; decimals: number }, which: 'before' | 'after') {
  const v = m[which];
  if (m.unit === '%') return `${(v * 100).toFixed(m.decimals)}%`;
  return `${v.toFixed(m.decimals)}${m.unit ? ' ' + m.unit : ''}`;
}

/** Minimum a row needs; both the frame counterfactual and the per-action
 *  impact payload satisfy this, so one component renders both. */
interface DeltaLike {
  path: string;
  label: string;
  unit: string;
  decimals: number;
  before: number;
  after: number;
  direction: 'better' | 'worse' | 'unchanged';
}

const DeltaRow: React.FC<{ m: DeltaLike }> = ({ m }) => {
  const better = m.direction === 'better';
  const Icon = better ? TrendingDown : TrendingUp;
  return (
    <div className="flex items-center justify-between gap-1 text-[9.5px] font-mono">
      <span className="text-slate-500 truncate">{m.label}</span>
      <span className="flex items-center gap-1 shrink-0">
        <span className="text-slate-400 line-through">{fmt(m, 'before')}</span>
        <ArrowRight className="w-2.5 h-2.5 text-slate-400" />
        <span className={`font-bold ${better ? 'text-emerald-700' : 'text-red-700'}`}>
          {fmt(m, 'after')}
        </span>
        <Icon className={`w-2.5 h-2.5 ${better ? 'text-emerald-600' : 'text-red-600'}`} />
      </span>
    </div>
  );
};

export const WhatIfPanel: React.FC = () => {
  const currentFrame = useTwinStore((s) => s.currentFrame);
  const activeInterventions = useTwinStore((s) => s.activeInterventions);
  const triggerWhatIf = useTwinStore((s) => s.triggerWhatIf);
  const resetWhatIf = useTwinStore((s) => s.resetWhatIf);
  const lastResult = useTwinStore((s) => s.lastWhatIfResult);
  const sendSeek = useTwinStore((s) => s.sendSeek);
  const beginner = useTwinStore((s) => s.uiMode) === 'beginner';

  const cf = currentFrame?.counterfactual;
  const zone = currentFrame?.next_corrugation_zone;
  const anyActive = Object.values(activeInterventions).some(Boolean);

  const doorState = currentFrame?.subsystems?.door?.cycle_state;
  const doorMoving = doorState === 'OPENING' || doorState === 'CLOSING';

  const health = cf?.metrics.find((m) => m.path === 'fleet_health_index');

  return (
    <div className="flex flex-col space-y-2">
      {/* ---------------------------------------------------------------- */}
      {/* Live measured impact banner                                      */}
      {/* ---------------------------------------------------------------- */}
      {anyActive && cf?.active && health && (
        <div className="p-2.5 rounded bg-emerald-50 border border-emerald-300">
          <div className="flex items-start justify-between gap-2">
            <div className="min-w-0">
              <div className="text-[8.5px] font-bold text-emerald-700">
                Measured effect of your changes
              </div>
              <div className="flex items-baseline gap-1.5 mt-0.5">
                <span className="font-mono text-sm text-slate-400 line-through">
                  {(health.before * 100).toFixed(0)}%
                </span>
                <ArrowRight className="w-3 h-3 text-emerald-600" />
                <span className="font-mono text-lg font-bold text-emerald-700">
                  {(health.after * 100).toFixed(0)}%
                </span>
                <span className="text-label text-slate-500">overall health</span>
              </div>
            </div>
            <button
              onClick={resetWhatIf}
              className="shrink-0 flex items-center gap-1 text-label font-mono px-1.5 py-1 rounded bg-white border border-slate-300 text-slate-600 hover:text-red-700 hover:border-red-300 transition-colors"
              title="Undo all simulated maintenance"
            >
              <RotateCcw className="w-2.5 h-2.5" />
              UNDO ALL
            </button>
          </div>

          <div className="mt-2 pt-2 border-t border-emerald-200 space-y-1">
            {cf.metrics
              .filter((m) => m.direction !== 'unchanged' && m.path !== 'fleet_health_index')
              .slice(0, 8)
              .map((m) => (
                <DeltaRow key={m.path} m={m} />
              ))}
          </div>
        </div>
      )}

      {/* Nothing changed yet, explain why rather than looking broken */}
      {anyActive && cf?.active && !health && (
        <div className="p-2 rounded bg-slate-50 border border-slate-200 text-label text-slate-600">
          Maintenance applied, but nothing measurable changed at this location.
        </div>
      )}

      {lastResult?.measured_impact?.has_measurable_effect === false &&
        lastResult.measured_impact.note && (
          <div className="p-2 rounded bg-amber-50 border border-amber-200 flex items-start gap-1.5">
            <Info className="w-3 h-3 text-amber-600 shrink-0 mt-0.5" />
            <div className="text-label text-amber-900 leading-relaxed">
              {lastResult.measured_impact.note}
            </div>
          </div>
        )}

      {/* ---------------------------------------------------------------- */}
      {/* Rail grinding is location-sensitive: offer to go somewhere useful */}
      {/* ---------------------------------------------------------------- */}
      {zone && !zone.is_inside_zone && (
        <button
          onClick={() => sendSeek(zone.kp_centre)}
          className="w-full p-2 rounded bg-sky-50 border border-sky-200 hover:border-sky-400 transition-colors text-left flex items-start gap-1.5"
        >
          <MapPin className="w-3 h-3 text-sky-600 shrink-0 mt-0.5" />
          <span className="text-label text-sky-900 leading-relaxed">
            The track here is already smooth, so grinding will barely register.{' '}
            <strong>Jump to the rough section at KP {zone.kp_start.toFixed(3)}</strong> (
            {zone.distance_km.toFixed(1)} km ahead) to see the real effect.
          </span>
        </button>
      )}

      {zone?.is_inside_zone && (
        <div className="p-2 rounded bg-red-50 border border-red-200 flex items-start gap-1.5">
          <MapPin className="w-3 h-3 text-red-600 shrink-0 mt-0.5" />
          <span className="text-label text-red-900 leading-relaxed">
            Train is on a <strong>known rough section</strong> (KP {zone.kp_start.toFixed(3)}–
            {zone.kp_end.toFixed(3)}). Rail grinding will have a large effect here.
          </span>
        </div>
      )}

      {/* ---------------------------------------------------------------- */}
      {/* Action cards                                                      */}
      {/* ---------------------------------------------------------------- */}
      <div className="flex items-center justify-between text-label px-0.5 pt-1">
        <span className="font-bold text-slate-600 flex items-center gap-1">
          <Wrench className="w-3 h-3 text-red-600" />
          Try a repair
        </span>
        <InfoTip
          title="What is this?"
          whatItIs="A sandbox. Switching a repair on recalculates the live readings as if that maintenance had just been carried out."
          whyItMatters="It lets you compare the benefit of competing jobs before committing crews and budget to one."
          side="right"
        />
      </div>

      {ACTIONS.map((item) => {
        const isActive = !!activeInterventions[item.action];
        const prefixes = ACTION_METRIC_PREFIX[item.action];
        const own = (cf?.metrics ?? []).filter(
          (m) => m.direction !== 'unchanged' && prefixes.some((p) => m.path.startsWith(p))
        );

        // A parked door draws no current, so lubrication cannot show a saving
        // at this instant. Say so, rather than implying the door is healthy.
        const doorParked =
          item.action === 'ACTION_LUBRICATE_DOOR' && !doorMoving && doorState !== undefined;

        // Per-action figures returned when this specific card was toggled.
        const ownResult =
          lastResult?.action === item.action ? lastResult.measured_impact : undefined;

        return (
          <div
            key={item.action}
            className={`p-2 rounded border transition-colors duration-150 ${
              isActive ? 'bg-emerald-50/70 border-emerald-300' : 'bg-white border-slate-200 hover:border-slate-300'
            }`}
          >
            <div className="flex items-start justify-between gap-2">
              <div className="min-w-0">
                <div className="text-label font-bold text-slate-800 leading-snug">
                  {beginner ? item.plainTitle : item.technicalTitle}
                </div>
                <div className="text-label text-slate-500 mt-0.5 leading-relaxed">
                  {beginner ? item.plainDescription : item.technicalDescription}
                </div>
              </div>

              <button
                onClick={() => triggerWhatIf(item.action, !isActive)}
                aria-pressed={isActive}
                className={`text-label font-mono font-bold px-2 py-1 rounded transition-colors duration-150 flex items-center gap-1 shrink-0 ${
                  isActive
                    ? 'bg-emerald-600 text-white'
                    : 'bg-slate-100 text-slate-700 hover:bg-red-600 hover:text-white'
                }`}
              >
                {isActive && <Check className="w-2.5 h-2.5 stroke-[3]" />}
                <span>{isActive ? 'DONE' : 'APPLY'}</span>
              </button>
            </div>

            {/* Measured result for THIS action, or an honest explanation */}
            {isActive && (
              <div className="mt-1.5 pt-1.5 border-t border-emerald-200 space-y-1">
                {own.length > 0 ? (
                  own.map((m) => <DeltaRow key={m.path} m={m} />)
                ) : ownResult?.changed_metrics?.length ? (
                  <>
                    {ownResult.changed_metrics
                      .filter((m) => prefixes.some((p) => m.path.startsWith(p)))
                      .map((m) => (
                        <DeltaRow key={m.path} m={m} />
                      ))}
                    {ownResult.note && (
                      <div className="text-label text-slate-500 italic leading-relaxed">
                        {ownResult.note}
                      </div>
                    )}
                  </>
                ) : doorParked ? (
                  <div className="text-[9.5px] text-slate-500 leading-relaxed">
                    The door is {plainTerm(doorState).toLowerCase()} right now, so its motor is
                    idle. The saving appears while the door is opening or closing.
                  </div>
                ) : (
                  <div className="text-[9.5px] text-slate-500 leading-relaxed">
                    {item.locationSensitive
                      ? 'No measurable gain here — this stretch of track is already smooth.'
                      : 'No measurable gain — this system was already in good condition.'}
                  </div>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};
