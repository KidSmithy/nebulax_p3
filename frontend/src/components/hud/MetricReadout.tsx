import React from 'react';
import { MetricStatus } from '../../types/telemetry';
import {
  METRIC_GLOSSARY,
  STATUS_SHORT,
  STATUS_STYLES,
  classifyMetric,
  formatMetric,
  gaugePosition,
} from '../../lib/metricGlossary';
import { InfoTip } from './InfoTip';

interface MetricReadoutProps {
  path: string;
  value: number | undefined | null;
  /** Verdict from the backend frame; preferred over local classification. */
  status?: MetricStatus;
  /** Beginner shows the everyday name, Expert the engineering name. */
  beginner: boolean;
  /** Secondary line, e.g. a comparison against nominal. */
  footnote?: string;
  compact?: boolean;
  side?: 'left' | 'right';
}

/**
 * One number, made self-explanatory: plain-language name, the value, a verdict
 * in words rather than only colour, a gauge showing where the value sits
 * between healthy and critical, and an explanation on demand.
 *
 * Colour alone is never the only carrier of meaning here - the verdict is also
 * spelled out as text, so the panel still reads correctly for colour-blind
 * users and in screen readers.
 */
export const MetricReadout: React.FC<MetricReadoutProps> = ({
  path,
  value,
  status,
  beginner,
  footnote,
  compact = false,
  side = 'left',
}) => {
  const def = METRIC_GLOSSARY[path];
  const verdict: MetricStatus = status ?? classifyMetric(path, value ?? null);
  const styles = STATUS_STYLES[verdict];

  if (!def) return null;

  const label = beginner ? def.shortName ?? def.plainName : def.technicalName;
  const hasGauge = def.good < 9000 && value !== undefined && value !== null;
  const pos = hasGauge ? gaugePosition(path, value as number) : 0;
  const goodPos = gaugePosition(path, def.good);
  const warnPos = gaugePosition(path, def.warn);

  const rangeText = def.higherIsBetter
    ? `Healthy at or above ${formatMetric(path, def.good)}`
    : `Healthy at or below ${formatMetric(path, def.good)}`;

  return (
    <div className={`rounded border ${styles.border} ${styles.bg} ${compact ? 'p-1.5' : 'p-2'}`}>
      <div className="flex items-start justify-between gap-1">
        <div className="flex items-center gap-1 min-w-0">
          <span className="text-[9.5px] font-semibold text-slate-600 leading-tight">
            {label}
          </span>
          <InfoTip
            title={`${def.plainName}${beginner ? '' : ` (${def.technicalName})`}`}
            whatItIs={def.whatItIs}
            whyItMatters={def.whyItMatters}
            analogy={def.analogy}
            range={def.good < 9000 ? rangeText : undefined}
            side={side}
          />
        </div>
        <span
          className={`text-[8.5px] font-bold px-1 py-px rounded shrink-0 ${styles.text} ${styles.bg} border ${styles.border}`}
        >
          {STATUS_SHORT[verdict]}
        </span>
      </div>

      <div className="mt-0.5 flex items-baseline gap-1.5">
        <span className={`font-mono font-bold ${compact ? 'text-xs' : 'text-sm'} ${styles.text}`}>
          {formatMetric(path, value)}
        </span>
        {beginner && (
          <span className="text-label text-slate-500 truncate">
            {verdict === 'GOOD'
              ? 'normal'
              : verdict === 'WATCH'
              ? 'slightly high'
              : verdict === 'ACTION_NEEDED'
              ? 'outside safe range'
              : ''}
          </span>
        )}
      </div>

      {hasGauge && (
        <div
          className="mt-1 relative h-1 rounded-full bg-slate-200 overflow-visible"
          role="img"
          aria-label={`${formatMetric(path, value)}, ${rangeText}`}
        >
          {/* Healthy band shading */}
          <div
            className="absolute inset-y-0 bg-emerald-200/70 rounded-full"
            style={
              def.higherIsBetter
                ? { left: `${goodPos * 100}%`, right: 0 }
                : { left: 0, width: `${goodPos * 100}%` }
            }
          />
          {/* Concern threshold marker */}
          <div
            className="absolute -top-0.5 h-2 w-px bg-slate-400"
            style={{ left: `${warnPos * 100}%` }}
          />
          {/* Current value */}
          <div
            className={`absolute -top-[3px] w-1.5 h-1.5 rounded-full ring-1 ring-white ${styles.bar}`}
            style={{ left: `calc(${pos * 100}% - 3px)` }}
          />
        </div>
      )}

      {footnote && (
        <div className="mt-1 text-label text-slate-500 font-mono truncate">{footnote}</div>
      )}
    </div>
  );
};
