import React, { useEffect, useRef, useState } from 'react';
import { HelpCircle, Info } from 'lucide-react';

interface InfoTipProps {
  title: string;
  whatItIs: string;
  whyItMatters?: string;
  analogy?: string;
  /** Extra line such as the healthy range. */
  range?: string;
  /** Rendered instead of the default question-mark icon. */
  children?: React.ReactNode;
  className?: string;
  side?: 'left' | 'right';
}

/**
 * Explanation bubble opened on hover or on click/keyboard focus.
 *
 * Click-to-pin matters for accessibility and for touch devices, where hover
 * does not exist. The trigger is a real <button> so it is keyboard reachable.
 */
export const InfoTip: React.FC<InfoTipProps> = ({
  title,
  whatItIs,
  whyItMatters,
  analogy,
  range,
  children,
  className = '',
  side = 'left',
}) => {
  const [open, setOpen] = useState(false);
  const [pinned, setPinned] = useState(false);
  const wrapRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (!pinned) return;
    const onDocClick = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) {
        setPinned(false);
        setOpen(false);
      }
    };
    const onEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setPinned(false);
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', onDocClick);
    document.addEventListener('keydown', onEsc);
    return () => {
      document.removeEventListener('mousedown', onDocClick);
      document.removeEventListener('keydown', onEsc);
    };
  }, [pinned]);

  const visible = open || pinned;

  return (
    <span ref={wrapRef} className={`relative inline-flex items-center ${className}`}>
      <button
        type="button"
        aria-label={`What does "${title}" mean?`}
        aria-expanded={visible}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => !pinned && setOpen(false)}
        onClick={(e) => {
          e.stopPropagation();
          setPinned((p) => !p);
        }}
        className="inline-flex items-center text-slate-400 hover:text-ink-900 focus:text-ink-900 focus:outline-none focus-visible:ring-2 focus-visible:ring-red-400 rounded transition-colors"
      >
        {children ?? <HelpCircle className="w-3 h-3" />}
      </button>

      {visible && (
        <span
          role="tooltip"
          className={`absolute z-50 top-full mt-1.5 w-64 p-2.5 rounded bg-white border border-slate-200 shadow-xl text-left normal-case tracking-normal ${
            side === 'left' ? 'left-0' : 'right-0'
          }`}
        >
          <span className="flex items-start space-x-1.5">
            <Info className="w-3 h-3 text-ink-500 shrink-0 mt-0.5" />
            <span className="text-label font-bold text-slate-900 leading-snug font-sans">
              {title}
            </span>
          </span>

          <span className="block mt-1 text-[10.5px] text-slate-700 leading-relaxed font-sans">
            {whatItIs}
          </span>

          {whyItMatters && (
            <span className="block mt-1.5 text-[10.5px] text-slate-600 leading-relaxed font-sans">
              <strong className="text-slate-800">Why it matters: </strong>
              {whyItMatters}
            </span>
          )}

          {analogy && (
            <span className="block mt-1.5 pt-1.5 border-t border-slate-100 text-[10.5px] text-slate-500 italic leading-relaxed font-sans">
              {analogy}
            </span>
          )}

          {range && (
            <span className="block mt-1.5 text-label text-slate-500 font-mono">{range}</span>
          )}
        </span>
      )}
    </span>
  );
};