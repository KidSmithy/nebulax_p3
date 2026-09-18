import React from 'react';
import { useTwinStore } from '../../store/useTwinStore';

/** The only camera control besides the preset buttons: the view is locked, so zoom is one axis. */
export const ZoomSlider: React.FC = () => {
  const cameraZoom = useTwinStore((s) => s.cameraZoom);
  const setCameraZoom = useTwinStore((s) => s.setCameraZoom);

  return (
    <div className="absolute bottom-6 right-4 z-20 pointer-events-auto glass-panel rounded px-3 py-2 w-[240px]">
      <label htmlFor="camera-zoom" className="sr-only">
        Camera zoom
      </label>
      <div className="flex items-center gap-2.5">
        <span className="text-label font-bold text-slate-500 shrink-0">All cars</span>
        <input
          id="camera-zoom"
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={1 - cameraZoom}
          onChange={(e) => setCameraZoom(1 - Number(e.target.value))}
          className="flex-1 min-w-0 accent-ink-900"
        />
        <span className="text-label font-bold text-slate-500 shrink-0">Close</span>
      </div>
    </div>
  );
};
