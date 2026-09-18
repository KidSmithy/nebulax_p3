import React from 'react';
import { Play, Pause, RotateCcw, Navigation } from 'lucide-react';
import { useTwinStore } from '../../store/useTwinStore';

export const TimelineScrubber: React.FC = () => {
  const isPlaying = useTwinStore((state) => state.isPlaying);
  const setPlaying = useTwinStore((state) => state.setPlaying);
  const playbackSpeed = useTwinStore((state) => state.playbackSpeed);
  const setPlaybackSpeed = useTwinStore((state) => state.setPlaybackSpeed);
  const currentFrame = useTwinStore((state) => state.currentFrame);
  const sendSeek = useTwinStore((state) => state.sendSeek);

  const currentKp = currentFrame?.track_chainage_km ?? 12.5;
  const minKp = 12.5;
  const maxKp = 35.0;

  const handleSeekChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    sendSeek(val);
  };

  return (
    <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-20 pointer-events-none w-full max-w-xl px-4">
      <div className="glass-panel px-4 py-2.5 rounded-2xl pointer-events-auto border-white/10 flex items-center space-x-4 shadow-2xl">
        {/* SMRT Red Play / Pause Toggle */}
        <button
          onClick={() => setPlaying(!isPlaying)}
          className={`w-8 h-8 rounded-xl flex items-center justify-center transition-all shrink-0 ${
            isPlaying
              ? 'bg-red-600 text-white font-bold shadow-md shadow-red-600/40 hover:bg-red-500'
              : 'bg-amber-500 text-slate-950 font-bold shadow-md shadow-amber-500/40 hover:bg-amber-400'
          }`}
          title={isPlaying ? 'Pause Simulation' : 'Resume Simulation'}
        >
          {isPlaying ? <Pause className="w-4 h-4 fill-current" /> : <Play className="w-4 h-4 fill-current ml-0.5" />}
        </button>

        {/* Reset to Start */}
        <button
          onClick={() => sendSeek(minKp)}
          className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800/60 transition-colors shrink-0"
          title="Reset to Track Start (KP 12.500)"
        >
          <RotateCcw className="w-3.5 h-3.5" />
        </button>

        {/* Speed Multiplier Selectors */}
        <div className="flex items-center space-x-0.5 bg-dark-850/90 p-0.5 rounded-lg border border-white/5 font-mono text-[10px] shrink-0">
          {[1.0, 2.0, 5.0, 10.0].map((spd) => (
            <button
              key={spd}
              onClick={() => setPlaybackSpeed(spd)}
              className={`px-2 py-0.5 rounded transition-all font-semibold ${
                playbackSpeed === spd
                  ? 'bg-red-600/25 text-red-300 border border-red-500/50'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {spd}x
            </button>
          ))}
        </div>

        {/* Track Chainage Timeline Slider */}
        <div className="flex-1 flex flex-col space-y-0.5 min-w-0">
          <div className="flex justify-between items-center text-[10px] font-mono text-slate-400">
            <span className="flex items-center space-x-1 text-slate-300">
              <Navigation className="w-3 h-3 text-red-500" />
              <span>SMRT LINE CHAINAGE</span>
            </span>
            <span className="text-red-400 font-bold">
              KP {currentKp.toFixed(3)}
            </span>
          </div>

          <input
            type="range"
            min={minKp}
            max={maxKp}
            step={0.05}
            value={currentKp}
            onChange={handleSeekChange}
            className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-red-600 hover:accent-red-500 transition-all"
          />
        </div>
      </div>
    </div>
  );
};
