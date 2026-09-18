import { Line } from '../types/telemetry';

export const LINE_IDS: Line[] = ['NSL', 'EWL'];

/** Per-line identity. Both lines share one simulated telemetry stream; only the livery, findings and Conductor differ. */
export const LINES: Record<
  Line,
  { tab: string; longName: string; setId: string; color: string; textClass: string; dotClass: string; light: string }
> = {
  NSL: {
    tab: 'NSL MRT',
    longName: 'NORTH-SOUTH LINE',
    setId: 'C151B-SET-402',
    color: '#ED1C24',
    textClass: 'text-red-600',
    dotClass: 'bg-red-600',
    light: '#f87171',
  },
  EWL: {
    tab: 'EWL MRT',
    longName: 'EAST-WEST LINE',
    setId: 'C151B-SET-517',
    color: '#009645',
    textClass: 'text-green-700',
    dotClass: 'bg-green-600',
    light: '#4ade80',
  },
};

/** Car the scene focuses on until an ACV finding names a faulty one. */
export const DEFAULT_CAR = 4;
