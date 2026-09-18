/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Brand mark is neutral ink. Red is severity only (see `status`).
        ink: {
          900: '#0f172a',
          700: '#334155',
          500: '#64748b',
          200: '#e2e8f0',
        },
        dark: {
          900: '#070b12',
          850: '#0d131f',
          800: '#111827',
          700: '#1f2937',
        },
        smrt: {
          red: '#ED1C24',
          darkred: '#BE121A',
          hover: '#FF333B',
          glow: 'rgba(237, 28, 36, 0.35)',
        },
        // Four-step severity ramp. The ONLY sanctioned use of red.
        status: {
          nominal: '#0f766e',
          watch: '#B45309',
          degraded: '#C2410C',
          fault: '#ED1C24',
        },
        lta: {
          green: '#009645',
          red: '#D42E12',
          amber: '#FF9E1B',
          blue: '#005EC4',
          purple: '#7B1FA2',
        },
      },
      // One radius. `full` stays for genuinely circular things (dots, avatars).
      borderRadius: {
        none: '0px',
        sm: '3px',
        DEFAULT: '4px',
        md: '4px',
        lg: '4px',
        xl: '4px',
        '2xl': '6px',
        '3xl': '6px',
      },
      // Real scale. Nothing below 11px.
      fontSize: {
        label: ['11px', { lineHeight: '14px', letterSpacing: '0.01em' }],
        body: ['13px', { lineHeight: '18px' }],
        readout: ['15px', { lineHeight: '18px' }],
        display: ['22px', { lineHeight: '26px' }],
        xs: ['12px', { lineHeight: '16px' }],
        sm: ['13px', { lineHeight: '18px' }],
      },
      // Two elevations. Chrome is flat; only canvas-floating things cast.
      boxShadow: {
        none: 'none',
        sm: '0 1px 2px rgba(15, 23, 42, 0.06)',
        DEFAULT: '0 1px 2px rgba(15, 23, 42, 0.06)',
        md: '0 2px 6px rgba(15, 23, 42, 0.08)',
        lg: '0 2px 6px rgba(15, 23, 42, 0.08)',
        xl: '0 2px 6px rgba(15, 23, 42, 0.08)',
        '2xl': '0 2px 6px rgba(15, 23, 42, 0.08)',
      },
    },
  },
  plugins: [],
}
