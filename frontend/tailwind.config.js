/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
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
        lta: {
          green: '#009645',    // LTA EWL & SG Bus Lush Green
          red: '#D42E12',      // LTA NSL Red Line
          amber: '#FF9E1B',    // LTA Circle Line Orange
          blue: '#005EC4',     // LTA Downtown Line Blue
          purple: '#7B1FA2',   // SBS Transit / NEL Purple
        },
      }
    },
  },
  plugins: [],
}
