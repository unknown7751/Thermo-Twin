/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      animation: {
        'pulse-slow': 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'slide-in':   'slideIn 0.35s ease forwards',
        'flash-red':  'flashRed 0.6s ease forwards',
      },
      keyframes: {
        slideIn: {
          from: { opacity: 0, transform: 'translateY(-8px)' },
          to:   { opacity: 1, transform: 'translateY(0)' },
        },
        flashRed: {
          '0%':   { boxShadow: '0 0 0 0 rgba(220,38,38,0.6)' },
          '70%':  { boxShadow: '0 0 0 16px rgba(220,38,38,0)' },
          '100%': { boxShadow: 'none' },
        },
      },
    },
  },
  plugins: [],
}
