import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/stream':  { target: 'http://localhost:5000', changeOrigin: true },
      '/twin':    { target: 'http://localhost:5000', changeOrigin: true },
      '/fleet':   { target: 'http://localhost:5000', changeOrigin: true },
      '/alerts':  { target: 'http://localhost:5000', changeOrigin: true },
      '/health':  { target: 'http://localhost:5000', changeOrigin: true },
      '/demo':    { target: 'http://localhost:5000', changeOrigin: true },
      '/signal':  { target: 'http://localhost:5000', changeOrigin: true },
      '/ws': {
        target: 'ws://localhost:5000',
        ws: true,
        changeOrigin: true,
      },
    },
  },
})
