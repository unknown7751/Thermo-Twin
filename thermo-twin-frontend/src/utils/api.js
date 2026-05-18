import axios from 'axios'

// Empty baseURL → requests use the current origin (Vite dev proxy forwards to Flask).
// Set VITE_API_URL to an explicit http:// URL only for production deploys.
const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || '',
  timeout: 8000,
})

export default api
