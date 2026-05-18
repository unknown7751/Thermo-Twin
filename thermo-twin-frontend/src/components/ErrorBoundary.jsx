import { Component } from 'react'

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { error: null, info: null }
  }

  static getDerivedStateFromError(error) {
    return { error }
  }

  componentDidCatch(error, info) {
    this.setState({ info })
    // eslint-disable-next-line no-console
    console.error('ErrorBoundary caught:', error, info)
  }

  render() {
    if (this.state.error) {
      return (
        <div className="min-h-screen bg-slate-950 text-slate-100 p-8 font-mono text-sm">
          <h1 className="text-red-400 text-lg font-bold mb-3">⚠ React render error</h1>
          <pre className="bg-slate-900 border border-red-800 rounded p-4 overflow-auto whitespace-pre-wrap text-red-300">
            {String(this.state.error?.stack || this.state.error)}
          </pre>
          {this.state.info?.componentStack && (
            <pre className="bg-slate-900 border border-slate-800 rounded p-4 mt-3 overflow-auto whitespace-pre-wrap text-slate-400">
              {this.state.info.componentStack}
            </pre>
          )}
          <button
            onClick={() => this.setState({ error: null, info: null })}
            className="mt-4 px-4 py-2 rounded border border-slate-700 hover:border-cyan-500 text-cyan-300"
          >
            Try to recover
          </button>
        </div>
      )
    }
    return this.props.children
  }
}
