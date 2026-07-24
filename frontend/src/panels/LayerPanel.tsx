import { AlertTriangle, Eye, EyeOff, Layers, MapPinned, Shapes } from 'lucide-react'
import type { MapLayerMeta } from '../types/dashboard'

interface Props {
  layers: MapLayerMeta[]
  onToggle: (id: string, visible: boolean) => void
  onOpacity: (id: string, opacity: number) => void
}

function iconFor(type: MapLayerMeta['type']) {
  if (type === 'point' || type === 'hub') return <MapPinned size={15} />
  if (type === 'risk') return <AlertTriangle size={15} />
  return <Shapes size={15} />
}

export function LayerPanel({ layers, onToggle, onOpacity }: Props) {
  return (
    <aside className="left-panel">
      <header className="panel-header">
        <div>
          <span className="eyebrow">GIS Layers</span>
          <h2>ERCOT Map Stack</h2>
        </div>
        <Layers size={17} />
      </header>
      <div className="layer-list">
        {layers.map((layer) => (
          <section key={layer.id} className={`layer-card ${!layer.enabled ? 'disabled' : ''}`}>
            <div className="layer-row">
              <button
                disabled={!layer.enabled}
                className="visibility-btn"
                aria-label={`Toggle ${layer.name}`}
                onClick={() => onToggle(layer.id, !layer.visible)}
              >
                {layer.visible && layer.enabled ? <Eye size={15} /> : <EyeOff size={15} />}
              </button>
              <div className="layer-type">{iconFor(layer.type)}</div>
              <div className="layer-title">
                <strong>{layer.name}</strong>
                <span>{layer.sourceStatus === 'demo' ? 'Demo Coordinates' : layer.sourceStatus}</span>
              </div>
            </div>
            <div className="layer-meta">
              <span>Variable</span><b>{layer.variable}</b>
              <span>Range</span><b>{layer.range}</b>
            </div>
            <div className="opacity-row">
              <span>Opacity</span>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={layer.opacity}
                disabled={!layer.enabled}
                onChange={(e) => onOpacity(layer.id, Number(e.target.value))}
              />
              <b>{Math.round(layer.opacity * 100)}%</b>
            </div>
            <div className="legend-row">
              {layer.legend.map((item) => <span key={item}>{item}</span>)}
            </div>
            {layer.reason && <p className="unavailable-reason">{layer.reason}</p>}
          </section>
        ))}
      </div>
    </aside>
  )
}
