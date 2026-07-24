import { useEffect, useMemo, useState } from 'react'
import { IconRail, type RailTab } from './components/IconRail'
import { MapCanvas } from './map/MapCanvas'
import { DecisionPanel } from './panels/DecisionPanel'
import { LayerPanel } from './panels/LayerPanel'
import { ContextPanel } from './panels/ContextPanel'
import { BottomAnalysisPanel } from './panels/BottomAnalysisPanel'
import { defaultLayers, demoSnapshot } from './mock/demoSnapshot'
import { loadWorkbenchData } from './services/api'
import type { DataMode, DashboardSnapshot, MapLayerMeta, WeatherLocation } from './types/dashboard'

export function App() {
  const [activeRail, setActiveRail] = useState<RailTab>('layers')
  const [mode, setMode] = useState<DataMode>('auto')
  const [snapshot, setSnapshot] = useState<DashboardSnapshot>(demoSnapshot)
  const [layers, setLayers] = useState<MapLayerMeta[]>(defaultLayers)
  const [apiHealthy, setApiHealthy] = useState(false)
  const [source, setSource] = useState<'api' | 'mock'>('mock')
  const [selected, setSelected] = useState<WeatherLocation | undefined>(demoSnapshot.locations[0])
  const [weatherVariable, setWeatherVariable] = useState('Temperature')
  const [marketVariable, setMarketVariable] = useState('Predicted RT−DA Spread')
  const [deliveryDate, setDeliveryDate] = useState('2025-07-15')
  const [deliveryHour, setDeliveryHour] = useState(12)

  useEffect(() => {
    let cancelled = false
    loadWorkbenchData(mode).then((bundle) => {
      if (cancelled) return
      setSnapshot(bundle.snapshot)
      setLayers(bundle.layers)
      setApiHealthy(bundle.apiHealthy)
      setSource(bundle.source)
      setSelected(bundle.snapshot.locations[0])
      setDeliveryDate(parseDeliveryDate(bundle.snapshot.metadata.delivery_time))
      setDeliveryHour(parseDeliveryHour(bundle.snapshot.metadata.delivery_time))
    })
    return () => { cancelled = true }
  }, [mode])

  const timeseries = useMemo(() => snapshot.timeseries_24h, [snapshot])

  const toggleLayer = (id: string, visible: boolean) => {
    setLayers((items) => items.map((item) => item.id === id && item.enabled ? { ...item, visible } : item))
  }
  const setOpacity = (id: string, opacity: number) => {
    setLayers((items) => items.map((item) => item.id === id ? { ...item, opacity } : item))
  }

  return (
    <div className="app-shell">
      <IconRail active={activeRail} onChange={setActiveRail} />
      {activeRail === 'layers' ? (
        <LayerPanel layers={layers} onToggle={toggleLayer} onOpacity={setOpacity} />
      ) : (
        <ContextPanel active={activeRail} snapshot={snapshot} selected={selected} weatherVariable={weatherVariable} onWeatherVariable={setWeatherVariable} />
      )}
      <MapCanvas
        snapshot={snapshot}
        layers={layers}
        selectedLocation={selected}
        weatherVariable={weatherVariable}
        marketVariable={marketVariable}
        deliveryDate={deliveryDate}
        deliveryHour={deliveryHour}
        onSelectedLocation={setSelected}
        onLayerToggle={toggleLayer}
        onReset={() => undefined}
      />
      <DecisionPanel snapshot={snapshot} selected={selected} />
      <BottomAnalysisPanel timeseries={timeseries} deliveryHour={deliveryHour} />
      <div className="mode-switcher">
        <label>Data</label>
        <select value={mode} onChange={(e) => setMode(e.target.value as DataMode)}>
          <option value="auto">Auto API→Mock</option>
          <option value="production">Production API</option>
          <option value="demo">Demo Snapshot</option>
        </select>
        <span className={apiHealthy ? 'ok' : 'warn'}>{source === 'api' ? 'FastAPI' : 'Mock JSON'}</span>
      </div>
      <div className="hidden-controls">
        <select value={weatherVariable} onChange={(e) => setWeatherVariable(e.target.value)}>
          <option>Temperature</option>
          <option>Extreme Risk</option>
        </select>
        <select value={marketVariable} onChange={(e) => setMarketVariable(e.target.value)}>
          <option>Predicted RT−DA Spread</option>
          <option>System Load</option>
          <option>Net Load</option>
        </select>
        <input type="date" value={deliveryDate} onChange={(e) => setDeliveryDate(e.target.value)} />
        <input type="range" min="0" max="23" value={deliveryHour} onChange={(e) => setDeliveryHour(Number(e.target.value))} />
      </div>
    </div>
  )
}

function parseDeliveryDate(value?: string): string {
  return value?.slice(0, 10) ?? '2025-07-15'
}

function parseDeliveryHour(value?: string): number {
  const match = value?.match(/(\d{2}):\d{2}:\d{2}/)
  return match ? Number(match[1]) : 12
}
