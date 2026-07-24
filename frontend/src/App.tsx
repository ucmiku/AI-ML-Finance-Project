import { useEffect, useMemo, useState } from 'react'
import { IconRail, type RailTab } from './components/IconRail'
import { MapCanvas } from './map/MapCanvas'
import { ContextPanel } from './panels/ContextPanel'
import { defaultLayers, demoSnapshot } from './mock/demoSnapshot'
import { loadWorkbenchData } from './services/api'
import type { DataMode, DashboardSnapshot, MapLayerMeta, WeatherLocation } from './types/dashboard'

export function App() {
  const [activeRail, setActiveRail] = useState<RailTab>('weather')
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

  const toggleLayer = (id: string, visible: boolean) => {
    setLayers((items) => items.map((item) => item.id === id && item.enabled ? { ...item, visible } : item))
  }

  return (
    <div className="app-shell">
      {/* 第 1 列：左侧图标导航栏 (48px) */}
      <IconRail active={activeRail} onChange={setActiveRail} />
      
      {/* 第 2 列：必须加回这个左侧图层数据面板 (230px)，否则地图会挤到这里来 */}
      <ContextPanel 
        active={activeRail} 
        snapshot={snapshot} 
        selected={selected} 
        weatherVariable={weatherVariable} 
        onWeatherVariable={setWeatherVariable} 
      />

      {/* 第 3 列：地图接管剩余的全部空间 (1fr) */}
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

      {/* 底部数据模式切换 */}
      <div className="mode-switcher">
        <label>Data</label>
        <select value={mode} onChange={(e) => setMode(e.target.value as DataMode)}>
          <option value="auto">Auto API→Mock</option>
          <option value="production">Production API</option>
          <option value="demo">Demo Snapshot</option>
        </select>
        <span className={apiHealthy ? 'ok' : 'warn'}>{source === 'api' ? 'FastAPI' : 'Mock JSON'}</span>
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