import { defaultLayers, demoSnapshot } from '../mock/demoSnapshot'
import type { ApiBundle, DataMode, DashboardSnapshot, MapLayerMeta, TimePoint } from '../types/dashboard'

const API_BASE = import.meta.env.VITE_ERCOT_API_BASE ?? ''

async function getJson<T>(path: string, timeoutMs = 1600): Promise<T> {
  const controller = new AbortController()
  const timer = window.setTimeout(() => controller.abort(), timeoutMs)
  try {
    const res = await fetch(`${API_BASE}${path}`, { signal: controller.signal })
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
    return (await res.json()) as T
  } finally {
    window.clearTimeout(timer)
  }
}

export async function loadWorkbenchData(mode: DataMode): Promise<ApiBundle> {
  if (mode === 'demo') {
    return mockBundle(false)
  }

  try {
    await getJson<{ status: string }>('/api/v1/health')
    const [snapshot, timeseries, layers] = await Promise.all([
      getJson<DashboardSnapshot>('/api/v1/dashboard/snapshot'),
      getJson<TimePoint[]>('/api/v1/dashboard/timeseries'),
      getJson<MapLayerMeta[]>('/api/v1/map/layers'),
    ])
    return { snapshot, timeseries, layers, source: 'api', apiHealthy: true }
  } catch {
    return mockBundle(false)
  }
}

async function mockBundle(apiHealthy: boolean): Promise<ApiBundle> {
  const snapshot = await loadLocalDemoSnapshot()
  return {
    snapshot,
    timeseries: snapshot.timeseries_24h,
    layers: defaultLayers,
    source: 'mock',
    apiHealthy,
  }
}

async function loadLocalDemoSnapshot(): Promise<DashboardSnapshot> {
  try {
    const res = await fetch('/mock/demo_snapshot.json')
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
    return (await res.json()) as DashboardSnapshot
  } catch {
    return demoSnapshot
  }
}

export function displayName(field: string): string {
  const names: Record<string, string> = {
    temperature_dfw_mean_c: 'Temperature',
    temperature_wichita_c: 'Wichita Temperature',
    humidity_dfw_mean_pct: 'Humidity',
    wind_speed_dfw_mean_ms: 'Wind Speed',
    wind_gust_dfw_mean_ms: 'Wind Gust',
    precipitation_dfw_mean_mm: 'Precipitation',
    load_system_total_mw: 'System Load',
    net_load_st_forecast_system_mw: 'Net Load',
    wind_stwpf_system_wide_mw: 'Wind Forecast',
    wind_wgrpp_system_wide_mw: 'Wind Potential',
    solar_stppf_system_mw: 'Solar Forecast',
    solar_pvgrpp_system_mw: 'Solar Potential',
    gas_price: 'Natural Gas',
    predicted_spread: 'Predicted RT−DA Spread',
    fixed_extreme_weather_flag: 'Extreme Weather Flag',
    net_load_ramp_3h_mw: '3h Net Load Ramp',
    extreme_heat_hour_flag: 'Extreme Heat Flag',
    wind_gap_system_mw: 'Wind Forecast Gap',
    gas_price_z30: 'Gas Price Z-Score',
  }
  return names[field] ?? field.replace(/_/g, ' ').replace(/\b\w/g, (m) => m.toUpperCase())
}

export function formatNumber(value: unknown, digits = 1): string {
  if (typeof value !== 'number' || Number.isNaN(value)) return 'Unavailable'
  return new Intl.NumberFormat('en-US', { maximumFractionDigits: digits }).format(value)
}
