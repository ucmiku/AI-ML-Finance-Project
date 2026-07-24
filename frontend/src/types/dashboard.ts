export type RiskLevel = 'low' | 'medium' | 'high'
export type TradeSignal = 'INC' | 'DEC' | 'NO_TRADE' | 'No Trade' | 'NO TRADE'
export type DataMode = 'auto' | 'production' | 'demo'

export interface DashboardSnapshot {
  metadata: {
    hub?: string
    mode?: string
    delivery_time?: string
    delivery_time_utc?: string
    decision_time?: string
    forecast_issue_time?: string
    data_status?: string
    model_version?: string
    source?: string
    [key: string]: unknown
  }
  locations: WeatherLocation[]
  weather: Record<string, unknown>
  load: Record<string, number>
  wind: Record<string, number>
  solar: Record<string, number>
  renewable: Record<string, number>
  gas: Record<string, number | boolean>
  extreme_weather: Record<string, number>
  prediction: {
    predicted_spread?: number
    signal?: TradeSignal
    confidence?: number
    risk_level?: RiskLevel | string
    target_available?: boolean
    actual_spread?: number
  }
  drivers: Driver[]
  timeseries_24h: TimePoint[]
  warnings?: string[]
}

export interface WeatherLocation {
  location: string
  latitude: number
  longitude: number
  risk_level?: RiskLevel | string
  coordinate_status?: string
  values: Record<string, number>
  units?: Record<string, string>
}

export interface Driver {
  feature: string
  impact: number
  label?: string
  status?: 'official' | 'illustrative' | 'unavailable'
}

export interface TimePoint {
  delivery_time: string
  [key: string]: number | string
}

export interface MapLayerMeta {
  id: string
  name: string
  type: 'reference' | 'boundary' | 'polygon' | 'point' | 'risk' | 'hub' | 'label'
  enabled: boolean
  visible: boolean
  opacity: number
  variable: string
  range: string
  legend: string[]
  reason?: string
  sourceStatus: 'available' | 'unavailable' | 'demo'
}

export interface ApiBundle {
  snapshot: DashboardSnapshot
  timeseries: TimePoint[]
  layers: MapLayerMeta[]
  source: 'api' | 'mock'
  apiHealthy: boolean
}
