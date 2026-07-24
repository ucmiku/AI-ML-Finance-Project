import type { RailTab } from '../components/IconRail'
import type { DashboardSnapshot, WeatherLocation } from '../types/dashboard'
import { displayName, formatNumber } from '../services/api'

export function ContextPanel({
  active,
  snapshot,
  selected,
  weatherVariable,
  onWeatherVariable,
}: {
  active: RailTab
  snapshot: DashboardSnapshot
  selected?: WeatherLocation
  weatherVariable: string
  onWeatherVariable: (value: string) => void
}) {
  if (active === 'layers') return null
  return (
    <aside className="left-panel context-panel">
      <header className="panel-header">
        <div>
          <span className="eyebrow">{active}</span>
          <h2>{title(active)}</h2>
        </div>
      </header>
      {active === 'weather' && (
        <>
          <label className="field-label">Weather Variable</label>
          <select value={weatherVariable} onChange={(e) => onWeatherVariable(e.target.value)}>
            <option>Temperature</option>
            <option>Extreme Risk</option>
            <option>Humidity</option>
            <option>Wind Speed</option>
            <option>Precipitation</option>
          </select>
          <div className="dense-grid">
            <Metric label="Selected Temperature" value={`${formatNumber(selected?.values.temperature_dfw_mean_c)} °C`} />
            <Metric label="Humidity" value={`${formatNumber(selected?.values.humidity_dfw_mean_pct)} %`} />
            <Metric label="Wind Gust" value={`${formatNumber(selected?.values.wind_gust_dfw_mean_ms)} m/s`} />
            <Metric label="Precipitation" value={`${formatNumber(selected?.values.precipitation_dfw_mean_mm)} mm`} />
          </div>
        </>
      )}
      {active === 'load' && (
        <div className="dense-grid">
          <Metric label="System Load" value={`${formatNumber(snapshot.load.load_system_total_mw, 0)} MW`} />
          <Metric label="Net Load" value={`${formatNumber(snapshot.load.net_load_st_forecast_system_mw, 0)} MW`} />
          <Metric label="1h Ramp" value={`${formatNumber(snapshot.load.load_ramp_1h_mw, 0)} MW`} />
          <Metric label="3h Ramp" value={`${formatNumber(snapshot.load.load_ramp_3h_mw, 0)} MW`} />
        </div>
      )}
      {active === 'renewables' && (
        <div className="dense-grid">
          <Metric label="Renewable Share" value={`${formatNumber((snapshot.renewable.renewable_st_share_of_load ?? 0) * 100)} %`} />
          <Metric label="Wind Gap" value={`${formatNumber(snapshot.wind.wind_gap_system_mw, 0)} MW`} />
          <Metric label="Solar Gap" value={`${formatNumber(snapshot.solar.solar_gap_system_mw, 0)} MW`} />
          <Metric label="Forecast MW" value={`${formatNumber(snapshot.renewable.renewable_st_forecast_system_mw, 0)} MW`} />
        </div>
      )}
      {active === 'risk' && (
        <div className="dense-grid">
          {Object.entries(snapshot.extreme_weather).slice(0, 8).map(([key, value]) => (
            <Metric key={key} label={displayName(key)} value={formatNumber(value, 2)} />
          ))}
        </div>
      )}
      {active === 'model' && (
        <div className="small-copy">
          <p>Model output is read from dashboard snapshot JSON. This workbench does not retrain, mutate, or load production parquet in React.</p>
          <p>Top drivers are shown as Unavailable unless official SHAP values are present. Demo drivers are labeled Illustrative.</p>
        </div>
      )}
      {active === 'info' && (
        <div className="small-copy">
          <p>Map-first ERCOT workbench v3.</p>
          <p>Spatial boundary status: trusted ERCOT/load/weather zone GeoJSON not found in local inventory, so those layers remain disabled.</p>
          <p>Data source: {snapshot.metadata.source}</p>
        </div>
      )}
    </aside>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return <div className="micro-metric"><span>{label}</span><b>{value}</b></div>
}

function title(tab: RailTab): string {
  return {
    layers: 'Layers',
    weather: 'Weather Controls',
    load: 'Load Diagnostics',
    renewables: 'Renewables',
    risk: 'Extreme Weather',
    model: 'Model Notes',
    info: 'Data Provenance',
  }[tab]
}
