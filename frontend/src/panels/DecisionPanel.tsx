import { AlertTriangle, BadgeCheck, MinusCircle, TrendingDown, TrendingUp } from 'lucide-react'
import type { DashboardSnapshot, WeatherLocation } from '../types/dashboard'
import { displayName, formatNumber } from '../services/api'

export function DecisionPanel({ snapshot, selected }: { snapshot: DashboardSnapshot; selected?: WeatherLocation }) {
  const signal = normalizeSignal(snapshot.prediction.signal)
  const risk = String(snapshot.prediction.risk_level ?? 'unavailable').toLowerCase()
  const hasOfficialDrivers = snapshot.drivers.some((d) => d.status === 'official')

  return (
    <aside className="right-panel">
      <section className="decision-section">
        <div className="section-title">Trading Decision</div>
        <div className={`trade-card ${signal.toLowerCase().replace(' ', '-')}`}>
          <div>
            <span>Predicted RT−DA Spread</span>
            <strong>{formatNumber(snapshot.prediction.predicted_spread, 2)} $/MWh</strong>
          </div>
          <div className="signal">
            {signal === 'INC' ? <TrendingUp size={22} /> : signal === 'DEC' ? <TrendingDown size={22} /> : <MinusCircle size={22} />}
            {signal}
          </div>
        </div>
        <div className="decision-grid">
          <Metric label="Confidence" value={`${formatNumber(snapshot.prediction.confidence)} %`} />
          <Metric label="Extreme Weather Risk" value={risk.toUpperCase()} danger={risk === 'high'} />
        </div>
      </section>

      <section className="decision-section">
        <div className="section-title">Selected Location</div>
        <div className="selected-name">{selected?.location ?? 'No location selected'}</div>
        <div className="kv-list">
          <Row label="Temperature" value={`${formatNumber(selected?.values.temperature_dfw_mean_c)} °C`} />
          <Row label="Humidity" value={`${formatNumber(selected?.values.humidity_dfw_mean_pct)} %`} />
          <Row label="Wind Speed" value={`${formatNumber(selected?.values.wind_speed_dfw_mean_ms)} m/s`} />
          <Row label="Wind Gust" value={`${formatNumber(selected?.values.wind_gust_dfw_mean_ms)} m/s`} />
          <Row label="Precipitation" value={`${formatNumber(selected?.values.precipitation_dfw_mean_mm)} mm`} />
          <Row label="Forecast Time" value={snapshot.metadata.forecast_issue_time ?? 'Unavailable'} />
        </div>
      </section>

      <section className="decision-section">
        <div className="section-title">Market Conditions</div>
        <div className="kv-list">
          <Row label="System Load" value={`${formatNumber(snapshot.load.load_system_total_mw, 0)} MW`} />
          <Row label="Net Load" value={`${formatNumber(snapshot.load.net_load_st_forecast_system_mw, 0)} MW`} />
          <Row label="Natural Gas" value={`$${formatNumber(snapshot.gas.gas_price, 2)}`} />
          <Row label="Renewable Share" value={`${formatNumber((snapshot.renewable.renewable_st_share_of_load ?? 0) * 100)} %`} />
          <Row label="Wind Gap" value={`${formatNumber(snapshot.wind.wind_gap_system_mw, 0)} MW`} />
          <Row label="Solar Gap" value={`${formatNumber(snapshot.solar.solar_gap_system_mw, 0)} MW`} />
        </div>
      </section>

      <section className="decision-section">
        <div className="section-title">
          Top Drivers
          {!hasOfficialDrivers && <span className="badge-muted">Illustrative / SHAP unavailable</span>}
        </div>
        <div className="driver-bars">
          {snapshot.drivers.length === 0 ? (
            <div className="unavailable-inline"><AlertTriangle size={14} /> Unavailable</div>
          ) : snapshot.drivers.map((driver) => {
            const width = Math.min(100, Math.abs(driver.impact) / 3 * 100)
            return (
              <div key={driver.feature} className="driver-row">
                <span>{driver.label ?? displayName(driver.feature)}</span>
                <div className="driver-track">
                  <i className={driver.impact >= 0 ? 'positive' : 'negative'} style={{ width: `${width}%` }} />
                </div>
                <b>{driver.impact > 0 ? '+' : ''}{formatNumber(driver.impact, 2)}</b>
              </div>
            )
          })}
        </div>
      </section>
      <footer className="right-foot"><BadgeCheck size={14} /> JSON-only frontend · no parquet reads</footer>
    </aside>
  )
}

function normalizeSignal(signal: unknown): 'INC' | 'DEC' | 'NO TRADE' {
  const text = String(signal ?? '').toUpperCase().replace('_', ' ')
  if (text === 'INC') return 'INC'
  if (text === 'DEC') return 'DEC'
  return 'NO TRADE'
}

function Metric({ label, value, danger }: { label: string; value: string; danger?: boolean }) {
  return <div className={`metric-tile ${danger ? 'danger' : ''}`}><span>{label}</span><b>{value}</b></div>
}

function Row({ label, value }: { label: string; value: string }) {
  return <div className="kv-row"><span>{label}</span><b title={value}>{value}</b></div>
}
