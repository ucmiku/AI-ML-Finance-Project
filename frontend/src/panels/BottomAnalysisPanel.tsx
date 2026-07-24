import { useState, type MouseEvent as ReactMouseEvent } from 'react'
import { ChevronDown, ChevronUp, GripHorizontal, X } from 'lucide-react'
import type { TimePoint } from '../types/dashboard'
import { EChartPanel } from '../charts/EChartPanel'

const tabs = [
  'Weather Forecast',
  'Load & Net Load',
  'Wind & Solar',
  'Natural Gas',
  'Extreme Weather',
  'Model Decision',
] as const

export type AnalysisTab = typeof tabs[number]

const fields: Record<AnalysisTab, string[]> = {
  'Weather Forecast': ['temperature_dfw_mean_c', 'temperature_wichita_c'],
  'Load & Net Load': ['load_system_total_mw', 'net_load_st_forecast_system_mw'],
  'Wind & Solar': ['wind_stwpf_system_wide_mw', 'solar_stppf_system_mw'],
  'Natural Gas': ['gas_price'],
  'Extreme Weather': ['fixed_extreme_weather_flag'],
  'Model Decision': ['predicted_spread'],
}

export function BottomAnalysisPanel({
  timeseries,
  deliveryHour,
}: {
  timeseries: TimePoint[]
  deliveryHour: number
}) {
  const [active, setActive] = useState<AnalysisTab>('Weather Forecast')
  const [collapsed, setCollapsed] = useState(false)
  const [closed, setClosed] = useState(false)
  const [height, setHeight] = useState(300)

  if (closed) {
    return <button className="reopen-bottom" onClick={() => setClosed(false)}>Open Analysis</button>
  }

  return (
    <section className={`bottom-analysis ${collapsed ? 'collapsed' : ''}`} style={{ height: collapsed ? 44 : height }}>
      <div className="resize-grip" onMouseDown={(event) => beginResize(event, setHeight)}><GripHorizontal size={16} /></div>
      <header className="bottom-header">
        <div className="bottom-tabs">
          {tabs.map((tab) => <button key={tab} className={active === tab ? 'active' : ''} onClick={() => setActive(tab)}>{tab}</button>)}
        </div>
        <div className="bottom-actions">
          <button onClick={() => setCollapsed((v) => !v)}>{collapsed ? <ChevronUp size={15} /> : <ChevronDown size={15} />}</button>
          <button onClick={() => setClosed(true)}><X size={15} /></button>
        </div>
      </header>
      {!collapsed && <EChartPanel title={active} timeseries={timeseries} fields={fields[active]} deliveryHour={deliveryHour} />}
    </section>
  )
}

function beginResize(event: ReactMouseEvent<HTMLElement>, setHeight: (height: number) => void) {
  event.preventDefault()
  const startY = event.clientY
  const startHeight = (event.currentTarget.closest('.bottom-analysis') as HTMLElement)?.offsetHeight ?? 300
  const onMove = (moveEvent: MouseEvent) => {
    const next = Math.min(430, Math.max(190, startHeight + (startY - moveEvent.clientY)))
    setHeight(next)
  }
  const onUp = () => {
    window.removeEventListener('mousemove', onMove)
    window.removeEventListener('mouseup', onUp)
  }
  window.addEventListener('mousemove', onMove)
  window.addEventListener('mouseup', onUp)
}
