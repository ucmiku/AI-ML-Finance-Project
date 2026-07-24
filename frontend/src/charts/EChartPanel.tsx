import { useEffect, useRef } from 'react'
import * as echarts from 'echarts'
import type { TimePoint } from '../types/dashboard'
import { displayName } from '../services/api'

export function EChartPanel({
  title,
  timeseries,
  fields,
  deliveryHour,
}: {
  title: string
  timeseries: TimePoint[]
  fields: string[]
  deliveryHour: number
}) {
  const ref = useRef<HTMLDivElement | null>(null)
  useEffect(() => {
    if (!ref.current) return
    const chart = echarts.init(ref.current, 'dark')
    const labels = timeseries.map((row) => hourLabel(String(row.delivery_time)))
    const markerIndex = Math.max(0, timeseries.findIndex((row) => hourFromTime(String(row.delivery_time)) === deliveryHour))
    chart.setOption({
      backgroundColor: 'transparent',
      grid: { left: 52, right: 24, top: 36, bottom: 34 },
      title: { text: title, textStyle: { color: '#f3f6fa', fontSize: 13, fontWeight: 600 }, left: 8, top: 4 },
      legend: { top: 6, right: 10, textStyle: { color: '#9aa7b5' } },
      tooltip: { trigger: 'axis', backgroundColor: '#111820', borderColor: 'rgba(255,255,255,0.12)', textStyle: { color: '#f3f6fa' } },
      xAxis: {
        type: 'category',
        data: labels,
        axisLine: { lineStyle: { color: 'rgba(255,255,255,0.18)' } },
        axisLabel: { color: '#9aa7b5' },
      },
      yAxis: {
        type: 'value',
        splitLine: { lineStyle: { color: 'rgba(255,255,255,0.08)' } },
        axisLabel: { color: '#9aa7b5' },
      },
      series: fields.map((field, index) => ({
        name: displayName(field),
        type: 'line',
        smooth: true,
        symbol: 'circle',
        symbolSize: 6,
        lineStyle: { width: 2 },
        itemStyle: { color: ['#2f80ed', '#20b486', '#f2a93b', '#e55353'][index % 4] },
        data: timeseries.map((row) => row[field] ?? null),
        markLine: index === 0 ? {
          symbol: 'none',
          label: { formatter: 'Delivery Hour', color: '#f3f6fa' },
          lineStyle: { color: '#e55353', width: 2, type: 'solid' },
          data: [{ xAxis: labels[markerIndex] }],
        } : undefined,
      })),
    })
    const resize = () => chart.resize()
    window.addEventListener('resize', resize)
    return () => {
      window.removeEventListener('resize', resize)
      chart.dispose()
    }
  }, [deliveryHour, fields.join('|'), timeseries, title])

  return <div ref={ref} className="echart-host" />
}

function hourLabel(value: string): string {
  const match = value.match(/(\d{2}):\d{2}:\d{2}/)
  return match ? `${match[1]}:00` : value
}

function hourFromTime(value: string): number {
  const match = value.match(/(\d{2}):\d{2}:\d{2}/)
  return match ? Number(match[1]) : -1
}
