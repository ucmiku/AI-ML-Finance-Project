export const TEXAS_CENTER: [number, number] = [-99.9018, 31.9686]
export const TEXAS_BOUNDS: [[number, number], [number, number]] = [
  [-107.4, 25.5],
  [-93.1, 36.7],
]

export const cartoDarkMatterStyle =
  'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json'

export function riskColor(risk?: string): string {
  if (risk?.toLowerCase() === 'high') return '#e55353'
  if (risk?.toLowerCase() === 'medium') return '#f2a93b'
  return '#20b486'
}

export function temperatureColor(value: number): string {
  if (value >= 38) return '#e55353'
  if (value >= 35) return '#f2a93b'
  if (value >= 30) return '#ffd166'
  return '#2f80ed'
}
