import { useEffect, useMemo, useRef, useState } from 'react'
import maplibregl, { GeoJSONSource, Map as MLMap, Popup } from 'maplibre-gl'
import type { FeatureCollection, Point } from 'geojson'
import { Camera, Maximize2, RotateCcw } from 'lucide-react'
import { cartoDarkMatterStyle, riskColor, temperatureColor, TEXAS_BOUNDS, TEXAS_CENTER } from './mapStyle'
import type { DashboardSnapshot, MapLayerMeta, WeatherLocation } from '../types/dashboard'

interface Props {
  snapshot: DashboardSnapshot
  layers: MapLayerMeta[]
  selectedLocation?: WeatherLocation
  weatherVariable: string
  marketVariable: string
  deliveryDate: string
  deliveryHour: number
  onSelectedLocation: (location: WeatherLocation) => void
  onLayerToggle: (id: string, visible: boolean) => void
  onReset: () => void
}

interface WeatherPointProperties {
  id: string
  location: string
  risk: string
  temp: number
  humidity: number
  selected: boolean
  color: string
}

export function MapCanvas({
  snapshot,
  layers,
  selectedLocation,
  weatherVariable,
  marketVariable,
  deliveryDate,
  deliveryHour,
  onSelectedLocation,
  onLayerToggle,
  onReset,
}: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const mapRef = useRef<MLMap | null>(null)
  const popupRef = useRef<Popup | null>(null)
  const [ready, setReady] = useState(false)

  const weatherVisible = layers.find((layer) => layer.id === 'weather-points')?.visible ?? true
  const riskVisible = layers.find((layer) => layer.id === 'extreme-risk')?.visible ?? true
  const hubVisible = layers.find((layer) => layer.id === 'settlement-hub')?.visible ?? true
  const labelsVisible = layers.find((layer) => layer.id === 'labels')?.visible ?? true
  const weatherOpacity = layers.find((layer) => layer.id === 'weather-points')?.opacity ?? 1
  const riskOpacity = layers.find((layer) => layer.id === 'extreme-risk')?.opacity ?? 0.65
  const hubOpacity = layers.find((layer) => layer.id === 'settlement-hub')?.opacity ?? 1
  const labelsOpacity = layers.find((layer) => layer.id === 'labels')?.opacity ?? 1

  const points = useMemo<FeatureCollection<Point, WeatherPointProperties>>(() => ({
    type: 'FeatureCollection',
    features: snapshot.locations.map((location) => ({
      type: 'Feature',
      properties: {
        id: location.location,
        location: location.location,
        risk: String(location.risk_level ?? 'low'),
        temp: location.values.temperature_dfw_mean_c ?? 0,
        humidity: location.values.humidity_dfw_mean_pct ?? 0,
        selected: selectedLocation?.location === location.location,
        color: weatherVariable === 'Extreme Risk'
          ? riskColor(location.risk_level)
          : temperatureColor(location.values.temperature_dfw_mean_c ?? 0),
      },
      geometry: { type: 'Point', coordinates: [location.longitude, location.latitude] },
    })),
  }), [snapshot.locations, selectedLocation, weatherVariable])

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return
    const map = new maplibregl.Map({
      container: containerRef.current,
      // 🌟 替换这里：使用自带浅米黄色的亮色底图 (CARTO Voyager)
      style: 'https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json',
      center: TEXAS_CENTER,
      zoom: 5.05,
      pitch: 0,
      attributionControl: false,
      preserveDrawingBuffer: true,
    })
    map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), 'bottom-right')
    map.addControl(new maplibregl.ScaleControl({ unit: 'imperial' }), 'bottom-left')
    popupRef.current = new maplibregl.Popup({ closeButton: false, closeOnClick: false, className: 'ercot-popup' })

    map.on('load', () => {
      map.addSource('ercot-weather-points', { type: 'geojson', data: points })
      map.addLayer({
        id: 'weather-risk-halo',
        type: 'circle',
        source: 'ercot-weather-points',
        paint: {
          'circle-radius': ['case', ['==', ['get', 'selected'], true], 36, 28],
          'circle-color': ['get', 'color'],
          'circle-opacity': riskVisible ? 0.16 : 0,
          'circle-blur': 0.55,
        },
      })
      map.addLayer({
        id: 'weather-point-outer',
        type: 'circle',
        source: 'ercot-weather-points',
        paint: {
          'circle-radius': ['case', ['==', ['get', 'selected'], true], 12, 9],
          // 🌟 原本是 '#090d12'，现在改为白色或浅色
          'circle-color': '#ffffff', 
          'circle-stroke-color': ['get', 'color'],
          'circle-stroke-width': ['case', ['==', ['get', 'selected'], true], 3, 2],
          'circle-opacity': 0.98,
        },
      })
      map.addLayer({
        id: 'weather-point-inner',
        type: 'circle',
        source: 'ercot-weather-points',
        paint: {
          'circle-radius': ['case', ['==', ['get', 'selected'], true], 6, 4.8],
          'circle-color': ['get', 'color'],
          'circle-opacity': weatherVisible ? 0.96 : 0,
        },
      })
      map.addLayer({
        id: 'weather-point-label',
        type: 'symbol',
        source: 'ercot-weather-points',
        layout: {
          'text-field': ['concat', ['get', 'location'], '\n', ['to-string', ['round', ['get', 'temp']]], '\u00B0C'],
          'text-size': 11,
          'text-offset': [0, 1.55],
          'text-anchor': 'top',
          'text-allow-overlap': true,
        },
        paint: {
          // 🌟 文字本身改为深灰色
          'text-color': '#333333',
          // 🌟 文字描边改为白色，以便在复杂的地图背景上凸显
          'text-halo-color': '#ffffff',
          'text-halo-width': 1.5,
          'text-opacity': labelsVisible ? 1 : 0,
        },
      })
      map.addSource('settlement-hub', {
        type: 'geojson',
        data: {
          type: 'FeatureCollection',
          features: [{
            type: 'Feature',
            properties: { name: snapshot.metadata.hub ?? 'HB_NORTH' },
            geometry: { type: 'Point', coordinates: [-97.05, 32.9] },
          }],
        },
      })
      map.addLayer({
        id: 'settlement-hub-symbol',
        type: 'symbol',
        source: 'settlement-hub',
        layout: { 'text-field': '\u25C6', 'text-size': 22, 'text-allow-overlap': true },
        paint: {
          'text-color': '#2f80ed',
          // 🌟 描边改为白色
          'text-halo-color': '#ffffff',
          'text-halo-width': 2,
          'text-opacity': hubVisible ? 1 : 0,
        },
      })
      setReady(true)
    })

    map.on('mousemove', 'weather-point-inner', (event) => {
      map.getCanvas().style.cursor = 'pointer'
      const feature = event.features?.[0]
      if (!feature) return
      const coords = (feature.geometry as Point).coordinates.slice() as [number, number]
      popupRef.current
        ?.setLngLat(coords)
        .setHTML(`<div class="tip-title">${feature.properties?.location}</div><div>Temperature: ${feature.properties?.temp} \u00B0C</div><div>Risk: ${feature.properties?.risk}</div>`)
        .addTo(map)
    })
    map.on('mouseleave', 'weather-point-inner', () => {
      map.getCanvas().style.cursor = ''
      popupRef.current?.remove()
    })
    map.on('click', 'weather-point-inner', (event) => {
      const id = event.features?.[0]?.properties?.location
      const location = snapshot.locations.find((item) => item.location === id)
      if (location) onSelectedLocation(location)
    })
    mapRef.current = map
    return () => {
      popupRef.current?.remove()
      map.remove()
      mapRef.current = null
    }
  }, [])

  useEffect(() => {
    const source = mapRef.current?.getSource('ercot-weather-points') as GeoJSONSource | undefined
    if (source) source.setData(points)
  }, [points])

  useEffect(() => {
    const map = mapRef.current
    if (!ready || !map) return
    for (const id of ['weather-point-outer', 'weather-point-inner']) {
      if (map.getLayer(id)) map.setPaintProperty(id, 'circle-opacity', weatherVisible ? weatherOpacity : 0)
    }
    if (map.getLayer('weather-risk-halo')) map.setPaintProperty('weather-risk-halo', 'circle-opacity', riskVisible ? 0.24 * riskOpacity : 0)
    if (map.getLayer('settlement-hub-symbol')) map.setPaintProperty('settlement-hub-symbol', 'text-opacity', hubVisible ? hubOpacity : 0)
    if (map.getLayer('weather-point-label')) map.setPaintProperty('weather-point-label', 'text-opacity', labelsVisible ? labelsOpacity : 0)
  }, [ready, weatherVisible, riskVisible, hubVisible, labelsVisible, weatherOpacity, riskOpacity, hubOpacity, labelsOpacity])

  const reset = () => {
    mapRef.current?.fitBounds(TEXAS_BOUNDS, { padding: 32, duration: 650 })
    onReset()
  }

  const fullscreen = () => {
    containerRef.current?.requestFullscreen?.()
  }

  const screenshot = () => {
    const url = mapRef.current?.getCanvas().toDataURL('image/png')
    if (!url) {
      window.alert('Screenshot is unavailable because the map canvas is not ready.')
      return
    }
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = 'ercot-map-workbench.png'
    anchor.click()
  }

  return (
    <main className="map-stage">
      <div className="map-toolbar">
        <div className="toolbar-pill strong">Scenario: {snapshot.metadata.mode ?? 'Snapshot'}</div>
        <div className="toolbar-pill">Delivery Date <span>{deliveryDate}</span></div>
        <div className="toolbar-pill">Hour <span>{String(deliveryHour).padStart(2, '0')}:00</span></div>
        <div className="toolbar-pill">Weather <span>{weatherVariable}</span></div>
        <div className="toolbar-pill">Market <span>{marketVariable}</span></div>
        <button onClick={reset}><RotateCcw size={15} /> Reset</button>
        <button onClick={fullscreen}><Maximize2 size={15} /> Fullscreen</button>
        <button onClick={screenshot}><Camera size={15} /> Screenshot</button>
      </div>
      <div ref={containerRef} className="map-host" />
      <div className="map-status-label">Spatial status: Texas reference from basemap · ERCOT boundaries unavailable · Demo Coordinates</div>
      <div className="map-legend">
        <div className="legend-title">{weatherVariable === 'Extreme Risk' ? 'Extreme Risk' : 'Temperature'}</div>
        {weatherVariable === 'Extreme Risk' ? (
          <>
            <span><i style={{ background: '#20b486' }} />Low</span>
            <span><i style={{ background: '#f2a93b' }} />Medium</span>
            <span><i style={{ background: '#e55353' }} />High</span>
          </>
        ) : (
          <div className="gradient-legend"><b>27°C</b><em /><b>39°C</b></div>
        )}
      </div>
      <div className="quick-layer-strip">
        {layers.filter((layer) => layer.enabled).map((layer) => (
          <button key={layer.id} className={layer.visible ? 'active' : ''} onClick={() => onLayerToggle(layer.id, !layer.visible)}>
            {layer.name}
          </button>
        ))}
      </div>
    </main>
  )
}
