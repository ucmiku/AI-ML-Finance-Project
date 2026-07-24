import { CloudSun, Gauge, Info, LineChart, Radiation, Wind } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'

// 1. 从 RailTab 类型中移除 'layers'
export type RailTab = 'weather' | 'load' | 'renewables' | 'risk' | 'model' | 'info'

// 2. 从图标数组中直接删掉第一项（id: 'layers'）
const items: Array<{ id: RailTab; label: string; icon: LucideIcon }> = [
  { id: 'weather', label: 'Weather', icon: CloudSun },
  { id: 'load', label: 'Load', icon: Gauge },
  { id: 'renewables', label: 'Renewables', icon: Wind },
  { id: 'risk', label: 'Extreme Risk', icon: Radiation },
  { id: 'model', label: 'Model', icon: LineChart },
  { id: 'info', label: 'Information', icon: Info },
]

export function IconRail({ active, onChange }: { active: RailTab; onChange: (tab: RailTab) => void }) {
  return (
    <nav className="icon-rail" aria-label="Workbench sections">
      <div className="rail-mark">E</div>
      {items.map((item) => {
        const Icon = item.icon
        return (
          <button
            key={item.id}
            className={active === item.id ? 'active' : ''}
            title={item.label}
            aria-label={item.label}
            onClick={() => onChange(item.id)}
          >
            <Icon size={18} />
          </button>
        )
      })}
    </nav>
  )
}