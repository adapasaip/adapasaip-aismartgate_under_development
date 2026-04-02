interface StatsCardProps {
  title: string;
  value: string | number;
  icon: string;
  iconColor: string;
  subtitle?: string;
  trend?: string;
  trendColor?: string;
}

export default function StatsCard({
  title,
  value,
  icon,
  iconColor,
  subtitle,
  trend,
  trendColor = "text-green-600"
}: StatsCardProps) {
  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-slate-600">{title}</p>
          <p className="text-3xl font-bold text-slate-800" data-testid={`stat-${title.toLowerCase().replace(/\s+/g, '-')}`}>
            {value}
          </p>
        </div>
        <div className={`w-12 h-12 ${iconColor} rounded-lg flex items-center justify-center`}>
          <i className={`${icon} text-2xl`}></i>
        </div>
      </div>
      {(trend || subtitle) && (
        <div className="mt-4 flex items-center text-sm">
          {trend && <span className={trendColor}>{trend}</span>}
          {subtitle && <span className="text-slate-600 ml-2">{subtitle}</span>}
        </div>
      )}
    </div>
  );
}
