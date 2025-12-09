import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { 
  LayoutDashboard, 
  TrendingUp, 
  Brain, 
  BarChart3, 
  Settings, 
  Users, 
  Activity,
  Database,
  Shield,
  Sparkles,
  Wifi
} from 'lucide-react';
import { cn } from '@/lib/utils';

const sidebarItems = [
  {
    name: 'Dashboard',
    href: '/dashboard',
    icon: LayoutDashboard,
    roles: ['admin', 'trader', 'analyst', 'ai_agent']
  },
  {
    name: 'Market Data',
    href: '/market',
    icon: TrendingUp,
    roles: ['admin', 'trader', 'analyst', 'ai_agent']
  },
  {
    name: 'Trading',
    href: '/trading',
    icon: BarChart3,
    roles: ['admin', 'trader']
  },
  {
    name: 'AI Agents',
    href: '/agents',
    icon: Brain,
    roles: ['admin', 'analyst', 'ai_agent']
  },
  {
    name: 'Reasoning Bank',
    href: '/reasoning',
    icon: Database,
    roles: ['admin', 'analyst', 'ai_agent']
  },
  {
    name: 'System Monitor',
    href: '/system',
    icon: Activity,
    roles: ['admin', 'analyst']
  },
  {
    name: 'Users',
    href: '/users',
    icon: Users,
    roles: ['admin']
  },
  {
    name: 'Settings',
    href: '/settings',
    icon: Settings,
    roles: ['admin', 'trader', 'analyst']
  }
];

export function Sidebar() {
  const location = useLocation();

  return (
    <aside className="relative hidden lg:flex w-72 flex-col border-r border-white/5 bg-slate-900/70 backdrop-blur-xl shadow-xl shadow-cyan-500/5">
      <div className="absolute inset-0 bg-gradient-to-b from-cyan-500/5 via-slate-900/40 to-slate-950" />
      <div className="relative p-6 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 bg-gradient-to-br from-cyan-400 to-blue-600 rounded-2xl flex items-center justify-center shadow-lg shadow-cyan-500/30">
            <Shield className="w-5 h-5 text-white" />
          </div>
          <div>
            <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Fenix AI</p>
            <span className="text-lg font-semibold text-white">Trading Control</span>
          </div>
        </div>
        <div className="flex items-center text-xs text-cyan-200 bg-cyan-500/10 px-3 py-1 rounded-full border border-cyan-400/20">
          <Wifi className="w-3 h-3 mr-1" /> Live
        </div>
      </div>

      <nav className="relative px-4 pb-8 space-y-3 overflow-y-auto">
        <div className="px-3 text-xs uppercase tracking-[0.2em] text-slate-400">Navigation</div>
        <ul className="space-y-1">
          {sidebarItems.map((item) => {
            const active = location.pathname === item.href;
            return (
              <li key={item.href}>
                <Link
                  to={item.href}
                  className={cn(
                    "group flex items-center justify-between px-4 py-3 rounded-xl border transition-all duration-200",
                    active
                      ? "border-cyan-400/40 bg-gradient-to-r from-cyan-500/20 to-blue-500/10 text-white shadow-lg shadow-cyan-500/10"
                      : "border-white/5 text-slate-200 hover:border-cyan-400/30 hover:bg-slate-800/60"
                  )}
                >
                  <div className="flex items-center space-x-3">
                    <div className={cn(
                      "p-2 rounded-lg border",
                      active ? "border-cyan-400/50 bg-cyan-500/10" : "border-white/5 bg-slate-800/60 group-hover:border-cyan-400/30"
                    )}>
                      <item.icon className={cn("w-4 h-4", active ? "text-cyan-200" : "text-slate-300")}/>
                    </div>
                    <span className="font-medium tracking-tight">{item.name}</span>
                  </div>
                  {active && (
                    <Sparkles className="w-4 h-4 text-cyan-200 drop-shadow" />
                  )}
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>
    </aside>
  );
}