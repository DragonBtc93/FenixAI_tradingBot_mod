import React from 'react';
import { LucideIcon, TrendingUp, TrendingDown } from 'lucide-react';
import { cn } from '@/lib/utils';

export interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  change?: number;
  changeType?: 'positive' | 'negative' | 'neutral';
  icon: LucideIcon;
  className?: string;
}

export function MetricCard({ 
  title, 
  value, 
  subtitle, 
  change, 
  changeType = 'neutral', 
  icon: Icon,
  className 
}: MetricCardProps) {
  const changeColor = {
    positive: 'text-green-600',
    negative: 'text-red-600',
    neutral: 'text-gray-600'
  }[changeType];

  const bgColor = {
    positive: 'bg-green-50',
    negative: 'bg-red-50',
    neutral: 'bg-gray-50'
  }[changeType];

  return (
    <div className={cn("bg-white rounded-lg shadow-sm border border-gray-200 p-6", className)}>
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className="text-2xl font-bold text-gray-900 mt-1">{value}</p>
          {subtitle && (
            <p className="text-sm text-gray-500 mt-1">{subtitle}</p>
          )}
          {change !== undefined && (
            <div className={cn("flex items-center mt-2 text-sm", changeColor)}>
              {changeType === 'positive' ? (
                <TrendingUp className="w-4 h-4 mr-1" />
              ) : changeType === 'negative' ? (
                <TrendingDown className="w-4 h-4 mr-1" />
              ) : null}
              <span>{Math.abs(change)}%</span>
              <span className="ml-1 text-gray-500">vs last period</span>
            </div>
          )}
        </div>
        <div className={cn("p-3 rounded-lg", bgColor)}>
          <Icon className={cn("w-6 h-6", changeColor)} />
        </div>
      </div>
    </div>
  );
}