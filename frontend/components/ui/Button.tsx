import React from 'react';
import { Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger' | 'outline' | 'success' | 'warning';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  icon?: React.ReactNode;
  fullWidth?: boolean;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'primary', size = 'md', loading = false, icon, fullWidth = false, children, disabled, ...props }, ref) => {
    return (
      <button
        className={cn(
          'inline-flex items-center justify-center gap-2 font-medium rounded-lg transition-all focus:outline-none focus:ring-2 focus:ring-offset-2',
          {
            'bg-blue-600 text-white hover:bg-blue-700 focus:ring-blue-500 disabled:bg-blue-400': variant === 'primary',
            'bg-gray-200 text-gray-900 hover:bg-gray-300 focus:ring-gray-500 disabled:bg-gray-100': variant === 'secondary',
            'text-gray-700 hover:bg-gray-100 focus:ring-gray-500': variant === 'ghost',
            'bg-red-600 text-white hover:bg-red-700 focus:ring-red-500 disabled:bg-red-400': variant === 'danger',
            'border border-gray-300 text-gray-700 hover:bg-gray-50 focus:ring-gray-500': variant === 'outline',
            'bg-green-600 text-white hover:bg-green-700 focus:ring-green-500 disabled:bg-green-400': variant === 'success',
            'bg-yellow-600 text-white hover:bg-yellow-700 focus:ring-yellow-500 disabled:bg-yellow-400': variant === 'warning',
            'px-3 py-1.5 text-sm': size === 'sm',
            'px-4 py-2 text-base': size === 'md',
            'px-6 py-3 text-lg': size === 'lg',
            'opacity-50 cursor-not-allowed': disabled || loading,
            'w-full': fullWidth,
          },
          className
        )}
        ref={ref}
        disabled={disabled || loading}
        {...props}
      >
        {loading && <Loader2 className="w-4 h-4 animate-spin" />}
        {icon && !loading && icon}
        {children}
      </button>
    );
  },
);

Button.displayName = 'Button';