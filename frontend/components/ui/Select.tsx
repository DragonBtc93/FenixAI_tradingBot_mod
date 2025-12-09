import React from 'react';

interface SelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
  className?: string;
  error?: string;
  label?: string;
  children: React.ReactNode;
}

export const Select = React.forwardRef<HTMLSelectElement, SelectProps>(
  ({ className = '', error, label, children, ...props }, ref) => {
    return (
      <div className="w-full">
        {label && (
          <label className="block text-sm font-medium text-gray-700 mb-1">
            {label}
          </label>
        )}
        <select
          ref={ref}
          className={`
            w-full px-3 py-2 border rounded-md shadow-sm text-sm bg-white
            focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500
            ${error ? 'border-red-300 text-red-900' : 'border-gray-300'}
            ${className}
          `}
          {...props}
        >
          {children}
        </select>
        {error && (
          <p className="mt-1 text-sm text-red-600">{error}</p>
        )}
      </div>
    );
  }
);

Select.displayName = 'Select';