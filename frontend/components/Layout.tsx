import React, { ReactNode } from 'react';
import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { Header } from './Header';
import { useAuthStore } from '@/stores/authStore';

interface LayoutProps {
  children?: ReactNode;
}

export function Layout({ children }: LayoutProps) {
  const { user } = useAuthStore();

  if (!user) {
    return <Outlet />;
  }

  return (
    <div className="relative flex h-screen w-full overflow-hidden bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_20%_20%,rgba(56,189,248,0.08),transparent_35%),radial-gradient(circle_at_80%_0%,rgba(14,165,233,0.12),transparent_30%)]" />
      <Sidebar />
      <div className="relative z-10 flex-1 flex flex-col overflow-hidden">
        <Header />
        <main className="flex-1 overflow-y-auto px-4 py-4 sm:px-8 lg:px-10">
          <div className="max-w-7xl mx-auto space-y-6">
            {children}
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
}