'use client';

import { useEffect } from 'react';
import { usePathname, useRouter } from 'next/navigation';
import { useAuth } from '@/lib/auth-context';
import { Sidebar } from '@/components/nav/sidebar';
import { MobileNav } from '@/components/nav/mobile-nav';

export function LayoutWrapper({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const { isAuthenticated, isLoading } = useAuth();

  // Pages that don't require authentication
  const publicPages = ['/welcome', '/auth'];
  const isPublicPage = publicPages.some((page) => pathname.startsWith(page));

  useEffect(() => {
    // Wait for auth check to complete
    if (isLoading) return;

    // If user is not authenticated and trying to access a protected page
    if (!isAuthenticated && !isPublicPage && pathname !== '/') {
      router.push('/welcome');
    }

    // If user is not authenticated and on root, redirect to welcome
    if (!isAuthenticated && pathname === '/') {
      router.push('/welcome');
    }

    // If user is authenticated and on welcome/auth, redirect to dashboard
    if (isAuthenticated && isPublicPage) {
      router.push('/');
    }
  }, [isAuthenticated, isLoading, pathname, isPublicPage, router]);

  // Show loading state
  if (isLoading) {
    return (
      <div className="flex h-screen items-center justify-center bg-background">
        <div className="text-center">
          <div className="mb-4 flex justify-center">
            <div className="h-12 w-12 animate-spin rounded-full border-4 border-muted border-t-foreground"></div>
          </div>
          <p className="text-muted-foreground">Loading...</p>
        </div>
      </div>
    );
  }

  // For public pages (welcome, auth), show full width
  if (isPublicPage) {
    return <>{children}</>;
  }

  // For authenticated users, show sidebar + dashboard
  if (isAuthenticated) {
    return (
      <div className="flex h-screen overflow-hidden">
        <div className="hidden lg:block">
          <Sidebar />
        </div>
        <main className="app-shell flex flex-1 flex-col overflow-hidden pb-16 lg:pb-0">
          {children}
        </main>
        <MobileNav />
      </div>
    );
  }

  // Fallback (shouldn't reach here due to redirects above)
  return <>{children}</>;
}
