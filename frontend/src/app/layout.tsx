import type { Metadata } from "next";
import "./globals.css";
import { ClerkProvider } from "@clerk/nextjs";
import { LayoutWrapper } from "@/components/layout-wrapper";
import { ThemeProvider } from "@/components/theme-provider";

export const metadata: Metadata = {
  title: "LTV Prediction Dashboard",
  description: "Customer Lifetime Value Intelligence Platform",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <ClerkProvider>
          <ThemeProvider attribute="class" defaultTheme="dark" enableSystem disableTransitionOnChange>
            <LayoutWrapper>
              {children}
            </LayoutWrapper>
          </ThemeProvider>
        </ClerkProvider>
      </body>
    </html>
  );
}
