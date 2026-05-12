import type { Metadata } from "next";
import { Fraunces, Space_Grotesk } from "next/font/google";
import "./globals.css";
import { Sidebar } from "@/components/nav/sidebar";

const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-sans",
});
const fraunces = Fraunces({
  subsets: ["latin"],
  variable: "--font-display",
});

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
    <html lang="en">
      <body className={`${spaceGrotesk.variable} ${fraunces.variable}`}>
        <div className="flex h-screen overflow-hidden">
          <Sidebar />
          <main className="app-shell flex flex-1 flex-col overflow-hidden">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}