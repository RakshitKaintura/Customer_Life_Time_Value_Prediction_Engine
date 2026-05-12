import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  experimental: {
    // React 19 Server Components
  },
  // Allow images from Supabase storage
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "*.supabase.co",
      },
    ],
  },
  // Proxy API requests to FastAPI backend
  async rewrites() {
    return [
      {
        source: "/api/ltv/:path*",
        destination: `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/:path*`,
      },
    ];
  },
};

export default nextConfig; 