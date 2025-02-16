import type { Config } from "next";

const config: Config = {
  output: 'standalone',
  images: {
    domains: ['images.unsplash.com'],
  },
  eslint: {
    ignoreDuringBuilds: true,
  },
};

export default config; 