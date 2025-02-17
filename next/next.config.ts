/** @type {import('next').NextConfig} */
const config = {
  output: 'standalone',
  images: {
    domains: ['images.unsplash.com'],
    unoptimized: true
  },
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  experimental: {
    serverActions: {
      allowedOrigins: ["*"]
    }
  },
  env: {
    NEXT_PUBLIC_GEMINI_API_KEY_POUR_1: process.env.NEXT_PUBLIC_GEMINI_API_KEY_POUR_1,
    NEXT_PUBLIC_GEMINI_API_KEY_POUR_2: process.env.NEXT_PUBLIC_GEMINI_API_KEY_POUR_2,
    NEXT_PUBLIC_GEMINI_API_KEY_CONTRE_1: process.env.NEXT_PUBLIC_GEMINI_API_KEY_CONTRE_1,
    NEXT_PUBLIC_GEMINI_API_KEY_CONTRE_2: process.env.NEXT_PUBLIC_GEMINI_API_KEY_CONTRE_2,
    NEXT_PUBLIC_QWEN_API_KEY_1: process.env.NEXT_PUBLIC_QWEN_API_KEY_1,
    NEXT_PUBLIC_QWEN_API_KEY_2: process.env.NEXT_PUBLIC_QWEN_API_KEY_2,
    NEXT_PUBLIC_DEEPSEEK_API_KEY_1: process.env.NEXT_PUBLIC_DEEPSEEK_API_KEY_1,
    NEXT_PUBLIC_DEEPSEEK_API_KEY_2: process.env.NEXT_PUBLIC_DEEPSEEK_API_KEY_2
  },
  async rewrites() {
    return {
      beforeFiles: [
        {
          source: '/:path*',
          has: [
            {
              type: 'host',
              value: 'jeremie-nunez.com',
            },
          ],
          destination: 'https://www.jeremie-nunez.com/:path*',
        },
      ],
    };
  },
};

export default config; 