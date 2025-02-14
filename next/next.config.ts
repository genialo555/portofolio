/** @type {import('next').NextConfig} */
const config = {
  output: 'standalone',
  images: {
    domains: ['images.unsplash.com'],
  },
  eslint: {
    // Désactive la vérification ESLint pendant le build
    ignoreDuringBuilds: true,
  },
  typescript: {
    // Désactive la vérification TypeScript pendant le build
    ignoreBuildErrors: true,
  },
  env: {
    NEXT_PUBLIC_GEMINI_API_KEY_POUR_1: 'AIzaSyAfpC2_s7EBfc6nskbG5Spx5ID4ZO4Rd88',
    NEXT_PUBLIC_GEMINI_API_KEY_POUR_2: 'AIzaSyDvA5VuWhgiAf2JJsX13sspiJLPfeQf5QM',
    NEXT_PUBLIC_GEMINI_API_KEY_CONTRE_1: 'AIzaSyATCNdzvw2LMS3huBrSDnPgrlOC5ASsa5s',
    NEXT_PUBLIC_GEMINI_API_KEY_CONTRE_2: 'AIzaSyDKhGZqsCpxIrxNo2gE9T78pbOCdHjbUDw',
  },
  // Ajouter les domaines autorisés
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
