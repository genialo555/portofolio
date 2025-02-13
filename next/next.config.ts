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
  // Configuration du serveur
  server: {
    port: parseInt(process.env.PORT || '3000', 10),
    host: '0.0.0.0'
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
