interface EnvConfig {
  isDev: boolean;
  isProduction: boolean;
  apiKeys: {
    qwen: {
      key1: string;
      key2: string;
    };
    deepseek: {
      key1: string;
      key2: string;
    };
  };
}

const getEnvVar = (key: string, fallbackKey?: string): string => {
  // Essayer d'abord la clé principale
  let value = process.env[key];
  
  // Si la clé principale n'existe pas et qu'une clé de secours est fournie, essayer celle-ci
  if (!value && fallbackKey) {
    value = process.env[fallbackKey];
  }
  
  // Avertissement en production si aucune valeur n'est trouvée
  if (!value && process.env.NODE_ENV === 'production') {
    console.warn(`La variable d'environnement ${key} n'est pas définie en production, utilisation d'une valeur par défaut`);
    return 'not-configured';
  }
  
  return value || '';
};

export const envConfig: EnvConfig = {
  isDev: process.env.NODE_ENV === 'development',
  isProduction: process.env.NODE_ENV === 'production',
  apiKeys: {
    qwen: {
      // Utiliser les variables côté serveur avec fallback sur les variables publiques
      key1: getEnvVar('QWEN_API_KEY_1', 'NEXT_PUBLIC_QWEN_API_KEY_1'),
      key2: getEnvVar('QWEN_API_KEY_2', 'NEXT_PUBLIC_QWEN_API_KEY_2')
    },
    deepseek: {
      // Utiliser les variables côté serveur avec fallback sur les variables publiques
      key1: getEnvVar('DEEPSEEK_API_KEY_1', 'NEXT_PUBLIC_DEEPSEEK_API_KEY_1'),
      key2: getEnvVar('DEEPSEEK_API_KEY_2', 'NEXT_PUBLIC_DEEPSEEK_API_KEY_2')
    }
  }
}; 