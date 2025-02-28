import { randomBytes } from 'crypto';
import { envConfig } from '../../config/env.config';

interface APIKeys {
  key1: string;
  key2: string;
}

interface EncryptedKeys {
  [modelId: string]: {
    encryptedKey1: string;
    encryptedKey2: string;
    iv: string;
  };
}

class SecurityManager {
  private static instance: SecurityManager;
  private encryptionKey: Buffer;
  private encryptedKeys: EncryptedKeys = {};
  private readonly isDev: boolean;

  private constructor() {
    this.isDev = envConfig.isDev;
    // Génère une clé de chiffrement unique pour cette instance du serveur
    this.encryptionKey = randomBytes(32);
    // Initialise les clés en production
    if (envConfig.isProduction) {
      this.initProductionKeys();
    }
  }

  private initProductionKeys(): void {
    // Qwen
    this.setAPIKeys('qwen-max', {
      key1: envConfig.apiKeys.qwen.key1,
      key2: envConfig.apiKeys.qwen.key2
    });
    this.setAPIKeys('qwen-plus', {
      key1: envConfig.apiKeys.qwen.key1,
      key2: envConfig.apiKeys.qwen.key2
    });
    this.setAPIKeys('qwen-turbo', {
      key1: envConfig.apiKeys.qwen.key1,
      key2: envConfig.apiKeys.qwen.key2
    });

    // DeepSeek
    this.setAPIKeys('deepseek', {
      key1: envConfig.apiKeys.deepseek.key1,
      key2: envConfig.apiKeys.deepseek.key2
    });
    this.setAPIKeys('deepseek-reasoner', {
      key1: envConfig.apiKeys.deepseek.key1,
      key2: envConfig.apiKeys.deepseek.key2
    });
  }

  public static getInstance(): SecurityManager {
    if (!SecurityManager.instance) {
      SecurityManager.instance = new SecurityManager();
    }
    return SecurityManager.instance;
  }

  private encrypt(text: string): { encrypted: string; iv: string } {
    const iv = randomBytes(16);
    const cipher = require('crypto').createCipheriv('aes-256-gcm', this.encryptionKey, iv);
    let encrypted = cipher.update(text, 'utf8', 'hex');
    encrypted += cipher.final('hex');
    return {
      encrypted: encrypted + cipher.getAuthTag().toString('hex'),
      iv: iv.toString('hex')
    };
  }

  private decrypt(encrypted: string, iv: string): string {
    const authTag = Buffer.from(encrypted.slice(-32), 'hex');
    const encryptedText = encrypted.slice(0, -32);
    const decipher = require('crypto').createDecipheriv(
      'aes-256-gcm',
      this.encryptionKey,
      Buffer.from(iv, 'hex')
    );
    decipher.setAuthTag(authTag);
    let decrypted = decipher.update(encryptedText, 'hex', 'utf8');
    decrypted += decipher.final('utf8');
    return decrypted;
  }

  public setAPIKeys(modelId: string, keys: APIKeys): void {
    if (!keys.key1 || !keys.key2) {
      if (envConfig.isProduction) {
        throw new Error(`Clés API manquantes pour le modèle ${modelId} en production`);
      }
      if (this.isDev) {
        console.warn(`Attention : Clés API manquantes pour le modèle ${modelId} en développement`);
      }
    }

    const key1Encrypted = this.encrypt(keys.key1);
    const key2Encrypted = this.encrypt(keys.key2);

    this.encryptedKeys[modelId] = {
      encryptedKey1: key1Encrypted.encrypted,
      encryptedKey2: key2Encrypted.encrypted,
      iv: key1Encrypted.iv
    };
  }

  public getAPIKeys(modelId: string): APIKeys {
    const encryptedData = this.encryptedKeys[modelId];
    if (!encryptedData) {
      throw new Error(`Clés API non trouvées pour le modèle ${modelId}`);
    }

    return {
      key1: this.decrypt(encryptedData.encryptedKey1, encryptedData.iv),
      key2: this.decrypt(encryptedData.encryptedKey2, encryptedData.iv)
    };
  }

  public validateAPIKey(apiKey: string): boolean {
    if (!apiKey) return false;
    // Vérifie le format UUID attendu
    return /^517-[a-f0-9]{16}-[1-8]$/.test(apiKey);
  }

  public rotateKeys(): void {
    // Régénère la clé de chiffrement
    this.encryptionKey = randomBytes(32);
    // Réinitialise les clés chiffrées
    this.encryptedKeys = {};
    // Réinitialise les clés en production
    if (envConfig.isProduction) {
      this.initProductionKeys();
    }
  }
}

export const securityManager = SecurityManager.getInstance(); 