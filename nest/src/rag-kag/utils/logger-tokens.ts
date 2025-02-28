import { Provider } from '@nestjs/common';

/**
 * Interface pour le service de logging
 */
export interface ILogger {
  /**
   * Log un message de niveau debug
   * @param message Message à logger
   * @param context Contexte optionnel (objet)
   */
  debug(message: string, context?: Record<string, any>): void;
  
  /**
   * Log un message de niveau info
   * @param message Message à logger
   * @param context Contexte optionnel (objet)
   */
  info(message: string, context?: Record<string, any>): void;
  
  /**
   * Log un message de niveau warning
   * @param message Message à logger
   * @param context Contexte optionnel (objet)
   */
  warn(message: string, context?: Record<string, any>): void;
  
  /**
   * Log un message de niveau error
   * @param message Message à logger
   * @param context Contexte optionnel (objet)
   */
  error(message: string, context?: Record<string, any>): void;
}

/**
 * Token pour l'injection du logger
 */
export const LOGGER_TOKEN = 'LOGGER_TOKEN';

/**
 * Provider pour le logger (à utiliser dans un module)
 */
export const LoggerProvider: Provider = {
  provide: LOGGER_TOKEN,
  useValue: {
    debug: (message: string, context?: Record<string, any>) => {
      console.debug(`[DEBUG] ${message}`, context ? context : '');
    },
    info: (message: string, context?: Record<string, any>) => {
      console.log(`[INFO] ${message}`, context ? context : '');
    },
    warn: (message: string, context?: Record<string, any>) => {
      console.warn(`[WARN] ${message}`, context ? context : '');
    },
    error: (message: string, context?: Record<string, any>) => {
      console.error(`[ERROR] ${message}`, context ? context : '');
    }
  }
}; 