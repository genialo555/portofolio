import { Module, Global } from '@nestjs/common';
import { Logger, LogLevel } from '@nestjs/common';
import { LOGGER_TOKEN } from './logger-tokens';

@Global()
@Module({
  providers: [
    {
      provide: LOGGER_TOKEN,
      useFactory: () => {
        // Récupérer le niveau de log depuis les variables d'environnement
        // ou utiliser INFO par défaut
        const logLevelString = process.env.LOG_LEVEL || 'info';
        
        // Convertir la chaîne de niveau de log en LogLevel NestJS
        let logLevel: LogLevel;
        
        switch (logLevelString.toLowerCase()) {
          case 'debug':
            logLevel = 'debug';
            break;
          case 'verbose':
            logLevel = 'verbose';
            break;
          case 'warn':
            logLevel = 'warn';
            break;
          case 'error':
            logLevel = 'error';
            break;
          case 'info':
          default:
            logLevel = 'log';
            break;
        }
        
        // Créer et configurer le logger
        const logger = new Logger('RAG-KAG');
        
        // Dans un contexte réel, on pourrait configurer ici d'autres aspects du logger,
        // comme la sortie vers un fichier, un service de monitoring, etc.
        
        // Retourner notre logger
        return {
          debug: (message: string, context?: any) => {
            if (logLevel === 'debug') {
              logger.debug(`${message}${context ? ' ' + JSON.stringify(context) : ''}`);
            }
          },
          info: (message: string, context?: any) => {
            if (['debug', 'verbose', 'log'].includes(logLevel)) {
              logger.log(`${message}${context ? ' ' + JSON.stringify(context) : ''}`);
            }
          },
          warn: (message: string, context?: any) => {
            if (['debug', 'verbose', 'log', 'warn'].includes(logLevel)) {
              logger.warn(`${message}${context ? ' ' + JSON.stringify(context) : ''}`);
            }
          },
          error: (message: string, context?: any) => {
            logger.error(`${message}${context ? ' ' + JSON.stringify(context) : ''}`);
          },
        };
      }
    }
  ],
  exports: [LOGGER_TOKEN]
})
export class LoggerModule {} 