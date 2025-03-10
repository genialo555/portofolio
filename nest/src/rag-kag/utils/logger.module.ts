import { Module, Provider, Logger } from '@nestjs/common';
import { LOGGER_TOKEN, ILogger } from './logger-tokens';

/**
 * Implémentation de l'interface ILogger avec la classe Logger de NestJS
 */
export class LoggerProvider implements ILogger {
  private readonly logger = new Logger();

  debug(message: string, context?: Record<string, any>): void {
    this.logger.debug(message, context ? JSON.stringify(context) : undefined);
  }

  info(message: string, context?: Record<string, any>): void {
    this.logger.log(message, context ? JSON.stringify(context) : undefined);
  }

  log(message: string, context?: Record<string, any>): void {
    // Assurer que la méthode log existe et est un alias pour info/log
    this.info(message, context);
  }

  warn(message: string, context?: Record<string, any>): void {
    this.logger.warn(message, context ? JSON.stringify(context) : undefined);
  }

  error(message: string, context?: Record<string, any>): void {
    this.logger.error(message, context ? JSON.stringify(context) : undefined);
  }
}

/**
 * Provider pour le service de logging
 */
const loggerProvider: Provider = {
  provide: LOGGER_TOKEN,
  useClass: LoggerProvider,
};

/**
 * Module de gestion des logs
 */
@Module({
  providers: [loggerProvider],
  exports: [loggerProvider],
})
export class LoggerModule {} 