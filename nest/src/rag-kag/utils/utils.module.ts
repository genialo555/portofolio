import { Module } from '@nestjs/common';
import { LoggerModule } from './logger.module';

/**
 * Ce module est maintenant obsolète, utilisez plutôt CommonModule
 * Conservé uniquement pour compatibilité avec l'existant
 */
@Module({
  imports: [
    LoggerModule,
  ],
  exports: [
    LoggerModule,
  ]
})
export class UtilsModule {} 