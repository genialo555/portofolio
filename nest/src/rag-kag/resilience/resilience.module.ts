import { Module } from '@nestjs/common';
import { LoggerModule } from '../utils/logger.module';

/**
 * Ce module est maintenant obsolète, utilisez plutôt CommonModule
 * Conservé uniquement pour compatibilité avec l'existant
 */
@Module({
  imports: [LoggerModule],
  exports: []
})
export class ResilienceModule {} 