import { Module } from '@nestjs/common';
import { SynthesisService } from './synthesis.service';
import { LoggerModule } from '../utils/logger.module';
import { PromptsModule } from '../prompts/prompts.module';

@Module({
  imports: [
    LoggerModule,
    PromptsModule,
  ],
  providers: [
    SynthesisService,
  ],
  exports: [
    SynthesisService,
  ],
})
export class SynthesisModule {} 