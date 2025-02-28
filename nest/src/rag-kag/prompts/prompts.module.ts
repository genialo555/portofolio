import { Module } from '@nestjs/common';
import { PromptsService } from './prompts.service';
import { LoggerModule } from '../utils/logger.module';

@Module({
  imports: [LoggerModule],
  providers: [PromptsService],
  exports: [PromptsService],
})
export class PromptsModule {} 