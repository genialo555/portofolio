import { Module } from '@nestjs/common';
import { DebateService } from './debate.service';
import { KagEngineService } from './kag-engine.service';
import { RagEngineService } from './rag-engine.service';
import { LoggerModule } from '../utils/logger.module';
import { PromptsModule } from '../prompts/prompts.module';
import { ApisModule } from '../apis/apis.module';

@Module({
  imports: [
    LoggerModule,
    PromptsModule,
    ApisModule,
  ],
  providers: [
    DebateService,
    KagEngineService,
    RagEngineService,
  ],
  exports: [
    DebateService,
  ],
})
export class DebateModule {} 