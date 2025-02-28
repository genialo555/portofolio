import { Module } from '@nestjs/common';
import { GoogleAiService } from './google-ai.service';
import { QwenAiService } from './qwen-ai.service';
import { DeepseekAiService } from './deepseek-ai.service';
import { ApiProviderFactory } from './api-provider-factory.service';
import { LoggerModule } from '../utils/logger.module';

@Module({
  imports: [
    LoggerModule,
  ],
  providers: [
    GoogleAiService,
    QwenAiService,
    DeepseekAiService,
    ApiProviderFactory,
  ],
  exports: [
    ApiProviderFactory,
  ],
})
export class ApisModule {} 