import { Module } from '@nestjs/common';
import { AgentFactoryService } from './agent-factory.service';
import { ApisModule } from '../apis/apis.module';
import { LoggerModule } from '../utils/logger.module';
import { PromptsModule } from '../prompts/prompts.module';

@Module({
  imports: [
    LoggerModule,
    PromptsModule,
    ApisModule,
  ],
  providers: [
    AgentFactoryService,
  ],
  exports: [
    AgentFactoryService,
  ],
})
export class AgentsModule {} 