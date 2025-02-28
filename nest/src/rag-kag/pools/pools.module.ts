import { Module } from '@nestjs/common';
import { CommercialPoolService } from './commercial-pool.service';
import { MarketingPoolService } from './marketing-pool.service';
import { SectorialPoolService } from './sectorial-pool.service';
import { PoolManagerService } from './pool-manager.service';
import { AgentsModule } from '../agents/agents.module';
import { PromptsModule } from '../prompts/prompts.module';
import { LoggerModule } from '../utils/logger.module';

@Module({
  imports: [
    LoggerModule,
    AgentsModule,
    PromptsModule
  ],
  providers: [
    CommercialPoolService,
    MarketingPoolService,
    SectorialPoolService,
    PoolManagerService
  ],
  exports: [
    PoolManagerService
  ],
})
export class PoolsModule {} 