import { Module } from '@nestjs/common';
import { OrchestratorService } from './orchestrator.service';
import { RouterService } from './router.service';
import { OutputCollectorService } from './output-collector.service';
import { LoggerModule } from '../utils/logger.module';
import { PoolsModule } from '../pools/pools.module';
import { DebateModule } from '../debate/debate.module';
import { SynthesisModule } from '../synthesis/synthesis.module';

@Module({
  imports: [
    LoggerModule,
    PoolsModule,
    DebateModule,
    SynthesisModule,
  ],
  providers: [
    OrchestratorService,
    RouterService,
    OutputCollectorService,
  ],
  exports: [
    OrchestratorService,
  ],
})
export class OrchestratorModule {} 