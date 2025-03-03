import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { ScheduleModule } from '@nestjs/schedule';
import { RagKagModule } from './rag-kag/rag-kag.module';

@Module({
  imports: [
    // Chargement des variables d'environnement
    ConfigModule.forRoot({
      isGlobal: true,
      envFilePath: ['.env', '.env.local']
    }),
    
    // Module de planification pour les tâches périodiques
    ScheduleModule.forRoot(),
    
    // Module principal RAG/KAG
    RagKagModule
  ],
  controllers: [],
  providers: [],
})
export class AppModule {} 