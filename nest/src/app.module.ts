import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { RagKagModule } from './rag-kag/rag-kag.module';

@Module({
  imports: [
    // Chargement des variables d'environnement
    ConfigModule.forRoot({
      isGlobal: true,
      envFilePath: ['.env', '.env.local']
    }),
    
    // Module principal RAG/KAG
    RagKagModule
  ],
  controllers: [],
  providers: [],
})
export class AppModule {} 