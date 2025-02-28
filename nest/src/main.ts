import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import { SwaggerModule, DocumentBuilder } from '@nestjs/swagger';
import { Logger } from './utils/logger';

async function bootstrap() {
  // Création d'une instance de logger
  const logger = new Logger({ 
    level: 2, // INFO
    colorize: true,
    timestamp: true,
    outputToConsole: true
  });

  try {
    // Création de l'application NestJS
    const app = await NestFactory.create(AppModule, {
      logger: ['error', 'warn', 'log']
    });

    // Configuration CORS pour permettre les appels depuis le frontend
    app.enableCors({
      origin: ['http://localhost:3000', 'http://localhost:4200'], 
      methods: 'GET,HEAD,PUT,PATCH,POST,DELETE,OPTIONS',
      credentials: true,
    });

    // Configuration Swagger pour la documentation API
    const config = new DocumentBuilder()
      .setTitle('RAG/KAG API')
      .setDescription('API pour le système mixte RAG/KAG')
      .setVersion('1.0')
      .addTag('rag-kag')
      .build();
    
    const document = SwaggerModule.createDocument(app, config);
    SwaggerModule.setup('api/docs', app, document);

    // Démarrage du serveur
    const port = process.env.PORT || 3001;
    await app.listen(port);
    
    logger.info(`Application démarrée sur le port ${port}`);
    logger.info(`Documentation API disponible sur: http://localhost:${port}/api/docs`);
  } catch (error) {
    logger.error(`Erreur lors du démarrage de l'application: ${error.message}`);
    process.exit(1);
  }
}

bootstrap(); 