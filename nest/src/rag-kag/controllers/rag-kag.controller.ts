import { Controller, Post, Body, Get } from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse } from '@nestjs/swagger';

/**
 * DTO pour la requête de l'utilisateur
 */
export class QueryRequestDto {
  query: string;
  contextInfo?: any;
  expertiseLevel?: 'BEGINNER' | 'INTERMEDIATE' | 'ADVANCED';
  useSimplifiedProcess?: boolean;
}

/**
 * DTO pour la réponse
 */
export interface FinalResponse {
  content: string;
  metaData: {
    sourceTypes: string[];
    confidenceLevel: string;
    processingTime: number;
    usedAgentCount: number;
    expertiseLevel: string;
    topicsIdentified: string[];
  };
  suggestedFollowUp?: string[];
  error?: string;
}

/**
 * Token pour le logger
 */
export const LOGGER_TOKEN = 'LOGGER_SERVICE';

/**
 * Contrôleur principal pour les requêtes RAG/KAG
 */
@Controller('api/rag-kag')
@ApiTags('rag-kag')
export class RagKagController {
  constructor() {}

  /**
   * Endpoint pour traiter une requête utilisateur
   * @param queryRequest Requête de l'utilisateur
   * @returns Résultat du traitement
   */
  @Post('query')
  @ApiOperation({ summary: 'Traiter une requête utilisateur' })
  @ApiResponse({ 
    status: 200, 
    description: 'La requête a été traitée avec succès',
    type: Object 
  })
  async processQuery(@Body() queryRequest: QueryRequestDto): Promise<FinalResponse> {
    // Réponse simulée
    console.log(`Requête reçue: ${queryRequest.query}`);
    return {
      content: `Réponse à la question: ${queryRequest.query}`,
      metaData: {
        sourceTypes: ['KAG', 'RAG'],
        confidenceLevel: 'HIGH',
        processingTime: 500,
        usedAgentCount: 3,
        expertiseLevel: queryRequest.expertiseLevel || 'INTERMEDIATE',
        topicsIdentified: ['IA', 'RAG', 'KAG'],
      },
      suggestedFollowUp: [
        'Comment fonctionne le débat entre KAG et RAG?',
        'Quels sont les avantages du système hybride?'
      ]
    };
  }

  /**
   * Endpoint de santé pour vérifier que le service est opérationnel
   * @returns Message de statut
   */
  @Get('health')
  @ApiOperation({ summary: 'Vérifier l\'état du service' })
  @ApiResponse({ 
    status: 200, 
    description: 'Le service est opérationnel',
    schema: {
      type: 'object',
      properties: {
        status: { type: 'string' },
        timestamp: { type: 'string', format: 'date-time' },
        apis: { 
          type: 'array',
          items: { type: 'string' }
        }
      }
    }
  })
  async checkHealth() {
    return {
      status: 'ok',
      timestamp: new Date(),
      apis: ['query', 'health']
    };
  }
} 