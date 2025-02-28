import { ResolvedInsights, FinalResponse, ExpertiseLevel } from '../types';

/**
 * Formateur de réponse finale
 */
export const responseFormatter = {
  /**
   * Formate la réponse finale selon le niveau d'expertise
   * @param resolvedInsights Insights résolus
   * @param expertiseLevel Niveau d'expertise du destinataire
   * @returns Réponse finale
   */
  format(resolvedInsights: ResolvedInsights, expertiseLevel: ExpertiseLevel): FinalResponse {
    // Implémentation minimale pour résoudre l'erreur d'import
    return {} as FinalResponse;
  },

  /**
   * Formate directement une réponse simplifiée
   * @param simplifiedInsights Insights simplifiés
   * @param expertiseLevel Niveau d'expertise du destinataire
   * @returns Réponse finale
   */
  formatSimplified(simplifiedInsights: any, expertiseLevel: ExpertiseLevel): FinalResponse {
    // Implémentation minimale pour résoudre l'erreur d'import
    return {} as FinalResponse;
  }
}; 