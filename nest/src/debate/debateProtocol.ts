import { KagAnalysis, RagAnalysis, DebateResult } from '../types';

/**
 * Protocole de débat entre KAG et RAG
 */
export const debateProtocol = {
  /**
   * Organise un débat entre les analyses KAG et RAG
   * @param kagAnalysis Analyse KAG
   * @param ragAnalysis Analyse RAG
   * @returns Résultat du débat
   */
  async debate(kagAnalysis: KagAnalysis, ragAnalysis: RagAnalysis): Promise<DebateResult> {
    // Implémentation minimale pour résoudre l'erreur d'import
    return {} as DebateResult;
  }
}; 