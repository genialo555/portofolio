import { UserQuery, TargetPools } from '../types';

/**
 * Routeur qui détermine quels pools d'agents utiliser en fonction de la requête
 */
export const router = {
  /**
   * Détermine les pools cibles basés sur la requête
   * @param query Requête utilisateur
   * @returns Pools cibles à utiliser
   */
  determineTargetPools(query: UserQuery): TargetPools {
    // Implémentation minimale pour résoudre l'erreur d'import
    return {} as TargetPools;
  }
}; 