import { v4 as uuidv4 } from 'uuid';
import { CoordinationComponent, ComponentPriority, ComponentStatus } from '../handlers/coordination-handler';
import { Logger } from '../utils/logger';

/**
 * Types de composants disponibles dans le système
 */
export enum ComponentType {
  ORCHESTRATOR = 'ORCHESTRATOR',          // Composant d'orchestration
  POOL_SELECTOR = 'POOL_SELECTOR',        // Sélection des pools d'agents
  QUERY_ANALYZER = 'QUERY_ANALYZER',      // Analyse des requêtes
  COMMERCIAL_AGENT = 'COMMERCIAL_AGENT',  // Agent commercial
  MARKETING_AGENT = 'MARKETING_AGENT',    // Agent marketing
  SECTORAL_AGENT = 'SECTORAL_AGENT',      // Agent sectoriel
  DEBATE_ENGINE = 'DEBATE_ENGINE',        // Moteur de débat
  SYNTHESIS_ENGINE = 'SYNTHESIS_ENGINE',  // Moteur de synthèse
  ANOMALY_DETECTOR = 'ANOMALY_DETECTOR',  // Détecteur d'anomalies
  RAG_ENGINE = 'RAG_ENGINE',              // Moteur RAG
  KAG_ENGINE = 'KAG_ENGINE',              // Moteur KAG
  OUTPUT_FORMATTER = 'OUTPUT_FORMATTER',  // Formatage des sorties
  FEEDBACK_COLLECTOR = 'FEEDBACK_COLLECTOR' // Collecte de feedback
}

/**
 * Interface pour l'enregistrement des composants
 */
export interface ComponentRegistration {
  id: string;                 // ID unique du composant
  type: ComponentType;        // Type de composant
  name: string;               // Nom lisible
  description?: string;       // Description détaillée
  version: string;            // Version du composant
  priority: ComponentPriority;// Priorité d'exécution
  dependencies?: string[];    // IDs des composants dépendants
  metadata?: Record<string, any>; // Métadonnées additionnelles
  executeFunction: (context: any) => Promise<any>; // Fonction d'exécution
  isEnabled: boolean;         // Si le composant est activé
}

/**
 * Registre central des composants du système
 */
export class ComponentRegistry {
  private components: Map<string, ComponentRegistration>;
  private logger: Logger;

  /**
   * Crée une nouvelle instance du registre de composants
   * @param logger Instance du logger
   */
  constructor(logger: Logger) {
    this.components = new Map<string, ComponentRegistration>();
    this.logger = logger;
  }

  /**
   * Enregistre un nouveau composant dans le registre
   * @param registration Données d'enregistrement du composant
   * @returns ID du composant enregistré
   */
  public register(registration: Omit<ComponentRegistration, 'id'>): string {
    const id = registration.metadata?.id || uuidv4();
    
    if (this.components.has(id)) {
      this.logger.warn(`Composant avec ID ${id} déjà enregistré, écrasement`);
    }
    
    const fullRegistration: ComponentRegistration = {
      ...registration,
      id,
      isEnabled: registration.isEnabled !== undefined ? registration.isEnabled : true
    };
    
    this.components.set(id, fullRegistration);
    this.logger.info(`Composant enregistré: ${registration.name} (${id}), type: ${registration.type}`);
    
    return id;
  }

  /**
   * Récupère un composant par son ID
   * @param id ID du composant
   * @returns Enregistrement du composant ou undefined si non trouvé
   */
  public get(id: string): ComponentRegistration | undefined {
    return this.components.get(id);
  }

  /**
   * Récupère tous les composants enregistrés
   * @returns Map des composants
   */
  public getAll(): Map<string, ComponentRegistration> {
    return new Map(this.components);
  }

  /**
   * Récupère tous les composants d'un type spécifique
   * @param type Type de composant à récupérer
   * @returns Tableau des composants du type spécifié
   */
  public getByType(type: ComponentType): ComponentRegistration[] {
    return Array.from(this.components.values()).filter(comp => comp.type === type);
  }

  /**
   * Désactive un composant
   * @param id ID du composant à désactiver
   * @returns true si succès, false si le composant n'existe pas
   */
  public disable(id: string): boolean {
    const component = this.components.get(id);
    if (!component) {
      return false;
    }
    
    component.isEnabled = false;
    this.logger.info(`Composant désactivé: ${component.name} (${id})`);
    return true;
  }

  /**
   * Active un composant
   * @param id ID du composant à activer
   * @returns true si succès, false si le composant n'existe pas
   */
  public enable(id: string): boolean {
    const component = this.components.get(id);
    if (!component) {
      return false;
    }
    
    component.isEnabled = true;
    this.logger.info(`Composant activé: ${component.name} (${id})`);
    return true;
  }

  /**
   * Supprime un composant du registre
   * @param id ID du composant à supprimer
   * @returns true si succès, false si le composant n'existe pas
   */
  public unregister(id: string): boolean {
    if (!this.components.has(id)) {
      return false;
    }
    
    const component = this.components.get(id);
    this.components.delete(id);
    this.logger.info(`Composant supprimé: ${component.name} (${id})`);
    return true;
  }

  /**
   * Met à jour l'enregistrement d'un composant
   * @param id ID du composant à mettre à jour
   * @param updates Mises à jour à appliquer
   * @returns true si succès, false si le composant n'existe pas
   */
  public update(id: string, updates: Partial<Omit<ComponentRegistration, 'id'>>): boolean {
    const component = this.components.get(id);
    if (!component) {
      return false;
    }
    
    // Appliquer les mises à jour
    Object.assign(component, updates);
    this.logger.info(`Composant mis à jour: ${component.name} (${id})`);
    return true;
  }

  /**
   * Convertit les composants enregistrés en composants exécutables pour le handler
   * @returns Tableau de composants exécutables
   */
  public buildExecutableComponents(): CoordinationComponent[] {
    return Array.from(this.components.values())
      .filter(registration => registration.isEnabled)
      .map(registration => {
        return {
          id: registration.id,
          name: registration.name,
          description: registration.description,
          execute: registration.executeFunction,
          dependencies: registration.dependencies,
          priority: registration.priority,
          status: ComponentStatus.IDLE,
          retryable: true
        };
      });
  }

  /**
   * Génère un rapport sur les composants enregistrés
   * @returns Rapport texte
   */
  public generateReport(): string {
    const components = Array.from(this.components.values());
    const totalCount = components.length;
    const enabledCount = components.filter(c => c.isEnabled).length;
    const disabledCount = totalCount - enabledCount;
    
    // Comptage par type
    const typeCount: Record<string, number> = {};
    components.forEach(c => {
      typeCount[c.type] = (typeCount[c.type] || 0) + 1;
    });
    
    // Comptage par priorité
    const priorityCount: Record<string, number> = {};
    components.forEach(c => {
      priorityCount[c.priority] = (priorityCount[c.priority] || 0) + 1;
    });
    
    // Génération du rapport
    let report = `=== Rapport du Registre de Composants ===\n`;
    report += `Total des composants: ${totalCount} (${enabledCount} actifs, ${disabledCount} inactifs)\n\n`;
    
    report += `=== Par Type ===\n`;
    Object.entries(typeCount).sort((a, b) => b[1] - a[1]).forEach(([type, count]) => {
      report += `- ${type}: ${count}\n`;
    });
    
    report += `\n=== Par Priorité ===\n`;
    Object.entries(priorityCount).forEach(([priority, count]) => {
      report += `- ${priority}: ${count}\n`;
    });
    
    report += `\n=== Dépendances ===\n`;
    const componentWithDeps = components.filter(c => c.dependencies && c.dependencies.length > 0);
    report += `Composants avec dépendances: ${componentWithDeps.length}\n`;
    
    componentWithDeps.forEach(c => {
      report += `- ${c.name} (${c.id}) dépend de: ${c.dependencies.join(', ')}\n`;
    });
    
    return report;
  }

  /**
   * Vérifie les dépendances circulaires entre les composants
   * @returns Liste de cycles détectés, vide si aucun
   */
  public detectCircularDependencies(): string[][] {
    // Construire un graphe de dépendances
    const graph: Record<string, string[]> = {};
    Array.from(this.components.values()).forEach(comp => {
      graph[comp.id] = comp.dependencies || [];
    });
    
    // Fonction pour détecter les cycles dans un graphe dirigé
    const detectCycles = (graph: Record<string, string[]>): string[][] => {
      const visited = new Set<string>();
      const recursionStack = new Set<string>();
      const cycles: string[][] = [];
      
      const dfs = (node: string, path: string[] = []): void => {
        if (recursionStack.has(node)) {
          // Cycle trouvé
          const cycleStart = path.indexOf(node);
          if (cycleStart >= 0) {
            cycles.push(path.slice(cycleStart).concat(node));
          }
          return;
        }
        
        if (visited.has(node)) {
          return;
        }
        
        visited.add(node);
        recursionStack.add(node);
        
        const neighbors = graph[node] || [];
        for (const neighbor of neighbors) {
          dfs(neighbor, [...path, node]);
        }
        
        recursionStack.delete(node);
      };
      
      // Explorer tous les nœuds
      for (const node in graph) {
        if (!visited.has(node)) {
          dfs(node);
        }
      }
      
      return cycles;
    };
    
    return detectCycles(graph);
  }

  /**
   * Vérifier l'intégrité des dépendances
   * @returns Objet contenant les problèmes détectés
   */
  public validateDependencies(): {
    missingDependencies: Array<{componentId: string, dependsOn: string}>,
    circularDependencies: string[][]
  } {
    const missingDependencies: Array<{componentId: string, dependsOn: string}> = [];
    
    // Vérifier les dépendances manquantes
    Array.from(this.components.values()).forEach(comp => {
      if (comp.dependencies) {
        comp.dependencies.forEach(depId => {
          if (!this.components.has(depId)) {
            missingDependencies.push({
              componentId: comp.id,
              dependsOn: depId
            });
          }
        });
      }
    });
    
    // Vérifier les dépendances circulaires
    const circularDependencies = this.detectCircularDependencies();
    
    return {
      missingDependencies,
      circularDependencies
    };
  }
} 