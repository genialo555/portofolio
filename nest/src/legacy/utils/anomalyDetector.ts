import * as fs from 'fs';
import { Injectable, Inject } from '@nestjs/common';
import { PoolOutputs, AnomalyReport, Anomaly, AnomalyType, AnomalySeverity, AgentOutput } from '../../types';
import { ILogger, LOGGER_TOKEN } from '../../rag-kag/utils/logger-tokens';

/**
 * Service de détection d'anomalies dans les outputs des agents
 * Utilise des techniques avancées pour identifier les incohérences, erreurs factuelles,
 * et autres problèmes dans les sorties des différents pools d'agents
 */
@Injectable()
export class AnomalyDetectorService {
  private readonly logger: ILogger;
  
  constructor(@Inject(LOGGER_TOKEN) logger: ILogger) {
    this.logger = logger;
  }

  /**
   * Analyse les outputs des pools pour détecter différentes catégories d'anomalies
   * @param poolOutputs Outputs consolidés des différents pools d'agents
   * @returns Rapport d'anomalies détaillé
   */
  async detectAnomalies(poolOutputs: PoolOutputs): Promise<AnomalyReport> {
    this.logger.info('Démarrage de la détection d\'anomalies', { 
      poolCount: this.countActivePools(poolOutputs),
      outputsCount: this.countTotalOutputs(poolOutputs)
    });

    // Collecter les anomalies par catégorie et sévérité
    const logicalInconsistencies = await this.detectLogicalInconsistencies(poolOutputs);
    const factualErrors = await this.detectFactualErrors(poolOutputs);
    const methodologicalFlaws = this.detectMethodologicalFlaws(poolOutputs);
    const biases = this.detectCognitiveBiases(poolOutputs);
    const statisticalErrors = this.detectStatisticalErrors(poolOutputs);
    const citationIssues = this.detectCitationIssues(poolOutputs);

    // Agréger toutes les anomalies détectées
    const allAnomalies = [
      ...logicalInconsistencies,
      ...factualErrors,
      ...methodologicalFlaws,
      ...biases,
      ...statisticalErrors,
      ...citationIssues
    ];

    // Répartir les anomalies par niveau de sévérité
    const highPriorityAnomalies = allAnomalies.filter(a => a.severity === 'HIGH');
    const mediumPriorityAnomalies = allAnomalies.filter(a => a.severity === 'MEDIUM');
    const lowPriorityAnomalies = allAnomalies.filter(a => a.severity === 'LOW');

    // Calculer la fiabilité globale (diminue avec le nombre et la sévérité des anomalies)
    const overallReliability = this.calculateReliability(allAnomalies);
    
    // Générer un rapport narratif sur les anomalies détectées
    const report = this.generateAnomalyReport(allAnomalies, overallReliability);

    // Identifier les patterns systémiques (problèmes récurrents)
    const systemicPatterns = this.identifySystemicPatterns(allAnomalies);

    this.logger.info('Détection d\'anomalies terminée', {
      highCount: highPriorityAnomalies.length,
      mediumCount: mediumPriorityAnomalies.length,
      lowCount: lowPriorityAnomalies.length,
      reliability: overallReliability
    });

    return {
      highPriorityAnomalies,
      mediumPriorityAnomalies,
      minorIssues: lowPriorityAnomalies,
      lowPriorityAnomalies,
      overallReliability,
      report,
      systemicPatterns
    };
  }

  /**
   * Détecte les incohérences logiques entre les outputs ou au sein d'un même output
   */
  private async detectLogicalInconsistencies(poolOutputs: PoolOutputs): Promise<Anomaly[]> {
    const anomalies: Anomaly[] = [];
    const allOutputs = this.getAllOutputs(poolOutputs);
    
    // 1. Détecter les contradictions directes entre agents
    for (let i = 0; i < allOutputs.length; i++) {
      for (let j = i + 1; j < allOutputs.length; j++) {
        const contradictions = this.findContradictions(allOutputs[i], allOutputs[j]);
        if (contradictions.length > 0) {
          anomalies.push({
            type: 'LOGICAL_INCONSISTENCY',
            description: `Contradiction entre agents: ${contradictions[0]}`,
            severity: 'HIGH',
            location: {
              agentId: allOutputs[i].agentId,
              poolType: allOutputs[i].poolType,
              contentFragment: this.extractFragment(allOutputs[i].content, contradictions[0])
            },
            suggestedResolution: 'Clarifier cette contradiction en comparant les sources'
          });
        }
      }
    }
    
    // 2. Détecter les incohérences internes dans chaque output
    for (const output of allOutputs) {
      const internalInconsistencies = this.findInternalInconsistencies(output.content);
      for (const inconsistency of internalInconsistencies) {
        anomalies.push({
          type: 'LOGICAL_INCONSISTENCY',
          description: `Incohérence interne: ${inconsistency}`,
          severity: 'MEDIUM',
          location: {
            agentId: output.agentId,
            poolType: output.poolType,
            contentFragment: this.extractFragment(output.content, inconsistency)
          }
        });
      }
    }
    
    return anomalies;
  }

  /**
   * Détecte les erreurs factuelles en comparant avec des sources fiables
   */
  private async detectFactualErrors(poolOutputs: PoolOutputs): Promise<Anomaly[]> {
    // Simulation: dans une implémentation réelle, cette méthode interrogerait 
    // une base de connaissances ou ferait des vérifications avec des API externes
    const anomalies: Anomaly[] = [];
    const allOutputs = this.getAllOutputs(poolOutputs);
    
    for (const output of allOutputs) {
      // Recherche de faits vérifiables dans le contenu
      const facts = this.extractFactualClaims(output.content);
      
      for (const fact of facts) {
        // Ici, vous intégreriez une vérification réelle avec une base de connaissances
        const isFactCorrect = this.simulateFactCheck(fact);
        
        if (!isFactCorrect) {
          anomalies.push({
            type: 'FACTUAL_ERROR',
            description: `Erreur factuelle potentielle: "${fact}"`,
            severity: 'HIGH',
            location: {
              agentId: output.agentId,
              poolType: output.poolType,
              contentFragment: fact
            },
            suggestedResolution: 'Vérifier cette affirmation avec des sources fiables'
          });
        }
      }
    }
    
    return anomalies;
  }

  /**
   * Détecte les biais cognitifs dans les réponses
   */
  public detectCognitiveBiases(poolOutputs: PoolOutputs): Anomaly[] {
    const anomalies: Anomaly[] = [];
    const biasPatterns = [
      { 
        name: 'Biais de confirmation', 
        pattern: /(?:confirme|prouve|démontre clairement|évident que|sans aucun doute)/i,
        severity: 'MEDIUM' 
      },
      { 
        name: 'Biais d\'autorité', 
        pattern: /(?:experts sont unanimes|la science a prouvé|tous les spécialistes)/i,
        severity: 'MEDIUM' 
      },
      { 
        name: 'Généralisation excessive', 
        pattern: /(?:toujours|jamais|tous|aucun|dans tous les cas|systématiquement)/i,
        severity: 'HIGH' 
      }
    ];
    
    const allOutputs = this.getAllOutputs(poolOutputs);
    
    for (const output of allOutputs) {
      for (const bias of biasPatterns) {
        const matches = output.content.match(bias.pattern);
        if (matches && matches.length > 0) {
          anomalies.push({
            type: 'COGNITIVE_BIAS',
            description: `${bias.name} détecté`,
            severity: bias.severity as AnomalySeverity,
            location: {
              agentId: output.agentId,
              poolType: output.poolType,
              contentFragment: matches[0]
            },
            suggestedResolution: 'Reformuler avec plus de nuance'
          });
        }
      }
    }
    
    return anomalies;
  }

  /**
   * Détecte les failles méthodologiques dans les réponses
   */
  public detectMethodologicalFlaws(poolOutputs: PoolOutputs): Anomaly[] {
    const anomalies: Anomaly[] = [];
    const methodologicalPatterns = [
      { 
        name: 'Analyse superficielle', 
        pattern: /(?:en résumé|pour faire simple|sans entrer dans les détails|en gros)/i,
        severity: 'LOW' 
      },
      { 
        name: 'Conclusion hâtive', 
        pattern: /(?:donc on peut conclure|il est clair que|cela prouve que|on en déduit que)/i,
        severity: 'MEDIUM' 
      }
    ];
    
    const allOutputs = this.getAllOutputs(poolOutputs);
    
    for (const output of allOutputs) {
      for (const flaw of methodologicalPatterns) {
        const matches = output.content.match(flaw.pattern);
        if (matches && matches.length > 0) {
          anomalies.push({
            type: 'METHODOLOGICAL_FLAW',
            description: `${flaw.name} détectée`,
            severity: flaw.severity as AnomalySeverity,
            location: {
              agentId: output.agentId,
              poolType: output.poolType,
              contentFragment: matches[0]
            }
          });
        }
      }
    }
    
    return anomalies;
  }

  /**
   * Détecte les erreurs statistiques dans les réponses
   */
  public detectStatisticalErrors(poolOutputs: PoolOutputs): Anomaly[] {
    const anomalies: Anomaly[] = [];
    // Recherche de patterns comme "X% des cas" sans source, ou des statistiques incohérentes
    const statPatterns = /\d{1,3}(?:[.,]\d+)?%|(?:la plupart|majorité|minorité) sans précision/gi;
    
    const allOutputs = this.getAllOutputs(poolOutputs);
    
    for (const output of allOutputs) {
      const matches = [...output.content.matchAll(statPatterns)];
      
      if (matches.length > 0 && !output.content.includes('source') && !output.content.includes('selon')) {
        anomalies.push({
          type: 'STATISTICAL_ERROR',
          description: 'Statistiques mentionnées sans source',
          severity: 'MEDIUM',
          location: {
            agentId: output.agentId,
            poolType: output.poolType,
            contentFragment: matches[0][0]
          },
          suggestedResolution: 'Ajouter les sources des statistiques citées'
        });
      }
    }
    
    return anomalies;
  }

  /**
   * Détecte les problèmes de citation dans les réponses
   */
  public detectCitationIssues(poolOutputs: PoolOutputs): Anomaly[] {
    const anomalies: Anomaly[] = [];
    const citationPatterns = [
      { 
        name: 'Citation sans source', 
        pattern: /(?:selon|d'après|comme l'a dit|comme mentionné par)(?!(?: \w+){1,4}(?: \d{4}| et al\.))/i,
        severity: 'MEDIUM' 
      },
      { 
        name: 'Référence à une étude non spécifiée', 
        pattern: /(?:une étude|des recherches|des études) (?!de|par|menée|publiée)/i,
        severity: 'LOW' 
      }
    ];
    
    const allOutputs = this.getAllOutputs(poolOutputs);
    
    for (const output of allOutputs) {
      for (const citation of citationPatterns) {
        const matches = output.content.match(citation.pattern);
        if (matches && matches.length > 0) {
          anomalies.push({
            type: 'CITATION_ISSUE',
            description: `${citation.name} détectée`,
            severity: citation.severity as AnomalySeverity,
            location: {
              agentId: output.agentId,
              poolType: output.poolType,
              contentFragment: this.extractFragment(output.content, matches[0])
            },
            suggestedResolution: 'Préciser la source de cette citation'
          });
        }
      }
    }
    
    return anomalies;
  }

  /**
   * Calcule la fiabilité globale en fonction des anomalies détectées
   */
  private calculateReliability(anomalies: Anomaly[]): number {
    // Partir d'une fiabilité de 100%
    let reliability = 1.0;
    
    // Réduire la fiabilité en fonction du nombre et de la sévérité des anomalies
    for (const anomaly of anomalies) {
      switch (anomaly.severity) {
        case 'HIGH':
          reliability -= 0.15; // -15% par anomalie haute
          break;
        case 'MEDIUM':
          reliability -= 0.08; // -8% par anomalie moyenne
          break;
        case 'LOW':
          reliability -= 0.03; // -3% par anomalie basse
          break;
      }
    }
    
    // Ne pas descendre en dessous de 10%
    return Math.max(0.1, reliability);
  }

  /**
   * Génère un rapport textuel sur les anomalies détectées
   */
  private generateAnomalyReport(anomalies: Anomaly[], reliability: number): string {
    if (anomalies.length === 0) {
      return "Aucune anomalie détectée. Les informations semblent fiables.";
    }

    const highCount = anomalies.filter(a => a.severity === 'HIGH').length;
    const mediumCount = anomalies.filter(a => a.severity === 'MEDIUM').length;
    const lowCount = anomalies.filter(a => a.severity === 'LOW').length;
    
    let reliabilityText = "";
    if (reliability > 0.8) {
      reliabilityText = "L'ensemble des informations semble globalement fiable malgré quelques points d'attention.";
    } else if (reliability > 0.5) {
      reliabilityText = "La fiabilité des informations est modérée, avec plusieurs éléments nécessitant une vérification.";
    } else {
      reliabilityText = "La fiabilité des informations est faible, avec de nombreuses anomalies critiques détectées.";
    }
    
    return `Rapport d'anomalies: ${highCount} anomalies critiques, ${mediumCount} anomalies modérées et ${lowCount} anomalies mineures détectées. ${reliabilityText}`;
  }

  /**
   * Identifie les patterns systémiques dans les anomalies
   */
  private identifySystemicPatterns(anomalies: Anomaly[]): string[] {
    const patterns: string[] = [];
    
    // Regrouper les anomalies par type
    const anomaliesByType = anomalies.reduce((acc, anomaly) => {
      acc[anomaly.type] = (acc[anomaly.type] || []).concat(anomaly);
      return acc;
    }, {} as Record<string, Anomaly[]>);
    
    // Identifier les types d'anomalies récurrents
    for (const [type, typeAnomalies] of Object.entries(anomaliesByType)) {
      if (typeAnomalies.length >= 3) {
        patterns.push(`Pattern récurrent de ${type}: ${typeAnomalies.length} occurrences`);
      }
    }
    
    // Regrouper par agent/pool pour détecter des problèmes spécifiques à certains agents
    const anomaliesByAgent = anomalies.reduce((acc, anomaly) => {
      const key = `${anomaly.location.agentId}`;
      acc[key] = (acc[key] || []).concat(anomaly);
      return acc;
    }, {} as Record<string, Anomaly[]>);
    
    for (const [agentId, agentAnomalies] of Object.entries(anomaliesByAgent)) {
      if (agentAnomalies.length >= 3) {
        patterns.push(`Agent problématique: ${agentId} avec ${agentAnomalies.length} anomalies`);
      }
    }
    
    return patterns;
  }

  /**
   * Recherche des contradictions entre deux outputs d'agents
   */
  private findContradictions(outputA: AgentOutput, outputB: AgentOutput): string[] {
    // Cette méthode serait idéalement implémentée avec NLP
    // Ici on simule une détection basique
    const contradictions: string[] = [];
    
    // Détection de phrases contradictoires basée sur des mots clés opposés
    const affirmationPatternsA = this.extractAffirmations(outputA.content);
    const affirmationPatternsB = this.extractAffirmations(outputB.content);
    
    for (const affirmA of affirmationPatternsA) {
      for (const affirmB of affirmationPatternsB) {
        if (this.areContradictory(affirmA, affirmB)) {
          contradictions.push(`"${affirmA}" vs "${affirmB}"`);
        }
      }
    }
    
    return contradictions;
  }

  /**
   * Extrait des affirmations d'un texte
   */
  private extractAffirmations(content: string): string[] {
    // Dans une implémentation réelle, utilisez NLP pour extraire des affirmations
    // Ici on divise simplement par phrases et on prend celles qui semblent être des affirmations
    const sentences = content.split(/[.!?]/).filter(s => s.trim().length > 0);
    return sentences.filter(s => {
      return !s.includes('?') && 
             !s.includes('peut-être') && 
             !s.includes('probablement') &&
             !s.toLowerCase().includes('je pense') &&
             s.length > 20;
    });
  }

  /**
   * Détermine si deux affirmations sont contradictoires
   */
  private areContradictory(affirmA: string, affirmB: string): boolean {
    // Détection simple basée sur la présence de négations opposées
    const negationA = this.containsNegation(affirmA);
    const negationB = this.containsNegation(affirmB);
    
    // Si une contient une négation et l'autre non, et qu'elles parlent du même sujet
    if (negationA !== negationB) {
      const subjectA = this.extractSubject(affirmA);
      const subjectB = this.extractSubject(affirmB);
      
      // Si les sujets sont similaires, on considère que c'est contradictoire
      return this.areSimilarSubjects(subjectA, subjectB);
    }
    
    return false;
  }

  /**
   * Extrait le sujet d'une phrase
   */
  private extractSubject(sentence: string): string {
    // Simplifié: dans une implémentation réelle, utiliser NLP pour extraire le sujet
    const words = sentence.toLowerCase().split(' ');
    return words.slice(0, Math.min(5, words.length)).join(' ');
  }

  /**
   * Vérifie si deux sujets sont similaires
   */
  private areSimilarSubjects(subjectA: string, subjectB: string): boolean {
    // Simplifié: comparer les mots clés des sujets
    const wordsA = new Set(subjectA.toLowerCase().split(' '));
    const wordsB = subjectB.toLowerCase().split(' ');
    
    let commonWords = 0;
    for (const word of wordsB) {
      if (wordsA.has(word) && word.length > 3) {
        commonWords++;
      }
    }
    
    return commonWords >= 2; // Au moins 2 mots communs significatifs
  }

  /**
   * Vérifie si une phrase contient une négation
   */
  private containsNegation(sentence: string): boolean {
    const negationPatterns = [
      /\bne\s+\w+\s+pas\b/i,
      /\bn['|']est\s+pas\b/i,
      /\baucun\b/i,
      /\bjamais\b/i,
      /\bnon\b/i,
      /\bcontre\b/i
    ];
    
    return negationPatterns.some(pattern => pattern.test(sentence));
  }

  /**
   * Recherche des incohérences internes dans un texte
   */
  private findInternalInconsistencies(content: string): string[] {
    const inconsistencies: string[] = [];
    const affirmations = this.extractAffirmations(content);
    
    for (let i = 0; i < affirmations.length; i++) {
      for (let j = i + 1; j < affirmations.length; j++) {
        if (this.areContradictory(affirmations[i], affirmations[j])) {
          inconsistencies.push(`"${affirmations[i]}" vs "${affirmations[j]}"`);
        }
      }
    }
    
    return inconsistencies;
  }

  /**
   * Extrait des affirmations factuelles d'un texte
   */
  private extractFactualClaims(content: string): string[] {
    // Patterns pour identifier des affirmations factuelles
    const factualPatterns = [
      /\ben (\d{4})\b/gi,                          // Années
      /\b(\d+(?:[.,]\d+)?) ?%\b/gi,                // Pourcentages
      /\b(plus|moins) de (\d+(?:[.,]\d+)?)\b/gi,   // Comparaisons numériques
      /\bselon .{3,30},/gi,                        // Citations de sources
      /\b([A-Z][a-z]+ (?:[A-Z][a-z]+)?)(?: a| est| était)/gi // Personnes ou entités
    ];
    
    const facts: string[] = [];
    
    for (const pattern of factualPatterns) {
      const matches = [...content.matchAll(pattern)];
      for (const match of matches) {
        // Extraire la phrase contenant ce fait
        const sentenceStart = content.lastIndexOf('.', match.index) + 1;
        const sentenceEnd = content.indexOf('.', match.index);
        if (sentenceEnd > sentenceStart) {
          const sentence = content.substring(sentenceStart, sentenceEnd).trim();
          facts.push(sentence);
        }
      }
    }
    
    return facts;
  }

  /**
   * Simulation de vérification des faits
   * Dans une vraie implémentation, cette fonction appellerait une base de connaissances
   */
  private simulateFactCheck(fact: string): boolean {
    // Pour la démonstration, on considère que 15% des affirmations sont incorrectes
    return Math.random() > 0.15;
  }

  /**
   * Extrait un fragment de texte autour d'un marqueur donné
   */
  private extractFragment(content: string, marker: string): string {
    if (!content.includes(marker)) return marker;
    
    const index = content.indexOf(marker);
    const start = Math.max(0, index - 20);
    const end = Math.min(content.length, index + marker.length + 20);
    
    return content.substring(start, end).replace(/^\S*\s/, '...').replace(/\s\S*$/, '...');
  }

  /**
   * Récupère tous les outputs d'agents à partir de PoolOutputs
   */
  private getAllOutputs(poolOutputs: PoolOutputs): AgentOutput[] {
    const outputs: AgentOutput[] = [];
    
    if (poolOutputs.commercial) {
      outputs.push(...poolOutputs.commercial);
    }
    
    if (poolOutputs.marketing) {
      outputs.push(...poolOutputs.marketing);
    }
    
    if (poolOutputs.sectoriel) {
      outputs.push(...poolOutputs.sectoriel);
    }
    
    return outputs;
  }

  /**
   * Compte le nombre de pools actifs
   */
  private countActivePools(poolOutputs: PoolOutputs): number {
    let count = 0;
    if (poolOutputs.commercial && poolOutputs.commercial.length > 0) count++;
    if (poolOutputs.marketing && poolOutputs.marketing.length > 0) count++;
    if (poolOutputs.sectoriel && poolOutputs.sectoriel.length > 0) count++;
    return count;
  }

  /**
   * Compte le nombre total d'outputs
   */
  private countTotalOutputs(poolOutputs: PoolOutputs): number {
    let count = 0;
    if (poolOutputs.commercial) count += poolOutputs.commercial.length;
    if (poolOutputs.marketing) count += poolOutputs.marketing.length;
    if (poolOutputs.sectoriel) count += poolOutputs.sectoriel.length;
    return count;
  }
} 