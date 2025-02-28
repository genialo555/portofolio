import { CoordinationSystem } from '../core/coordination-system';
import { ExecutionMode } from '../handlers/coordination-handler';
import { LogLevel } from '../utils/logger';
import { QueryAnalyzerComponent } from '../components/impl/query-analyzer';
import { RagEngine } from '../components/impl/rag-engine';
import { KagEngine } from '../components/impl/kag-engine';

/**
 * Exemple démontrant l'enregistrement et l'utilisation des composants personnalisés
 */
export async function runComponentsExample(): Promise<void> {
  console.log('=== Démarrage du système avec composants personnalisés ===');
  
  // Créer une instance du système de coordination
  const system = new CoordinationSystem({
    logLevel: LogLevel.DEBUG,
    defaultExecutionMode: ExecutionMode.ADAPTIVE,
    // Désactiver l'enregistrement automatique pour utiliser nos propres composants
    autoRegisterComponents: false
  });
  
  // Accéder au logger du système (via une méthode qui devrait être ajoutée au système)
  const logger = system['logger']; // Dans une implémentation réelle, utilisez une méthode publique
  
  // Créer et enregistrer un analyseur de requête
  const queryAnalyzer = new QueryAnalyzerComponent(logger);
  const queryAnalyzerId = system.registerComponent(queryAnalyzer.createRegistration());
  console.log(`Composant analyseur de requête enregistré avec ID: ${queryAnalyzerId}`);
  
  // Créer et enregistrer un moteur RAG
  const ragEngine = new RagEngine(logger);
  const ragEngineId = system.registerComponent(ragEngine.createRegistration());
  console.log(`Composant moteur RAG enregistré avec ID: ${ragEngineId}`);
  
  // Créer et enregistrer un moteur KAG
  const kagEngine = new KagEngine(logger);
  const kagEngineId = system.registerComponent(kagEngine.createRegistration());
  console.log(`Composant moteur KAG enregistré avec ID: ${kagEngineId}`);
  
  // Attendre que le système soit initialisé
  await new Promise(resolve => setTimeout(resolve, 500));
  
  // Afficher le rapport du système
  console.log('\n=== État du système avec composants personnalisés ===');
  const systemReport = system.generateSystemReport();
  console.log(JSON.stringify(systemReport.status, null, 2));
  
  // Exemple de requête simple
  const query = "Quelles sont les stratégies marketing efficaces pour une entreprise B2B?";
  
  console.log(`\n=== Traitement d'une requête avec nos composants ===`);
  console.log(`Requête: "${query}"`);
  
  const result = await system.processQuery(query, {
    executionMode: ExecutionMode.SEQUENTIAL,
    includeDebugInfo: true,
    enableAnomalyDetection: true
  });
  
  console.log(`Résultat: ${result.success ? 'Succès' : 'Échec'}`);
  console.log(`Durée: ${result.duration}ms`);
  console.log(`TraceID: ${result.traceId}`);
  
  if (result.result) {
    console.log('\nRésultat principal:');
    console.log(JSON.stringify(result.result, null, 2));
  }
  
  if (result.debugInfo) {
    console.log('\nChemin d\'exécution:');
    console.log(result.debugInfo.executionPath.join(' → '));
  }
  
  // Exemple de traitement d'une requête plus complexe
  const complexQuery = "Comment optimiser les entonnoirs de conversion commerciale dans le e-commerce tout en analysant les données clients avec l'IA?";
  
  console.log(`\n=== Traitement d'une requête complexe ===`);
  console.log(`Requête: "${complexQuery}"`);
  
  const complexResult = await system.processQuery(complexQuery, {
    executionMode: ExecutionMode.PARALLEL,
    includeDebugInfo: true
  });
  
  console.log(`Résultat: ${complexResult.success ? 'Succès' : 'Échec'}`);
  console.log(`Durée: ${complexResult.duration}ms`);
  
  if (complexResult.result && complexResult.result.contextualKnowledge) {
    console.log('\nConnaissances contextuelles:');
    console.log(complexResult.result.contextualKnowledge);
    
    console.log('\nSources pertinentes:');
    for (const source of complexResult.result.mostRelevantSources || []) {
      console.log(`- ${source}`);
    }
  }
  
  // Exemple de requête nécessitant le KAG
  const kagQuery = "Quelles tendances futures pourraient transformer l'expérience client et comment s'y préparer stratégiquement?";
  
  console.log(`\n=== Traitement d'une requête nécessitant KAG ===`);
  console.log(`Requête: "${kagQuery}"`);
  
  const kagResult = await system.processQuery(kagQuery, {
    executionMode: ExecutionMode.PARALLEL,
    includeDebugInfo: true,
    useComponents: ['KAG_ENGINE'] // Spécifier les composants à utiliser
  });
  
  console.log(`Résultat: ${kagResult.success ? 'Succès' : 'Échec'}`);
  console.log(`Durée: ${kagResult.duration}ms`);
  
  if (kagResult.result && kagResult.result.synthesizedResponse) {
    console.log('\nRéponse synthétisée:');
    console.log(kagResult.result.synthesizedResponse);
    
    if (kagResult.result.generatedKnowledge) {
      console.log('\nConnaissances générées:');
      for (const knowledge of kagResult.result.generatedKnowledge.slice(0, 2)) {
        console.log(`- [${knowledge.sourceType}] ${knowledge.content} (confiance: ${(knowledge.confidenceScore * 100).toFixed(1)}%)`);
      }
      
      console.log('\nScores de confiance:');
      console.log(JSON.stringify(kagResult.result.confidenceScores, null, 2));
    }
  }
  
  console.log('\n=== Système de coordination arrêté ===');
}

// Pour exécuter cet exemple, utilisez:
// import { runComponentsExample } from './components-usage';
// runComponentsExample().catch(console.error); 