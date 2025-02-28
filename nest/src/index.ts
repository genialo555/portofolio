import { CoordinationSystem, QueryOptions } from './core/coordination-system';
import { ExecutionMode } from './handlers/coordination-handler';
import { LogLevel } from './utils/logger';

/**
 * Exemple d'utilisation du système de coordination
 */
async function main() {
  console.log('=== Démarrage du système de coordination RAG/KAG ===');
  
  // Créer une instance du système de coordination
  const system = new CoordinationSystem({
    logLevel: LogLevel.DEBUG,
    defaultExecutionMode: ExecutionMode.ADAPTIVE,
    autoRegisterComponents: true
  });
  
  // Attendre que le système soit initialisé
  await new Promise(resolve => setTimeout(resolve, 500));
  
  // Afficher le rapport initial du système
  console.log('\n=== État initial du système ===');
  const initialReport = system.generateSystemReport();
  console.log(JSON.stringify(initialReport.status, null, 2));
  
  // Exemple de requête simple
  const simpleQuery = "Quelle est la meilleure stratégie marketing pour une petite entreprise?";
  
  const simpleOptions: QueryOptions = {
    executionMode: ExecutionMode.SEQUENTIAL,
    includeDebugInfo: true,
    abortOnFailure: false
  };
  
  console.log(`\n=== Traitement d'une requête simple ===`);
  console.log(`Requête: "${simpleQuery}"`);
  
  const simpleResult = await system.processQuery(simpleQuery, simpleOptions);
  
  console.log(`Résultat: ${simpleResult.success ? 'Succès' : 'Échec'}`);
  console.log(`Durée: ${simpleResult.duration}ms`);
  console.log(`TraceID: ${simpleResult.traceId}`);
  
  if (simpleResult.result) {
    console.log('\nRésultat principal:');
    console.log(JSON.stringify(simpleResult.result, null, 2));
  }
  
  if (simpleResult.debugInfo) {
    console.log('\nChemin d\'exécution:');
    console.log(simpleResult.debugInfo.executionPath.join(' → '));
  }
  
  // Exemple de requête complexe
  const complexQuery = "Analysez l'impact des stratégies omnicanal sur la fidélisation client dans le secteur du luxe, avec une attention particulière aux marchés émergents et aux tendances post-pandémie.";
  
  const complexOptions: QueryOptions = {
    executionMode: ExecutionMode.PARALLEL,
    timeout: 60000,
    includeDebugInfo: true,
    enableAnomalyDetection: true
  };
  
  console.log(`\n=== Traitement d'une requête complexe ===`);
  console.log(`Requête: "${complexQuery}"`);
  
  const complexResult = await system.processQuery(complexQuery, complexOptions);
  
  console.log(`Résultat: ${complexResult.success ? 'Succès' : 'Échec'}`);
  console.log(`Durée: ${complexResult.duration}ms`);
  console.log(`TraceID: ${complexResult.traceId}`);
  
  if (complexResult.debugInfo && complexResult.debugInfo.metrics) {
    console.log('\nMétriques d\'exécution:');
    console.log(`Durée totale: ${complexResult.debugInfo.metrics.totalDuration}ms`);
    
    if (complexResult.debugInfo.metrics.bottlenecks.length > 0) {
      console.log(`Goulots d'étranglement détectés: ${complexResult.debugInfo.metrics.bottlenecks.join(', ')}`);
    }
    
    if (complexResult.debugInfo.metrics.optimizationSuggestions.length > 0) {
      console.log('\nSuggestions d\'optimisation:');
      for (const suggestion of complexResult.debugInfo.metrics.optimizationSuggestions) {
        console.log(`- ${suggestion}`);
      }
    }
  }
  
  console.log('\n=== Système de coordination arrêté ===');
}

// Exécuter l'exemple
main().catch(error => {
  console.error('Erreur lors de l\'exécution:', error);
  process.exit(1);
}); 