/**
 * Script pour créer un modèle TensorFlow.js minimal
 * Exécuter avec: npx ts-node src/scripts/create-model.ts
 */

import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'fs';
import * as path from 'path';

async function createSimpleModel(modelName: string, vocabSize: number = 10000, embeddingDim: number = 128) {
  console.log(`Création d'un modèle simple pour ${modelName}...`);
  
  // Créer le dossier du modèle s'il n'existe pas
  const modelDir = path.join(process.cwd(), 'models', modelName);
  if (!fs.existsSync(modelDir)) {
    fs.mkdirSync(modelDir, { recursive: true });
  }
  
  // Créer un modèle simple de génération de texte
  const model = tf.sequential();
  
  // Couche d'embedding
  model.add(tf.layers.embedding({
    inputDim: vocabSize,
    outputDim: embeddingDim,
    inputLength: 50, // Longueur de séquence d'entrée
    name: 'embedding'
  }));
  
  // Couche LSTM
  model.add(tf.layers.lstm({
    units: 128,
    returnSequences: true,
    name: 'lstm_1'
  }));
  
  // Couche LSTM 2
  model.add(tf.layers.lstm({
    units: 128,
    returnSequences: false,
    name: 'lstm_2'
  }));
  
  // Couche Dense
  model.add(tf.layers.dense({
    units: 128,
    activation: 'relu',
    name: 'dense_1'
  }));
  
  // Couche de sortie
  model.add(tf.layers.dense({
    units: vocabSize,
    activation: 'softmax',
    name: 'output'
  }));
  
  // Compiler le modèle
  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });
  
  // Résumé du modèle
  model.summary();
  
  // Sauvegarder le modèle
  const modelPath = `file://${modelDir}`;
  await model.save(modelPath);
  
  console.log(`Modèle ${modelName} créé et sauvegardé dans ${modelDir}`);
  
  // Créer un fichier de métadonnées
  const metadata = {
    name: modelName,
    version: '1.0.0',
    description: 'Modèle TensorFlow.js simple pour simulation',
    vocabSize,
    embeddingDim,
    date: new Date().toISOString()
  };
  
  fs.writeFileSync(
    path.join(modelDir, 'metadata.json'),
    JSON.stringify(metadata, null, 2)
  );
  
  console.log(`Métadonnées du modèle créées`);
}

async function main() {
  try {
    // Créer des modèles pour chaque type
    await createSimpleModel('phi-3-mini', 10000, 128);
    await createSimpleModel('llama-3-8b', 12000, 256);
    await createSimpleModel('mistral-7b-fr', 12000, 256);
    
    console.log('Tous les modèles ont été créés avec succès!');
  } catch (error) {
    console.error('Erreur lors de la création des modèles:', error);
  }
}

main(); 