import { Injectable, OnModuleInit, Inject } from '@nestjs/common';
import { LOGGER_TOKEN, ILogger } from '../utils/logger-tokens';
import { TokenizerService } from './tokenizer.service';
import * as fs from 'fs';
import * as path from 'path';

/**
 * Service utilitaire pour la gestion des modèles TensorFlow
 * Ce service fournit des méthodes pour initialiser, charger et sauvegarder des modèles
 */
@Injectable()
export class ModelUtilsService implements OnModuleInit {
  private readonly modelDir = './models';
  private readonly vocabDir = './models/vocab';
  private readonly logger: ILogger;
  private tensorflowLoaded = false;
  private tf: any;

  constructor(
    @Inject(LOGGER_TOKEN) logger: ILogger,
    private readonly tokenizerService: TokenizerService
  ) {
    this.logger = logger;
  }

  /**
   * Initialisation du service
   */
  async onModuleInit() {
    this.logger.info('Initialisation du service de gestion des modèles');
    this.ensureDirectories();
    
    try {
      this.tf = require('@tensorflow/tfjs-node');
      this.tensorflowLoaded = true;
      this.logger.info('TensorFlow.js chargé avec succès');
    } catch (error) {
      this.logger.error('Erreur lors du chargement de TensorFlow.js', { error });
    }
  }

  /**
   * Vérifie et crée les répertoires nécessaires
   */
  private ensureDirectories() {
    try {
      if (!fs.existsSync(this.modelDir)) {
        fs.mkdirSync(this.modelDir, { recursive: true });
      }
      
      if (!fs.existsSync(this.vocabDir)) {
        fs.mkdirSync(this.vocabDir, { recursive: true });
      }
      
      this.logger.info('Répertoires des modèles créés avec succès');
    } catch (error) {
      this.logger.error('Erreur lors de la création des répertoires des modèles', { error });
    }
  }

  /**
   * Vérifie si un modèle existe
   * @param modelName Nom du modèle
   * @returns true si le modèle existe
   */
  modelExists(modelName: string): boolean {
    const modelPath = path.join(this.modelDir, modelName, 'model.json');
    return fs.existsSync(modelPath);
  }

  /**
   * Enregistre un vocabulaire ou tokenizer pour un modèle
   * @param modelName Nom du modèle
   * @param vocabulary Vocabulaire à enregistrer
   */
  saveVocabulary(modelName: string, vocabulary: Record<string, number>): void {
    try {
      const vocabPath = path.join(this.vocabDir, `${modelName}_vocab.json`);
      fs.writeFileSync(vocabPath, JSON.stringify(vocabulary, null, 2));
      this.logger.info(`Vocabulaire enregistré pour ${modelName}`);
    } catch (error) {
      this.logger.error(`Erreur lors de l'enregistrement du vocabulaire pour ${modelName}`, { error });
    }
  }

  /**
   * Charge un vocabulaire pour un modèle
   * @param modelName Nom du modèle
   * @returns Vocabulaire chargé ou null en cas d'erreur
   */
  loadVocabulary(modelName: string): Record<string, number> | null {
    try {
      const vocabPath = path.join(this.vocabDir, `${modelName}_vocab.json`);
      if (fs.existsSync(vocabPath)) {
        const vocabData = fs.readFileSync(vocabPath, 'utf8');
        return JSON.parse(vocabData);
      }
      
      this.logger.warn(`Aucun vocabulaire trouvé pour ${modelName}`);
      return null;
    } catch (error) {
      this.logger.error(`Erreur lors du chargement du vocabulaire pour ${modelName}`, { error });
      return null;
    }
  }

  /**
   * Crée un modèle simple pour les tests
   * @param modelName Nom du modèle
   * @param inputDim Dimension d'entrée
   * @param outputDim Dimension de sortie
   */
  async createSimpleModel(modelName: string, inputDim: number, outputDim: number): Promise<any> {
    if (!this.tensorflowLoaded) {
      throw new Error('TensorFlow.js non disponible');
    }
    
    this.logger.info(`Création d'un modèle simple pour ${modelName}`);
    
    try {
      // Créer un simple modèle séquentiel
      const model = this.tf.sequential();
      
      // Couche d'embedding
      model.add(this.tf.layers.embedding({
        inputDim: inputDim, // Taille du vocabulaire
        outputDim: 128,     // Dimension de l'embedding
        inputLength: 100    // Longueur de la séquence
      }));
      
      // Couche LSTM
      model.add(this.tf.layers.lstm({
        units: 64,
        returnSequences: false
      }));
      
      // Couche Dense
      model.add(this.tf.layers.dense({
        units: outputDim,
        activation: 'softmax'
      }));
      
      // Compiler le modèle
      model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
      });
      
      this.logger.info(`Modèle simple créé pour ${modelName}`);
      
      // Sauvegarde du modèle
      const modelSavePath = `file://${path.join(this.modelDir, modelName)}`;
      await model.save(modelSavePath);
      
      this.logger.info(`Modèle simple enregistré pour ${modelName} à ${modelSavePath}`);
      
      return model;
    } catch (error) {
      this.logger.error(`Erreur lors de la création du modèle simple pour ${modelName}`, { error });
      throw error;
    }
  }

  /**
   * Charge un modèle TensorFlow
   * @param modelName Nom du modèle
   * @returns Le modèle chargé ou null en cas d'erreur
   */
  async loadModel(modelName: string): Promise<any> {
    if (!this.tensorflowLoaded) {
      throw new Error('TensorFlow.js non disponible');
    }
    
    const modelPath = `file://${path.join(this.modelDir, modelName)}`;
    
    try {
      this.logger.info(`Chargement du modèle ${modelName} depuis ${modelPath}`);
      const model = await this.tf.loadLayersModel(modelPath);
      this.logger.info(`Modèle ${modelName} chargé avec succès`);
      
      return model;
    } catch (error) {
      this.logger.error(`Erreur lors du chargement du modèle ${modelName}`, { error });
      
      // Si le modèle n'existe pas, on retourne null (et non une exception)
      if (error.message && error.message.includes('Could not find model')) {
        return null;
      }
      
      throw error;
    }
  }

  /**
   * Retourne une référence à l'objet TensorFlow
   * @returns L'objet TensorFlow
   */
  getTensorflow() {
    if (!this.tensorflowLoaded) {
      throw new Error('TensorFlow.js non disponible');
    }
    
    return this.tf;
  }

  /**
   * Fonction utilitaire pour tokeniser un texte (version simplifiée)
   * @param text Texte à tokeniser
   * @param maxLength Longueur maximale 
   * @returns Tableau d'entiers représentant les tokens
   */
  tokenizeText(text: string, maxLength: number = 100): number[] {
    // Version simplifiée de tokenisation pour démonstration
    const tokens = Array.from(text)
      .map(c => c.charCodeAt(0) % 5000); // Utiliser les codes de caractères comme tokens
    
    // Tronquer si nécessaire
    if (tokens.length > maxLength) {
      return tokens.slice(0, maxLength);
    }
    
    // Paddé avec des zéros
    return [...tokens, ...Array(maxLength - tokens.length).fill(0)];
  }

  /**
   * Crée un jeu de données TensorFlow à partir d'exemples textuels
   * @param inputs Textes d'entrée
   * @param outputs Textes de sortie
   * @param modelName Nom du modèle
   * @param maxLength Longueur maximale des séquences
   * @returns Dataset TensorFlow
   */
  createDataset(inputs: string[], outputs: string[], modelName: string = 'phi-3-mini', maxLength: number = 100): any {
    if (!this.tensorflowLoaded) {
      throw new Error('TensorFlow.js non disponible');
    }
    
    this.logger.info(`Création du dataset pour ${modelName} avec ${inputs.length} exemples`);
    
    // Tokenisation avec le service dédié
    const tokenizedInputs = inputs.map(text => 
      this.tokenizerService.tokenize(text, modelName, { maxLength, padding: true, truncation: true })
    );
    
    const tokenizedOutputs = outputs.map(text => 
      this.tokenizerService.tokenize(text, modelName, { maxLength, padding: true, truncation: true })
    );
    
    // Création des tenseurs
    const inputTensor = this.tf.tensor2d(tokenizedInputs);
    const outputTensor = this.tf.tensor2d(tokenizedOutputs);
    
    // Création du dataset
    const dataset = this.tf.data.zip({
      xs: this.tf.data.array(tokenizedInputs),
      ys: this.tf.data.array(tokenizedOutputs)
    }).batch(32);
    
    return {
      dataset,
      inputTensor,
      outputTensor
    };
  }
  
  /**
   * Prédit une réponse à partir d'un prompt en utilisant un modèle chargé
   * @param model Modèle TensorFlow
   * @param prompt Prompt d'entrée
   * @param modelName Nom du modèle pour la tokenization
   * @param maxLength Longueur maximale de sortie
   * @returns Texte prédit
   */
  async predictText(model: any, prompt: string, modelName: string, maxLength: number = 100): Promise<string> {
    if (!this.tensorflowLoaded) {
      throw new Error('TensorFlow.js non disponible');
    }
    
    try {
      // Tokenizer le prompt
      const tokenizedPrompt = this.tokenizerService.tokenize(prompt, modelName, { 
        maxLength, 
        padding: true, 
        truncation: true 
      });
      
      // Créer un tenseur et prédire
      const inputTensor = this.tf.tensor2d([tokenizedPrompt]);
      const prediction = await model.predict(inputTensor);
      
      // Convertir la prédiction en tokens (prendre l'argmax pour chaque position)
      const predictedTokens = Array.from(
        await prediction.argMax(-1).array()
      )[0] as number[];
      
      // Détokenizer la prédiction
      const predictedText = this.tokenizerService.detokenize(predictedTokens, modelName);
      
      return predictedText;
    } catch (error) {
      this.logger.error(`Erreur lors de la prédiction avec ${modelName}`, { error });
      return `Erreur de prédiction: ${error.message}`;
    }
  }
} 