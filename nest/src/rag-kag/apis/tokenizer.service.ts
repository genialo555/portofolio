import { Injectable, OnModuleInit, Inject } from '@nestjs/common';
import { LOGGER_TOKEN, ILogger } from '../utils/logger-tokens';
import * as fs from 'fs';
import * as path from 'path';

interface Vocabulary {
  [token: string]: number;
}

interface TokenizerOptions {
  maxLength?: number;
  padding?: boolean;
  truncation?: boolean;
}

/**
 * Service de tokenization pour convertir du texte en séquences de tokens
 * Compatible avec TensorFlow.js
 */
@Injectable()
export class TokenizerService implements OnModuleInit {
  private readonly vocabDir = './models/vocab';
  private vocabularies: Map<string, Vocabulary> = new Map();
  private inverseVocabularies: Map<string, Map<number, string>> = new Map();
  
  constructor(@Inject(LOGGER_TOKEN) private readonly logger: ILogger) {}
  
  async onModuleInit() {
    this.logger.info('Initialisation du service de tokenization');
    this.ensureDirectories();
    await this.loadDefaultVocabularies();
  }
  
  /**
   * Crée les répertoires nécessaires
   */
  private ensureDirectories() {
    try {
      if (!fs.existsSync(this.vocabDir)) {
        fs.mkdirSync(this.vocabDir, { recursive: true });
      }
      this.logger.info('Répertoire de vocabulaire créé avec succès');
    } catch (error) {
      this.logger.error('Erreur lors de la création du répertoire de vocabulaire', { error });
    }
  }
  
  /**
   * Charge les vocabulaires par défaut pour les modèles
   */
  private async loadDefaultVocabularies() {
    // Si les vocabulaires n'existent pas, on crée des vocabulaires simples pour démonstration
    const defaultModels = ['phi-3-mini', 'llama-3-8b', 'mistral-7b-fr', 'deepseek-r1'];
    
    for (const model of defaultModels) {
      const vocabPath = path.join(this.vocabDir, `${model}_vocab.json`);
      
      if (!fs.existsSync(vocabPath)) {
        this.logger.info(`Création d'un vocabulaire par défaut pour ${model}`);
        // Créer un vocabulaire simple pour démonstration
        const vocab = this.createSimpleVocabulary();
        await this.saveVocabulary(model, vocab);
      } else {
        // Charger le vocabulaire existant
        await this.loadVocabulary(model);
      }
    }
  }
  
  /**
   * Crée un vocabulaire simple basé sur les caractères ASCII
   * Pour démonstration uniquement
   */
  private createSimpleVocabulary(): Vocabulary {
    const vocab: Vocabulary = {};
    
    // Caractères spéciaux
    vocab['[PAD]'] = 0;
    vocab['[UNK]'] = 1;
    vocab['[BOS]'] = 2;
    vocab['[EOS]'] = 3;
    
    // Caractères ASCII de base
    for (let i = 32; i <= 126; i++) {
      vocab[String.fromCharCode(i)] = i - 28; // Décalage pour commencer après les tokens spéciaux
    }
    
    // Ajout de quelques caractères accentués et étendus pour le français
    const extendedChars = 'àáâäæçèéêëìíîïñòóôöùúûüÿÀÁÂÄÆÇÈÉÊËÌÍÎÏÑÒÓÔÖÙÚÛÜŸ';
    let id = 127 - 28;
    
    for (const char of extendedChars) {
      vocab[char] = id++;
    }
    
    return vocab;
  }
  
  /**
   * Sauvegarde un vocabulaire
   */
  async saveVocabulary(modelName: string, vocabulary: Vocabulary): Promise<void> {
    try {
      const vocabPath = path.join(this.vocabDir, `${modelName}_vocab.json`);
      fs.writeFileSync(vocabPath, JSON.stringify(vocabulary, null, 2));
      
      // Mettre à jour les caches
      this.vocabularies.set(modelName, vocabulary);
      
      // Construire et mettre en cache le vocabulaire inversé
      const inverseVocab = new Map<number, string>();
      for (const [token, id] of Object.entries(vocabulary)) {
        inverseVocab.set(id, token);
      }
      this.inverseVocabularies.set(modelName, inverseVocab);
      
      this.logger.info(`Vocabulaire sauvegardé pour ${modelName}`);
    } catch (error) {
      this.logger.error(`Erreur lors de la sauvegarde du vocabulaire pour ${modelName}`, { error });
      throw error;
    }
  }
  
  /**
   * Charge un vocabulaire
   */
  async loadVocabulary(modelName: string): Promise<Vocabulary> {
    try {
      const vocabPath = path.join(this.vocabDir, `${modelName}_vocab.json`);
      if (!fs.existsSync(vocabPath)) {
        throw new Error(`Le vocabulaire pour ${modelName} n'existe pas`);
      }
      
      const vocabData = fs.readFileSync(vocabPath, 'utf8');
      const vocabulary = JSON.parse(vocabData) as Vocabulary;
      
      // Mettre à jour les caches
      this.vocabularies.set(modelName, vocabulary);
      
      // Construire et mettre en cache le vocabulaire inversé
      const inverseVocab = new Map<number, string>();
      for (const [token, id] of Object.entries(vocabulary)) {
        inverseVocab.set(id, token);
      }
      this.inverseVocabularies.set(modelName, inverseVocab);
      
      this.logger.info(`Vocabulaire chargé pour ${modelName}`);
      return vocabulary;
    } catch (error) {
      this.logger.error(`Erreur lors du chargement du vocabulaire pour ${modelName}`, { error });
      throw error;
    }
  }
  
  /**
   * Tokenize un texte
   * @param text Texte à tokenizer
   * @param modelName Nom du modèle dont le vocabulaire sera utilisé
   * @param options Options de tokenization
   */
  tokenize(text: string, modelName: string, options: TokenizerOptions = {}): number[] {
    const {
      maxLength = 512,
      padding = true, 
      truncation = true
    } = options;
    
    try {
      // Vérifier si le vocabulaire est disponible
      if (!this.vocabularies.has(modelName)) {
        this.logger.warn(`Vocabulaire non chargé pour ${modelName}, chargement à la volée`);
        // On crée un vocabulaire par défaut
        const vocab = this.createSimpleVocabulary();
        this.vocabularies.set(modelName, vocab);
        
        const inverseVocab = new Map<number, string>();
        for (const [token, id] of Object.entries(vocab)) {
          inverseVocab.set(id, token);
        }
        this.inverseVocabularies.set(modelName, inverseVocab);
      }
      
      const vocabulary = this.vocabularies.get(modelName);
      const tokens: number[] = [];
      
      // Ajout du token de début si nécessaire
      if (vocabulary['[BOS]'] !== undefined) {
        tokens.push(vocabulary['[BOS]']);
      }
      
      // Tokenization caractère par caractère (simplifié pour démonstration)
      for (const char of text) {
        if (vocabulary[char] !== undefined) {
          tokens.push(vocabulary[char]);
        } else {
          // Token inconnu
          tokens.push(vocabulary['[UNK]'] || 1);
        }
      }
      
      // Ajout du token de fin si nécessaire
      if (vocabulary['[EOS]'] !== undefined) {
        tokens.push(vocabulary['[EOS]']);
      }
      
      // Tronquer si nécessaire
      let result = tokens;
      if (truncation && tokens.length > maxLength) {
        result = tokens.slice(0, maxLength);
        
        // S'assurer que le dernier token est EOS si disponible
        if (vocabulary['[EOS]'] !== undefined) {
          result[maxLength - 1] = vocabulary['[EOS]'];
        }
      }
      
      // Padding si nécessaire
      if (padding && result.length < maxLength) {
        const padToken = vocabulary['[PAD]'] || 0;
        result = [...result, ...Array(maxLength - result.length).fill(padToken)];
      }
      
      return result;
    } catch (error) {
      this.logger.error(`Erreur lors de la tokenization pour ${modelName}`, { error });
      // En cas d'erreur, retourner une séquence vide ou paddée
      return Array(maxLength).fill(0);
    }
  }
  
  /**
   * Détokenize une séquence de tokens
   */
  detokenize(tokens: number[], modelName: string): string {
    try {
      // Vérifier si le vocabulaire inversé est disponible
      if (!this.inverseVocabularies.has(modelName)) {
        this.logger.warn(`Vocabulaire inversé non disponible pour ${modelName}`);
        throw new Error(`Vocabulaire inversé non disponible pour ${modelName}`);
      }
      
      const inverseVocab = this.inverseVocabularies.get(modelName);
      let text = '';
      
      // Filtrer les tokens spéciaux
      const specialTokens = new Set([0, 2, 3]); // PAD, BOS, EOS
      
      for (const token of tokens) {
        // Ignorer les tokens spéciaux
        if (specialTokens.has(token)) {
          continue;
        }
        
        // Récupérer le caractère correspondant
        const char = inverseVocab.get(token);
        if (char !== undefined) {
          text += char;
        } else {
          // Token inconnu
          text += '[?]';
        }
      }
      
      return text;
    } catch (error) {
      this.logger.error(`Erreur lors de la détokenization pour ${modelName}`, { error });
      return '[ERREUR DE DETOKENIZATION]';
    }
  }
} 