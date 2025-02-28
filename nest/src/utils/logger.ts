/**
 * Niveaux de journalisation supportés
 */
export enum LogLevel {
  ERROR = 0,
  WARN = 1,
  INFO = 2,
  DEBUG = 3,
  TRACE = 4
}

/**
 * Configuration du logger
 */
export interface LoggerConfig {
  level: LogLevel;                  // Niveau de journalisation
  colorize?: boolean;               // Utiliser la coloration ANSI
  timestamp?: boolean;              // Inclure un timestamp
  includeTraceId?: boolean;         // Inclure l'ID de trace pour le suivi
  outputToConsole?: boolean;        // Afficher dans la console
  outputToFile?: boolean;           // Enregistrer dans un fichier
  logFilePath?: string;             // Chemin du fichier de log
  maxFileSizeMB?: number;           // Taille maximale du fichier avant rotation
  maxFiles?: number;                // Nombre maximal de fichiers de rotation
  formatFn?: (entry: LogEntry) => string; // Fonction personnalisée de formatage
  tag?: string;                     // Tag pour identifier la source des logs
}

/**
 * Entrée de log
 */
export interface LogEntry {
  timestamp: Date;                  // Timestamp de l'entrée
  level: LogLevel;                  // Niveau de journalisation
  message: string;                  // Message de log
  traceId?: string;                 // ID de trace (pour le suivi)
  metadata?: Record<string, any>;   // Métadonnées supplémentaires
}

/**
 * Classe de journalisation pour le système de coordination
 */
export class Logger {
  private config: LoggerConfig;
  private levelNames: Record<LogLevel, string> = {
    [LogLevel.ERROR]: 'ERROR',
    [LogLevel.WARN]: 'WARN',
    [LogLevel.INFO]: 'INFO',
    [LogLevel.DEBUG]: 'DEBUG',
    [LogLevel.TRACE]: 'TRACE'
  };
  
  private colors: Record<LogLevel, string> = {
    [LogLevel.ERROR]: '\x1b[31m', // Rouge
    [LogLevel.WARN]: '\x1b[33m',  // Jaune
    [LogLevel.INFO]: '\x1b[36m',  // Cyan
    [LogLevel.DEBUG]: '\x1b[35m', // Magenta
    [LogLevel.TRACE]: '\x1b[90m'  // Gris
  };
  
  private resetColor: string = '\x1b[0m';
  
  /**
   * Crée une nouvelle instance du logger
   * @param config Configuration du logger
   */
  constructor(config: Partial<LoggerConfig> = {}) {
    // Configuration par défaut
    this.config = {
      level: LogLevel.INFO,
      colorize: true,
      timestamp: true,
      includeTraceId: true,
      outputToConsole: true,
      outputToFile: false,
      logFilePath: './logs/coordination.log',
      maxFileSizeMB: 10,
      maxFiles: 5,
      ...config
    };
  }
  
  /**
   * Journal un message au niveau ERROR
   * @param message Message à journaliser
   * @param metadata Métadonnées additionnelles
   * @param traceId ID de trace optionnel
   */
  public error(message: string, metadata?: Record<string, any>, traceId?: string): void {
    this.log(LogLevel.ERROR, message, metadata, traceId);
  }
  
  /**
   * Journal un message au niveau WARN
   * @param message Message à journaliser
   * @param metadata Métadonnées additionnelles
   * @param traceId ID de trace optionnel
   */
  public warn(message: string, metadata?: Record<string, any>, traceId?: string): void {
    this.log(LogLevel.WARN, message, metadata, traceId);
  }
  
  /**
   * Journal un message au niveau INFO
   * @param message Message à journaliser
   * @param metadata Métadonnées additionnelles
   * @param traceId ID de trace optionnel
   */
  public info(message: string, metadata?: Record<string, any>, traceId?: string): void {
    this.log(LogLevel.INFO, message, metadata, traceId);
  }
  
  /**
   * Journal un message au niveau DEBUG
   * @param message Message à journaliser
   * @param metadata Métadonnées additionnelles
   * @param traceId ID de trace optionnel
   */
  public debug(message: string, metadata?: Record<string, any>, traceId?: string): void {
    this.log(LogLevel.DEBUG, message, metadata, traceId);
  }
  
  /**
   * Journal un message au niveau TRACE
   * @param message Message à journaliser
   * @param metadata Métadonnées additionnelles
   * @param traceId ID de trace optionnel
   */
  public trace(message: string, metadata?: Record<string, any>, traceId?: string): void {
    this.log(LogLevel.TRACE, message, metadata, traceId);
  }
  
  /**
   * Modifie le niveau de journalisation
   * @param level Nouveau niveau de journalisation
   */
  public setLevel(level: LogLevel): void {
    this.config.level = level;
  }
  
  /**
   * Journal un message à un niveau spécifique
   * @param level Niveau de journalisation
   * @param message Message à journaliser
   * @param metadata Métadonnées additionnelles
   * @param traceId ID de trace optionnel
   */
  private log(level: LogLevel, message: string, metadata?: Record<string, any>, traceId?: string): void {
    // Vérifier si ce niveau doit être journalisé
    if (level > this.config.level) {
      return;
    }
    
    const entry: LogEntry = {
      timestamp: new Date(),
      level,
      message,
      traceId: traceId || this.extractTraceId(message),
      metadata
    };
    
    // Formatage et sortie
    const formattedMessage = this.formatLogEntry(entry);
    this.output(formattedMessage, entry);
  }
  
  /**
   * Extrait l'ID de trace d'un message si présent
   * Recherche les formats courants comme [Coordination:abc-123]
   * @param message Message à analyser
   * @returns ID de trace si trouvé
   */
  private extractTraceId(message: string): string | undefined {
    const match = message.match(/\[([^:]+):([^\]]+)\]/);
    return match ? match[2] : undefined;
  }
  
  /**
   * Formate une entrée de log selon la configuration
   * @param entry Entrée à formater
   * @returns Message formaté
   */
  private formatLogEntry(entry: LogEntry): string {
    // Utiliser un formateur personnalisé si disponible
    if (this.config.formatFn) {
      return this.config.formatFn(entry);
    }
    
    const parts: string[] = [];
    
    // Timestamp
    if (this.config.timestamp) {
      const timestamp = entry.timestamp.toISOString();
      parts.push(`[${timestamp}]`);
    }
    
    // Niveau de log
    const levelName = this.levelNames[entry.level] || 'UNKNOWN';
    const formattedLevel = this.config.colorize 
      ? `${this.colors[entry.level]}${levelName.padEnd(5, ' ')}${this.resetColor}` 
      : levelName.padEnd(5, ' ');
    parts.push(formattedLevel);
    
    // ID de trace
    if (this.config.includeTraceId && entry.traceId) {
      parts.push(`[${entry.traceId}]`);
    }
    
    // Message principal
    parts.push(entry.message);
    
    // Métadonnées
    if (entry.metadata && Object.keys(entry.metadata).length > 0) {
      try {
        const metadataString = JSON.stringify(entry.metadata);
        parts.push(metadataString);
      } catch (e) {
        parts.push(`[Métadonnées non sérialisables]`);
      }
    }
    
    return parts.join(' ');
  }
  
  /**
   * Envoie le message formaté vers les destinations configurées
   * @param formattedMessage Message formaté
   * @param entry Entrée de log originale
   */
  private output(formattedMessage: string, entry: LogEntry): void {
    // Sortie console
    if (this.config.outputToConsole) {
      if (entry.level <= LogLevel.ERROR) {
        console.error(formattedMessage);
      } else if (entry.level === LogLevel.WARN) {
        console.warn(formattedMessage);
      } else {
        console.log(formattedMessage);
      }
    }
    
    // Sortie fichier (implémentation simplifiée)
    if (this.config.outputToFile && this.config.logFilePath) {
      // Dans une implémentation réelle, vous utiliseriez un module comme 'winston'
      // pour gérer la rotation des fichiers et le formatage avancé
      // Ici, nous simulons juste le comportement
      
      // Note: remplacer ce bloc par l'implémentation réelle
      // de journalisation dans des fichiers si nécessaire
      this.simulateFileLogging(formattedMessage, entry);
    }
  }
  
  /**
   * Simule la journalisation dans un fichier (à remplacer par l'implémentation réelle)
   * @param formattedMessage Message formaté
   * @param entry Entrée de log originale
   */
  private simulateFileLogging(formattedMessage: string, entry: LogEntry): void {
    // Cette méthode est un placeholder
    // Dans une vraie implémentation, vous écririez dans un fichier,
    // avec gestion de la rotation des logs, etc.
    
    // console.log(`[FILE LOG SIMULATION] Would write to ${this.config.logFilePath}: ${formattedMessage}`);
  }
} 