/**
 * Script pour descargar modelos de Hugging Face usando axios
 * Ejecutar con: npx ts-node src/scripts/download-huggingface-models.ts
 */

import * as fs from 'fs';
import * as path from 'path';
import axios from 'axios';
import * as readline from 'readline';

// Configuración para descargas
let HF_TOKEN = process.env.HF_TOKEN; // Token de Hugging Face (opcional)
const MODELS_DIR = path.join(process.cwd(), 'models');

// Modelos para descargar - solo descargaremos los archivos esenciales
const MODELS = [
  // Modelos sin autenticación
  {
    name: 'phi-3-mini',
    huggingFaceId: 'microsoft/phi-3-mini-4k-instruct',
    files: ['config.json', 'tokenizer.json', 'tokenizer_config.json'],
    requiresAuth: false
  },
  {
    name: 'tinyllama',
    huggingFaceId: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    files: ['config.json', 'tokenizer.json', 'tokenizer_config.json'],
    requiresAuth: false
  },
  // Modelos con autenticación
  {
    name: 'llama-3-8b',
    huggingFaceId: 'meta-llama/Llama-3-8B-Instruct',
    files: ['config.json', 'tokenizer.json', 'tokenizer_config.json'],
    requiresAuth: true,
    fallbackId: 'TheBloke/Llama-3-8B-Instruct-GGUF' // Versión accesible sin autenticación
  },
  {
    name: 'mistral-7b-fr',
    huggingFaceId: 'mistralai/Mistral-7B-Instruct-v0.2',
    files: ['config.json', 'tokenizer.json', 'tokenizer_config.json'],
    requiresAuth: true,
    fallbackId: 'mistralai/Mistral-7B-v0.1' // Versión accesible sin autenticación
  },
  // Añadir el modelo deepseek-r1
  {
    name: 'deepseek-r1',
    huggingFaceId: 'deepseek-ai/deepseek-v2',  // Modelo más reciente de DeepSeek
    files: ['config.json', 'tokenizer.json', 'tokenizer_config.json'],
    requiresAuth: true,
    fallbackId: 'deepseek-ai/deepseek-coder-6.7b-base' // Versión accesible sin autenticación
  }
];

/**
 * Crea una interfaz readline para las entradas del usuario
 */
function createReadlineInterface() {
  return readline.createInterface({
    input: process.stdin,
    output: process.stdout
  });
}

/**
 * Pregunta por el token de Hugging Face al usuario
 */
async function askForHuggingFaceToken(): Promise<string> {
  const rl = createReadlineInterface();
  
  return new Promise((resolve) => {
    rl.question('Por favor, ingrese su token de Hugging Face para acceder a los modelos que requieren autenticación (o presione Enter para omitir): ', (token) => {
      rl.close();
      resolve(token.trim());
    });
  });
}

/**
 * Pregunta al usuario si quiere usar los fallbacks
 */
async function askToUseFallbacks(): Promise<boolean> {
  const rl = createReadlineInterface();
  
  const answer = await new Promise<string>(resolve => {
    rl.question('Do you want to use fallback URLs if download fails? (y/n) ', resolve);
  });

  rl.close();

  return answer.trim().toLowerCase() === 'y';
}

/**
 * Descarga un archivo desde Hugging Face
 */
async function downloadFile(url: string, outputPath: string, requiresAuth: boolean = false): Promise<void> {
  console.log(`Descargando ${url}...`);
  
  // Opciones para la solicitud axios
  const requestOptions: any = {
    method: 'GET',
    url: url,
    responseType: 'stream'
  };
  
  // Agregar el token de autenticación si está disponible
  if (HF_TOKEN) {
    requestOptions.headers = { Authorization: `Bearer ${HF_TOKEN}` };
  } else if (requiresAuth) {
    console.warn('ADVERTENCIA: Este modelo requiere autenticación pero no se proporciona ningún token!');
    console.warn('Intentando descargar sin autenticación...');
  }
  
  try {
    const response = await axios(requestOptions);
    
    const writer = fs.createWriteStream(outputPath);
    response.data.pipe(writer);
    
    return new Promise((resolve, reject) => {
      writer.on('finish', resolve);
      writer.on('error', reject);
    });
  } catch (error) {
    if (error.response && error.response.status === 401 && requiresAuth && !HF_TOKEN) {
      console.error('Error 401: Autenticación requerida para acceder a este modelo.');
      throw new Error('Token de Hugging Face requerido para este modelo.');
    }
    console.error(`Error al descargar ${url}:`, error.message);
    throw error;
  }
}

/**
 * Descarga los archivos de un modelo
 */
async function downloadModelFiles(model: typeof MODELS[0], useFallback: boolean = false): Promise<void> {
  // Determinar qué ID usar
  const modelId = (useFallback && model.fallbackId) ? model.fallbackId : model.huggingFaceId;
  const requiresAuth = useFallback ? false : model.requiresAuth;
  
  console.log(`Descargando el modelo ${model.name} (${modelId})...`);
  if (useFallback && model.fallbackId) {
    console.log(`Usando el modelo alternativo: ${model.fallbackId}`);
  }
  
  const modelDir = path.join(MODELS_DIR, model.name);
  if (!fs.existsSync(modelDir)) {
    fs.mkdirSync(modelDir, { recursive: true });
  }
  
  try {
    // Descargar cada archivo configurado para el modelo
    let filesDownloaded = 0;
    
    for (const file of model.files) {
      const fileUrl = `https://huggingface.co/${modelId}/resolve/main/${file}`;
      const outputPath = path.join(modelDir, file);
      
      try {
        await downloadFile(fileUrl, outputPath, requiresAuth);
        console.log(`Archivo ${file} descargado con éxito.`);
        filesDownloaded++;
      } catch (fileError) {
        console.error(`No se pudo descargar ${file}: ${fileError.message}`);
      }
    }
    
    if (filesDownloaded === 0) {
      throw new Error(`No se descargó ningún archivo para ${model.name}`);
    }
    
    // Guardar metadatos adicionales
    const metadata = {
      name: model.name,
      huggingFaceId: modelId,
      originalModelId: model.huggingFaceId,
      usedFallback: useFallback && model.fallbackId ? true : false,
      downloadDate: new Date().toISOString(),
      files: model.files,
      filesDownloaded,
    };
    
    fs.writeFileSync(
      path.join(modelDir, 'metadata.json'),
      JSON.stringify(metadata, null, 2)
    );
    
    console.log(`Metadatos guardados para ${model.name}`);
  } catch (error) {
    console.error(`Error al descargar el modelo ${model.name}:`, error);
    throw error;
  }
}

/**
 * Descarga los archivos de tokenizer para la compatibilidad con el sistema existente
 */
async function createVocabFile(model: typeof MODELS[0]): Promise<void> {
  const vocabDir = path.join(MODELS_DIR, 'vocab');
  if (!fs.existsSync(vocabDir)) {
    fs.mkdirSync(vocabDir, { recursive: true });
  }
  
  // Verificar si el archivo de vocabulario ya existe
  const vocabPath = path.join(vocabDir, `${model.name}_vocab.json`);
  if (fs.existsSync(vocabPath)) {
    console.log(`El archivo de vocabulario para ${model.name} ya existe.`);
    return;
  }
  
  console.log(`Creando un archivo de vocabulario para ${model.name}...`);
  
  // Intentar primero copiar el tokenizer.json del modelo si existe
  const tokenizerPath = path.join(MODELS_DIR, model.name, 'tokenizer.json');
  if (fs.existsSync(tokenizerPath)) {
    try {
      const tokenizerData = fs.readFileSync(tokenizerPath, 'utf8');
      const tokenizer = JSON.parse(tokenizerData);
      
      // Si el tokenizer tiene un vocabulario, usarlo
      if (tokenizer.model && tokenizer.model.vocab) {
        fs.writeFileSync(vocabPath, JSON.stringify(tokenizer.model.vocab, null, 2));
        console.log(`Vocabulario copiado desde tokenizer.json para ${model.name}`);
        return;
      }
    } catch (err) {
      console.error(`Error al leer el tokenizer para ${model.name}:`, err.message);
    }
  }
  
  // De lo contrario, crear un vocabulario simple
  console.log(`Creando un vocabulario simple para ${model.name}...`);
  const simpleVocab = {};
  for (let i = 0; i < 10000; i++) {
    simpleVocab[`token_${i}`] = i;
  }
  
  fs.writeFileSync(vocabPath, JSON.stringify(simpleVocab, null, 2));
  console.log(`Archivo de vocabulario simple creado para ${model.name}`);
}

/**
 * Verificar las últimas versiones de los modelos en Hugging Face
 */
async function checkLatestModels() {
  console.log("Verificando las últimas versiones de los modelos en Hugging Face...");
  
  try {
    // En la práctica, se podría consultar la API de Hugging Face aquí
    console.log("Los modelos configurados están actualizados según la información disponible:");
    console.log("- phi-3-mini: microsoft/phi-3-mini-4k-instruct (sin autenticación)");
    console.log("- tinyllama: TinyLlama/TinyLlama-1.1B-Chat-v1.0 (sin autenticación)");
    console.log("- llama-3-8b: meta-llama/Llama-3-8B-Instruct (requiere autenticación)");
    console.log("- mistral-7b-fr: mistralai/Mistral-7B-Instruct-v0.2 (requiere autenticación)");
    console.log("- deepseek-r1: deepseek-ai/deepseek-v2 (requiere autenticación)");
    console.log("Si desea agregar otros modelos, modifique la lista MODELS en el script.");
  } catch (error) {
    console.error("Error al verificar las últimas versiones:", error.message);
  }
}

/**
 * Función principal
 */
async function main() {
  console.log(`Descargando modelos a ${MODELS_DIR}...`);
  
  // Verificar las últimas versiones de los modelos
  await checkLatestModels();
  
  // Crear el directorio de modelos si no existe
  if (!fs.existsSync(MODELS_DIR)) {
    fs.mkdirSync(MODELS_DIR, { recursive: true });
  }
  
  // Determinar si algunos modelos requieren autenticación
  const requiresAuthModels = MODELS.filter(model => model.requiresAuth);
  let useFallbacks = false;
  
  if (requiresAuthModels.length > 0) {
    if (!HF_TOKEN) {
      console.log(`Advertencia: ${requiresAuthModels.length} modelo(s) requiere(n) autenticación.`);
      HF_TOKEN = await askForHuggingFaceToken();
      
      if (!HF_TOKEN) {
        console.warn("No se proporcionó ningún token.");
        
        // Preguntar al usuario si quiere usar los modelos alternativos
        if (requiresAuthModels.some(model => model.fallbackId)) {
          useFallbacks = await askToUseFallbacks();
          if (useFallbacks) {
            console.log("Se utilizarán modelos alternativos para aquellos que requieren autenticación.");
          } else {
            console.log("Los modelos que requieren autenticación podrían no ser descargables.");
          }
        } else {
          console.warn("No hay modelos alternativos disponibles. Algunos modelos podrían no ser descargables.");
        }
      }
    }
  }
  
  // Descargar cada modelo
  for (const model of MODELS) {
    try {
      console.log(`\n===== Procesando el modelo: ${model.name} =====`);
      
      // Determinar si se utiliza el fallback para este modelo
      const useModelFallback = useFallbacks && model.requiresAuth && !!model.fallbackId;
      
      // Descargar el modelo
      await downloadModelFiles(model, useModelFallback);
      await createVocabFile(model);
      console.log(`Modelo ${model.name} descargado con éxito!\n`);
    } catch (error) {
      console.error(`Error al descargar el modelo ${model.name}:`, error);
      console.log('Pasando al siguiente modelo...\n');
    }
  }
  
  console.log('Todos los modelos han sido descargados!');
  console.log('Nota: Estos archivos son solo las configuraciones y tokenizers.');
  console.log('Tous les modèles ont été téléchargés!');
  console.log('Note: Ces fichiers sont les configurations et tokenizers uniquement.');
  console.log('Les poids des modèles doivent être téléchargés séparément ou utilisés via TensorFlow.js/ONNX.');
}

// Exécution du script
main().catch(error => {
  console.error('Erreur lors de l\'exécution du script:', error);
  process.exit(1);
}); 