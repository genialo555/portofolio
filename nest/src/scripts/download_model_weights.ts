import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'fs';
import * as path from 'path';

const models = [
  {
    name: 'phi-3-mini',
    url: 'https://huggingface.co/microsoft/phi-3-mini-4k-instruct/resolve/main/model.json',
  },
  {
    name: 'llama-3-8b',
    url: 'https://huggingface.co/meta-llama/Llama-3-8B-Instruct/resolve/main/model.json',
  },
  {
    name: 'mistral-7b-fr',
    url: 'https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/main/model.json',
  },
  {
    name: 'deepseek-r1',
    url: 'https://huggingface.co/deepseek-ai/deepseek-v2/resolve/main/model.json',
  },
];

async function downloadModel(modelName: string, modelUrl: string) {
  const localModelPath = `file://${path.join(process.cwd(), 'models', modelName)}`;

  try {
    const model = await tf.loadGraphModel(modelUrl);
    await model.save(localModelPath);
    console.log(`Model ${modelName} downloaded and saved to ${localModelPath}`);
  } catch (err) {
    console.error(`Failed to download model ${modelName}:`, err);
  }
}

async function main() {
  for (const model of models) {
    await downloadModel(model.name, model.url);
  }
}

main(); 