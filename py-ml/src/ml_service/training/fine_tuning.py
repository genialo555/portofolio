from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
from peft import (
    LoraConfig, 
    get_peft_model,
    prepare_model_for_kbit_training
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
import datasets
from accelerate import Accelerator

@dataclass
class FineTuningConfig:
    """Configuration pour le fine-tuning des agents."""
    
    base_models = {
        'teacher': 'Qwen/Qwen2.5-7B-Chat',
        'debate': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
        'specialist': 'codellama/CodeLlama-34b-Instruct-hf'
    }
    
    lora_config = {
        'r': 64,  # Rang de l'adaptation
        'lora_alpha': 32,
        'target_modules': ['q_proj', 'v_proj'],
        'bias': 'none',
        'task_type': 'CAUSAL_LM'
    }
    
    training_args = {
        'per_device_train_batch_size': 2,
        'gradient_accumulation_steps': 4,
        'warmup_steps': 100,
        'max_steps': 1000,
        'learning_rate': 2e-4,
        'fp16': True,
        'logging_steps': 10,
        'output_dir': 'checkpoints',
        'optim': 'adamw_torch'
    }

class ModelFineTuner:
    """Gestionnaire de fine-tuning pour les agents."""
    
    def __init__(self, dataset_path: str):
        self.config = FineTuningConfig()
        self.accelerator = Accelerator()
        self.dataset = datasets.load_from_disk(dataset_path)
        
    async def prepare_model(self, model_key: str):
        """Prépare un modèle pour le fine-tuning."""
        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_models[model_key],
            load_in_4bit=True,
            device_map='auto',
            torch_dtype=torch.float16
        )
        
        # Préparation pour QLoRA
        model = prepare_model_for_kbit_training(model)
        
        # Configuration LoRA
        lora_config = LoraConfig(
            **self.config.lora_config
        )
        
        # Application de LoRA
        model = get_peft_model(model, lora_config)
        
        return model
        
    async def train(self, model_key: str, subset_filter: Optional[Dict] = None):
        """Lance le fine-tuning d'un modèle."""
        model = await self.prepare_model(model_key)
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_models[model_key]
        )
        
        # Filtrage du dataset si nécessaire
        train_dataset = self.dataset
        if subset_filter:
            train_dataset = train_dataset.filter(
                lambda x: all(x[k] == v for k, v in subset_filter.items())
            )
            
        # Configuration de l'entraînement
        training_args = TrainingArguments(
            **self.config.training_args
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer
        )
        
        # Lancement de l'entraînement
        trainer.train()
        
        # Sauvegarde du modèle
        await self.save_model(model, model_key) 