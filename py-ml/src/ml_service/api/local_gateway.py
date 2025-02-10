from typing import Dict, Any
import torch
from pathlib import Path
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import bitsandbytes as bnb

class LocalModelGateway:
    """Passerelle pour les modèles en local optimisés pour RTX 4050 (6GB VRAM)."""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_memory = 6  # 6GB VRAM sur RTX 4050
        
        # Configurations optimisées pour la latence
        self.model_configs = {
            'phi': {  # Le plus rapide
                'path': 'microsoft/phi-2',
                'quantization': None,  # FP16 natif
                'max_batch': 24,
                'gpu_layers': 'all',
                'expected_latency': '100-200ms'  # Très rapide
            },
            'mistral-small': {  # Version rapide de Mistral
                'path': 'mistralai/Mistral-7B-Instruct-v0.1',
                'quantization': '8bit',
                'max_batch': 1,  # Réduit pour la latence
                'gpu_layers': 32,
                'expected_latency': '200-400ms'
            },
            'codellama-small': {  # Version optimisée pour le code
                'path': 'codellama/CodeLlama-7b-Instruct-hf',
                'quantization': '8bit',
                'max_batch': 1,
                'gpu_layers': 32,
                'expected_latency': '200-400ms'
            }
        }
        
        # Optimisations CPU/RAM
        self.cpu_threads = os.cpu_count()
        self.ram_limit = int(32 * 0.8)  # On peut utiliser 80% de la RAM
        
        # Activation du mode faible latence
        self.low_latency_mode = True
        
    async def load_model(self, model_name: str):
        """Charge un modèle avec les optimisations RTX 4050 6GB."""
        config = self.model_configs[model_name]
        
        if model_name == 'phi':
            return await self._load_phi_optimized()
        elif model_name == 'codellama':
            return await self._load_codellama()
        else:
            return await self._load_quantized_model(config)
            
    async def _load_phi_optimized(self):
        """Charge Phi-2 en FP16 complet sur GPU."""
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",
            torch_dtype=torch.float16,
            device_map="auto",
            use_flash_attention_2=True  # Optimisation mémoire
        )
        return model
        
    async def _load_codellama(self):
        """Charge CodeLlama optimisé pour la programmation."""
        model = AutoModelForCausalLM.from_pretrained(
            "codellama/CodeLlama-7b-Instruct-hf",
            load_in_8bit=True,
            device_map="auto",
            use_flash_attention_2=True
        )
        return model
        
    async def _load_quantized_model(self, config: Dict[str, Any]):
        """Charge un modèle avec quantification optimisée."""
        model = AutoModelForCausalLM.from_pretrained(
            config['path'],
            load_in_8bit=config['quantization'] == '8bit',
            load_in_4bit=config['quantization'] == '4bit',
            device_map="auto",
            use_flash_attention_2=True,
            torch_dtype=torch.float16
        )
        return model
        
    def optimize_inference(self, model: AutoModelForCausalLM):
        """Optimise le modèle pour l'inférence."""
        if torch.cuda.is_available():
            # Activation des optimisations CUDA
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Optimisation mémoire
            model.config.use_cache = True
            
        return model 

    def optimize_for_latency(self, model: AutoModelForCausalLM):
        """Optimisations spécifiques pour réduire la latence."""
        if self.low_latency_mode:
            # Optimisations CUDA
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True  # Auto-tuning
            
            # Réduction de la précision pour la vitesse
            if hasattr(model, 'half'):
                model = model.half()
                
            # Désactivation des calculs non nécessaires
            model.config.use_cache = True
            
            # Optimisation de la mémoire tampon
            torch.cuda.empty_cache()
            
            # Préchauffage du modèle
            self._warmup_model(model)
            
        return model
        
    def _warmup_model(self, model: AutoModelForCausalLM):
        """Préchauffage du modèle pour réduire la latence initiale."""
        dummy_input = torch.zeros(1, 32).long().to(self.device)
        with torch.no_grad():
            model(dummy_input) 