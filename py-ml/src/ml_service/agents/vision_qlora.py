#!/usr/bin/env python3
"""
Module d'adaptation et fine-tuning de modèles Gemma avec QLoRA pour les tâches de vision.
Optimisé pour Apple Silicon (M1, M2, M3, M4).

QLoRA (Quantized Low-Rank Adaptation) est une technique qui permet de fine-tuner
efficacement les grands modèles de langage en quantifiant le modèle de base à 4 bits
et en ajoutant seulement un petit nombre de paramètres entraînables.
"""

import os
import logging
import torch
import platform
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
import numpy as np
from PIL import Image
import json

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    CLIPImageProcessor,
    CLIPVisionModel
)

try:
    from peft import (
        LoraConfig, 
        get_peft_model, 
        prepare_model_for_kbit_training,
        TaskType
    )
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

try:
    from trl import SFTTrainer
    HAS_TRL = True
except ImportError:
    HAS_TRL = False

from ..utils.memory_manager import get_memory_manager
from ..config import settings

# Configuration du logging
logger = logging.getLogger("ml_service.vision_qlora")

@dataclass
class QLoRAConfig:
    """Configuration pour l'adaptation QLoRA."""
    r: int = 8  # Rang de la décomposition (généralement entre 4 et 32)
    lora_alpha: int = 16  # Hyperparamètre d'échelle
    lora_dropout: float = 0.05  # Taux de dropout pour la régularisation
    bias: str = "none"  # Stratégie pour les biais ('none', 'all', 'lora_only')
    task_type: str = "CAUSAL_LM"  # Type de tâche (CAUSAL_LM, SEQ_CLS, etc.)
    target_modules: Optional[List[str]] = None  # Modules cibles pour LoRA, si None, détection automatique

    # Paramètres spécifiques au training
    batch_size: int = 1  # Taille de batch pour l'entraînement
    gradient_accumulation_steps: int = 4  # Pas d'accumulation du gradient
    learning_rate: float = 2e-4  # Taux d'apprentissage
    num_train_epochs: int = 2  # Nombre d'époques d'entraînement
    max_steps: int = -1  # Nombre maximum d'étapes, -1 signifie compléter toutes les époques
    save_steps: int = 100  # Fréquence de sauvegarde
    
    # Paramètres de quantification
    load_in_4bit: bool = True  # Charger le modèle en 4-bit
    bnb_4bit_compute_dtype: str = "float16"  # Type de données pour le calcul
    bnb_4bit_quant_type: str = "nf4"  # Type de quantification (nf4 ou fp4)
    bnb_4bit_use_double_quant: bool = True  # Utiliser la double quantification
    
    # Optimisations pour Apple Silicon
    use_fp16: bool = True  # Utiliser la précision mixte float16
    use_8bit_adam: bool = False  # Utiliser Adam 8-bit pour réduire l'usage mémoire
    
    # Options avancées
    use_gradient_checkpointing: bool = True  # Utiliser le checkpoint de gradient pour économiser la mémoire
    max_grad_norm: float = 0.3  # Norme maximale pour clipping de gradient
    warmup_ratio: float = 0.03  # Ratio de warmup
    
    # Paramètres spécifiques à la vision
    vision_tower: str = "openai/clip-vit-large-patch14"  # Modèle de vision à utiliser
    max_seq_length: int = 512  # Longueur maximale de séquence
    image_token_id: int = 32000  # ID du token d'image (à ajuster selon le modèle)
    image_token: str = "<image>"  # Token d'image dans le texte


class VisionQLoRAAdapter:
    """
    Adaptateur QLoRA pour fine-tuning de modèles Gemma avec capacités de vision.
    Optimisé pour Apple Silicon (M1, M2, M3, M4).
    """
    
    def __init__(
        self,
        base_model_id: str = "google/gemma-3-12b-it",
        qlora_config: Optional[QLoRAConfig] = None,
        output_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialise l'adaptateur QLoRA pour la vision.
        
        Args:
            base_model_id: ID du modèle de base (Hugging Face)
            qlora_config: Configuration QLoRA
            output_dir: Répertoire de sortie pour les adaptateurs
            cache_dir: Répertoire de cache pour les modèles
        """
        if not HAS_PEFT:
            raise ImportError("La bibliothèque 'peft' n'est pas installée. Installez-la avec 'pip install peft'")
        
        if not HAS_TRL:
            raise ImportError("La bibliothèque 'trl' n'est pas installée. Installez-la avec 'pip install trl'")
        
        self.base_model_id = base_model_id
        self.qlora_config = qlora_config or QLoRAConfig()
        self.output_dir = output_dir or os.path.join(settings.MODEL_PATH, "vision_qlora_adapters")
        self.cache_dir = cache_dir or str(settings.MODEL_PATH)
        
        # Créer les répertoires
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Détection pour Apple Silicon
        self.is_apple_silicon = platform.processor() == 'arm' and platform.system() == 'Darwin'
        self.has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        
        # Configurer la quantification
        self.memory_manager = get_memory_manager(cache_dir=self.cache_dir)
        
        # Initialiser les modèles de vision
        self.vision_tower = None
        self.image_processor = None
        
        logger.info(f"Vision QLoRA Adapter initialisé pour {self.base_model_id}")
    
    def _initialize_vision_components(self):
        """Initialise les composants de vision (CLIP)."""
        vision_tower_id = self.qlora_config.vision_tower
        logger.info(f"Initialisation des composants de vision: {vision_tower_id}")
        
        # Charger le processeur d'image CLIP
        self.image_processor = CLIPImageProcessor.from_pretrained(
            vision_tower_id,
            cache_dir=self.cache_dir
        )
        
        # Charger le modèle de vision CLIP
        self.vision_tower = CLIPVisionModel.from_pretrained(
            vision_tower_id,
            cache_dir=self.cache_dir,
            torch_dtype=torch.float16 if self.qlora_config.use_fp16 else torch.float32
        )
        
        # Mettre le modèle de vision en mode évaluation
        self.vision_tower.eval()
        
        # Déplacer sur le bon device
        if self.is_apple_silicon and self.has_mps:
            self.vision_tower = self.vision_tower.to("mps")
        elif torch.cuda.is_available():
            self.vision_tower = self.vision_tower.to("cuda")
        
        logger.info("Composants de vision initialisés avec succès")
    
    def prepare_model_for_training(self):
        """
        Prépare le modèle de base pour l'entraînement QLoRA avec vision.
        
        Returns:
            Tuple (model, tokenizer) prêts pour l'entraînement
        """
        logger.info(f"Chargement du modèle de base {self.base_model_id}")
        
        # Initialiser les composants de vision si nécessaire
        if self.vision_tower is None:
            self._initialize_vision_components()
        
        # Configuration de quantification pour QLoRA
        compute_dtype = getattr(torch, self.qlora_config.bnb_4bit_compute_dtype)
        
        # Configuration BitsAndBytes pour la quantification 4-bit
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.qlora_config.load_in_4bit,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=self.qlora_config.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=self.qlora_config.bnb_4bit_quant_type
        )
        
        # Charger le modèle avec quantification
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            cache_dir=self.cache_dir,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            use_cache=False  # Désactiver le KV cache pour l'entraînement
        )
        
        # Charger le tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_id,
            cache_dir=self.cache_dir,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True
        )
        
        # S'assurer que le tokenizer a un padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Padding token configuré comme EOS token: {tokenizer.pad_token}")
        
        # Ajouter le token d'image au tokenizer si nécessaire
        if self.qlora_config.image_token not in tokenizer.get_vocab():
            logger.info(f"Ajout du token d'image '{self.qlora_config.image_token}' au tokenizer")
            tokenizer.add_special_tokens({"additional_special_tokens": [self.qlora_config.image_token]})
            # Redimensionner l'embedding du modèle
            model.resize_token_embeddings(len(tokenizer))
        
        # Préparer le modèle pour la quantification et QLoRA
        model = prepare_model_for_kbit_training(model)
        
        # Déterminer les modules cibles pour LoRA
        target_modules = self.qlora_config.target_modules
        if target_modules is None:
            # Détection automatique des modules
            if "gemma" in self.base_model_id.lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            elif "llama" in self.base_model_id.lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            elif "mistral" in self.base_model_id.lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        logger.info(f"Modules cibles pour QLoRA: {target_modules}")
        
        # Configurer LoRA
        peft_config = LoraConfig(
            r=self.qlora_config.r,
            lora_alpha=self.qlora_config.lora_alpha,
            lora_dropout=self.qlora_config.lora_dropout,
            bias=self.qlora_config.bias,
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules
        )
        
        # Appliquer LoRA au modèle
        model = get_peft_model(model, peft_config)
        
        logger.info(f"Modèle préparé pour QLoRA: {model.peft_config}")
        logger.info(f"Nombre de paramètres entraînables: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        
        return model, tokenizer
    
    def process_image(self, image_path):
        """
        Traite une image pour l'intégrer dans le modèle.
        
        Args:
            image_path: Chemin vers l'image
            
        Returns:
            Embeddings de l'image
        """
        if self.image_processor is None or self.vision_tower is None:
            self._initialize_vision_components()
        
        # Charger l'image
        image = Image.open(image_path).convert("RGB")
        
        # Prétraiter l'image
        image_inputs = self.image_processor(images=image, return_tensors="pt")
        
        # Déplacer sur le bon device
        if self.is_apple_silicon and self.has_mps:
            image_inputs = {k: v.to("mps") for k, v in image_inputs.items()}
        elif torch.cuda.is_available():
            image_inputs = {k: v.to("cuda") for k, v in image_inputs.items()}
        
        # Obtenir les embeddings de l'image
        with torch.no_grad():
            image_features = self.vision_tower(**image_inputs).last_hidden_state
        
        return image_features
    
    def prepare_vision_dataset(self, dataset, image_column="image_path", text_column="text"):
        """
        Prépare un dataset pour l'entraînement avec vision.
        
        Args:
            dataset: Dataset à préparer
            image_column: Nom de la colonne contenant les chemins d'images
            text_column: Nom de la colonne contenant le texte
            
        Returns:
            Dataset préparé pour l'entraînement
        """
        if self.image_processor is None:
            self._initialize_vision_components()
        
        # Fonction de prétraitement pour chaque exemple
        def preprocess_function(examples):
            # Traiter les images
            images = [Image.open(img_path).convert("RGB") for img_path in examples[image_column]]
            image_inputs = self.image_processor(images=images, return_tensors="pt", padding=True)
            
            # Formater le texte avec le token d'image
            texts = [f"{self.qlora_config.image_token} {text}" for text in examples[text_column]]
            
            return {
                "image_inputs": image_inputs,
                "texts": texts
            }
        
        # Appliquer le prétraitement au dataset
        processed_dataset = dataset.map(preprocess_function, batched=True)
        
        return processed_dataset
    
    def train(
        self,
        train_dataset,
        val_dataset=None,
        custom_training_args=None
    ):
        """
        Entraîne un adaptateur QLoRA sur un dataset de vision.
        
        Args:
            train_dataset: Dataset d'entraînement
            val_dataset: Dataset de validation (optionnel)
            custom_training_args: Arguments personnalisés pour le training
            
        Returns:
            Modèle entraîné avec QLoRA
        """
        model, tokenizer = self.prepare_model_for_training()
        
        # Configurer les arguments d'entraînement
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.qlora_config.batch_size,
            gradient_accumulation_steps=self.qlora_config.gradient_accumulation_steps,
            learning_rate=self.qlora_config.learning_rate,
            num_train_epochs=self.qlora_config.num_train_epochs,
            max_steps=self.qlora_config.max_steps,
            save_steps=self.qlora_config.save_steps,
            save_total_limit=3,
            logging_steps=10,
            remove_unused_columns=False,
            push_to_hub=False,
            warmup_ratio=self.qlora_config.warmup_ratio,
            max_grad_norm=self.qlora_config.max_grad_norm,
            fp16=self.qlora_config.use_fp16 and not self.is_apple_silicon,  # Désactivé pour Apple Silicon car nous utilisons MPS
            optim="adamw_torch",
            evaluation_strategy="steps" if val_dataset is not None else "no",
            load_best_model_at_end=val_dataset is not None,
            eval_steps=100 if val_dataset is not None else None,
        )
        
        # Fusionner avec les arguments personnalisés
        if custom_training_args:
            for key, value in custom_training_args.items():
                setattr(training_args, key, value)
        
        # Créer le SFTTrainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            max_seq_length=self.qlora_config.max_seq_length,
            packing=False,  # Désactiver le packing pour les tâches de vision
        )
        
        logger.info("Début de l'entraînement QLoRA avec vision")
        trainer.train()
        
        # Sauvegarder l'adaptateur QLoRA
        adapter_path = os.path.join(self.output_dir, "final_vision_qlora_adapter")
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        
        # Sauvegarder la configuration de vision
        vision_config = {
            "vision_tower": self.qlora_config.vision_tower,
            "image_token": self.qlora_config.image_token,
            "image_token_id": tokenizer.convert_tokens_to_ids(self.qlora_config.image_token)
        }
        
        with open(os.path.join(adapter_path, "vision_config.json"), "w") as f:
            json.dump(vision_config, f, indent=2)
        
        logger.info(f"Adaptateur Vision QLoRA sauvegardé dans {adapter_path}")
        
        return model, tokenizer, adapter_path
    
    def load_adapter(self, adapter_path: str):
        """
        Charge un adaptateur QLoRA préalablement entraîné.
        
        Args:
            adapter_path: Chemin vers l'adaptateur QLoRA
            
        Returns:
            Modèle avec l'adaptateur QLoRA appliqué
        """
        # Charger la configuration de vision
        vision_config_path = os.path.join(adapter_path, "vision_config.json")
        if os.path.exists(vision_config_path):
            with open(vision_config_path, "r") as f:
                vision_config = json.load(f)
                
            # Mettre à jour la configuration
            self.qlora_config.vision_tower = vision_config.get("vision_tower", self.qlora_config.vision_tower)
            self.qlora_config.image_token = vision_config.get("image_token", self.qlora_config.image_token)
            self.qlora_config.image_token_id = vision_config.get("image_token_id", self.qlora_config.image_token_id)
        
        # Initialiser les composants de vision
        self._initialize_vision_components()
        
        # Charger le modèle de base
        compute_dtype = getattr(torch, self.qlora_config.bnb_4bit_compute_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.qlora_config.load_in_4bit,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=self.qlora_config.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=self.qlora_config.bnb_4bit_quant_type
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            cache_dir=self.cache_dir,
            device_map="auto",
            quantization_config=bnb_config
        )
        
        # Charger le tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            adapter_path,
            cache_dir=self.cache_dir,
            use_fast=True
        )
        
        # Charger l'adaptateur QLoRA
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, adapter_path)
        
        logger.info(f"Adaptateur Vision QLoRA chargé depuis {adapter_path}")
        
        return model, tokenizer
    
    def generate_from_image(self, model, tokenizer, image_path, prompt=None, max_length=256):
        """
        Génère du texte à partir d'une image.
        
        Args:
            model: Modèle avec adaptateur QLoRA
            tokenizer: Tokenizer
            image_path: Chemin vers l'image
            prompt: Prompt textuel (optionnel)
            max_length: Longueur maximale de génération
            
        Returns:
            Texte généré
        """
        # Traiter l'image
        image_features = self.process_image(image_path)
        
        # Préparer le prompt
        if prompt is None:
            prompt = f"{self.qlora_config.image_token} Décrivez cette image en détail."
        elif self.qlora_config.image_token not in prompt:
            prompt = f"{self.qlora_config.image_token} {prompt}"
        
        # Tokenizer le prompt
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Déplacer sur le bon device
        if self.is_apple_silicon and self.has_mps:
            inputs = {k: v.to("mps") for k, v in inputs.items()}
            image_features = image_features.to("mps")
        elif torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            image_features = image_features.to("cuda")
        
        # Générer le texte
        with torch.no_grad():
            # Remplacer les tokens d'image par les embeddings d'image
            input_ids = inputs["input_ids"][0]
            image_token_id = tokenizer.convert_tokens_to_ids(self.qlora_config.image_token)
            image_token_indices = (input_ids == image_token_id).nonzero(as_tuple=True)[0]
            
            # Générer avec les embeddings d'image
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                image_features=image_features,
                image_token_indices=image_token_indices
            )
        
        # Décoder le texte généré
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text

# Exemple d'utilisation:
if __name__ == "__main__":
    # Créer l'adaptateur QLoRA pour Gemma
    vision_adapter = VisionQLoRAAdapter(
        base_model_id="google/gemma-3-12b-it",
        qlora_config=QLoRAConfig(
            r=16,  # Rang plus élevé pour une meilleure adaptation
            lora_alpha=32,
            batch_size=1,  # Personnaliser selon votre GPU/RAM
            vision_tower="openai/clip-vit-large-patch14"
        )
    )
    
    # Exemple d'entraînement (à remplacer par votre dataset)
    from datasets import Dataset
    
    # Créer un dataset d'exemple
    data = {
        "image_path": ["chemin/vers/image1.jpg", "chemin/vers/image2.jpg"],
        "text": ["Description du produit 1", "Description du produit 2"]
    }
    dataset = Dataset.from_dict(data)
    
    # Entraîner l'adaptateur QLoRA
    model, tokenizer, adapter_path = vision_adapter.train(dataset)
    
    # Générer une description à partir d'une image
    generated_text = vision_adapter.generate_from_image(
        model, 
        tokenizer, 
        "chemin/vers/nouvelle_image.jpg",
        prompt="Décrivez ce produit en détail."
    )
    
    print(generated_text) 