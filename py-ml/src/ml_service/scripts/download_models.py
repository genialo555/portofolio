#!/usr/bin/env python3
"""
Script pour télécharger les modèles pré-entraînés utilisés par le service ML.
Ce script télécharge automatiquement les modèles depuis HuggingFace et les stocke
dans le répertoire de cache configuré.
"""

import os
import argparse
import logging
from pathlib import Path
import torch
from tqdm import tqdm
import sys

# Ajout du répertoire parent au path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ml_service.config import settings

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("download_models")

# Modèles à télécharger avec leurs paramètres
MODELS_TO_DOWNLOAD = {
    "text": {
        "all-MiniLM-L6-v2": {  # Modèle d'embedding pour RAG
            "type": "sentence-transformers",
            "required": True,
        },
        "openai/whisper-tiny": {  # Petit modèle de transcription (exemple)
            "type": "transformers",
            "class": "AutoModelForSpeechSeq2Seq",
            "processor": "AutoProcessor",
            "required": False,
        },
        "facebook/opt-125m": {  # Petit modèle de langage (exemple/test)
            "type": "transformers",
            "class": "AutoModelForCausalLM",
            "tokenizer": "AutoTokenizer",
            "required": True,
        },
    },
    "image": {
        "stabilityai/stable-diffusion-2-base": {  # Modèle de génération d'images (comme substitut SDXL)
            "type": "diffusers",
            "class": "StableDiffusionPipeline",
            "required": False,
        },
    },
}

def download_sentence_transformers_model(model_name, cache_dir):
    """Télécharge un modèle sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer
        
        logger.info(f"Téléchargement du modèle sentence-transformers: {model_name}")
        model = SentenceTransformer(model_name, cache_folder=cache_dir)
        logger.info(f"Modèle {model_name} téléchargé et mis en cache avec succès")
        return True
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement du modèle {model_name}: {str(e)}")
        if "ModuleNotFoundError" in str(e):
            logger.error("Veuillez installer sentence-transformers: pip install sentence-transformers")
        return False

def download_transformers_model(model_name, model_class, tokenizer_or_processor, cache_dir):
    """Télécharge un modèle transformers avec son tokenizer/processor."""
    try:
        import transformers
        
        logger.info(f"Téléchargement du modèle transformers: {model_name}")
        
        # Télécharger le tokenizer ou processor
        if tokenizer_or_processor:
            processor_class = getattr(transformers, tokenizer_or_processor)
            processor = processor_class.from_pretrained(model_name, cache_dir=cache_dir)
            logger.info(f"Tokenizer/Processor {tokenizer_or_processor} pour {model_name} téléchargé")
        
        # Télécharger le modèle
        model_class = getattr(transformers, model_class)
        
        # Utiliser un dtype approprié
        dtype = torch.float16 if torch.cuda.is_available() else None
        
        # Télécharger le modèle
        model = model_class.from_pretrained(
            model_name, 
            cache_dir=cache_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        )
        
        logger.info(f"Modèle {model_name} téléchargé et mis en cache avec succès")
        return True
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement du modèle {model_name}: {str(e)}")
        if "ModuleNotFoundError" in str(e):
            logger.error("Veuillez installer transformers: pip install transformers")
        return False

def download_diffusers_model(model_name, model_class, cache_dir):
    """Télécharge un modèle diffusers."""
    try:
        import diffusers
        
        logger.info(f"Téléchargement du modèle diffusers: {model_name}")
        
        # Télécharger le pipeline
        pipeline_class = getattr(diffusers, model_class)
        
        # Utiliser un dtype approprié
        dtype = torch.float16 if torch.cuda.is_available() else None
        
        # Télécharger le modèle
        pipeline = pipeline_class.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=dtype,
            safety_checker=None  # Désactiver pour un téléchargement plus rapide
        )
        
        logger.info(f"Modèle {model_name} téléchargé et mis en cache avec succès")
        return True
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement du modèle {model_name}: {str(e)}")
        if "ModuleNotFoundError" in str(e):
            logger.error("Veuillez installer diffusers: pip install diffusers")
        return False

def download_models(models_to_download, only_required=False, cache_dir=None):
    """Télécharge tous les modèles spécifiés."""
    cache_dir = cache_dir or str(settings.MODEL_PATH)
    logger.info(f"Utilisation du répertoire de cache: {cache_dir}")
    
    # Créer le répertoire de cache s'il n'existe pas
    os.makedirs(cache_dir, exist_ok=True)
    
    # Télécharger les modèles par catégorie
    success_count = 0
    failure_count = 0
    skip_count = 0
    
    for category, models in models_to_download.items():
        logger.info(f"Téléchargement des modèles de la catégorie: {category}")
        category_dir = os.path.join(cache_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        
        for model_name, model_info in models.items():
            # Ignorer les modèles non requis si only_required est True
            if only_required and not model_info.get("required", False):
                logger.info(f"Ignoré modèle non requis: {model_name}")
                skip_count += 1
                continue
            
            model_type = model_info.get("type")
            success = False
            
            if model_type == "sentence-transformers":
                success = download_sentence_transformers_model(model_name, category_dir)
            elif model_type == "transformers":
                model_class = model_info.get("class")
                tokenizer_or_processor = model_info.get("tokenizer") or model_info.get("processor")
                success = download_transformers_model(model_name, model_class, tokenizer_or_processor, category_dir)
            elif model_type == "diffusers":
                model_class = model_info.get("class")
                success = download_diffusers_model(model_name, model_class, category_dir)
            else:
                logger.warning(f"Type de modèle inconnu: {model_type}")
                continue
            
            if success:
                success_count += 1
            else:
                failure_count += 1
    
    logger.info(f"Téléchargement terminé. Succès: {success_count}, Échecs: {failure_count}, Ignorés: {skip_count}")
    return success_count, failure_count, skip_count

def main():
    parser = argparse.ArgumentParser(description="Télécharge les modèles pré-entraînés requis")
    parser.add_argument("--cache-dir", type=str, help="Répertoire de cache pour les modèles")
    parser.add_argument("--only-required", action="store_true", help="Télécharger uniquement les modèles requis")
    args = parser.parse_args()
    
    cache_dir = args.cache_dir or str(settings.MODEL_PATH)
    
    print(f"===== Téléchargement des modèles pré-entraînés =====")
    print(f"Répertoire de cache: {cache_dir}")
    print(f"Modèles requis uniquement: {'Oui' if args.only_required else 'Non'}")
    print("=" * 50)
    
    # Télécharger les modèles
    success, failure, skipped = download_models(
        MODELS_TO_DOWNLOAD, 
        only_required=args.only_required, 
        cache_dir=cache_dir
    )
    
    print("\n" + "=" * 50)
    print(f"Téléchargement terminé!")
    print(f"  - Modèles téléchargés avec succès: {success}")
    print(f"  - Modèles avec erreurs: {failure}")
    print(f"  - Modèles ignorés: {skipped}")
    print("=" * 50)
    
    # Retourner un code d'erreur si des téléchargements ont échoué
    return 0 if failure == 0 else 1

if __name__ == "__main__":
    sys.exit(main()) 