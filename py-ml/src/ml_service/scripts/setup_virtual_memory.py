#!/usr/bin/env python3
"""
Script pour configurer la mémoire virtuelle et télécharger les modèles optimisés.
Ce script prépare l'environnement pour exécuter des grands modèles comme Phi-4 et Qwen 32B 
sur des machines avec moins de mémoire que normalement requis.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import shutil
import json
import psutil
import torch
import platform
import gc

# Ajouter le répertoire parent au path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ml_service.utils.memory_manager import get_memory_manager
from ml_service.config import settings
from ml_service.scripts.download_models import download_models, MODELS_TO_DOWNLOAD

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("setup_virtual_memory")

def create_virtual_memory(swap_size_gb, swap_path=None):
    """
    Configure la mémoire virtuelle pour les modèles de grande taille.
    
    Args:
        swap_size_gb: Taille du fichier d'échange en Go
        swap_path: Chemin où créer le fichier d'échange (par défaut: dossier temporaire)
    """
    # Obtenir le gestionnaire de mémoire avec la taille de swap spécifiée
    memory_manager = get_memory_manager(
        cache_dir=settings.MODEL_PATH,
        swap_file_size_gb=swap_size_gb,
        offload_folder=swap_path
    )
    
    # Afficher les informations sur la mémoire système
    mem_info = psutil.virtual_memory()
    total_gb = mem_info.total / (1024**3)
    available_gb = mem_info.available / (1024**3)
    
    # Informations GPU si disponible
    gpu_info = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_info[f"gpu_{i}"] = {
                "name": props.name,
                "vram_total_gb": props.total_memory / (1024**3),
                "compute_capability": f"{props.major}.{props.minor}"
            }
    
    # Afficher le résumé de configuration
    logger.info("=== Configuration de la mémoire virtuelle ===")
    logger.info(f"RAM totale: {total_gb:.2f} GB")
    logger.info(f"RAM disponible: {available_gb:.2f} GB")
    logger.info(f"VRAM virtuelle configurée: {swap_size_gb} GB")
    logger.info(f"Dossier d'offloading: {memory_manager.offload_folder}")
    
    if gpu_info:
        for gpu, info in gpu_info.items():
            logger.info(f"GPU {gpu}: {info['name']} ({info['vram_total_gb']:.2f} GB)")
    else:
        logger.info("Aucun GPU détecté, utilisation du CPU uniquement")
    
    # Sauvegarder la configuration pour référence
    config_path = os.path.join(memory_manager.offload_folder, "virtual_memory_config.json")
    config = {
        "ram_total_gb": total_gb,
        "ram_available_gb": available_gb,
        "virtual_vram_gb": swap_size_gb,
        "offload_folder": memory_manager.offload_folder,
        "gpu_info": gpu_info,
        "system_info": {
            "cpu_cores": psutil.cpu_count(),
            "cpu_physical_cores": psutil.cpu_count(logical=False),
            "platform": sys.platform,
            "python_version": sys.version
        }
    }
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Configuration sauvegardée dans {config_path}")
    logger.info("=== Fin de la configuration ===")
    
    return memory_manager

def download_optimized_models(models_to_download, cache_dir=None, only_required=True, sequential=True):
    """
    Télécharge les modèles avec des optimisations pour la mémoire virtuelle.
    
    Args:
        models_to_download: Dictionnaire des modèles à télécharger
        cache_dir: Répertoire de cache pour les modèles
        only_required: Ne télécharger que les modèles requis
        sequential: Télécharger les modèles séquentiellement pour économiser la mémoire
        
    Returns:
        Tuple contenant le nombre de succès, d'échecs et de modèles ignorés
    """
    logger.info("=== Téléchargement des modèles optimisés ===")
    
    # Obtenir le répertoire de cache
    cache_dir = cache_dir or str(settings.MODEL_PATH)
    
    # Vérifier si nous sommes sur Apple Silicon
    is_apple_silicon = platform.processor() == 'arm' and platform.system() == 'Darwin'
    
    if sequential and is_apple_silicon:
        logger.info("Mode séquentiel activé pour Apple Silicon - Téléchargement des modèles un par un")
        
        success_count = 0
        failure_count = 0
        skipped_count = 0
        
        # Télécharger les embeddings en premier (priorité car petits et essentiels)
        for category, models in models_to_download.items():
            if category == "text":
                for model_name, model_info in models.items():
                    if model_name == "all-MiniLM-L6-v2":
                        # Télécharger d'abord les embeddings
                        modified_models = {
                            "text": {
                                model_name: model_info
                            }
                        }
                        s, f, sk = download_models(
                            modified_models,
                            only_required=False,  # Toujours télécharger les embeddings
                            cache_dir=cache_dir
                        )
                        success_count += s
                        failure_count += f
                        skipped_count += sk
                        
                        # Forcer un nettoyage mémoire
                        gc.collect()
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        
                        # Pour MPS (Metal)
                        if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                            torch.mps.empty_cache()
        
        # Télécharger les petits modèles restants
        small_models = {}
        for category, models in models_to_download.items():
            small_models[category] = {}
            for model_name, model_info in models.items():
                if (model_name != "all-MiniLM-L6-v2" and  # Déjà téléchargé
                    model_name != "microsoft/Phi-4" and   # Gros modèle - après
                    model_name != "Qwen/Qwen1.5-32B"):    # Très gros modèle - après
                    small_models[category][model_name] = model_info
        
        # Télécharger les petits modèles
        if small_models:
            s, f, sk = download_models(
                small_models,
                only_required=only_required,
                cache_dir=cache_dir
            )
            success_count += s
            failure_count += f
            skipped_count += sk
            
            # Nettoyage mémoire
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
        
        # Télécharger Phi-4 séparément (modèle moyen à grand)
        phi4_model = {
            "text": {
                "microsoft/Phi-4": models_to_download["text"].get("microsoft/Phi-4", {"required": False})
            }
        }
        
        if "microsoft/Phi-4" in models_to_download.get("text", {}):
            # Vérifier si on doit télécharger Phi-4
            if not only_required or phi4_model["text"]["microsoft/Phi-4"].get("required", False):
                logger.info("Téléchargement du modèle Phi-4...")
                s, f, sk = download_models(
                    phi4_model,
                    only_required=False,  # Déjà filtré
                    cache_dir=cache_dir
                )
                success_count += s
                failure_count += f
                skipped_count += sk
                
                # Nettoyage mémoire
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
        
        # Télécharger Qwen 32B séparément (très gros modèle) si demandé
        qwen32b_model = {
            "text": {
                "Qwen/Qwen1.5-32B": models_to_download["text"].get("Qwen/Qwen1.5-32B", {"required": False})
            }
        }
        
        if "Qwen/Qwen1.5-32B" in models_to_download.get("text", {}):
            # Vérifier si on doit télécharger Qwen 32B
            if not only_required or qwen32b_model["text"]["Qwen/Qwen1.5-32B"].get("required", False):
                logger.info("Téléchargement du modèle Qwen 32B...")
                s, f, sk = download_models(
                    qwen32b_model,
                    only_required=False,  # Déjà filtré
                    cache_dir=cache_dir
                )
                success_count += s
                failure_count += f
                skipped_count += sk
        
        return success_count, failure_count, skipped_count
    else:
        # Téléchargement standard
        return download_models(
            models_to_download, 
            only_required=only_required,
            cache_dir=cache_dir
        )

def setup_for_apple_m4(swap_size_gb=32, models_to_download=None):
    """
    Configuration spécialement optimisée pour les puces Apple M4.
    
    Args:
        swap_size_gb: Taille du fichier d'échange en Go
        models_to_download: Dictionnaire des modèles à télécharger
    """
    is_m4 = False
    
    # Vérifier si nous sommes sur Apple Silicon
    if platform.processor() == 'arm' and platform.system() == 'Darwin':
        # Détecter la puce
        import subprocess
        try:
            chip_info = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
            if "M4" in chip_info:
                is_m4 = True
                logger.info(f"Puce Apple M4 détectée: {chip_info}")
        except Exception as e:
            logger.warning(f"Impossible de détecter le modèle de puce: {str(e)}")
    
    if not is_m4:
        logger.warning("Ce script est optimisé pour les puces Apple M4. Vous utilisez une autre puce.")
        return setup_everything(swap_size_gb, models_to_download, only_required=True)
    
    # Vérifier la disponibilité de MPS (Metal Performance Shaders)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logger.info("MPS disponible - Configuration pour utiliser l'Apple Neural Engine")
        # Configurer torch pour utiliser MPS
        try:
            torch.set_default_device('mps')
            logger.info("Torch configuré pour utiliser MPS par défaut")
        except Exception as e:
            logger.warning(f"Erreur lors de la configuration MPS: {str(e)}")
    
    # Configurations spécifiques pour M4
    logger.info("Configuration optimisée pour M4")
    
    # Créer le dossier d'offloading avec un sous-dossier spécial pour M4
    offload_folder = os.path.join(settings.ROOT_PATH, "m4_offload")
    os.makedirs(offload_folder, exist_ok=True)
    
    # Configurer la mémoire virtuelle
    memory_manager = create_virtual_memory(
        swap_size_gb=swap_size_gb,
        swap_path=offload_folder
    )
    
    # Télécharger les modèles
    if models_to_download is None:
        models_to_download = MODELS_TO_DOWNLOAD
    
    # Télécharger les modèles séquentiellement pour éviter les OOM
    download_optimized_models(
        models_to_download,
        cache_dir=memory_manager.cache_dir,
        only_required=True,
        sequential=True
    )
    
    logger.info("=== Configuration pour M4 complète ===")
    logger.info(f"Le système est prêt à exécuter de grands modèles sur M4 avec {swap_size_gb} GB de VRAM virtuelle")
    logger.info(f"Vous pouvez maintenant démarrer l'API avec la commande:")
    logger.info(f"   uvicorn src.ml_service.api.main:app --host 0.0.0.0 --port 8000 --reload")
    logger.info("=== Fin de la configuration ===")

def setup_everything(swap_size_gb=32, models_to_download=None, only_required=False):
    """
    Configure tout le système pour l'exécution des grands modèles.
    
    Args:
        swap_size_gb: Taille du fichier d'échange en Go
        models_to_download: Dictionnaire des modèles à télécharger
        only_required: Ne télécharger que les modèles requis
    """
    # Configurer la mémoire virtuelle
    memory_manager = create_virtual_memory(swap_size_gb)
    
    # Télécharger les modèles
    if models_to_download is None:
        models_to_download = MODELS_TO_DOWNLOAD
        
    # Télécharger les modèles
    download_optimized_models(models_to_download, cache_dir=memory_manager.cache_dir)
    
    logger.info("=== Configuration complète ===")
    logger.info(f"Le système est prêt à exécuter de grands modèles avec {swap_size_gb} GB de VRAM virtuelle")
    logger.info(f"Vous pouvez maintenant démarrer l'API avec la commande:")
    logger.info(f"   uvicorn src.ml_service.api.main:app --host 0.0.0.0 --port 8000 --reload")
    logger.info("=== Fin de la configuration ===")

def main():
    parser = argparse.ArgumentParser(description="Configure la mémoire virtuelle et télécharge les modèles optimisés")
    parser.add_argument("--swap-size", type=int, default=32, help="Taille du fichier d'échange en Go")
    parser.add_argument("--swap-path", type=str, help="Chemin où créer le fichier d'échange")
    parser.add_argument("--cache-dir", type=str, help="Répertoire de cache pour les modèles")
    parser.add_argument("--only-required", action="store_true", help="Ne télécharger que les modèles requis")
    parser.add_argument("--skip-download", action="store_true", help="Ne pas télécharger les modèles")
    parser.add_argument("--m4-mode", action="store_true", help="Mode optimisé pour Apple M4")
    parser.add_argument("--sequential", action="store_true", help="Téléchargement séquentiel des modèles")
    
    args = parser.parse_args()
    
    print(f"===== Configuration de la mémoire virtuelle pour grands modèles =====")
    print(f"Taille du fichier d'échange: {args.swap_size} GB")
    print(f"Chemin du fichier d'échange: {args.swap_path or 'Dossier temporaire'}")
    print(f"Répertoire de cache: {args.cache_dir or str(settings.MODEL_PATH)}")
    print(f"Modèles requis uniquement: {'Oui' if args.only_required else 'Non'}")
    print(f"Sauter le téléchargement: {'Oui' if args.skip_download else 'Non'}")
    print(f"Mode M4 optimisé: {'Oui' if args.m4_mode else 'Non'}")
    print(f"Téléchargement séquentiel: {'Oui' if args.sequential else 'Non'}")
    print("=" * 60)
    
    try:
        # Mode spécial pour Apple M4
        if args.m4_mode:
            setup_for_apple_m4(args.swap_size, MODELS_TO_DOWNLOAD)
            return 0
        
        # Configurer la mémoire virtuelle
        memory_manager = create_virtual_memory(args.swap_size, args.swap_path)
        
        # Télécharger les modèles si demandé
        if not args.skip_download:
            success, failure, skipped = download_optimized_models(
                MODELS_TO_DOWNLOAD, 
                cache_dir=args.cache_dir or memory_manager.cache_dir,
                only_required=args.only_required,
                sequential=args.sequential
            )
            if failure > 0:
                logger.warning(f"Certains modèles n'ont pas pu être téléchargés ({failure} échecs)")
        
        print("\n" + "=" * 60)
        print(f"Configuration terminée!")
        print(f"  - VRAM virtuelle configurée: {args.swap_size} GB")
        print(f"  - Dossier d'offloading: {memory_manager.offload_folder}")
        print(f"  - Cache des modèles: {memory_manager.cache_dir}")
        print("=" * 60)
        
        return 0
    except Exception as e:
        logger.error(f"Erreur lors de la configuration: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 