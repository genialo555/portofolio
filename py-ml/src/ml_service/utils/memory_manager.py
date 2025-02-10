import psutil
import torch
import py3nvml
from typing import Dict, List, Optional
import asyncio
import logging
from dataclasses import dataclass

@dataclass
class MemoryState:
    vram_used: float
    vram_total: float
    ram_used: float
    ram_total: float
    gpu_utilization: float
    loaded_models: List[str]

class MemoryManager:
    """Gestionnaire de mémoire intelligent pour les modèles."""
    
    def __init__(self, vram_limit: float = 5.5, ram_limit: float = 28.0):
        self.vram_limit = vram_limit  # Garde 0.5GB de marge sur 6GB
        self.ram_limit = ram_limit    # Garde 4GB de marge sur 32GB
        
        # Initialisation du monitoring NVIDIA
        py3nvml.nvmlInit()
        self.handle = py3nvml.nvmlDeviceGetHandleByIndex(0)
        
        self.loaded_models: Dict[str, torch.nn.Module] = {}
        self.model_sizes = {
            'phi': {'vram': 2.5, 'ram': 3.0},
            'mistral': {'vram': 4.0, 'ram': 5.0},
            'codellama': {'vram': 4.0, 'ram': 5.0}
        }
        
        self.logger = logging.getLogger(__name__)
        
    async def monitor_memory(self) -> MemoryState:
        """Monitore l'utilisation mémoire en temps réel."""
        # GPU Memory
        info = py3nvml.nvmlDeviceGetMemoryInfo(self.handle)
        vram_used = info.used / 1024**3  # Conversion en GB
        vram_total = info.total / 1024**3
        
        # GPU Utilization
        gpu_util = py3nvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
        
        # System RAM
        ram = psutil.virtual_memory()
        ram_used = ram.used / 1024**3
        ram_total = ram.total / 1024**3
        
        state = MemoryState(
            vram_used=vram_used,
            vram_total=vram_total,
            ram_used=ram_used,
            ram_total=ram_total,
            gpu_utilization=gpu_util,
            loaded_models=list(self.loaded_models.keys())
        )
        
        # Log warnings if approaching limits
        if vram_used > self.vram_limit * 0.9:
            self.logger.warning(f"VRAM usage high: {vram_used:.2f}GB/{self.vram_limit}GB")
        
        return state
        
    async def can_load_model(self, model_name: str) -> bool:
        """Vérifie si un modèle peut être chargé."""
        state = await self.monitor_memory()
        model_size = self.model_sizes[model_name]
        
        return (state.vram_used + model_size['vram'] <= self.vram_limit and
                state.ram_used + model_size['ram'] <= self.ram_limit)
                
    async def optimize_memory(self):
        """Optimise l'utilisation mémoire."""
        state = await self.monitor_memory()
        
        if state.vram_used > self.vram_limit * 0.8:
            # Libère les modèles non utilisés récemment
            for model_name in list(self.loaded_models.keys()):
                if model_name != 'phi':  # Garde toujours Phi-2
                    await self.unload_model(model_name)
                    
            # Force le garbage collector
            torch.cuda.empty_cache()
            
    async def load_model(self, model_name: str, model_loader) -> Optional[torch.nn.Module]:
        """Charge un modèle avec gestion mémoire."""
        if not await self.can_load_model(model_name):
            await self.optimize_memory()
            if not await self.can_load_model(model_name):
                self.logger.error(f"Cannot load {model_name}: insufficient memory")
                return None
                
        if model_name not in self.loaded_models:
            model = await model_loader(model_name)
            self.loaded_models[model_name] = model
            
        return self.loaded_models[model_name]
        
    async def unload_model(self, model_name: str):
        """Décharge un modèle de la mémoire."""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            torch.cuda.empty_cache() 