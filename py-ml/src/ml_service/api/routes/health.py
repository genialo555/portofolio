from fastapi import APIRouter, Depends
import psutil
import time
import os
from typing import Dict, Any

router = APIRouter()

async def get_system_stats():
    """Récupère les statistiques système actuelles."""
    return {
        "cpu_usage": psutil.cpu_percent(interval=0.1),
        "memory_usage": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent,
        "uptime": time.time() - psutil.boot_time(),
    }

@router.get("/")
async def health_check() -> Dict[str, Any]:
    """
    Vérification basique de la santé de l'API.
    
    Returns:
        Dict[str, Any]: Statut de santé de l'API
    """
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0"
    }

@router.get("/stats")
async def health_stats(stats: Dict[str, Any] = Depends(get_system_stats)) -> Dict[str, Any]:
    """
    Statistiques détaillées sur la santé de l'API et du système.
    
    Returns:
        Dict[str, Any]: Statistiques détaillées sur le système
    """
    return {
        "status": "healthy" if stats["cpu_usage"] < 90 and stats["memory_usage"] < 90 else "warning",
        "timestamp": time.time(),
        "version": "1.0.0",
        "system": stats,
        "environment": os.environ.get("ENVIRONMENT", "development")
    }

@router.get("/ping")
async def ping() -> Dict[str, str]:
    """
    Simple réponse ping-pong pour vérifier que l'API est réactive.
    
    Returns:
        Dict[str, str]: Message pong
    """
    return {"ping": "pong"} 