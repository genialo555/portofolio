from typing import Dict, Any
import asyncio
from pathlib import Path
import torch
from .dataset_loader import DatasetLoader
from ..pilpoul.debate_engine import PilpoulEngine
from ..agents.instagram.influencer_agent import InstagramInfluencerAgent

async def train_all_agents():
    """Script principal d'entraînement."""
    loader = DatasetLoader()
    
    # Préparation des datasets
    instagram_data = loader.prepare_instagram_dataset('data/instagram_dataset.json')
    pilpoul_data = loader.prepare_pilpoul_dataset('data/pilpoul_debates.json')
    
    # Configuration des agents
    instagram_agent = InstagramInfluencerAgent(niche='general')
    pilpoul_engine = PilpoulEngine()
    
    # Préparation pour l'entraînement
    training_configs = {
        'instagram': {
            'agent': instagram_agent,
            'dataset': instagram_data,
            'model_name': 'Qwen/Qwen2.5-7B-Chat'
        },
        'pilpoul': {
            'agent': pilpoul_engine,
            'dataset': pilpoul_data,
            'model_name': 'mistralai/Mixtral-8x7B-Instruct-v0.1'
        }
    }
    
    return training_configs 