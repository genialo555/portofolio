from typing import Dict, Any, List
import httpx
import asyncio
from dataclasses import dataclass
import os

@dataclass
class TeacherConfig:
    """Configuration avec uniquement des modèles open source puissants."""
    models = {
        # Modèles principaux
        'qwen25': 'Qwen/Qwen2.5-7B-Chat',  # Excellent rapport qualité/performance
        'mixtral': 'mistralai/Mixtral-8x7B-Instruct-v0.1',  # Très puissant, open source
        'yi': 'zero-one-ai/Yi-34B-Chat',  # Concurrent direct de GPT-4
        
        # Modèles spécialisés
        'codellama': 'codellama/CodeLlama-34b-Instruct-hf',  # Pour le code
        'neural-chat': 'intel/neural-chat-7b-v3-1',  # Très optimisé
        'openchat': 'openchat/openchat-3.5',  # Performances proches de GPT-3.5
        
        # Backups légers
        'phi': 'microsoft/phi-2',  # Excellent en petit modèle
        'mistral': 'mistralai/Mistral-7B-Instruct-v0.1'  # Base solide
    }
    
    endpoint = 'https://api-inference.huggingface.co/models'
    
    prompts = {
        'deep_analysis': """
        Analysez en profondeur cette réponse avec une approche talmudique :
        
        RÉPONSE À ANALYSER :
        {response}
        
        DIRECTIVES D'ANALYSE :
        1. Examinez chaque argument et sous-argument
        2. Identifiez les présupposés cachés
        3. Proposez des contre-arguments potentiels
        4. Explorez les implications logiques
        5. Recherchez les contradictions subtiles
        6. Suggérez des perspectives alternatives
        7. Évaluez la solidité du raisonnement
        
        FORMAT DE RÉPONSE :
        1. Analyse initiale (profonde et détaillée)
        2. Contre-arguments et réfutations
        3. Synthèse dialectique
        4. Recommandations d'amélioration
        """,
        
        'debate_synthesis': """
        Synthétisez ce débat avec une approche philosophique approfondie :
        
        ARGUMENTS GROUPE A :
        {perspective_a}
        
        ARGUMENTS GROUPE B :
        {perspective_b}
        
        HISTORIQUE DU DÉBAT :
        {debate_history}
        
        DIRECTIVES DE SYNTHÈSE :
        1. Identifiez la thèse et l'antithèse
        2. Examinez les arguments dialectiques
        3. Cherchez les points de convergence profonds
        4. Proposez une synthèse transcendante
        5. Explorez les implications philosophiques
        6. Suggérez des pistes de réflexion nouvelles
        """
    }

    load_config = {
        'qwen25': {
            'quantization': '4bit',
            'max_memory': {0: '5.2GB'},
            'device_map': 'auto',
            'use_flash_attention_2': True,
            'temperature': 0.3,  # Plus précis
            'max_tokens': 2048  # Réponses plus détaillées
        }
    }

class TeacherModel:
    """Modèle enseignant puissant utilisant les meilleurs modèles disponibles."""
    
    def __init__(self):
        self.config = TeacherConfig()
        self.hf_api_key = os.getenv('HF_API_KEY')
        
    async def evaluate_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        # Analyse approfondie avec Qwen 2.5
        deep_analysis = await self._query_model(
            'qwen25',
            self.config.prompts['deep_analysis'].format(
                response=response['content']
            ),
            max_time=120  # 2 minutes max pour l'analyse
        )
        
        # Vérification avec Mixtral
        verification = await self._query_model(
            'mixtral',
            f"Vérifiez et enrichissez cette analyse :\n{deep_analysis}",
            max_time=60
        )
        
        # Synthèse finale avec Phi-2
        synthesis = await self._query_model(
            'phi',
            f"Synthétisez ces analyses en une conclusion finale :\n{deep_analysis}\n{verification}",
            max_time=30
        )
        
        return {
            'deep_analysis': deep_analysis,
            'verification': verification,
            'final_synthesis': synthesis,
            'quality_metrics': self._evaluate_quality(deep_analysis, verification)
        }
        
    async def synthesize_debate(self, perspective_a: Dict, perspective_b: Dict, debate_history: List) -> Dict:
        """Synthèse avec modèles gratuits."""
        
        # Première synthèse avec Yi via Together.ai (gratuit)
        synthesis = await self._query_model(
            provider='together',
            model='yi',
            prompt=self.config.prompts['debate_synthesis'].format(
                perspective_a=perspective_a['content'],
                perspective_b=perspective_b['content'],
                debate_history=debate_history
            )
        )
        
        # Vérification locale avec Mistral via Ollama (gratuit)
        local_check = await self._query_model(
            provider='ollama',
            model='mistral',
            prompt=f"Verify and improve this synthesis:\n{synthesis}"
        )
        
        return {
            'synthesis': synthesis,
            'verification': local_check,
            'final_version': self._merge_insights(synthesis, local_check)
        }
        
    async def _query_model(self, model: str, prompt: str, max_time: int) -> str:
        """Interroge un modèle via HuggingFace."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.config.endpoint}/{self.config.models[model]}",
                headers={'Authorization': f"Bearer {self.hf_api_key}"},
                json={'inputs': prompt},
                timeout=max_time
            )
            return response.json()[0]['generated_text'] 