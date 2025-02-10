from typing import Dict, List
import json
from pathlib import Path
import datasets
from transformers import AutoTokenizer

class DatasetLoader:
    """Prépare les datasets pour l'entraînement."""
    
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Chat')
        
    def prepare_instagram_dataset(self, path: str) -> datasets.Dataset:
        """Prépare le dataset Instagram."""
        raw_data = self._load_json_data(path)
        
        # Formatage des données
        formatted_data = []
        for item in raw_data:
            formatted_data.append({
                'input_text': self._format_instagram_input(item),
                'target_text': self._format_instagram_output(item),
                'type': 'instagram_strategy',
                'metadata': {
                    'niche': item.get('niche', 'general'),
                    'followers': item.get('followers_count', 0),
                    'engagement_rate': item.get('engagement_rate', 0)
                }
            })
            
        return datasets.Dataset.from_dict({
            'input_text': [x['input_text'] for x in formatted_data],
            'target_text': [x['target_text'] for x in formatted_data],
            'type': [x['type'] for x in formatted_data],
            'metadata': [x['metadata'] for x in formatted_data]
        })
        
    def prepare_pilpoul_dataset(self, path: str) -> datasets.Dataset:
        """Prépare le dataset pour le débat pilpoul."""
        raw_data = self._load_json_data(path)
        
        formatted_data = []
        for debate in raw_data:
            # Format pour chaque étape du débat
            for stage in ['thesis', 'antithesis', 'synthesis']:
                formatted_data.append({
                    'input_text': self._format_debate_input(debate, stage),
                    'target_text': self._format_debate_output(debate, stage),
                    'type': f'pilpoul_{stage}',
                    'metadata': {
                        'topic': debate.get('topic', ''),
                        'complexity': debate.get('complexity', 'medium'),
                        'stage': stage
                    }
                })
                
        return datasets.Dataset.from_dict({
            'input_text': [x['input_text'] for x in formatted_data],
            'target_text': [x['target_text'] for x in formatted_data],
            'type': [x['type'] for x in formatted_data],
            'metadata': [x['metadata'] for x in formatted_data]
        })
        
    def _format_instagram_input(self, item: Dict) -> str:
        """Formate l'entrée pour le dataset Instagram."""
        return f"""
        PROFIL INSTAGRAM
        Niche: {item.get('niche', 'N/A')}
        Followers: {item.get('followers_count', 0)}
        Engagement: {item.get('engagement_rate', 0)}%
        
        CONTENU RÉCENT
        {item.get('recent_posts', [])}
        
        OBJECTIF
        {item.get('goal', 'Optimiser la stratégie')}
        """
        
    def _format_debate_input(self, debate: Dict, stage: str) -> str:
        """Formate l'entrée pour le dataset de débat."""
        if stage == 'thesis':
            return f"""
            SUJET DU DÉBAT
            {debate.get('topic', '')}
            
            CONTEXTE
            {debate.get('context', '')}
            
            GÉNÉRER UNE THÈSE INITIALE
            """
        elif stage == 'antithesis':
            return f"""
            THÈSE
            {debate.get('thesis', '')}
            
            GÉNÉRER UNE ANTITHÈSE CRITIQUE
            """
        else:  # synthesis
            return f"""
            THÈSE
            {debate.get('thesis', '')}
            
            ANTITHÈSE
            {debate.get('antithesis', '')}
            
            GÉNÉRER UNE SYNTHÈSE
            """
            
    def _load_json_data(self, path: str) -> List[Dict]:
        """Charge les données JSON."""
        data_path = Path(path)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {path}")
            
        with open(data_path, 'r') as f:
            return json.load(f) 