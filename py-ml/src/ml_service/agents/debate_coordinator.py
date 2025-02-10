from typing import Dict, Any, List
import asyncio
from .local_gateway import LocalModelGateway
from .model_gateway import FreeModelGateway
from ..models.image_teacher import ImageTeacherModel
from ..utils.memory_manager import MemoryManager
import logging

class DebateCoordinator:
    """Coordonne le débat entre les agents."""
    
    def __init__(self):
        self.local_models = LocalModelGateway()
        self.api_models = FreeModelGateway()
        self.image_teacher = ImageTeacherModel()
        self.memory_manager = MemoryManager()
        
        # Groupes d'agents spécialisés
        self.group_a = {
            'philosophy': self.local_models.load_model('phi'),
            'math': self.local_models.load_model('mistral-small'),
            'code': self.local_models.load_model('codellama-small')
        }
        
        self.group_b = {
            'image': self.image_teacher,
            'knowledge': self.api_models,
            'analysis': self.local_models.load_model('phi')
        }
        
        self.logger = logging.getLogger(__name__)
        
    async def process_query(self, query: str) -> Dict[str, Any]:
        """Traite une question avec débat entre agents."""
        
        # Monitore la mémoire avant le traitement
        memory_state = await self.memory_manager.monitor_memory()
        self.logger.info(f"Memory state before query: {memory_state}")
        
        try:
            # 1. Analyse initiale par les deux groupes
            group_a_response = await self._get_group_a_response(query)
            group_b_response = await self._get_group_b_response(query)
            
            # 2. Débat entre les groupes
            debate_history = []
            for round in range(3):  # 3 rounds de débat
                # Groupe A critique B
                a_critique = await self._generate_critique(
                    'group_a',
                    group_a_response,
                    group_b_response
                )
                
                # Groupe B défend et contre-critique
                b_defense = await self._generate_defense(
                    'group_b',
                    group_b_response,
                    a_critique
                )
                
                debate_history.append({
                    'round': round,
                    'a_critique': a_critique,
                    'b_defense': b_defense
                })
                
                # Mise à jour des réponses
                group_a_response = await self._refine_response('group_a', a_critique)
                group_b_response = await self._refine_response('group_b', b_defense)
                
            # 3. Synthèse finale
            synthesis = await self._generate_synthesis(
                query,
                group_a_response,
                group_b_response,
                debate_history
            )
            
            return {
                'query': query,
                'final_response': synthesis,
                'debate_history': debate_history,
                'group_a_final': group_a_response,
                'group_b_final': group_b_response
            }
        finally:
            # Optimise la mémoire après le traitement
            await self.memory_manager.optimize_memory()
        
    async def _get_group_a_response(self, query: str) -> Dict[str, Any]:
        """Obtient la réponse du groupe A (philosophie, math, code)."""
        responses = await asyncio.gather(
            self.group_a['philosophy'].generate(query),
            self.group_a['math'].generate(query),
            self.group_a['code'].generate(query)
        )
        return self._merge_responses(responses)
        
    async def _get_group_b_response(self, query: str) -> Dict[str, Any]:
        """Obtient la réponse du groupe B (image, connaissance, analyse)."""
        # Si la requête concerne une image
        if 'image' in query.lower():
            image_response = await self.group_b['image'].generate_image(query)
            return {'type': 'image', 'content': image_response}
        else:
            return {'type': 'text', 'content': await self.group_b['knowledge'].query_free_model('phi', query)} 

    async def _generate_critique(self, 
                               group: str, 
                               own_response: Dict[str, Any],
                               other_response: Dict[str, Any]) -> Dict[str, Any]:
        """Génère une critique de la réponse de l'autre groupe."""
        
        critique_prompt = f"""
        Analysez de manière critique la réponse suivante en tant qu'expert :
        
        Réponse à analyser : {other_response['content']}
        
        Points à considérer :
        1. Précision technique
        2. Complétude
        3. Logique du raisonnement
        4. Points manquants importants
        
        Votre réponse précédente : {own_response['content']}
        """
        
        if group == 'group_a':
            # Utilise Phi-2 pour une critique rapide et précise
            critique = await self.local_models.generate(
                model_name='phi',
                prompt=critique_prompt,
                max_tokens=300
            )
        else:
            # Utilise l'API pour une perspective différente
            critique = await self.api_models.query_free_model(
                'phi',
                critique_prompt
            )
            
        return {
            'group': group,
            'critique': critique,
            'points': self._extract_critique_points(critique)
        }
        
    async def _generate_defense(self,
                              group: str,
                              own_response: Dict[str, Any],
                              critique: Dict[str, Any]) -> Dict[str, Any]:
        """Génère une défense contre la critique reçue."""
        
        defense_prompt = f"""
        Défendez votre réponse contre la critique suivante :
        
        Votre réponse initiale : {own_response['content']}
        
        Critique reçue : {critique['critique']}
        
        Points spécifiques à adresser :
        {critique['points']}
        
        Formulez une défense constructive et proposez des améliorations.
        """
        
        if group == 'group_a':
            model = self.group_a['analysis']
        else:
            model = self.group_b['analysis']
            
        defense = await model.generate(defense_prompt)
        
        return {
            'group': group,
            'defense': defense,
            'improvements': self._extract_improvements(defense)
        }
        
    async def _refine_response(self,
                              group: str,
                              debate_input: Dict[str, Any]) -> Dict[str, Any]:
        """Raffine la réponse basée sur le débat."""
        
        refinement_prompt = f"""
        Améliorez la réponse en tenant compte du débat :
        
        Points de critique : {debate_input['points']}
        Défense : {debate_input['defense']}
        Améliorations proposées : {debate_input['improvements']}
        
        Générez une réponse améliorée qui intègre ces éléments.
        """
        
        if group == 'group_a':
            # Utilise CodeLlama pour les aspects techniques
            refined = await self.group_a['code'].generate(refinement_prompt)
        else:
            # Utilise Phi pour la synthèse générale
            refined = await self.group_b['analysis'].generate(refinement_prompt)
            
        return {
            'content': refined,
            'refinement_history': debate_input
        }
        
    async def _generate_synthesis(self,
                                query: str,
                                group_a_final: Dict[str, Any],
                                group_b_final: Dict[str, Any],
                                debate_history: List[Dict[str, Any]]) -> str:
        """Génère une synthèse finale des deux perspectives."""
        
        synthesis_prompt = f"""
        Question originale : {query}
        
        Perspective Groupe A : {group_a_final['content']}
        Perspective Groupe B : {group_b_final['content']}
        
        Historique du débat :
        {self._format_debate_history(debate_history)}
        
        Générez une synthèse qui :
        1. Combine les meilleures idées des deux groupes
        2. Résout les contradictions
        3. Fournit une réponse complète et équilibrée
        """
        
        # Utilise Phi-2 pour la synthèse finale
        return await self.local_models.generate(
            model_name='phi',
            prompt=synthesis_prompt,
            max_tokens=500
        ) 