from dataclasses import dataclass
from typing import List, Dict, Any
from enum import Enum

class DebateStage(Enum):
    THESIS = "thesis"          # Proposition initiale
    ANTITHESIS = "antithesis"  # Contre-arguments
    SYNTHESIS = "synthesis"    # Résolution
    REFINEMENT = "refinement"  # Amélioration

@dataclass
class PilpoulAnalysis:
    """Structure d'analyse pilpoul."""
    initial_question: str
    perspectives: List[str]
    contradictions: List[Dict[str, str]]
    resolutions: List[str]
    deeper_insights: List[str]
    practical_applications: List[str]

class PilpoulEngine:
    """Moteur de débat basé sur la méthode pilpoul."""
    
    async def analyze_instagram_strategy(self, profile_data: Dict) -> PilpoulAnalysis:
        """Analyse une stratégie Instagram avec la méthode pilpoul."""
        
        # 1. THESIS - Analyse initiale
        initial_analysis = await self._generate_thesis(profile_data)
        
        # 2. ANTITHESIS - Challenges et contradictions
        challenges = await self._generate_antithesis(initial_analysis)
        
        # 3. SYNTHESIS - Résolution des contradictions
        synthesis = await self._generate_synthesis(initial_analysis, challenges)
        
        # 4. REFINEMENT - Approfondissement
        refined = await self._refine_analysis(synthesis)
        
        return PilpoulAnalysis(
            initial_question="Comment optimiser la stratégie Instagram?",
            perspectives=initial_analysis['perspectives'],
            contradictions=challenges['points'],
            resolutions=synthesis['solutions'],
            deeper_insights=refined['insights'],
            practical_applications=refined['applications']
        )
    
    async def _generate_thesis(self, data: Dict) -> Dict[str, Any]:
        """Génère la thèse initiale."""
        prompt = f"""
        Analysez cette stratégie Instagram selon la méthode pilpoul :
        
        DONNÉES DU PROFIL :
        {data}
        
        DIRECTIVES D'ANALYSE :
        1. Identifiez les principes fondamentaux
        2. Explorez les présupposés
        3. Examinez les implications
        4. Trouvez les patterns cachés
        5. Proposez des perspectives multiples
        """
        
        return await self._query_model(prompt)
    
    async def _generate_antithesis(self, thesis: Dict) -> Dict[str, Any]:
        """Génère l'antithèse - challenges et contradictions."""
        prompt = f"""
        Challengez cette analyse avec rigueur :
        
        THÈSE INITIALE :
        {thesis}
        
        POINTS À CHALLENGER :
        1. Validité des présupposés
        2. Contradictions internes
        3. Cas limites
        4. Exceptions potentielles
        5. Perspectives alternatives
        """
        
        return await self._query_model(prompt)
    
    async def _generate_synthesis(self, thesis: Dict, antithesis: Dict) -> Dict[str, Any]:
        """Génère une synthèse résolvant les contradictions."""
        prompt = f"""
        Créez une synthèse dialectique :
        
        THÈSE : {thesis}
        ANTITHÈSE : {antithesis}
        
        OBJECTIFS :
        1. Résoudre les contradictions
        2. Intégrer les perspectives
        3. Transcender les oppositions
        4. Proposer une vision unifiée
        5. Identifier les principes émergents
        """
        
        return await self._query_model(prompt) 