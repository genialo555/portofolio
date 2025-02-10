class DebateAgent:
    """Agent gérant le débat entre les groupes."""
    
    def __init__(self, max_rounds: int = 3):
        self.max_rounds = max_rounds
        self.anomaly_detector = AnomalyDetector("debate_monitoring")
        
    async def conduct_debate(self,
                           group_a_response: Dict[str, Any],
                           group_b_response: Dict[str, Any]) -> Dict[str, Any]:
        """Conduit un débat entre les deux groupes."""
        debate_history = []
        
        for round in range(self.max_rounds):
            # Vérification d'anomalie pendant le débat
            debate_state = {
                "round": round,
                "group_a": group_a_response,
                "group_b": group_b_response
            }
            
            anomaly_check = self.anomaly_detector.check(debate_state)
            if anomaly_check.has_anomaly:
                break
                
            # Analyse des arguments
            comparison = self.compare_responses(
                group_a_response,
                group_b_response
            )
            
            debate_history.append({
                "round": round,
                "comparison": comparison,
                "group_a_response": group_a_response,
                "group_b_response": group_b_response
            })
            
            # Mise à jour des réponses pour le prochain round
            group_a_response = self.generate_counter_arguments(
                group_a_response,
                group_b_response
            )
            group_b_response = self.generate_counter_arguments(
                group_b_response,
                group_a_response
            )
            
        return {
            "debate_history": debate_history,
            "final_consensus": self.reach_consensus(debate_history)
        } 