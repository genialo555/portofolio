# Plan de travail: Système mixte RAG/KAG en pools d'agents

## Phase 1: Configuration de l'infrastructure

1. **Sélection et configuration des APIs LLM**
   - Configurer les accès API pour Qwen, Google et DeepSeek
   - Tester les limites de requêtes et la latence de chaque API
   - Établir le système de gestion des clés API et de rotation

2. **Définition des pools d'agents**
   - Pool A (Commercial): 4 configurations différentes
   - Pool B (Marketing): 4 configurations différentes
   - Pool C (Sectoriel): 4 configurations différentes

3. **Configuration des paramètres par agent**
   - Définir les variations de température
   - Configurer les paramètres top_p, top_k
   - Établir les longueurs de contexte variables

## Phase 2: Conception des prompts

1. **Développement des prompts de base**
   - Prompt principal pour le routage initial
   - Prompts spécifiques pour chaque pool d'agents
   - Prompts de débat pour les modèles KAG vs RAG

2. **Création des méta-prompts**
   - Prompt d'orchestration pour gérer le flux entre les pools
   - Prompt du détecteur d'anomalies
   - Prompt de synthèse finale

3. **Fine-tuning des prompts**
   - Collecte de données exemplaires pour chaque domaine
   - Optimisation des prompts selon des cas d'usage spécifiques
   - Tests A/B des variations de prompts

## Phase 3: Développement du pipeline

1. **Système d'orchestration**
   - Développer le routeur principal des requêtes
   - Implémenter la gestion des pools parallèles
   - Créer le système de collecte des outputs

2. **Mécanisme de débat KAG vs RAG**
   - Implémenter l'extraction de connaissances KAG
   - Développer le système de recherche RAG
   - Créer le protocole de débat et confrontation

3. **Module de synthèse**
   - Développer l'algorithme de synthèse multisource
   - Implémenter la détection et résolution de contradictions
   - Créer le formateur de réponse finale

## Phase 4: Tests et optimisation

1. **Tests unitaires**
   - Tester chaque pool d'agents séparément
   - Valider les différentes configurations de température
   - Évaluer la qualité des outputs individuels

2. **Tests d'intégration**
   - Tester le pipeline complet avec des cas simples
   - Valider le flux de débat KAG/RAG
   - Évaluer la qualité de la synthèse

3. **Optimisation basée sur les feedbacks**
   - Ajuster les prompts selon les résultats
   - Optimiser les paramètres des agents
   - Raffiner le processus de débat et synthèse 