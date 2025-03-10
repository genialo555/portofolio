"""
Module d'agents conversationnels.

Ce module fournit un système d'agents conversationnels basés sur des LLMs,
avec différentes spécialisations et capacités.
"""

# Imports de base
from .agent import Agent, AgentConfig, Message, Role
from .agent_manager import AgentManager, get_agent_manager
from .tools import Tool, ToolManager, get_tool_manager

# Agents spécialisés
from .agent_factory import (
    create_agent,
    create_teacher_agent,
    create_assistant_agent,
    create_rag_agent,
    create_hybrid_agent
)

# Nouveaux agents du système d'orchestration
from .specialized_agents import (
    ContentGeneratorAgent,
    SalesAgent,
    TechnicalSupportAgent,
    create_content_generator_agent,
    create_sales_agent,
    create_technical_support_agent
)

from .analyzer_agent import (
    RequestAnalyzerAgent,
    RequestType,
    create_request_analyzer_agent
)

from .coordinator_agent import (
    CoordinatorAgent,
    RouteConfig,
    create_coordinator_agent
)

# Exporter les éléments publics
__all__ = [
    # Base
    "Agent", "AgentConfig", "Message", "Role",
    "AgentManager", "get_agent_manager",
    "Tool", "ToolManager", "get_tool_manager",
    
    # Factory et agents de base
    "create_agent", "create_teacher_agent", "create_assistant_agent", 
    "create_rag_agent", "create_hybrid_agent",
    
    # Agents spécialisés
    "ContentGeneratorAgent", "SalesAgent", "TechnicalSupportAgent",
    "create_content_generator_agent", "create_sales_agent", "create_technical_support_agent",
    
    # Système d'orchestration
    "RequestAnalyzerAgent", "RequestType", "create_request_analyzer_agent",
    "CoordinatorAgent", "RouteConfig", "create_coordinator_agent"
] 