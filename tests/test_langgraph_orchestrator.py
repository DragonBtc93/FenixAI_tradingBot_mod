"""
Tests para el LangGraph Orchestrator.
"""
import pytest


class TestFenixAgentState:
    """Tests para el estado del agente."""

    def test_state_type_definition(self):
        """Verificar definición del tipo de estado."""
        from src.core.langgraph_orchestrator import FenixAgentState

        # FenixAgentState es un TypedDict
        assert hasattr(FenixAgentState, '__annotations__')

        annotations = FenixAgentState.__annotations__
        assert 'symbol' in annotations
        assert 'timeframe' in annotations
        assert 'current_price' in annotations


class TestHelperFunctions:
    """Tests para funciones auxiliares."""

    def test_merge_dicts(self):
        """Verificar merge de diccionarios."""
        from src.core.langgraph_orchestrator import merge_dicts

        a = {"key1": "value1"}
        b = {"key2": "value2"}
        result = merge_dicts(a, b)

        assert result == {"key1": "value1", "key2": "value2"}

    def test_merge_dicts_override(self):
        """Verificar que b sobreescribe a."""
        from src.core.langgraph_orchestrator import merge_dicts

        a = {"key": "old"}
        b = {"key": "new"}
        result = merge_dicts(a, b)

        assert result["key"] == "new"

    def test_append_lists(self):
        """Verificar concatenación de listas."""
        from src.core.langgraph_orchestrator import append_lists

        a = [1, 2]
        b = [3, 4]
        result = append_lists(a, b)

        assert result == [1, 2, 3, 4]


class TestReasoningBankHelpers:
    """Tests para helpers de ReasoningBank."""

    def test_get_agent_context_no_bank(self):
        """Verificar comportamiento sin ReasoningBank."""
        from src.core.langgraph_orchestrator import get_agent_context_from_bank

        result = get_agent_context_from_bank(
            reasoning_bank=None,
            agent_name="technical",
            current_prompt="Test prompt",
        )

        assert result == ""

    def test_store_agent_decision_no_bank(self):
        """Verificar almacenamiento sin ReasoningBank."""
        from src.core.langgraph_orchestrator import store_agent_decision

        result = store_agent_decision(
            reasoning_bank=None,
            agent_name="technical",
            prompt="Test",
            result={"action": "BUY"},
            raw_response="",
            backend="ollama",
            latency_ms=100.0,
        )

        assert result is None


class TestFenixTradingGraph:
    """Tests para el grafo de trading."""

    def test_langgraph_availability_check(self):
        """Verificar check de disponibilidad de LangGraph."""
        from src.core.langgraph_orchestrator import LANGGRAPH_AVAILABLE

        # Solo verificar que la variable existe
        assert isinstance(LANGGRAPH_AVAILABLE, bool)

    @pytest.mark.skipif(
        True,  # Skip por defecto, requiere LangGraph instalado
        reason="Requiere LangGraph instalado"
    )
    def test_graph_creation(self):
        """Verificar creación del grafo."""
        from src.core.langgraph_orchestrator import get_trading_graph

        graph = get_trading_graph()
        assert graph is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
