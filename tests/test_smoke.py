def test_health_stub():
    assert 2 + 2 == 4


def test_imports():
    """Ensure main packages import without error"""
    import orchestrator.langgraph_dag as dag
    import services.api_server as api

    assert callable(dag.initialize_orchestrator)
    assert api.app.title == "SERA API"
