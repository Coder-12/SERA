def test_health_stub():
    assert 2 + 2 == 4


def test_imports():
    """Ensure main packages import without error"""
    import sera.api.api_server as api
    import sera.orchestrator.langgraph_dag as dag

    assert callable(dag.initialize_orchestrator)
    assert api.app.title == "SERA API"
