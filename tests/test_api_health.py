from fastapi.testclient import TestClient

from services.api_server import app

client = TestClient(app)


def test_health_endpoint():
    """Check that the /health endpoint returns correct JSON."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "SERA" in data["message"]
