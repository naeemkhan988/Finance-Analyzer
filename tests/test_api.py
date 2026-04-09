"""
API Tests
=========
Tests for the FastAPI backend endpoints.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestHealthEndpoints:
    """Test health check endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from backend.main import app
        return TestClient(app)

    def test_root(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"

    def test_health(self, client):
        """Test health check."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestQueryEndpoints:
    """Test query endpoints."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        return TestClient(app)

    def test_empty_query(self, client):
        """Test query with empty question."""
        response = client.post("/api/query", json={"question": ""})
        assert response.status_code in [400, 422]

    def test_query_no_pipeline(self, client):
        """Test query when pipeline is not initialized."""
        response = client.post(
            "/api/query",
            json={"question": "What is the revenue?"},
        )
        # Should return 503 or graceful error
        assert response.status_code in [200, 503]
