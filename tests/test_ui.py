"""The retrieval page is a static file served by the app."""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_graph_retrieval.api.routes import ui


def test_ui_page_is_served():
    app = FastAPI()
    app.include_router(ui.router)
    response = TestClient(app).get("/ui")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert 'id="graph"' in response.text
    assert "/v1/graph/retrieve" in response.text
