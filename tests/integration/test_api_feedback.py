def test_feedback_export_empty(client):
    r = client.get("/feedback/export?limit=50")
    assert r.status_code == 200
    data = r.json()
    assert "items" in data
    assert isinstance(data["items"], list)


def test_feedback_summary(client):
    r = client.get("/feedback/summary")
    assert r.status_code == 200
    data = r.json()
    assert "feedback_total" in data
    assert "feedback_by_outcome" in data


def test_feedback_export_limit_param(client):
    r = client.get("/feedback/export?limit=1")
    assert r.status_code == 200
    data = r.json()
    assert "items" in data
    assert isinstance(data["items"], list)
    assert len(data["items"]) <= 1
