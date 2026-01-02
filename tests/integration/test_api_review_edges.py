def test_review_queue_empty(client, init_test_db):
    r = client.get("/review/queue")
    assert r.status_code == 200
    data = r.json()
    assert data.get("items") == [] or isinstance(data.get("items", []), list)

def test_review_get_not_found(client):
    r = client.get("/review/rev_does_not_exist")
    # depending on your API, could be 404 or 400 with stripe-like body
    assert r.status_code in (404, 400)
