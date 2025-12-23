from fastapi import APIRouter
from api.services.store import load_review_queue

router = APIRouter(tags=["review"])


@router.get("/review/queue")
def review_queue():
    return {"items": load_review_queue(limit=200)}
