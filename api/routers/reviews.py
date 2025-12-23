from __future__ import annotations

from fastapi import APIRouter, Query
from api.core.config import get_settings
from api.core.errors import ApiError
from api.services.store import init_db, list_reviews, get_review, update_review
from api.schemas.review import ReviewItem, ReviewUpdateRequest

router = APIRouter()


@router.get("/reviews", response_model=list[ReviewItem])
def list_all_reviews(status: str | None = Query(default=None)):
    settings = get_settings()
    init_db(settings.abs_sqlite_path())
    rows = list_reviews(settings.abs_sqlite_path(), status=status)
    return [ReviewItem(**r) for r in rows]


@router.get("/reviews/{review_id}", response_model=ReviewItem)
def get_one_review(review_id: str):
    settings = get_settings()
    init_db(settings.abs_sqlite_path())
    r = get_review(settings.abs_sqlite_path(), review_id)
    if not r:
        raise ApiError(404, "resource_missing", "Review not found.", param="review_id")
    return ReviewItem(**r)


@router.put("/reviews/{review_id}", response_model=ReviewItem)
def update_one_review(review_id: str, patch: ReviewUpdateRequest):
    settings = get_settings()
    init_db(settings.abs_sqlite_path())
    updated = update_review(settings.abs_sqlite_path(), review_id, patch.model_dump(exclude_none=True))
    if not updated:
        raise ApiError(404, "resource_missing", "Review not found.", param="review_id")
    return ReviewItem(**updated)
