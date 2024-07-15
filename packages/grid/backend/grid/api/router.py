"""
Add each api routes to the application main router.
Accesing a specific URL the user would be redirected to the correct router
and the specific request handler.
"""

# third party
from fastapi import APIRouter

# server absolute
from grid.api.new.new import router as new_router

api_router = APIRouter()
api_router.include_router(new_router)
