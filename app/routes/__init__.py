from fastapi import APIRouter
from . import user_routes, resume_routes, job_routes, enhancement_routes

router = APIRouter()

router.include_router(user_routes.router, prefix="/users")
router.include_router(resume_routes.router, prefix="/resumes")
router.include_router(job_routes.router, prefix="/job-descriptions")
router.include_router(enhancement_routes.router, prefix="/enhancements")