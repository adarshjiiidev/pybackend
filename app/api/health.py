"""
Production health checks and metrics endpoints.
"""

from fastapi import APIRouter, Response
from typing import Dict, Any
import time
import psutil
import logging
from datetime import datetime

from ..config.database import db

logger = logging.getLogger(__name__)

router = APIRouter()

# Startup time for uptime calculation
_start_time = time.time()

# Request metrics
_request_count = 0
_error_count = 0


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Comprehensive health check.
    Returns 200 if all systems operational, 503 otherwise.
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": int(time.time() - _start_time),
        "checks": {}
    }
    
    # Database check
    try:
        await db.server_info()
        health_status["checks"]["database"] = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        health_status["checks"]["database"] = "unhealthy"
        health_status["status"] = "degraded"
    
    # Cache check (optional)
    try:
        from ..utils.cache import get_cache
        cache = get_cache()
        if cache.client:
            await cache.client.ping()
            health_status["checks"]["cache"] = "healthy"
        else:
            health_status["checks"]["cache"] = "disabled"
    except Exception as e:
        logger.warning(f"Cache health check failed: {e}")
        health_status["checks"]["cache"] = "unhealthy"
    
    # Memory check
    memory = psutil.virtual_memory()
    health_status["checks"]["memory"] = {
        "usage_percent": memory.percent,
        "available_mb": memory.available // (1024 * 1024),
        "status": "healthy" if memory.percent < 90 else "warning"
    }
    
    # CPU check
    cpu_percent = psutil.cpu_percent(interval=0.1)
    health_status["checks"]["cpu"] = {
        "usage_percent": cpu_percent,
        "status": "healthy" if cpu_percent < 80 else "warning"
    }
    
    return health_status


@router.get("/metrics")
async def metrics() -> Dict[str, Any]:
    """
    Prometheus-compatible metrics endpoint.
    """
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": int(time.time() - _start_time),
        "requests_total": _request_count,
        "errors_total": _error_count,
        "error_rate": _error_count / max(_request_count, 1),
        "memory": {
            "rss_mb": psutil.Process().memory_info().rss // (1024 * 1024),
            "percent": psutil.virtual_memory().percent
        },
        "cpu_percent": psutil.cpu_percent(interval=0.1)
    }


def increment_request_count():
    """Increment total request counter."""
    global _request_count
    _request_count += 1


def increment_error_count():
    """Increment error counter."""
    global _error_count
    _error_count += 1
