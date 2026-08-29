"""Redis connection foundation.

Only client creation and a connectivity check live here for now. Celery and
background workers are introduced in Phase 7.
"""

from redis.asyncio import Redis

from evalyx.core.config import Settings


def create_redis_client(settings: Settings) -> Redis:
    """Create a Redis client from configuration."""
    return Redis.from_url(
        settings.redis_url,
        decode_responses=True,
    )


async def check_redis(client: Redis) -> bool:
    """Return True when Redis answers PING."""
    try:
        return bool(await client.ping())
    except Exception:
        return False
