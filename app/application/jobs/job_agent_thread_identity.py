from __future__ import annotations

import hashlib


def build_job_agent_thread_id(owner_id: str) -> str:
    """Return the one server-controlled conversation thread for an owner.

    The browser never provides this value.  Hashing prevents a database
    checkpoint row from exposing the internal owner UUID as its thread id.
    """
    normalized_owner_id = owner_id.strip()
    if not normalized_owner_id:
        raise ValueError("job agent owner_id must not be blank")
    digest = hashlib.sha256(
        normalized_owner_id.encode("utf-8")
    ).hexdigest()
    return f"jobscope-owner-{digest}"
