from dataclasses import dataclass

from fastapi import Header, HTTPException


DEFAULT_USER_ID = "default-user"
DEFAULT_PROJECT_ID = "default-project"
MAX_SCOPE_ID_LENGTH = 200


@dataclass(frozen=True)
class RequestScope:
    user_id: str
    project_id: str


def get_request_scope(
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    x_project_id: str | None = Header(default=None, alias="X-Project-Id"),
) -> RequestScope:
    user_id = (x_user_id or DEFAULT_USER_ID).strip()
    project_id = (x_project_id or DEFAULT_PROJECT_ID).strip()

    if not user_id or not project_id:
        raise HTTPException(
            status_code=400,
            detail="X-User-Id and X-Project-Id cannot be empty",
        )
    if len(user_id) > MAX_SCOPE_ID_LENGTH or len(project_id) > MAX_SCOPE_ID_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Scope IDs must be at most {MAX_SCOPE_ID_LENGTH} characters",
        )

    return RequestScope(user_id=user_id, project_id=project_id)
