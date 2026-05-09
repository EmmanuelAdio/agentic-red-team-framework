from OLD.app.core.container import ServiceContainer, get_container


def get_services() -> ServiceContainer:
    """FastAPI dependency wrapper for accessing shared services."""

    return get_container()
