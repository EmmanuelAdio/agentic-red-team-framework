from pymongo import MongoClient
from pymongo.database import Database

from OLD.app.core.settings import Settings


class MongoDatabaseFactory:
    """Creates a MongoDB database handle for repositories and services."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._client: MongoClient | None = None
        self._database: Database | None = None

    @property
    def database(self) -> Database:
        if self._database is None:
            self._database = self._create_database()
        return self._database

    def _create_database(self) -> Database:
        if self._settings.mongo_use_mock:
            import mongomock

            # Use default in-memory mongomock client to avoid SRV/DNS parsing
            # from real Mongo URIs during local tests.
            self._client = mongomock.MongoClient()
        else:
            self._client = MongoClient(self._settings.mongodb_uri)
        return self._client[self._settings.mongodb_db_name]

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
