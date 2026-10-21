"""Client Qdrant local avec fermeture idempotente.
"""
import threading
from qdrant_client import QdrantClient


class LocalQdrantClient(QdrantClient):
    def __init__(self, *args, **kwargs):
        # Initialiser avant super : __del__ peut aussi suivre un échec de création.
        self._rag_close_lock = threading.RLock()
        self._rag_cleanup_done = False
        super().__init__(*args, **kwargs)

    def close(self, grpc_grace=None, **kwargs):
        with self._rag_close_lock:
            if self._rag_cleanup_done:
                return
            super().close(grpc_grace=grpc_grace, **kwargs)
            # Une fermeture en échec n'est pas marquée réussie : le propriétaire
            # garde la possibilité de libérer la connexion correctement.
            self._rag_cleanup_done = True
