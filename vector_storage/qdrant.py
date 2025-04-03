import uuid
import numpy as np
import os
from typing import List, Dict, Any
from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams, Distance
from qdrant_client.http.models import PointsSelector
from interface.vector_storage import VectorStorage as VectorStorageInterface

class Qdrant(VectorStorageInterface):
    """
    Concrete implementation of the VectorStorage interface using Qdrant.
    """

    def __init__(
        self,
        url: str,
        collection_name: str,
        similariry_threshold: float,
        dimension: int = 512,
    ):
        """
        Initialize the Qdrant client and ensure the collection exists.
        
        Args:
            url (str): Qdrant server URL (e.g., "http://localhost:6333").
            collection_name (str): Name of the collection for face embeddings.
            dimension (int): dimension of vectors
        """
        self.collection_name = collection_name
        self.client = QdrantClient(url=url)
        self.dimension = dimension
        self.similarity_threshold = similariry_threshold
        self._initialize_collection()

    def _initialize_collection(self):
        """
        Initialize the collection in Qdrant only if it does not exist.
        """
        try:
            # Check if collection exists
            self.client.get_collection(self.collection_name)
        except Exception:
            # Collection does not exist, create it
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=Distance.COSINE
                )
            )


    def add_vector(self, vector: np.ndarray, metadata: Dict[str, Any]) -> str:
        """
        Add a feature vector with associated metadata to the Qdrant storage.
        
        Args:
            vector (np.ndarray): The facial embedding or feature vector.
            metadata (Dict[str, Any]): Additional metadata (e.g., person ID, timestamp).
        
        Returns:
            str: A unique identifier for the stored vector.
        """
        # Generate a unique ID for the vector.
        vector_id = str(uuid.uuid4())
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=vector_id,
                    payload=metadata,
                    vector=vector.tolist() if isinstance(vector, np.ndarray) else vector
                ),
            ],
        )
        print(f"Vector with ID {vector_id} added to Qdrant collection {self.collection_name}.")
        return vector_id

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for the most similar vectors to the query vector.
        
        Args:
            query_vector (np.ndarray): The query feature vector.
            top_k (int, optional): Number of top matches to return. Defaults to 5.
        
        Returns:
            List[Dict[str, Any]]: A list of matching records containing metadata and similarity score.
        """
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k,
            with_payload=True,
            with_vectors=False
        )
        # Process results to include metadata and similarity score.
        results = []
        for result in search_result:
            results.append({
                "id": result.id,
                "score": result.score,
                "metadata": result.payload
            })

        return results
