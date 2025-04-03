from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np

class VectorStorage(ABC):
    """
    Interface for a vector storage/database system.
    """

    @abstractmethod
    def add_vector(self, vector: np.ndarray, metadata: Dict[str, Any]) -> str:
        """
        Add a feature vector along with associated metadata to the storage.

        Args:
            vector (np.ndarray): The facial embedding or feature vector.
            metadata (Dict[str, Any]): Additional information such as person ID, timestamp, etc.
        
        Returns:
            str: A unique identifier for the stored vector.
        """
        pass

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for the most similar vectors to the given query vector.

        Args:
            query_vector (np.ndarray): The query vector to match against stored vectors.
            top_k (int, optional): The number of top matching results to return. Defaults to 5.
        
        Returns:
            List[Dict[str, Any]]: A list of matching records, each containing metadata and similarity score.
        """
        pass
