from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

class ReIdentifier(ABC):
    """
    Interface for a face re-identifier.
    """

    @abstractmethod
    def add_new_user(self, image: np.ndarray, user_data: dict) -> str:
        """
        Add a new user to the re-identification system.
        
        Args:
            image (np.ndarray): Input image of the user.
            user_data (dict): Metadata associated with the user.
            
        Returns:
            str: Unique identifier for the added user.
        """
        pass

    @abstractmethod
    def reidentify(self, image: np.ndarray) -> list:
        """
        Re-identify face in an image and return identified user data.
        
        Args:
            image (np.ndarray): Input image.
            
        Returns:
            list: Data of the identified face.
        """
        pass

