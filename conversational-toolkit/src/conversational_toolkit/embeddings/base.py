from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class EmbeddingsModel(ABC):
    """
    Abstract base class for embeddings models.

    Attributes:
        model_name (str): The name of the embeddings model.
        embedding_size (int): The size of the embedding vector.
    """

    @abstractmethod
    async def get_embeddings(self, texts: str | list[str]) -> NDArray[np.float64]:
        """
        Retrieves the embedding for the given text.

        Args:
            texts (list[str]): The input text for which the embedding needs to be retrieved.

        Returns:
            np.ndarray: The embedding vector for the input text.
        """
        pass
