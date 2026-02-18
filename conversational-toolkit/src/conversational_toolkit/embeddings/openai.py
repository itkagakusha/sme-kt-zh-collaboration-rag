from typing import Union
from loguru import logger
import numpy as np
from numpy.typing import NDArray

from conversational_toolkit.embeddings.base import EmbeddingsModel
from openai import OpenAI


class OpenAIEmbeddings(EmbeddingsModel):
    """
    OpenAI embeddings model.

    Attributes:
        model_name (str): The name of the embeddings model.
    """

    def __init__(self, model_name: str):
        self.client = OpenAI()
        self.model_name = model_name
        logger.debug(f"OpenAI embeddings model loaded: {model_name}")

    def get_embeddings(self, texts: Union[str, list[str]]) -> NDArray[np.float64]:  # type: ignore
        """
        Retrieves the embedding for the given text(s) using OpenAI.

        Args:
            texts (Union[str, list[str]]): The input text or list of texts for which the embedding needs to be retrieved.

        Returns:
            np.ndarray: The embedding vector(s) for the input text(s).
        """
        if isinstance(texts, str):
            texts = [texts]

        response = self.client.embeddings.create(input=texts, model=self.model_name, dimensions=1024)
        embeddings = np.asarray([response.data[i].embedding for i in range(len(response.data))])

        logger.info(f"OpenAI embeddings shape: {embeddings.shape}")

        return embeddings
