from typing import Union, Any
from loguru import logger
from numpy._typing import NDArray
from sentence_transformers.models import Transformer, Pooling
from sentence_transformers import SentenceTransformer
from conversational_toolkit.embeddings.base import EmbeddingsModel
import numpy as np


class CustomizeSentenceTransformer(SentenceTransformer):  # type:ignore
    def _load_auto_model(self, model_name_or_path: str, *args: Any, **kwargs: Any) -> list[Transformer | Pooling]:
        """
        Creates a simple Transformer + CLS Pooling model and returns the modules
        """
        token = kwargs.get("token", None)
        cache_folder = kwargs.get("cache_folder", None)
        revision = kwargs.get("revision", None)
        trust_remote_code = kwargs.get("trust_remote_code", False)
        if "token" in kwargs or "cache_folder" in kwargs or "revision" in kwargs or "trust_remote_code" in kwargs:
            transformer_model = Transformer(
                model_name_or_path,
                cache_dir=cache_folder,
                model_args={
                    "token": token,
                    "trust_remote_code": trust_remote_code,
                    "revision": revision,
                },
                tokenizer_args={
                    "token": token,
                    "trust_remote_code": trust_remote_code,
                    "revision": revision,
                },
            )
        else:
            transformer_model = Transformer(model_name_or_path)
        pooling_model = Pooling(transformer_model.get_word_embedding_dimension(), "cls")
        return [transformer_model, pooling_model]


class SentenceTransformerEmbeddings(EmbeddingsModel):
    def __init__(self, model_name: str, **kwargs: Any):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, **kwargs)
        self.model.eval()
        logger.debug(f"Sentence Transformer embeddings model loaded: {model_name} with kwargs: {kwargs}")

    async def get_embeddings(self, texts: Union[str, list[str]], **kwargs_encode: Any) -> NDArray[np.float64]:
        """
        Encode a string or a list of strings into embeddings using the model.

        Parameters:
        texts (Union[str, list[str]]): A single string or a list of strings to be encoded.

        Returns:
        np.ndarray: A numpy array of embeddings.
        """
        # Encode the texts using the model
        embedded_chunk = self.model.encode(texts, **kwargs_encode)

        # If a single string is given, convert the output to a numpy array of numpy arrays
        if isinstance(texts, str):
            embedded_chunk = np.array([embedded_chunk])

        logger.debug(f"{self.model_name} embeddings size: {embedded_chunk.shape}")
        return embedded_chunk  # type: ignore
