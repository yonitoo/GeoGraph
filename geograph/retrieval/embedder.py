import logging
from typing import List

import torch
from transformers import AutoModel, AutoTokenizer

from geograph.config import EmbeddingConfig

logger = logging.getLogger(__name__)

_SENTENCE_TRANSFORMER_PREFIXES = ("BAAI/bge",)


class Embedder:
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.device = self._select_device()
        self._use_sentence_transformer = config.model_name.startswith(_SENTENCE_TRANSFORMER_PREFIXES)
        logger.info("Loading model: %s", config.model_name)
        if self._use_sentence_transformer:
            from sentence_transformers import SentenceTransformer
            self.st_model = SentenceTransformer(config.model_name, device=str(self.device))
            self.st_model.max_seq_length = config.max_length
        else:
            self.model = AutoModel.from_pretrained(config.model_name).eval().to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        logger.info("Embedder ready on %s", self.device)

    @staticmethod
    def _select_device() -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def embed(self, text: str) -> List[float]:
        """Generate an embedding for a single text."""
        if self._use_sentence_transformer:
            return self.st_model.encode(text, normalize_embeddings=True).tolist()

        inputs = self.tokenizer.encode_plus(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        sequence_output = outputs[0]
        mask = inputs["attention_mask"].unsqueeze(-1).expand(sequence_output.size()).float()
        summed = torch.sum(sequence_output * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        mean_pooled = summed / counts

        return mean_pooled.cpu().numpy()[0].tolist()

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        if not texts:
            return []

        if self._use_sentence_transformer:
            embeddings = self.st_model.encode(
                texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
            )
            return embeddings.tolist()

        all_embeddings: List[List[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            try:
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs)
                sequence_output = outputs[0]
                mask = (
                    inputs["attention_mask"]
                    .unsqueeze(-1)
                    .expand(sequence_output.size())
                    .float()
                )
                summed = torch.sum(sequence_output * mask, dim=1)
                counts = torch.clamp(mask.sum(dim=1), min=1e-9)
                mean_pooled = (summed / counts).cpu().numpy()
                all_embeddings.extend(mean_pooled.tolist())
            except Exception as exc:
                logger.warning(
                    "Batch embed failed (start=%d, size=%d): %s — falling back to single",
                    start, len(batch), exc,
                )
                for text in batch:
                    all_embeddings.append(self.embed(text))

        return all_embeddings