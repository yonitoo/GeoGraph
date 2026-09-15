from typing import List, Any, Dict

import torch
from transformers import AutoModel, AutoTokenizer

from geograph.constants import EMBEDDING_MODEL_NAME, \
    TOKENIZER_NAME


def load_model_and_tokenizer(model_name=EMBEDDING_MODEL_NAME, tokenizer_name=TOKENIZER_NAME) -> Dict[str, Any]:
    """
    Load the specified pretrained embedding model and its tokenizer.
    Sets device priority: MPS > CUDA > CPU.
    """
    print(f"Loading model: {model_name}")
    print(f"Loading tokenizer: {tokenizer_name}")
    model = AutoModel.from_pretrained(model_name).eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device (Apple Silicon GPU).")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device.")
    else:
        device = torch.device("cpu")
        print("Using CPU device.")

    model.to(device)
    mtd = {'model': model, 'tokenizer': tokenizer, 'device': device}
    print("Model and tokenizer loaded successfully.")
    return mtd


def embed(mtd: Dict[str, Any], text: str) -> List[float]:
    """
    Generate embeddings for a given text using the specified model components.
    """
    model = mtd['model']
    tokenizer = mtd['tokenizer']
    device = mtd['device']

    inputs = tokenizer.encode_plus(
        text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    sequence_output = outputs[0]
    attention_mask = inputs['attention_mask']
    mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
    sum_embeddings = torch.sum(sequence_output * mask_expanded, 1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
    mean_pooled_embeddings = sum_embeddings / sum_mask

    return mean_pooled_embeddings.cpu().numpy()[0].tolist()
