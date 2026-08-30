import torch
from functools import lru_cache
from transformers import AutoTokenizer, AutoModel


@lru_cache(maxsize=4)
def _load_polybert(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.eval()
    return tokenizer, model


@lru_cache(maxsize=16384)
def _embedding_smiles_cached(model_path, smiles):
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    tokenizer, polyBERT = _load_polybert(model_path)

    encoded_input = tokenizer(smiles, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        model_output = polyBERT(**encoded_input)

    fingerprints = mean_pooling(model_output, encoded_input['attention_mask'])

    return fingerprints.detach().numpy().flatten()


def Embedding_smiles(model_path, smiles):
    # The environment does not mutate returned embeddings, but return a copy so
    # callers cannot modify the process-local cache accidentally.
    return _embedding_smiles_cached(str(model_path), smiles).copy()

