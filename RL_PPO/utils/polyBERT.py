import torch
from transformers import AutoTokenizer, AutoModel


def Embedding_smiles(model_path, smiles):
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    polyBERT = AutoModel.from_pretrained(model_path)

    encoded_input = tokenizer(smiles, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        model_output = polyBERT(**encoded_input)

    fingerprints = mean_pooling(model_output, encoded_input['attention_mask'])

    # print(fingerprints.detach().numpy().flatten())
    return fingerprints.detach().numpy().flatten()

