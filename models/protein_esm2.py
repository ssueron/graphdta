import torch
import torch.nn as nn
import esm


class ESM2ProteinEncoder(nn.Module):
    def __init__(self, model_name='esm2_t12_35M_UR50D', output_dim=128, freeze=True, **kwargs):
        super(ESM2ProteinEncoder, self).__init__()

        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.batch_converter = self.alphabet.get_batch_converter()

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.esm_dim = self.model.embed_dim
        self.projection = nn.Linear(self.esm_dim, output_dim)

    def forward(self, sequences):
        if isinstance(sequences, torch.Tensor):
            raise ValueError("ESM2ProteinEncoder requires raw amino acid sequences, not encoded tensors")

        batch_labels = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
        _, _, batch_tokens = self.batch_converter(batch_labels)
        batch_tokens = batch_tokens.to(next(self.model.parameters()).device)

        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[self.model.num_layers])

        embeddings = results["representations"][self.model.num_layers]
        embeddings = embeddings[:, 1:-1, :].mean(dim=1)

        output = self.projection(embeddings)
        return output
