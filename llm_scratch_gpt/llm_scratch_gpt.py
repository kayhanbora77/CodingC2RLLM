import torch
import torch.nn as nn
import torch.nn.functional as F


class CharTokenizer:
    def __init__(self, text):
        self.vocab = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, tokens):
        return "".join(self.itos[t] for t in tokens)


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, heads=4, layers=2):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(512, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)

        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T).unsqueeze(0)

        x = self.embed(x) + self.pos_embed(pos)
        x = self.transformer(x)
        return self.lm_head(x)


def train(model, data, epochs=200, lr=1e-3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()

        x = data[:, :-1]
        y = data[:, 1:]

        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")


@torch.no_grad()
def generate(model, tokenizer, prompt, max_tokens=100):
    model.eval()
    tokens = tokenizer.encode(prompt)

    for _ in range(max_tokens):
        x = torch.tensor(tokens).unsqueeze(0)
        logits = model(x)
        next_token = torch.argmax(logits[0, -1]).item()
        tokens.append(next_token)

    return tokenizer.decode(tokens)


def main():
    text = "hello world. this is a tiny language model."
    tokenizer = CharTokenizer(text)

    data = torch.tensor(tokenizer.encode(text)).unsqueeze(0)

    model = MiniGPT(vocab_size=len(tokenizer.vocab))

    train(model, data)

    print(generate(model, tokenizer, "hello"))


if __name__ == "__main__":
    main()
