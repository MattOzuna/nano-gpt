import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)
# ===================================================================================================#
# hyperparamters
batch_size = 32  # how many independent sequences will we process in parallel?
block_size = 8  # what is the maximum context length for predictions?
n_embd = 32
max_iters = 3000  # training cycles
eval_interval = 300  # how many chars generated when sampling
learning_rate = 1e-2
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"using {device} device")
eval_iters = 200  # how often you print the loss
dropout = 0.0
# ===================================================================================================#
# read data and sort data

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# print("length of dataset in characters: ", len(text))  # roughly 1 million chars
# print(text[:1000])

# sorted list of unique chars
chars = sorted(list(set(text)))
vocab_size = len(chars)
# print("".join(chars))
# print(vocab_size)

# creates a mapping from chars to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
# encoder: take a string, output a list of integers
encode = lambda s: [stoi[c] for c in s]
# decoder: take a list of integers, output a string
decode = lambda l: "".join([itos[i] for i in l])

# print(encode("hii there"))
# print(decode(encode("hii there")))

data = torch.tensor(encode(text), dtype=torch.long)
# print(data.shape, data.dtype)
# print(data[:1000])

# seperate training data and validation data
n = int(0.9 * len(data))  # first 90% of data is used for training
train_data = data[:n]
val_data = data[n:]

# ===================================================================================================#

block_size = 8  # what is the maximum context length for predictions?

x = train_data[:block_size]
y = train_data[1 : block_size + 1]

# shows how the context and the target corespond
# for t in range(block_size):
#     context = x[: t + 1]
#     target = y[t]
#     print(f"when input is {context} the target: {target}")

# ===================================================================================================#
# data loading

batch_size = 32  # how many independent sequences will we process in parallel?


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    # creates a batch size number of int between 0 and len - blocksize
    # this give the starting index for each batch
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # then we get the chars from each of these indexes
    x = torch.stack([data[i : i + block_size] for i in ix])
    # and the correct targets for each of those chars to be use later in the loss function
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# xb, yb = get_batch("train")
# print("inputs:")
# print(xb.shape)
# print(xb)
# print("targets:")
# print(yb.shape)
# print(yb)

# print("----")

# for b in range(batch_size):  # batch dimension
#     for t in range(block_size):  # time dimension
#         context = xb[b, : t + 1]
#         target = yb[b, t]
#         print(f"when input is {context.tolist()} the target: {target}")


# ===================================================================================================#
# loss estimation function
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# ===================================================================================================#
# Simple Bigram model

torch.manual_seed(1337)


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # This is the same as C from makemore
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # This truns them into B,T,C
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C) this works with broadcasting
        x = self.blocks(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            # torch's cross entropy function is expecting a B,C not a B,T,C
            logits = logits.view(B * T, C)
            # again the cross entropy function is expecting 1 dimesnional tensor and
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


# ===================================================================================================#

# Notes:
# - Attention is a **communication mechanism**. Can be seen as nodes in a directed graph looking at each other
# and aggregating information with a weighted sum from all nodes that point to them, with data-dependent weights.
# - There is no notion of space. Attention simply acts over a set of vectors. This is why we need to positionally encode tokens.
# - Each example across batch dimension is of course processed completely independently and never "talk" to each other
# - In an "encoder" attention block just delete the single line that does masking with `tril`,
# allowing all tokens to communicate. This block here is called a "decoder" attention block because it has triangular masking,
#  and is usually used in autoregressive settings, like language modeling.
# - "self-attention" just means that the keys and values are produced from the same source as queries.
# In "cross-attention", the queries still get produced from x, but the keys and values come from some other,
# external source (e.g. an encoder module)
# - "Scaled" attention additional divides `wei` by 1/sqrt(head_size).
# This makes it so when input Q,K are unit variance, wei will be unit variance too
# and Softmax will stay diffuse and not saturate too much. Illustration below

# ===================================================================================================#
# Attention implementation


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,C)
        q = self.query(x)  # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        #  makes sure future doesn't communicate with the past
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T),
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        # wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        # create multiple heads
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # run all of them in parrallel and concat into a single list on the channel (C) dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        # out = self.dropout(self.proj(out))
        return out


# ===================================================================================================#
# This layer helps the tokens further think about what their context is telling them


class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            # nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        # self.ln1 = nn.LayerNorm(n_embd)
        # self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(x)
        x = x + self.ffwd(x)
        # x = x + self.sa(self.ln1(x))
        # x = x + self.ffwd(self.ln2(x))
        return x


# ===================================================================================================#

model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
# for larger networs 3e-4 is a good starting point
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ===================================================================================================#
# Optimization

for iter in range(max_iters):
    # evaluate loss
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    # sample a batch
    xb, yb = get_batch("train")

    # evaulate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# ===================================================================================================#
# generate from the model

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
