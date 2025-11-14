import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import tiktoken
import gradio as gr
import math
from transformers import GPT2LMHeadModel # Only for loading pretrained config, not model class itself

# --- Model Definition (Copied from Model_Final.py) ---
# These classes are necessary to define the architecture for loading the state_dict

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean = 0.0, std = std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std = 0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024 # Match the block_size used during training

        model_config = GPTConfig(**config_args)
        model = cls(model_config) # Use cls(config) instead of GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

# --- Device Configuration ---
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"Using device: {device}")

# --- Load Model and Tokenizer ---
MODEL_PATH = 'fine_tuned_gpt2/gpt2_finetuned_best_loss_0.0941.pt' # Ensure this path is correct
MODEL_TYPE = 'gpt2' # The base model type used for fine-tuning

print(f"Loading GPT model from {MODEL_PATH}...")
# Create a GPTConfig for the model type to initialize the architecture
config_args = {
    'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
    'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
    'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
    'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
}[MODEL_TYPE]
config_args['vocab_size'] = 50257
config_args['block_size'] = 1024 # Match the block_size used during training

model_config = GPTConfig(**config_args)
model = GPT(model_config)

# Load the fine-tuned state dictionary
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval() # Set model to evaluation mode

# Load tokenizer
enc = tiktoken.get_encoding('gpt2')

# --- Text Generation Function ---
def generate_text(prompt: str, max_length: int, num_sequences: int, temperature: float = 0.7) -> str:
    if not prompt:
        return "Please enter a starting prompt."

    # Encode the prompt
    start_ids = enc.encode(prompt)
    if len(start_ids) == 0:
        return "Prompt resulted in no tokens. Please try a different prompt."

    x = (torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0).repeat(num_sequences, 1))

    # Generate text
    generated_sequences = []
    with torch.no_grad():
        while x.size(1) < max_length:
            logits = model(x)[0] # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size) - take the logits at the last position

            # Apply temperature
            logits = logits / temperature

            probs = F.softmax(logits, dim=-1) # (B, vocab_size)
            # Top-k sampling (default to 50 as in your script, but could be made a Gradio input)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1) # (B, 1)
            xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
            x = torch.cat((x, xcol), dim=1)

    # Decode and format results
    for i in range(num_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        generated_sequences.append(f"GENERATED SEQUENCE {i+1}:\n{decoded}\n")

    return "\n".join(generated_sequences)

# --- Gradio Interface ---
print("Setting up Gradio interface...")
iface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(lines=2, label="Enter your starting prompt:", placeholder="Type a sentence or two to get started..."),
        gr.Slider(minimum=10, maximum=100, value=50, step=5, label="Max Length of Generation (tokens):"),
        gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Number of Sequences to Generate:"),
        gr.Slider(minimum=0.1, maximum=1.5, value=0.7, step=0.1, label="Temperature (higher = more creative):"),
    ],
    # MODIFIED: Output Textbox is larger and non-interactive
    outputs=gr.Textbox(lines=15, label="Generated Text", interactive=False),
    title="🌌 Fine-tuned GPT-2 Text Generation App 🌌",
    description="Explore text generation with a GPT-2 model fine-tuned on your custom data. "
                "Enter a prompt, adjust parameters, and see the model's creative outputs!",
    allow_flagging="never",
    examples=[
        ["The quick brown fox jumps over the lazy dog", 50, 1, 0.7],
        ["In a land far, far away, there was a wizard", 70, 2, 0.8],
        ["The meaning of life is", 30, 1, 0.5],
    ]
)

if __name__ == "__main__":
    iface.launch()