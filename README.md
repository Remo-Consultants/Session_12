```markdown
# Fine-tuned GPT-2 Text Generation

## Project Overview

This project demonstrates how to fine-tune a pre-trained GPT-2 (124M parameters) model on a custom text dataset (`input.txt`) for text generation. The primary goal was to achieve a specific low loss (less than 0.09999) during training, which was successfully met, resulting in a model with a final best loss of **0.0941**. The training process utilizes the `AdamW` optimizer with the `OneCycleLR` learning rate scheduler for efficient convergence.

The project also outlines steps for hosting the code on GitHub and deploying an interactive AI application on Hugging Face Spaces.

## Features

*   **Pre-trained GPT-2 Fine-tuning:** Leverages the power of the GPT-2 124M parameter model as a strong starting point.
*   **Custom Dataset Training:** Fine-tunes the model on `input.txt` using a custom `DataLoaderLite`.
*   **Optimized Training:** Implements the `AdamW` optimizer alongside the `OneCycleLR` learning rate scheduler for improved training dynamics.
*   **Loss Monitoring & Model Saving:** Tracks training loss and automatically saves the best-performing model checkpoint to `fine_tuned_gpt2/`.
*   **Detailed Training Logs:** Captures comprehensive training progress, including loss and learning rate, to a timestamped log file in `training_logs/`.
*   **Text Generation:** Includes functionality to generate new text sequences based on a starting prompt after training.
*   **GPU Acceleration:** Configured to automatically detect and utilize CUDA or MPS (Apple Silicon) if available, falling back to CPU otherwise.

## Project Structure

```
.
├── .venv/                      # Python virtual environment
├── fine_tuned_gpt2/            # Directory to store best model checkpoints (e.g., gpt2_finetuned_best_loss_0.0941.pt)
├── training_logs/              # Directory to store training log files (e.g., training_log_YYYYMMDD-HHMMSS.txt)
├── input.txt                   # Your custom training data
├── Model_Final.py              # Main script for model training and generation
└── requirements.txt            # Python dependencies
```

## Setup and Model Training

Follow these steps to set up the project and train the model locally.

### 1. Clone the Repository (if starting from GitHub)

```bash
git clone <your-github-repo-url>
cd <your-repo-name>
```

### 2. Create and Activate a Python Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

**For Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**For macOS/Linux (Bash/Zsh):**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

Install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

**Important Note for GPU Usage:**
If you have an NVIDIA GPU, ensure you have a **CUDA-enabled PyTorch** installed. The `pip install torch` command might default to a CPU-only version. If your script reports using `cpu` despite having a GPU, you'll need to uninstall PyTorch and reinstall it with CUDA support. Visit [pytorch.org](https://pytorch.org/get-started/locally/) for the exact installation command tailored to your CUDA version (e.g., `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`).

### 4. Prepare Your Input Data

The model is trained on the text content provided in `input.txt`. This file should contain the data you want your GPT-2 model to learn from. The `DataLoaderLite` class in `Model_Final.py` handles reading and tokenizing this text.

**Example `input.txt` (first few lines):**
```
First sentence of your text.
Second sentence, providing more context.
Another line of interesting data for the model.
... (rest of your 40000+ lines of text)
```

### 5. Running Model Training and Text Generation

The `Model_Final.py` script automatically loads a pre-trained GPT-2 model, fine-tunes it on your `input.txt` data, saves the best checkpoint, and then performs text generation.

#### **Training Process Walkthrough (from `Model_Final.py`):**

The script utilizes a `DataLoaderLite` to process `input.txt`. This data loader:
*   Reads the entire `input.txt` file.
*   Tokenizes the text using `tiktoken` (GPT-2's tokenizer).
*   Prepares batches of input sequences (`x`) and their corresponding target sequences (`y`) for the model to predict the next token.

The training loop proceeds as follows:

```python
# From Model_Final.py: DataLoaderLite excerpt
class DataLoaderLite:
    def __init__(self, B, T):
        # ... existing code ...
        with open('input.txt', 'r') as f:
            text = f.read()
        self.enc = tiktoken.get_encoding('gpt2')
        tokens = self.enc.encode(text)
        self.tokens = torch.tensor(tokens)
        # ... existing code ...

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T) # inputs for the model
        y = (buf[1:]).view(B, T) # targets for the model
        # ... existing code ...
        return x, y

# From Model_Final.py: Training Loop excerpt
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
scheduler = OneCycleLR(optimizer, max_lr=learning_rate, total_steps=TOTAL_TRAINING_STEPS)

for i in range(TOTAL_TRAINING_STEPS):
    x, y = train_loader.next_batch() # Get next batch of input and target tokens
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x, y) # Forward pass: model predicts next token logits
    loss.backward() # Backward pass: compute gradients
    optimizer.step() # Update model weights using AdamW
    scheduler.step() # Adjust learning rate with OneCycleLR
    # ... logging and model saving ...
```

To initiate the training and subsequent generation:

```bash
python Model_Final.py
```

### Configuration Parameters (within `Model_Final.py`):

You can adjust these parameters to experiment with training:

*   `model_type`: `'gpt2'`, `'gpt2-medium'`, `'gpt2-large'`, `'gpt2-xl'`
*   `TOTAL_TRAINING_STEPS`: E.g., `10000`, `50000`
*   `learning_rate`: E.g., `3e-4`, `5e-4`
*   `TARGET_LOSS`: The desired loss to stop early. (Achieved: **0.0941**)
*   `B`: Batch size (e.g., `4`, `8`, `16`)
*   `T`: Sequence length / block size (e.g., `32`, `64`, `128`)

## Training Logs Showcase

The training progress, including the loss at each step and the dynamically adjusted learning rate from `OneCycleLR`, is meticulously logged.

**Example Log Entry:**

```
using device: cuda
loaded 40000 tokens
1 epoch = 312 batches
Loading pretrained GPT-2 model: gpt2
Using OneCycleLR scheduler with max_lr=0.0003 and total_steps=10000
step 0, loss: 10.8252, current_lr: 0.000013
step 1, loss: 10.5901, current_lr: 0.000013
...
step 995, loss: 3.3387, current_lr: 0.000084
...
step 5053, loss: 0.0941, current_lr: 0.000660
Saved best model with loss: 0.0941 to fine_tuned_gpt2\gpt2_finetuned_best_loss_0.0941.pt
Achieved target loss of 0.09999! Stopping training.
Final loss: 0.09410800784826279
```

This snippet shows the initial high loss, the gradual reduction, the dynamic learning rate, and the final achievement of the target loss at step 5053, with the best model saved.

## Sequence Generation Showcase

After training, the script automatically proceeds to generate text. The generation process starts with a `start_phrase`, and the model predicts subsequent tokens iteratively.

```python
# From Model_Final.py: Text Generation excerpt
start_phrase = "The quick brown fox" # The initial prompt for generation
# ... encoder and device setup ...
x = (torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0).repeat(num_return_sequences, 1))

log_message("\n" + "="*50 + "\nStarting text generation...\n" + "="*50)

while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)[0] # Model predicts logits for the next token
        logits = logits[:, -1, :] # Take logits for the last token in sequence
        probs = F.softmax(logits, dim=-1) # Convert logits to probabilities
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # Top-k sampling
        ix = torch.multinomial(topk_probs, 1) # Select a token based on probabilities
        xcol = torch.gather(topk_indices, -1, ix) # Get the actual token ID
        x = torch.cat((x, xcol), dim=1) # Append to the generated sequence

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    log_message(f"GENERATED TEXT {i+1}:")
    log_message(f"> {decoded}")
    # ... separator ...
```

**Example Generated Output:**

```
==================================================
Starting text generation...
==================================================
GENERATED TEXT 1:
> The quick brown foxes peering
Like lightning that make us wretched subjects!

GLOUCESTER:
A mankind witch, that
--------------------------------------------------
GENERATED TEXT 2:
> The quick brown foxes peering out of the forest at a sudden,
sharp, barking sound. A small, dark creature darted from a bush and
--------------------------------------------------
```
*(Note: Your actual generated text will vary based on your `input.txt` content and random sampling during generation.)*

## Deployment to GitHub

This project is designed to be easily hosted on GitHub.

1.  **Initialize Git (if not already):**
    ```bash
    git init
    ```
2.  **Add `.gitignore`:** Create a `.gitignore` file to prevent unwanted files from being committed:
    ```
    .venv/
    __pycache__/
    *.pt
    ```
3.  **Add and Commit Files:**
    ```bash
    git add .
    git commit -m "Initial commit: Fine-tuned GPT-2 project"
    ```
4.  **Create GitHub Repository:** Go to GitHub.com, create a new public repository.
5.  **Link and Push:**
    ```bash
    git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
    git branch -M main
    git push -u origin main
    ```
    Ensure you push the `fine_tuned_gpt2/` directory with your best model checkpoint and the `input.txt` file.

## Deployment to Hugging Face Spaces (AI App)

To demonstrate your AI app, you can deploy it on Hugging Face Spaces.

1.  **Create an `app.py`:** Create a new Python file (e.g., `app.py`) in your project root. This file will define the user interface for your AI application, typically using `Gradio` or `Streamlit`.
    *   This `app.py` will need to:
        *   Load the `GPT` class definition (you can copy relevant classes from `Model_Final.py` or structure your project to import them).
        *   Load the `tiktoken` encoder.
        *   Load your saved model weights (e.g., `fine_tuned_gpt2/gpt2_finetuned_best_loss_0.0941.pt`) using `model.load_state_dict()`.
        *   Define an inference function that takes a text prompt and returns generated text.
        *   Create a Gradio/Streamlit interface to expose this function.

2.  **Update `requirements.txt`:** If using Gradio or Streamlit, add them to your `requirements.txt` file:
    ```
    torch
    transformers
    tiktoken
    gradio  # or streamlit
    ```

3.  **Create a Hugging Face Space:**
    *   Go to [Hugging Face Spaces](https://huggingface.co/spaces).
    *   Click "Create new Space."
    *   Choose a Space name (e.g., `my-gpt2-finetuned-app`), select the appropriate SDK (Gradio or Streamlit), and choose a free CPU hardware.
4.  **Clone your Space and Upload Files:**
    ```bash
    git clone https://huggingface.co/spaces/<your-username>/<your-space-name>
    cd <your-space-name>
    # Copy app.py, requirements.txt, and the fine_tuned_gpt2/ directory into this folder
    git add .
    git commit -m "Add Gradio app and fine-tuned model"
    git push
    ```
Your Hugging Face Space should then automatically build and deploy your interactive AI application.

## Customization and Further Exploration

*   **Hyperparameter Tuning:** Experiment with `TOTAL_TRAINING_STEPS`, `learning_rate`, `B` (batch size), and `T` (sequence length) to potentially achieve even better results or faster training.
*   **Model Size:** Try `gpt2-medium`, `gpt2-large`, or `gpt2-xl` in `Model_Final.py` if you have sufficient computational resources.
*   **Dataset Expansion:** Use a larger and more diverse `input.txt` for broader capabilities.
*   **Sampling Strategies:** Explore different text generation sampling strategies beyond top-k (e.g., nucleus sampling) for varied outputs.

