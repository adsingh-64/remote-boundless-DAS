# %%
"""
This script trains Boundless DAS on a remotely hosted nnsight model for a Yes/No
classification task. It expects a folder in the same working directory called `counterfactuals` 
containing CSV files with counterfactuals named `train.csv`, `val.csv`, and `test.csv`.
Each CSV should have the following three columns (they can have more):

    - "base_prompt": a base prompt string
    - "source_prompt": a source prompt string
    - "cf_label": a label string, either "yes" or "no"
"""

# %%
from typing import Type, TypeAlias, Literal, List, Dict
import torch as t
from torch import Tensor
from torch.utils.data import DataLoader
from datasets import Dataset as hf_Dataset
from datasets import load_dataset
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F
from tqdm import tqdm, trange
from jaxtyping import Float, Int
import nnsight
from nnsight import LanguageModel
from pyvene import BoundlessRotatedSpaceIntervention

# %%
model = LanguageModel("meta-llama/Meta-Llama-3.1-405B-Instruct") # Replace with your desired model. See available models here: https://nnsight.net/status/
remote = True

# %%
INTERVENTION_TOKEN_POS: int = -12  # Position in the token sequence to intervene
YES_TOKEN_ID: int = 7566 # Replace with your model tokenizer's correct ID. Beware of ' Yes' vs 'Yes'!
NO_TOKEN_ID: int = 2360

# %%
def calculate_loss(
    logits: Float[Tensor, "batch 2"],
    labels: Int[Tensor, "batch"],
    subspace_proj: BoundlessRotatedSpaceIntervention,
    mask_weight: float = 0.0
) -> Tensor:
    """
    Calculates the cross-entropy loss for binary classification (yes/no),
    plus an optional boundary regularization term from the subspace intervention.
    
    Args:
        logits: Tensor of shape (B, 2) containing log-probs for [No, Yes].
        labels: Tensor of shape (B,) with IDs in {NO_TOKEN_ID, YES_TOKEN_ID}.
        subspace_proj: BoundlessRotatedSpaceIntervention for boundary regularization.
        mask_weight: Weight for boundary loss term.
    
    Returns:
        A scalar Tensor representing the total loss.
    """
    label_classes = t.where(labels == NO_TOKEN_ID, 0, 1)
    loss = (logits[t.arange(logits.shape[0]), label_classes] * -1).mean()
    boundary_loss = mask_weight * subspace_proj.intervention_boundaries.sum()
    loss += boundary_loss
    return loss

# %%
def compute_metrics(
    eval_preds: List[Float[Tensor, "batch 2"]],
    eval_labels: List[Float[Tensor, "batch"]],
) -> Dict[str, float]:
    """
    Computes simple accuracy for binary classification over multiple batches.
    
    Args:
        eval_preds: List of (B, 2) Tensors of log probabilities for [No, Yes].
        eval_labels: List of (B,) Tensors with token IDs in {NO_TOKEN_ID, YES_TOKEN_ID}.
    
    Returns:
        Dictionary with key "accuracy" mapped to a float value.
    """
    total_count = 0
    correct_count = 0

    for log_probs, labels in zip(eval_preds, eval_labels):
        mapped_labels = []
        for token_id in labels:
            if token_id.item() == NO_TOKEN_ID:
                mapped_labels.append(0)
            elif token_id.item() == YES_TOKEN_ID:
                mapped_labels.append(1)
            else:
                mapped_labels.append(-100)

        mapped_labels_tensor = t.tensor(mapped_labels, device=log_probs.device, dtype=t.long)
        predicted = log_probs.argmax(dim=-1)
        valid_mask = mapped_labels_tensor != -100
        valid_preds = predicted[valid_mask]
        valid_labels = mapped_labels_tensor[valid_mask]

        total_count += valid_labels.size(0)
        correct_count += (valid_preds == valid_labels).sum().item()

    accuracy = 0.0 if total_count == 0 else correct_count / total_count
    return {"accuracy": round(accuracy, 2)}

# %%
# Load training data
train_data = load_dataset('csv', data_files='counterfactuals/val.csv')['train']

tokenized_base_inputs = model.tokenizer(train_data['base_prompt'], padding=True, return_tensors="pt")
tokenized_source_inputs = model.tokenizer(train_data['source_prompt'], padding=True, return_tensors="pt")
tokenized_labels = model.tokenizer(train_data['cf_label'], padding=True, return_tensors="pt")

train_dataset = hf_Dataset.from_dict({
    "input_ids": tokenized_base_inputs['input_ids'],
    "source_input_ids": tokenized_source_inputs['input_ids'],
    "labels": tokenized_labels['input_ids'][:, -1]
}).with_format("torch")

train_dataloader = DataLoader(train_dataset, batch_size=8)

# %%
subspace_proj = BoundlessRotatedSpaceIntervention(
    embed_dim=model.config.hidden_size
)
intervention_layer: int = 0

# %%
def batch_subspace_swap(
    inputs: Dict[str, Int[Tensor, "batch seq"]],
    intervention_layer: int,
    model: LanguageModel,
    subspace_proj: BoundlessRotatedSpaceIntervention
) -> Tensor:
    """
    1. Forward pass: gather hidden states at `intervention_layer`.
    2. Interchange subspace for the positions of interest.
    3. Re-inject manipulated hidden states and compute final log-probs over [NO, YES].
    
    Args:
        inputs: Dictionary containing "input_ids" and "source_input_ids".
        intervention_layer: The layer index at which to apply the subspace intervention.
        model: A LanguageModel object from nnsight.
        subspace_proj: BoundlessRotatedSpaceIntervention object.
    
    Returns:
        A Tensor of shape (B, 2) containing log probabilities for [NO_TOKEN_ID, YES_TOKEN_ID].
    """
    base_prompt = inputs["input_ids"]
    source_prompt = inputs["source_input_ids"]

    with model.trace(remote=False) as tracer:
        with tracer.invoke(base_prompt):
            base_hidden = (
                model.model.layers[intervention_layer]
                .output[0][:, INTERVENTION_TOKEN_POS]
                .save()
            )

        with tracer.invoke(source_prompt):
            source_hidden = (
                model.model.layers[intervention_layer]
                .output[0][:, INTERVENTION_TOKEN_POS]
                .save()
            )

    # Merge base_hidden and source_hidden in the subspace
    mixed_out = subspace_proj(base_hidden, source_hidden)

    with model.trace(base_prompt, remote=False):
        # Re-inject manipulated vector
        model.model.layers[intervention_layer].output[0][:, INTERVENTION_TOKEN_POS] = mixed_out
        final_logits = model.lm_head.output[:, -1]  # (B, vocab_size)
        final_log_probs = F.log_softmax(final_logits, dim=-1)[:, [NO_TOKEN_ID, YES_TOKEN_ID]].save()

    return final_log_probs

# %%
epochs: int = 1
temperature_start: float = 50.0
temperature_end: float = 0.1

t_total: int = len(train_dataloader) * epochs
warm_up_steps: int = int(0.1 * t_total)

optimizer_params = [
    {"params": subspace_proj.rotate_layer.parameters(), "lr": 1e-3},
    {"params": subspace_proj.intervention_boundaries, "lr": 1e-2},
]
optimizer = t.optim.Adam(optimizer_params)

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warm_up_steps, num_training_steps=t_total
)

temperature_schedule = t.linspace(temperature_start, temperature_end, steps=t_total).float()

subspace_proj.train()
total_step: int = 0

for epoch in range(epochs):
    print(f"Epoch {epoch}")
    for step, inputs in enumerate(tqdm(train_dataloader)):
        logits_2way = batch_subspace_swap(inputs, intervention_layer, model, subspace_proj)
        eval_metrics = compute_metrics([logits_2way.cpu()], [inputs["labels"].cpu()])
        loss = calculate_loss(logits_2way, inputs["labels"], subspace_proj)

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if total_step < len(temperature_schedule):
            subspace_proj.set_temperature(temperature_schedule[total_step])

        total_step += 1

        if step % 10 == 0:
            print(
                f"Step {step} -> Loss: {loss.item():.4f}, "
                f"Acc: {eval_metrics['accuracy']:.2f}"
            )

# %%
# Test the trained intervention
test_data = load_dataset("csv", data_files="counterfactuals/test.csv")["train"]

tokenized_base_test = model.tokenizer(test_data["base_prompt"], padding=True, return_tensors="pt")
tokenized_source_test = model.tokenizer(test_data["source_prompt"], padding=True, return_tensors="pt")
tokenized_labels_test = model.tokenizer(test_data["cf_label"], padding=True, return_tensors="pt")

test_dataset = hf_Dataset.from_dict({
    "input_ids": tokenized_base_test["input_ids"],
    "source_input_ids": tokenized_source_test["input_ids"],
    "labels": tokenized_labels_test["input_ids"][:, -1],
}).with_format("torch")

test_dataloader = DataLoader(test_dataset, batch_size=8)

# %%
with t.no_grad():
    subspace_proj.eval()
    eval_labels = []
    eval_preds = []

    for step, inputs in enumerate(tqdm(test_dataloader, desc="Testing")):
        outputs = batch_subspace_swap(inputs, intervention_layer, model, subspace_proj)
        eval_labels.append(inputs["labels"].cpu())
        eval_preds.append(outputs.cpu())

    eval_metrics = compute_metrics(eval_preds, eval_labels)

print(f"Boundless DAS test accuracy: {eval_metrics['accuracy']}")

# %%
# Save the subspace intervention state
t.save(subspace_proj.state_dict(), "bdas_state_dict.pt")
