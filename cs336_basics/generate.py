from collections.abc import Generator
import torch
import yaml

from cs336_basics.config import instantiate
from cs336_basics.model.checkpointing import load_checkpoint
from cs336_basics.model.misc import softmax


class LLMServing:
    def __init__(self, model, tokenizer, checkpoint: str):
        self.tokenizer = tokenizer
        device = next(model.parameters()).device
        if "mps" in str(device):
            self.model = torch.compile(model, backend="aot_eager")
        else:
            self.model = torch.compile(model)

        load_checkpoint(checkpoint, model)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        *,
        stop_token: bytes = b"<|endoftext|>",
        temperature: float = 0.0,
        top_p: float | None = None,
    ) -> Generator[str, None, None]:
        prompt_tokens = self.tokenizer.encode(prompt)

        inputs = torch.tensor(prompt_tokens, dtype=torch.int32, device="mps:0").unsqueeze(dim=0)
        with torch.no_grad():
            for _ in range(max_tokens):
                last_token_logits = self.model(inputs).squeeze(dim=0)[-1]
                last_token_probs = softmax(last_token_logits, dim=0, temperature=temperature)
                if top_p is not None:
                    sorted_token_probs, indices = torch.sort(last_token_probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_token_probs, dim=-1)
                    mask = cumulative_probs <= top_p
                    mask[..., 0] = True  # make sure at least 1 token is selected
                    top_p_probs = sorted_token_probs * mask
                    top_p_probs_normalized = top_p_probs / top_p_probs.sum(dim=-1, keepdim=True)
                    sampled_idx = torch.multinomial(top_p_probs_normalized, num_samples=1)
                    next_token_idx = indices[sampled_idx].cpu().item()
                else:
                    next_token_idx = torch.argmax(last_token_probs).cpu().item()
                token_bytes = self.tokenizer.vocab[next_token_idx]
                if token_bytes == stop_token:
                    break
                yield token_bytes.decode(errors="replace")
                inputs = torch.cat(
                    [inputs, torch.tensor([[next_token_idx]], dtype=torch.int32, device="mps:0")],
                    dim=1,
                )
                if inputs.shape[1] > self.model.context_length:
                    inputs = inputs[:, -self.model.context_length :]


if __name__ == "__main__":
    with open("./cs336_basics/experiments/tiny_stories.yaml") as file:
        config = {k: v for k, v in yaml.safe_load(file).items() if k in ["tokenizer", "model"]}
    entities = instantiate(config)
    g = LLMServing(
        entities["model"],
        entities["tokenizer"],
        checkpoint="./scratch/checkpoints/checkpoint_003500",
    )
    generation_kwargs = dict(max_tokens=256, temperature=0.2, top_p=0.8)
    for chunk in g.generate("Once upon a time, in a warm and sunny place", **generation_kwargs):
        print(
            chunk,
            end="",
            flush=True,
        )
