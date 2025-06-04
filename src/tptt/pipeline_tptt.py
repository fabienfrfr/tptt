import torch
from transformers import Pipeline


class TpttPipeline(Pipeline):
    """Pipeline for TPTT model inference."""

    def __init__(self, model, tokenizer, device=None, **kwargs):
        """
        Initialize TpttPipeline.
        """
        super().__init__(model=model, tokenizer=tokenizer, device=device, **kwargs)

    def _sanitize_parameters(self, **kwargs):
        # No special parameter handling for now
        preprocess_kwargs = {}
        forward_kwargs = {}
        postprocess_kwargs = {}
        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def preprocess(self, prompt):
        # Tokenize the input prompt
        return self.tokenizer(prompt, return_tensors="pt", truncation=False)

    def _forward(self, model_inputs, **forward_params):
        # Move tensors to the correct device
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
        # Use generate for text generation
        with torch.no_grad():
            output = self.model.generate(
                **model_inputs,
                max_new_tokens=forward_params.get("max_new_tokens", 50),
                do_sample=forward_params.get("do_sample", False),
                # cache_implementation=forward_params.get("cache_implementation", "static"),
            )
        return {"generated_ids": output}

    def postprocess(self, model_outputs):
        # Decode the generated ids into text
        generated_ids = model_outputs["generated_ids"]
        return [
            {"generated_text": self.tokenizer.decode(ids, skip_special_tokens=True)}
            for ids in generated_ids
        ]
