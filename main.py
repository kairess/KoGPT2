# opyrator launch-ui main:generate_text
import torch
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
from pydantic import BaseModel, Field

tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2",
    bos_token='</s>',
    eos_token='</s>',
    unk_token='<unk>',
    pad_token='<pad>',
    mask_token='<mask>') 

model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')


class TextGenerationInput(BaseModel):
    text: str = Field(
        title='Text Input',
        max_length=128,
    )
    max_length: int = Field(
        128,
        ge=5,
        le=128,
    )
    repetition_penalty: float = Field(
        2.0,
        ge=0.0,
        le=2.0,
    )


class TextGenerationOutput(BaseModel):
    generated_text: str = Field(...)


def generate_text(input: TextGenerationInput) -> TextGenerationOutput:
    input_ids = tokenizer.encode(input.text)
    gen_ids = model.generate(
        torch.tensor([input_ids]),
            max_length=input.max_length,
            repetition_penalty=input.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            use_cache=True)

    generated = tokenizer.decode(gen_ids[0,:].tolist())

    return TextGenerationOutput(generated_text=generated)
