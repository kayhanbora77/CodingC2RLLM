from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "kayhan77/click2refund-phi35-mini-merged-4bit"  # or 16bit

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if "16bit" in model_id else torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,  # sometimes needed for Phi family
)
# Then same chat template + generate code
