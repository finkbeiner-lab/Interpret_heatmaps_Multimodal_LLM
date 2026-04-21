from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from peft import LoraConfig, get_peft_model
from peft import prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer, AutoProcessor, Qwen2VLForConditionalGeneration,
    get_scheduler
)
import torch
from torchvision import transforms

from accelerate import Accelerator


# ============ Accelerator Setup ============
#accelerator = Accelerator(mixed_precision="bf16")  # or "fp16"


model_name = "Qwen/Qwen2-VL-7B-Instruct"
lora_finetuned_model = "qwen2vl_lora_epoch99"
#device = "cuda" if torch.cuda.is_available() else "cpu"
accelerator = Accelerator(mixed_precision="bf16")  # or "fp16"
device = accelerator.device
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True )  #, min_pixels=min_pixels, max_pixels=max_pixels)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
    #device_map=device,  # handled by Accelerate,
)

# Prepare for 8-bit training if needed

##model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# Apply LoRA
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
)
#model = get_peft_model(model, lora_config)
model = PeftModel.from_pretrained(model, lora_finetuned_model)

#model =  model.to(device)

system_message = "You are a pathologist. Give answer in English"

def format_data(sample):
    return {
      "images": [sample["image"]],
      "messages": [

          {
              "role": "system",
              "content": [
                  {
                      "type": "text",
                      "text": system_message
                  }
              ],
          },
          {
              "role": "user",
              "content": [
                  {
                      "type": "image",
                      "image": sample["image"],
                  },
                  {
                      "type": "text",
                      "text": sample['query'],
                  }
              ],
          }
      ]
      }
    
item =    {
        "image": "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/crop_based/dino/output/attentions/DLB/Striatum/C110-115/DLB-2013-131-Striatum-C110-115_files_13_51.png",
        "query": "First image is original image, the brown aggregates are Lewy Bodies if they are on neurons and Lewy Neurites if they are on neurites, and second shows attention maps. Show color wise which cell types such as neurons, oligodendrites, lewy bodies, lewy neurites etc are highlighted in the right image, (red shows high activation , blue shows low)"
}

formatted_item = format_data(item)["messages"]
full_conversation = formatted_item
full_result = processor.apply_chat_template(formatted_item, tokenize=True, return_dict=True, return_tensors="pt",add_generation_prompt=True)
#print(full_result) 

full_result["input_ids"] = full_result["input_ids"].to(device)
full_result["attention_mask"] = full_result["attention_mask"].to(device)
full_result["pixel_values"] = full_result["pixel_values"].to(device)
full_result["image_grid_thw"] = full_result["image_grid_thw"].to(device)
model.eval()

"""
outputs = model(**full_result)


pred_ids = torch.argmax(outputs["logits"], dim=-1)  # [1, 358]

print(pred_ids)

print(len(pred_ids))


answer = processor.decode(pred_ids[0])

print(answer)
"""
with torch.no_grad():
    pred_ids = model.generate(
        input_ids=full_result["input_ids"],
        attention_mask=full_result["attention_mask"],
        pixel_values=full_result["pixel_values"],
        image_grid_thw=full_result["image_grid_thw"],
        max_new_tokens=512,
    )

# Decode only the newly generated tokens
input_len = full_result["input_ids"].shape[1]
answer = processor.decode(pred_ids[0][input_len:], skip_special_tokens=True)
print(answer)


#answer = processor.decode(pred_ids[1])