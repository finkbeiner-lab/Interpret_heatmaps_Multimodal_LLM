import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoProcessor, Qwen2VLForConditionalGeneration,
    get_scheduler
)
from torch.optim import AdamW

from peft import LoraConfig, get_peft_model
from peft import prepare_model_for_kbit_training

from accelerate import Accelerator
from PIL import Image
import json, os
import torch
from torchvision import transforms
from qwen_vl_utils import process_vision_info
from torch.nn.utils.rnn import pad_sequence
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import numpy as np

print(torch.cuda.device_count())

# ============ Accelerator Setup ============
accelerator = Accelerator(mixed_precision="bf16")  # or "fp16"
device = accelerator.device

#device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

min_pixels = 224*224
max_pixels = 2048*2048
IGNORE_INDEX=-100
# ============ Config ============
model_name = "Qwen/Qwen2-VL-7B-Instruct"
image_folder = "train_images"
#train_json = "lbd_qa_corpus.json"
train_json = "lbd_qa_corpus_big.json"
epochs = 100
lr = 2e-5
batch_size = 2

# ============ Load Model + Tokenizer + Processor ============
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True )  #, min_pixels=min_pixels, max_pixels=max_pixels)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    #device_map="auto"
    #device_map=device,  # handled by Accelerate,
)

# Prepare for 8-bit training if needed

#model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model.gradient_checkpointing_enable()

# Apply LoRA
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
)
model = get_peft_model(model, lora_config)

#model =  model.to(device)

# ============ Dataset ============
def load_json(path):
    """Load a standard JSON file (array of objects)."""
    with open(path, "r") as f:
        data = json.load(f)
    return data

# Example usage
annotations = load_json(train_json)
print(f"Loaded {len(annotations)} samples")

def load_test_data(test_json_path):
    with open(test_json_path, "r") as f:
        data = json.load(f)
    return data

test_data = load_test_data("lbd_qa_corpus_big_test.json")


system_message = "This is Monika's LBD bot. You are an English assistant. Respond only in English."

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
          },
          {
              "role": "assistant",
              "content": [
                  {
                      "type": "text",
                      "text": sample["label"]
                  }
              ],
          },
      ]
      }
    
def format_to_qwen_chat(full_conversation, assistant, processor):
    """
    Converts the structured message list into the single, tokenizable chat string
    and separates the full conversation from the input-only conversation.
    """
    # Create the full conversation string (User + Assistant) for input_ids/labels
    full_conversation = processor.apply_chat_template(
        full_conversation, 
        tokenize=False, 
        add_generation_prompt=False
    )
    
    # Create the input-only string (User turn only) for masking reference
    input_only_conversation = processor.apply_chat_template(
        assistant, # Exclude the last (assistant) turn
        tokenize=False, 
        add_generation_prompt=True # Add prompt to start generation
    )
    
    return full_conversation, input_only_conversation



# --- C. Tokenize and Mask Labels ---
def get_labels(full_result):
    """
    Tokenizes the text and applies masking to the labels tensor.
    """

    input_ids = full_result["input_ids"]
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids).unsqueeze(0)

    labels = torch.full_like(input_ids, IGNORE_INDEX)
    # Get token ids for reference
    user_token = tokenizer.convert_tokens_to_ids("user")
    assistant_token = tokenizer.convert_tokens_to_ids("assistant")
    #print("user_token",user_token)
    #print("assistant_token",assistant_token)
    eos_token = tokenizer.eos_token_id
    #print("eos_token",eos_token)
    input_ids_flat = input_ids[0].tolist()
    L = len(input_ids_flat)
    pos = 0
    while pos < L:
        #if input_ids_flat[pos] == 77091:
        if input_ids_flat[pos] == assistant_token:
            ans_start = pos + 2
            ans_end = ans_start
            #while ans_end < L and input_ids_flat[ans_end] != 151645:
            while ans_end < L and input_ids_flat[ans_end]!=eos_token:
                ans_end += 1
            if ans_end < L:
                #labels[0, ans_start : ans_end + 2] = input_ids[0, ans_start : ans_end + 2]
                labels[0, ans_start : ans_end + 1] = input_ids[0, ans_start : ans_end + 1]

                pos = ans_end
        pos += 1

    full_result["labels"] = labels
    full_result["input_ids"] = input_ids
    
    return full_result
    
class VLDataset(torch.utils.data.Dataset):
    def __init__(self, annotations, image_folder=None):
        self.ann = annotations
        self.image_folder = image_folder

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, idx):
        item = self.ann[idx]
        formatted_item = format_data(item)["messages"]
        full_conversation = formatted_item
        full_result = processor.apply_chat_template(formatted_item, tokenize=True, return_dict=True, return_tensors="pt")
        full_result = get_labels(full_result)
        # Return as a single dict
        return full_result

"""
def vl_collate_fn(batch):
    #print(batch)
    input_ids = [b["input_ids"] for b in batch]
    labels = [b["labels"] for b in batch]
    attention_mask = [b["attention_mask"] for b in batch]
    pixel_values = [b["pixel_values"] for b in batch]
    image_grid_thw = [b["image_grid_thw"] for b in batch]
    input_ids =torch.stack(input_ids)
    attention_mask =torch.stack(attention_mask)
    labels =  torch.stack(labels)
    # Stack images (processor already resized them)
    pixel_values = torch.stack(pixel_values)
    image_grid_thw = torch.stack(image_grid_thw)
    return {
        "input_ids": input_ids[0],
        "labels": labels[0],
        "attention_mask": attention_mask[0],
        "pixel_values": pixel_values[0],
        "image_grid_thw":image_grid_thw[0]
    }
"""


def vl_collate_fn(batch):
    input_ids = pad_sequence(
        [b["input_ids"].squeeze(0) for b in batch], 
        batch_first=True, 
        padding_value=tokenizer.pad_token_id
    )
    labels = pad_sequence(
        [b["labels"].squeeze(0) for b in batch], 
        batch_first=True, 
        padding_value=IGNORE_INDEX
    )
    attention_mask = pad_sequence(
        [b["attention_mask"].squeeze(0) for b in batch], 
        batch_first=True, 
        padding_value=0
    )
    # pixel_values and image_grid_thw need careful handling
    # if images are different sizes, they can't be naively stacked
    pixel_values = torch.cat([b["pixel_values"] for b in batch], dim=0)
    image_grid_thw = torch.cat([b["image_grid_thw"] for b in batch], dim=0)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }
    
train_dataset = VLDataset(annotations, image_folder=None)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=vl_collate_fn)

# ============ Optimizer & Scheduler ============
optimizer = AdamW(model.parameters(), lr=lr)
num_training_steps = len(train_dataloader) * epochs
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=100, num_training_steps=num_training_steps
)


# ============ Prepare everything with Accelerator ============
model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, lr_scheduler
)

#for step, batch in enumerate(train_dataloader):
#    print(batch)
# ============ Sample for Inference Test ============
test_sample = {
    "image": "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/crop_based/dino/output/attentions/DLB/Striatum/C110-115/DLB-2013-131-Striatum-C110-115_files_13_51.png",
    "query": "The first image shows the original histology slide, where brown aggregates represent Lewy Bodies when located on neurons and Lewy Neurites when found on neurites. The second image shows the corresponding attention map, where red indicates high activation and blue indicates low activation. Based on this color mapping, identify which cell types — such as neurons, oligodendrocytes, glial cells, Lewy Bodies, Lewy Neurites, and astrocytes — the model is focusing on in the attention map."
}

def run_inference_test(unwrapped_model, processor, test_sample, device, accelerator):
    """
    Runs a single inference pass on a test sample to verify the saved model works.
    """
    accelerator.print(f"\n[Inference Test] Running on test sample...")

    system_message = "You are a pathologist. Give answer in English"

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": test_sample["image"]},
                {"type": "text",  "text": test_sample["query"]},
            ],
        }
    ]

    # Tokenize
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True
    )

    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    unwrapped_model.eval()
    with torch.no_grad():
        pred_ids = unwrapped_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
            max_new_tokens=256,
        )

    # Decode only the newly generated tokens
    input_len = inputs["input_ids"].shape[1]
    answer = processor.decode(pred_ids[0][input_len:], skip_special_tokens=True)

    accelerator.print(f"[Inference Test] Query   : {test_sample['query']}")
    accelerator.print(f"[Inference Test] Response: {answer}")
    accelerator.print(f"[Inference Test] Done.\n")
    

    # Switch back to train mode
    unwrapped_model.train()


    return answer

# ============ BLEU ============
def compute_bleu(predictions, references):
    smoother = SmoothingFunction().method1
    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens  = ref.lower().split()
        score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoother)
        scores.append(score)
    return np.mean(scores)

# ============ ROUGE ============
def compute_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rL = [], [], []
    for pred, ref in zip(predictions, references):
        s = scorer.score(ref, pred)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rL.append(s["rougeL"].fmeasure)
    return {
        "rouge1": np.mean(r1),
        "rouge2": np.mean(r2),
        "rougeL": np.mean(rL)
    }



def compute_bertscore(predictions, references):
    P, R, F1 = bert_score(
        predictions, references,
        model_type="allenai/scibert_scivocab_uncased",  # Supported out of the box
        lang="en",
        verbose=False
    )
    return {
        "precision": P.mean().item(),
        "recall":    R.mean().item(),
        "f1":        F1.mean().item()
    }

# ============ Training Loop ============
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    num_steps = 0
    for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            #print(batch)
            #for k, v in batch.items():
            #    print(k, v.dtype, v.device, v.min().item(), v.max().item())
            #    break
            #print("loss", loss)
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item()
        num_steps += 1

        #if step % 50 == 0 and accelerator.is_main_process:
        #    accelerator.print(f"Epoch {epoch}, step {step}, loss: {loss.item():.4f}")
    avg_loss = total_loss / num_steps if num_steps > 0 else float("inf")
    accelerator.wait_for_everyone()
    #if accelerator.is_main_process and epoch % 10 == 0:  # Save every 10 epochs
    if accelerator.is_main_process and (epoch % 10 == 0 or epoch == epochs - 1):
        save_dir = f"./qwen_trained_models/qwen2vl_lora_epoch{epoch}"
        os.makedirs(save_dir, exist_ok=True)

        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        processor.save_pretrained(save_dir)

        # Save training state
        torch.save({
            "epoch": epoch,
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "avg_loss": avg_loss,
        }, os.path.join(save_dir, "training_state.pt"))

        accelerator.print(f"[Checkpoint] Saved at epoch {epoch}, avg_loss: {avg_loss:.4f}")

        # ============ Run Inference Test on Saved Checkpoint ============
        #try:
            # ============ Run on All Test Samples ============
        results = []
        for idx, sample in enumerate(test_data):
            print(f"[{idx+1}/{len(test_data)}] Running inference...")
            answer = run_inference_test(unwrapped_model, processor, sample, device, accelerator)
            
            results.append({
                "image":            sample["image"],
                "query":            sample["query"],
                "ground_truth":     sample["label"],
                "finetuned_answer": answer,
                #"base_answer":      base_answer,
            })
            
                # ============ Compute All Metrics ============
        ground_truths     = [r["ground_truth"]     for r in results]
        finetuned_answers = [r["finetuned_answer"] for r in results]
        #base_answers      = [r["base_answer"]      for r in results]

        print("\n========== Fine-tuned Model ==========")
        print(f"BLEU:       {compute_bleu(finetuned_answers, ground_truths):.4f}")
        rouge = compute_rouge(finetuned_answers, ground_truths)
        print(f"ROUGE-1:    {rouge['rouge1']:.4f}")
        print(f"ROUGE-2:    {rouge['rouge2']:.4f}")
        print(f"ROUGE-L:    {rouge['rougeL']:.4f}")
        #bscore = compute_bertscore(finetuned_answers, ground_truths)
        #print(f"BERTScore F1: {bscore['f1']:.4f}")
        
        """
        print("\n========== Base Model ==========")
        print(f"BLEU:       {compute_bleu(base_answers, ground_truths):.4f}")
        rouge_base = compute_rouge(base_answers, ground_truths)
        print(f"ROUGE-1:    {rouge_base['rouge1']:.4f}")
        print(f"ROUGE-2:    {rouge_base['rouge2']:.4f}")
        print(f"ROUGE-L:    {rouge_base['rougeL']:.4f}")
        bscore_base = compute_bertscore(base_answers, ground_truths)
        print(f"BERTScore F1: {bscore_base['f1']:.4f}")
        """
        