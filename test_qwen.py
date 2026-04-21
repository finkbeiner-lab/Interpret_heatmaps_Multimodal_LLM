from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers import AutoModelForCausalLM, AutoTokenizer
from glob import glob
from PIL import Image
import os
import json
import torch
print(torch.cuda.is_available())
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
#from qwen_vl_utils import process_vision_info

# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)


image = Image.open("/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/crop_based/dino/output/attentions/DLB/Striatum/C110-115/DLB-2013-131-Striatum-C110-115_files_13_51.png")

conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
            },
            {"type": "text", "text": "First image is original image, the brown aggregates are Lewy Bodies if they are on neurons and Lewy Neurites if they are on neurites, and second shows attention maps. Show color wise which cell types such as neurons, oligodendrites, glial cells, lewy bodies, lewy neurites, astrocytes etc are highlighted in the right image, (red shows high activation , blue shows low)"},
        ],
    }
]


# Preprocess the inputs
text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
# Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'

inputs = processor(
    text=[text_prompt], images=[image], padding=True, return_tensors="pt"
)
inputs = inputs.to("cuda")


output_ids = model.generate(**inputs, max_new_tokens=512)
generated_ids = [
    output_ids[len(input_ids) :]
    for input_ids, output_ids in zip(inputs.input_ids, output_ids)
]
output_text = processor.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
)
print(output_text)


jsons = []

imgs = glob(os.path.join('/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/crop_based/LLM/data/attn_images_combined','*.png'))

for img in imgs:
    image = Image.open(img)
    conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
            },
            {"type": "text", "text": "First image is original image, the brown aggregates are Lewy Bodies if they are on neurons and Lewy Neurites if they are on neurites, and second shows attention maps. Show color wise which cell types such as neurons, oligodendrites, glial cells, lewy bodies, lewy neurites, astrocytes etc are highlighted in the right image, (red shows high activation , blue shows low)"},
        ],
    }]
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(
    text=[text_prompt], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to("cuda")
    output_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    query_answer = {
        "image":img,
        "label":output_text
    }
    jsons.append(query_answer)
# Write to a JSON file
with open("generated_output.json", "w") as f:
    json.dump(jsons, f, indent=4)