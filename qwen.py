from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import pandas as pd
import os


def prompt_design(image_path, entity_name, entity_unit_map):
    plausible_units = entity_unit_map[entity_name]

    str_units = ""
    for _ in plausible_units:
        str_units+= _+", "

    prompt = f'''Please extract the item weight and its unit of measurement from the image, providing them separately. Ensure that the unit is one of the following: {str_units}. Format your response as follows:
    \nValue: <only the numerical value>
    \nUnit: <unit of measurement from the specified list>'''

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", 
                 "text": prompt},
            ],
        }
    ]
    return messages

def text(messages, model):
    # Preparation for inference
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    #inputs = inputs.to("cuda")

    # Set the device to GPU 1
    # device = torch.device("cuda:2")
    # model = model.to(device)
    # inputs = inputs.to(device)
    inputs = inputs.to('cuda')
    #input_ids = input_ids.to('cuda')
    #Output generation
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text

def process_csv(file_path, entity_unit_map, batch, model):
    df = pd.read_csv(file_path)
    df['answer'] = None 

    for start in range(0, len(df), batch):
        end = start + batch
        batch_df = df.iloc[start:end]
        messages = []
        for index, row in batch_df.iterrows():
            image_path = "/raid/home/ritwikm/avinash/qwen/AML/test_images/" + row['image_link'].split('/')[-1]
            prompt = prompt_design(image_path, row['entity_name'], entity_unit_map)
            messages.append(prompt)
        output_text = text(messages, model)
        df.loc[start:end-1, 'answer'] = output_text
        print(output_text, start, end)
    
    df.to_csv('/raid/home/ritwikm/avinash/qwen/updated_test.csv', index=False)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

entity_unit_map = {
    'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'item_weight': {'gram',
        'kilogram',
        'microgram',
        'milligram',
        'ounce',
        'pound',
        'ton'},
    'maximum_weight_recommendation': {'gram',
        'kilogram',
        'microgram',
        'milligram',
        'ounce',
        'pound',
        'ton'},
    'voltage': {'kilovolt', 'millivolt', 'volt'},
    'wattage': {'kilowatt', 'watt'},
    'item_volume': {'centilitre',
        'cubic foot',
        'cubic inch',
        'cup',
        'decilitre',
        'fluid ounce',
        'gallon',
        'imperial gallon',
        'litre',
        'microlitre',
        'millilitre',
        'pint',
        'quart'}
}

test_file_path = "/raid/home/ritwikm/avinash/qwen/ans_test.csv"

process_csv(test_file_path, entity_unit_map, 2, model)
