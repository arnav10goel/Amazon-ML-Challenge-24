# Importing the required libraries
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import pandas as pd
import os
from PIL import Image


# Defining the prompt design function. This function takes in the image path, entity name and entity unit map as input and returns the messages list.
# We are doing few-shot learning here, so we are providing some examples of the questions and answers for the user to understand the format of the response.
# The messages list contains the user and assistant messages for each example.
# The user messages contain the image and the question, while the assistant messages contain the answer.
# The last user message contains the image and the prompt for the user to extract the item and its unit of measurement.
# The messages list is returned by the function.
def prompt_design(image_path, entity_name, entity_unit_map):
    plausible_units = entity_unit_map[entity_name]

    str_units = ""
    for _ in plausible_units:
        str_units+= _+", "
    prompt = f'''Please extract the item {entity_name} and its unit of measurement from the image, providing them separately. Ensure that the unit is one of the following: {str_units}. Format your response as follows:
        \nValue: <only the numerical value>
        \nUnit: <unit of measurement from the specified list>'''
    if entity_name == 'width':
        question1 = "Please extract the item width and its unit of measurement from the image, providing them separately. Ensure that the unit is one of the following: {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'}. Format your response as follows:\nValue: <only the numerical value>\nUnit: <unit of measurement from the specified list>" 
        image1 = '../test/11gHj8dhhrL.jpg'
        answer1 = "Value: 10.50\nUnit: centimetre"
        question2 = "Please extract the item width and its unit of measurement from the image, providing them separately. Ensure that the unit is one of the following: {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'}. Format your response as follows:\nValue: <only the numerical value>\nUnit: <unit of measurement from the specified list>" 
        image2 = '../test/A1tnTUPyr7L.jpg'
        answer2 = "Value: \nUnit: "
        question3 = "Please extract the item width and its unit of measurement from the image, providing them separately. Ensure that the unit is one of the following: {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'}. Format your response as follows:\nValue: <only the numerical value>\nUnit: <unit of measurement from the specified list>"
        image3 = '../test/619006tSnML.jpg'
        answer3 = "Value: 5\nUnit: centimetre"
        messages = [{"role": "user", "content": [{"type": "image", "image": image1}, {"type": "text", "text": question1}]},
                    {"role": "assistant", "content": [{"type": "text", "text": answer1}]},
                    {"role": "user", "content": [{"type": "image", "image": image2}, {"type": "text", "text": question2}]}, 
                    {"role": "assistant", "content": [{"type": "text", "text": answer2}]}, 
                    {"role": "user", "content": [{"type": "image", "image": image3}, {"type": "text", "text": question3}]},
                    {"role": "assistant", "content": [{"type": "text", "text": answer3}]},
                    {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt}]}]
    elif entity_name == 'height':
        question1 = "Please extract the item height and its unit of measurement from the image, providing them separately. Ensure that the unit is one of the following: {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'}. Format your response as follows:\nValue: <only the numerical value>\nUnit: <unit of measurement from the specified list>" 
        image1 = '../test/11gHj8dhhrL.jpg'
        answer1 = "Value: \nUnit: "
        question2 = "Please extract the item height and its unit of measurement from the image, providing them separately. Ensure that the unit is one of the following: {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'}. Format your response as follows:\nValue: <only the numerical value>\nUnit: <unit of measurement from the specified list>" 
        image2 = '../test/A1tnTUPyr7L.jpg'
        answer2 = "Value: \nUnit: "
        question3 = "Please extract the item height and its unit of measurement from the image, providing them separately. Ensure that the unit is one of the following: {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'}. Format your response as follows:\nValue: <only the numerical value>\nUnit: <unit of measurement from the specified list>"
        image3 = '../test/619006tSnML.jpg'
        answer3 = "Value: 22\nUnit: centimetre"
        messages = [{"role": "user", "content": [{"type": "image", "image": image1}, {"type": "text", "text": question1}]},
                    {"role": "assistant", "content": [{"type": "text", "text": answer1}]},
                    {"role": "user", "content": [{"type": "image", "image": image2}, {"type": "text", "text": question2}]}, 
                    {"role": "assistant", "content": [{"type": "text", "text": answer2}]}, 
                    {"role": "user", "content": [{"type": "image", "image": image3}, {"type": "text", "text": question3}]},
                    {"role": "assistant", "content": [{"type": "text", "text": answer3}]},
                    {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt}]}]

    elif entity_name == 'depth':
        question1 = "Please extract the item depth and its unit of measurement from the image, providing them separately. Ensure that the unit is one of the following: {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'}. Format your response as follows:\nValue: <only the numerical value>\nUnit: <unit of measurement from the specified list>" 
        image1 = '../test/11gHj8dhhrL.jpg'
        answer1 = "Value: \nUnit: "
        question2 = "Please extract the item depth and its unit of measurement from the image, providing them separately. Ensure that the unit is one of the following: {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'}. Format your response as follows:\nValue: <only the numerical value>\nUnit: <unit of measurement from the specified list>" 
        image2 = '../test/A1tnTUPyr7L.jpg'
        answer2 = "Value: \nUnit: "
        question3 = "Please extract the item depth and its unit of measurement from the image, providing them separately. Ensure that the unit is one of the following: {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'}. Format your response as follows:\nValue: <only the numerical value>\nUnit: <unit of measurement from the specified list>"
        image3 = '../test/717517zezwL.jpg'
        answer3 = "Value: 28.25\nUnit: inch"
        messages = [{"role": "user", "content": [{"type": "image", "image": image1}, {"type": "text", "text": question1}]},
                    {"role": "assistant", "content": [{"type": "text", "text": answer1}]},
                    {"role": "user", "content": [{"type": "image", "image": image2}, {"type": "text", "text": question2}]}, 
                    {"role": "assistant", "content": [{"type": "text", "text": answer2}]}, 
                    {"role": "user", "content": [{"type": "image", "image": image3}, {"type": "text", "text": question3}]},
                    {"role": "assistant", "content": [{"type": "text", "text": answer3}]},
                    {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt}]}]
    elif entity_name == 'item_weight':
        question1 = "Please extract the item item_weight and its unit of measurement from the image, providing them separately. Ensure that the unit is one of the following: {'gram', 'kilogram', 'microgram','milligram','ounce','pound','ton'}. Format your response as follows:\nValue: <only the numerical value>\nUnit: <unit of measurement from the specified list>" 
        image1 = '../test/11gHj8dhhrL.jpg'
        answer1 = "Value: \nUnit: "
        question2 = "Please extract the item item_weight and its unit of measurement from the image, providing them separately. Ensure that the unit is one of the following: {'gram', 'kilogram', 'microgram','milligram','ounce','pound','ton'}. Format your response as follows:\nValue: <only the numerical value>\nUnit: <unit of measurement from the specified list>" 
        image2 = '../test/A1tnTUPyr7L.jpg'
        answer2 = "Value: \nUnit: "
        question3 = "Please extract the item item_weight and its unit of measurement from the image, providing them separately. Ensure that the unit is one of the following: {'gram', 'kilogram', 'microgram','milligram','ounce','pound','ton'}. Format your response as follows:\nValue: <only the numerical value>\nUnit: <unit of measurement from the specified list>"
        image3 = '../test/A1rVsIzEtkL.jpg'
        answer3 = "Value: 1500\nUnit: pound"
        messages = [{"role": "user", "content": [{"type": "image", "image": image1}, {"type": "text", "text": question1}]},
                    {"role": "assistant", "content": [{"type": "text", "text": answer1}]},
                    {"role": "user", "content": [{"type": "image", "image": image2}, {"type": "text", "text": question2}]}, 
                    {"role": "assistant", "content": [{"type": "text", "text": answer2}]}, 
                    {"role": "user", "content": [{"type": "image", "image": image3}, {"type": "text", "text": question3}]},
                    {"role": "assistant", "content": [{"type": "text", "text": answer3}]},
                    {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt}]}]
    elif entity_name == 'maximum_weight_recommendation':

        question1 = "Please extract the item maximum_weight_recommendation and its unit of measurement from the image, providing them separately. Ensure that the unit is one of the following: {'gram', 'kilogram', 'microgram','milligram','ounce','pound','ton'}. Format your response as follows:\nValue: <only the numerical value>\nUnit: <unit of measurement from the specified list>" 
        image1 = '../test/11gHj8dhhrL.jpg'
        answer1 = "Value: \nUnit: "
        question2 = "Please extract the item maximum_weight_recommendation and its unit of measurement from the image, providing them separately. Ensure that the unit is one of the following: {'gram', 'kilogram', 'microgram','milligram','ounce','pound','ton'}. Format your response as follows:\nValue: <only the numerical value>\nUnit: <unit of measurement from the specified list>" 
        image2 = '../test/A1tnTUPyr7L.jpg'
        answer2 = "Value: \nUnit: "
        question3 = "Please extract the item maximum_weight_recommendation and its unit of measurement from the image, providing them separately. Ensure that the unit is one of the following: {'gram', 'kilogram', 'microgram','milligram','ounce','pound','ton'}. Format your response as follows:\nValue: <only the numerical value>\nUnit: <unit of measurement from the specified list>"
        image3 = '../test/A1rVsIzEtkL.jpg'
        answer3 = "Value: 1500\nUnit: pound"
        messages = [{"role": "user", "content": [{"type": "image", "image": image1}, {"type": "text", "text": question1}]},
                    {"role": "assistant", "content": [{"type": "text", "text": answer1}]},
                    {"role": "user", "content": [{"type": "image", "image": image2}, {"type": "text", "text": question2}]}, 
                    {"role": "assistant", "content": [{"type": "text", "text": answer2}]}, 
                    {"role": "user", "content": [{"type": "image", "image": image3}, {"type": "text", "text": question3}]},
                    {"role": "assistant", "content": [{"type": "text", "text": answer3}]},
                    {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt}]}]
    elif entity_name == 'voltage':
        question1 = "Please extract the item voltage and its unit of measurement from the image, providing them separately. Ensure that the unit is one of the following: {'kilovolt', 'millivolt', 'volt'}. Format your response as follows:\nValue: <only the numerical value>\nUnit: <unit of measurement from the specified list>" 
        image1 = '../test/81381OOxI9L.jpg'
        answer1 = "Value: 20\nUnit: volt"


        question2 = "Please extract the item voltage and its unit of measurement from the image, providing them separately. Ensure that the unit is one of the following: {'kilovolt', 'millivolt', 'volt'}. Format your response as follows:\nValue: <only the numerical value>\nUnit: <unit of measurement from the specified list>" 
        image2 = '../test/81284AM5C9L.jpg'
        answer2 = "Value: 0.5\nUnit: volt"

        question3 = "Please extract the item voltage and its unit of measurement from the image, providing them separately. Ensure that the unit is one of the following: {'kilovolt', 'millivolt', 'volt'}. Format your response as follows:\nValue: <only the numerical value>\nUnit: <unit of measurement from the specified list>" 
        image3 = '../test/619310eNEkL.jpg'
        answer3 = "Value: \nUnit: "

        question4 = "Please extract the item wattage and its unit of measurement from the image, providing them separately. Ensure that the unit is one of the following: {'kilowatt', 'watt'}. Format your response as follows:\nValue: <only the numerical value>\nUnit: <unit of measurement from the specified list>" 
        image4 = '../test/21+i52HRW4L.jpg'
        answer4 = "Value: \nUnit: "
        messages = [{"role": "user", "content": [{"type": "image", "image": image1}, {"type": "text", "text": question1}]},
                    {"role": "assistant", "content": [{"type": "text", "text": answer1}]},
                    {"role": "user", "content": [{"type": "image", "image": image2}, {"type": "text", "text": question2}]}, 
                    {"role": "assistant", "content": [{"type": "text", "text": answer2}]}, 
                    {"role": "user", "content": [{"type": "image", "image": image3}, {"type": "text", "text": question3}]},
                    {"role": "assistant", "content": [{"type": "text", "text": answer3}]},
                    {"role": "user", "content": [{"type": "image", "image": image4}, {"type": "text", "text": question4}]},
                    {"role": "assistant", "content": [{"type": "text", "text": answer4}]},
                    {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt}]}]
    
    elif entity_name == 'wattage':
        question1 = "Please extract the item wattage and its unit of measurement from the image, providing them separately. Ensure that the unit is one of the following: {'kilowatt', 'watt'}. Format your response as follows:\nValue: <only the numerical value>\nUnit: <unit of measurement from the specified list>" 
        image1 = '../test/A1T6uvLe7oL.jpg'
        answer1 = "Value: 600\nUnit: watt"
        question3 = "Please extract the item wattage and its unit of measurement from the image, providing them separately. Ensure that the unit is one of the following: {'kilowatt', 'watt'}. Format your response as follows:\nValue: <only the numerical value>\nUnit: <unit of measurement from the specified list>" 
        image3 = '../test/21+i52HRW4L.jpg'
        answer3 = "Value: \nUnit: "
        question4 = "Please extract the item wattage and its unit of measurement from the image, providing them separately. Ensure that the unit is one of the following: {'kilowatt', 'watt'}. Format your response as follows:\nValue: <only the numerical value>\nUnit: <unit of measurement from the specified list>" 
        image4 = '../test/31b4reYTRzL.jpg'
        answer4 = "Value: \nUnit: "
        messages = [{"role": "user", "content": [{"type": "image", "image": image1}, {"type": "text", "text": question1}]},
                    {"role": "assistant", "content": [{"type": "text", "text": answer1}]},
                    {"role": "user", "content": [{"type": "image", "image": image3}, {"type": "text", "text": question3}]},
                    {"role": "assistant", "content": [{"type": "text", "text": answer3}]},
                    {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt}]}]
    elif entity_name == 'item_volume':
        question1 = "Please extract the item item_volume and its unit of measurement from the image, providing them separately. Ensure that the unit is one of the following: {'centilitre','cubic foot','cubic inch','cup,'decilitre','fluid ounce','gallon','imperial gallon','litre','microlitre','millilitre','pint','quart'}. Format your response as follows:\nValue: <only the numerical value>\nUnit: <unit of measurement from the specified list>" 
        image1 = '../test/21DZ7BAZ6-L.jpg'
        answer1 = "Value: 1.5\nUnit: millilitre"
        question2 = "Please extract the item item_volume and its unit of measurement from the image, providing them separately. Ensure that the unit is one of the following: {'centilitre','cubic foot','cubic inch','cup,'decilitre','fluid ounce','gallon','imperial gallon','litre','microlitre','millilitre','pint','quart'}. Format your response as follows:\nValue: <only the numerical value>\nUnit: <unit of measurement from the specified list>" 
        image2 = '../test/31-Jdk3SMUL.jpg'
        answer2 = "Value: 59\nUnit: millilitre"
        question3 = "Please extract the item item_volume and its unit of measurement from the image, providing them separately. Ensure that the unit is one of the following: {'centilitre','cubic foot','cubic inch','cup,'decilitre','fluid ounce','gallon','imperial gallon','litre','microlitre','millilitre','pint','quart'}. Format your response as follows:\nValue: <only the numerical value>\nUnit: <unit of measurement from the specified list>" 
        image3 = '../test/21+i52HRW4L.jpg'
        answer3 = "Value: \nUnit: "
        question4 = "Please extract the item item_volume and its unit of measurement from the image, providing them separately. Ensure that the unit is one of the following: {'centilitre','cubic foot','cubic inch','cup,'decilitre','fluid ounce','gallon','imperial gallon','litre','microlitre','millilitre','pint','quart'}. Format your response as follows:\nValue: <only the numerical value>\nUnit: <unit of measurement from the specified list>" 
        image4 = '../test/31b4reYTRzL.jpg'
        answer4 = "Value: \nUnit: "
        messages = [{"role": "user", "content": [{"type": "image", "image": image1}, {"type": "text", "text": question1}]},
                    {"role": "assistant", "content": [{"type": "text", "text": answer1}]},
                    {"role": "user", "content": [{"type": "image", "image": image2}, {"type": "text", "text": question2}]}, 
                    {"role": "assistant", "content": [{"type": "text", "text": answer2}]}, 
                    {"role": "user", "content": [{"type": "image", "image": image3}, {"type": "text", "text": question3}]},
                    {"role": "assistant", "content": [{"type": "text", "text": answer3}]},
                    {"role": "user", "content": [{"type": "image", "image": image4}, {"type": "text", "text": question4}]},
                    {"role": "assistant", "content": [{"type": "text", "text": answer4}]},
                    {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt}]}]

    return messages

from tqdm import tqdm

# Defining the text function. This function takes in the messages and model as input and returns the output text.
# The function processes the messages, generates the output text using the model, and returns the output text.
# The function also clears the cache after processing the messages.
def text(messages, model):
    torch.cuda.empty_cache()
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
    inputs = inputs.to('cuda')
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    del inputs
    del image_inputs
    del video_inputs
    return output_text

# Defining the process_csv function. This function takes in the file path, entity unit map, batch size, and model as input.
# The function reads the CSV file, processes the data in batches, and generates the output text using the model.
# The function saves the output text to a CSV file named 'few_shot_test.csv'.
def process_csv(file_path, entity_unit_map, batch, model):
    df = pd.read_csv(file_path)
    
    df = df.head(200)
    df['answer'] = None 

    for start in tqdm(range(0, len(df), batch)):
        torch.cuda.empty_cache()
        end = start + batch
        batch_df = df.iloc[start:end]
        messages = []
        for index, row in batch_df.iterrows():
            image_path = "../student_resource/test/" + row['image_link'].split('/')[-1]
            prompt = prompt_design(image_path, row['entity_name'], entity_unit_map)
            messages.append(prompt)
        output_text = text(messages, model)
        df.loc[start:end-1, 'answer'] = output_text
        print(output_text, start, end)
    
    df.to_csv('few_shot_test.csv', index=False)

# Defining the model, processor, entity_unit_map, and test_file_path.
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/data/poornash/AML/LLaMA-Factory/qwen2vl_2b_instruct_lora_merged",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
model = model.to("cuda")

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

test_file_path = "test.csv"

process_csv(test_file_path, entity_unit_map, 2, model)