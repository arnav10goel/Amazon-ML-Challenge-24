from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import torch
import os
import pickle
import nest_asyncio
nest_asyncio.apply()

# Map of the entity names and values
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

image_dir = "/data/poornash/AML/66e31d6ee96cd_student_resource_3/student_resource 3/test"  # Replace with your actual image directory
download_dir = "/data/poornash/AML/models"
prompt_format = f'''Extract the item and its unit of measurement from the image, providing them separately. Ensure that the unit is one of the following: str_units. Format your response as follows:
    \nValue: <only the numerical value>
    \nUnit: <unit of measurement from the specified list>'''

batch_size = 16
test = True
device = 1
save_dir = "/data/poornash/AML/save_data_internvlm_units/"
start = device 
file_name = f"data_{start}.pkl"

os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
os.environ["WORLD_SIZE"] = "1"

# Loading the csv
load_path ="/data/poornash/AML/data/test.csv"
df = pd.read_csv(load_path)
if not test:
    df_train, df = train_test_split(df, test_size=0.08, random_state=1, shuffle=False)

# Start and end indices
start_ind = start*len(df)//6
end_ind = min((start+1)*len(df)//6, len(df))
df = df[:end_ind][start_ind:]
# Fetching images from image path
images = df["image_link"].apply(lambda x: image_dir + "/" + x.split("/")[-1]).tolist()
# Prompt design
prompts = df["entity_name"].apply(lambda x: prompt_format.split("item")[0].strip() + " " + ' '.join(x.split("_")) + 
                                  " " + prompt_format.split("item")[1].strip().split("str_units")[0].strip() +
                                  " " + ' , '.join(entity_unit_map[x]) + 
                                  " " + prompt_format.split("item")[1].strip().split("str_units")[1].strip()).tolist()

model = "openbmb/MiniCPM-V-2_6"

pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=None, download_dir=download_dir))

responses = []
count = 0

for i in tqdm(range(0, len(prompts), batch_size)):
    torch.cuda.empty_cache()
    # Batchinf the prompts and then process the batch and get responses
    response = pipe([(prompts[i + j], images[i + j]) for j in range(min(batch_size, len(prompts) - i))])
    
    # Append the responses (adjusting the indexing)
    for j in range(min(batch_size, len(prompts) - i)):
        # Appening the required response text
        responses.append(response[j].text)
    
    # Increment counter
    count += 1

# Final save of all responses after the loop finishes
with open(save_dir + file_name, "wb") as f:
    pickle.dump(responses, f)


