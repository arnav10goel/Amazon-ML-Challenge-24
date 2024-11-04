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


image_dir = "AML/66e31d6ee96cd_student_resource_3/student_resource 3/test"
download_dir = "AML/models"
# Format of the prompt
prompt_format = '''Extract the best (only one) item and its unit from the image in the format - \nValue: <only numbers> \nUnit: <only alphabets>
'''
batch_size = 16
test = True
# CUDA devices
device = 0
save_dir = "AML/save_data_minicpm/"
# Counter
start = 0

file_name = f"data_{start}.pkl"
# Ensuring appropriate GPU allocation
os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
os.environ["WORLD_SIZE"] = "1"

# Loading the CSV
load_path ="AML/data/test.csv"
df = pd.read_csv(load_path)

if not test:
    df_train, df = train_test_split(df, test_size=0.08, random_state=1, shuffle=False)

# Based on start and end indexes
df = df[:min((start+1)*len(df)//4, len(df))][start*len(df)//4:]

images = df["image_link"].apply(lambda x: image_dir + "/" + x.split("/")[-1]).tolist()
prompts = df["entity_name"].apply(lambda x: prompt_format.split("item")[0].strip() + " " + ' '.join(x.split("_")) + " " + prompt_format.split("item")[1].strip()).tolist()

model = "openbmb/MiniCPM-V-2_6"
# Model pipeline
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=None, download_dir=download_dir))

# List of responses
responses = []
count = 0
for i in tqdm(range(0, len(prompts), batch_size)):
    torch.cuda.empty_cache()
    # Batching the prompt+images and then processing the batch and get responses
    response = pipe([(prompts[i + j], images[i + j]) for j in range(min(batch_size, len(prompts) - i))])
    
    # Append the responses (adjusting the indexing)
    for j in range(min(batch_size, len(prompts) - i)):
        # Assuming response[j].text holds the required text, append it
        responses.append(response[j].text)
    
    # Increment counter
    count += 1
    print(count)

# Final save of all responses after the loop finishes
with open(save_dir + file_name, "wb") as f:
    pickle.dump(responses, f)
