import os 
os.environ['CUDA_VISIBLE_DEVICES'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import torch
import json
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
from evaluate import load 
import tensorflow_datasets as tfds
# set seeds
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device_map = {"": 0} 
# device_map = "auto"
# model_name = 'google/gemma-7b-it'
# model_name = 'bigscience/bloomz-7b1'
# model_name = 'sarvamai/sarvam-2b-v0.5'
# lang = "de"
# file_name_to_save = "bloomz_base"

MODEL_TYPE = sys.argv[1]
NAME = sys.argv[2]
CHECKPOINT = int(sys.argv[3])
lang = sys.argv[4]
model_name = f'models/{MODEL_TYPE}/{NAME}/checkpoint-{CHECKPOINT}'
file_name_to_save = f"{MODEL_TYPE}_{NAME}_ckpt-{CHECKPOINT}"
print(model_name)
print(lang)

tokenizer = AutoTokenizer.from_pretrained(model_name, token="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")
model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
        token="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"
    )
pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer = tokenizer, 
    torch_dtype=torch.bfloat16, 
    device_map=device_map
)

if lang in ["or", "te", "hi", "mr", "en"]:
    with open(f"datasets/xquad_in/{lang}/xquad_{lang}_test.json") as f:
        data = json.load(f)["examples"]
    print("Data Length: ", len(data))
elif lang in ["es", "de"]:
    dataset = tfds.load(f"xquad/{lang}", split="test")
    data = []
    for sample_item in dataset:
        assert(len(sample_item["answers"]["text"]) == 1)
        data.append({
            "answers": [{
                "answer_start": -1,
                "text": sample_item["answers"]["text"][0].numpy().decode("utf-8")
            }],
            "context": sample_item["context"].numpy().decode("utf-8"),
            "question": sample_item["question"].numpy().decode("utf-8"),
            "id": sample_item["id"].numpy().decode("utf-8")
        })
    print("Data Length: ", len(data))
else:
    raise ValueError("Language not supported")

def create_input(dataitem):
    if dataitem["answers"][0]["answer_start"] != -1:
        print(f"Data ID: {dataitem['id']} has answer_start != -1")
    if len(dataitem["answers"]) != 1:
        print(f"Data ID: {dataitem['id']} has more than one answer") 
    prompt = f"{dataitem['context']}\n\nQ: {dataitem['question']}\nA:"
    return prompt

def check(i, temperature, max_new_tokens, prompt):
    sequences = pipe(
        prompt,
        do_sample=True,
        max_new_tokens=max_new_tokens, 
        temperature=temperature, 
        top_k=50, 
        top_p=0.95,
        num_return_sequences=1,
    )

    return sequences[0]["generated_text"], data[i]['answers'][0]['text']

# Evaluation
temperature = 0.1
predictions = []
references = []
num_tokens = 100 

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

for i in range(len(data)):
    prompt = create_input(data[i])
    output, true_answer = check(i,temperature, num_tokens, prompt)
    output_answer = (" ".join(output.split(prompt)[1:])).strip()
    output_answer = output_answer.split("\n")[0]
    predictions.append({
        "prediction_text" : output_answer,
        "id" : data[i]["id"],
        'no_answer_probability': 0
    })
    references.append({
        "answers" : data[i]["answers"],
        "id" : data[i]["id"]
    })  
    if i % 100 == 0:
        print(f"Done with {i}")


squad_metric = load("squad_v2")
results = squad_metric.compute(predictions=predictions, references=references)
print(results)

# Save the results
to_save = {}
for i in range(len(references)):
    to_save[references[i]["id"]] = {
        "prediction_text" : predictions[i]["prediction_text"],
        "references" : references[i]["answers"]
    }

os.system(f"mkdir -p generations/xquad_{lang}_test")

# with open(f"generations/xquad_{lang}_test/{file_name_to_save}.json", "w") as f:
#     json.dump(to_save, f, indent=4)

with open(f"generations/xquad_{lang}_test/{file_name_to_save}.csv", "w") as f:
    f.write("metric, value\n")
    for key in results:
        f.write(f"{key}, {results[key]}\n")   