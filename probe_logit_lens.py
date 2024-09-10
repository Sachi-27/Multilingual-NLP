import os 
import torch 
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluate import load
import tensorflow_datasets as tfds
import json 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import langid
import random
import os
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

plt.rcParams["font.family"] = "sans-serif"

device_map = {"": 0}
model = "models/bloomz/bloomz_de_RA_DS=103_lora_qkv"
model_type = 'bloomz'
checkpoint = 0
# model_name = f"{model}/checkpoint-{checkpoint}"
model_name = "bigscience/bloomz-7b1"
task = "xquad"
task_lang = "de"
os.system(f"mkdir -p logit_lens/{model}")
os.system(f"mkdir -p logit_lens/{model}/{checkpoint}")
npy_file_to_save = f"logit_lens/{model}/{checkpoint}/{task_lang}_{task}.npy"
dtst_sz = 150
seed = 42

tokenizer = AutoTokenizer.from_pretrained(model_name, token="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")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device_map,
    trust_remote_code=True,
    output_hidden_states=True,
    token="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"
)

def load_dataset(task, lang):
    if task == "xquad":
        if lang in ["or", "te", "hi", "mr", "en"]:
            with open(f"datasets/xquad_in/{lang}/xquad_{lang}_test.json") as f:
                data = json.load(f)["examples"]
            print("Data Length: ", len(data))
            return data
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
            return data
    else:
        raise NotImplementedError

# To be used for or/en/te/mr/hi
def create_input(dataitem):
    if dataitem["answers"][0]["answer_start"] != -1:
        print(f"Data ID: {dataitem['id']} has answer_start != -1")
    if len(dataitem["answers"]) != 1:
        print(f"Data ID: {dataitem['id']} has more than one answer") 
    prompt = f"{dataitem['context']}\n\nQ: {dataitem['question']}\nA:"
    return prompt, dataitem["answers"][0]["text"]

with open(f"logit_lens/tokenid_to_lang/{model_type}.json") as f:
    tokenid_to_lang = json.load(f)
lang_to_tokenids = {}
for id_ in tokenid_to_lang:
    lang = tokenid_to_lang[id_]
    if lang not in lang_to_tokenids:
        lang_to_tokenids[lang] = []
    lang_to_tokenids[lang].append(int(id_))

data = load_dataset(task, task_lang)
def update_mean(mean, new_val, n):
    if n == 0:
        return new_val
    else: return mean + (new_val - mean) / n
mean = 0

lang_to_langid = {}
i = 0
for lang in lang_to_tokenids:
    lang_to_langid[lang] = i
    i += 1

random.seed(seed)
random.shuffle(data)
for i in tqdm(range(dtst_sz)):
    dataitem = data[i]
    prompt, answer = create_input(dataitem)
    # print("Prompt: ", prompt)
    # print("Answer: ", answer)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model(**inputs)
    all_logits = [torch.nn.functional.softmax(model.lm_head(model.transformer.ln_f(outputs.hidden_states[i])), dim=-1) for i in range(len(outputs.hidden_states)-1)] + [torch.nn.functional.softmax(model.lm_head(outputs.hidden_states[-1]), dim=-1)]
    last_token_logits = [logits[0, -1, :] for logits in all_logits]
    layer_lang_probs = np.zeros((len(last_token_logits), len(lang_to_tokenids)))
    for i, logits in enumerate(last_token_logits):
        for lang, tokenids in lang_to_tokenids.items():
            layer_lang_probs[i, lang_to_langid[lang]] = logits[tokenids].sum().item()

    mean = update_mean(mean, layer_lang_probs, i)

with open(npy_file_to_save, "wb") as f:
    np.save(f, mean)