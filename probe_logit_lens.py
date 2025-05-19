import os 
import torch 
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluate import load
import tensorflow_datasets as tfds
from datasets import load_dataset as hf_load_dataset
import json 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import langid
import random
import argparse
import os
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'

device_map = "auto"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Logit lens token id to language mapping")
    parser.add_argument("--model_path", type=str, default="models/bloomz/xquad/bloomz_de_RA_DS=103_lora_qkv", help="Model name")
    parser.add_argument("--model_type", type=str, default="bloomz", help="Model type")
    parser.add_argument("--checkpoint", type=int, default=0, help="Checkpoint number")
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face token")
    parser.add_argument("--task", type=str, required=True, help="Task name", choices=["xquad_2.1", "xquad_2.2", "flores", "xsum"])
    parser.add_argument("--task_lang", type=str, required=True, help="Task language")
    parser.add_argument("--task_type", type=str, required=True, help="Task type", choices=["normal", "incorrect", "correct", "normally_incorrect", "normally_correct"])
    args = parser.parse_args()

    model = args.model_path
    model_type = args.model_type
    checkpoint = args.checkpoint
    task = args.task
    model_name_for_the_probing_new_plots = model.split("/")[-1]
    if checkpoint == 0: 
        if model_type == 'bloomz': model_name = f"bigscience/bloomz-7b1"
        elif model_type == 'gemma': model_name = f"google/gemma-7b"
        elif model_type == "mistral": model_name = f"mistralai/Mistral-7B-Instruct-v0.3"
        elif model_type == "llama": model_name = f"meta-llama/Llama-3.1-8B-Instruct"
        elif model_type == "polylm": model_name = f"DAMO-NLP-MT/polylm-13b"
        else: raise NotImplementedError
    else: model_name = f"{model}/checkpoint-{checkpoint}"

    task_lang = args.task_lang # Accepts iso_639_1
    task_type = args.task_type # Accepts normal, incorrect, correct, normally_incorrect, normally_correct
    os.system(f"mkdir -p logit_lens/{model}")
    os.system(f"mkdir -p logit_lens/{model}/{checkpoint}")

    if task == "xquad_2.1" or task == "flores": task_type = "normal"

    if task_type == "normal":
        npy_file_to_save = f"logit_lens/{model}/{checkpoint}/{task_lang}_{task}_corrected.npy"
    elif task_type == "incorrect" or task_type == "correct" or task_type == "normally_incorrect" or task_type == "normally_correct":
        npy_file_to_save = f"logit_lens/{model}/{checkpoint}/{task_lang}_{task}_corrected_{task_type}.npy"
    else:
        raise NotImplementedError  
    
    # xquad_2.1_corrected.npy -- All samples
    # xquad_2.2_corrected.npy -- intersection of correct predictions of bloomz and bloomz_base
    # xquad_2.2_corrected_incorrect.npy -- Earlier correct predictions of bloomz_base which are now incorrect
    # xquad_2.2_corrected_correct.npy -- Earlier incorrect predictions of bloomz_base which are now correct
    # xquad_2.2_corrected_normally_incorrect.npy -- Earlier incorrect predictions of bloomz_base
    # xquad_2.2_corrected_normally_correct.npy -- Earlier correct predictions of bloomz_base

    dtst_sz = 2000 # For xquad and flores # setting to 2000 to consider all data items as the total number of data items is less than 2000
    # dtst_sz = 500 # For xsum
    seed = 42

    print(model_name)
    print(task_lang)

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=args.hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
        output_hidden_states=True,
        token=args.hf_token,
    )
    model.resize_token_embeddings(tokenizer.vocab_size)
    print(model)
    with open("flores_lang_map.json", "r") as f:
        flores_lang_map = json.load(f)
    flores_lang_map = {v: k for k, v in flores_lang_map.items()}

    def load_dataset(task, lang):
        if task == "flores":
            DATASET_FULL = hf_load_dataset("openlanguagedata/flores_plus", split="devtest")
            # if lang in DATASET_FULL.unique("iso_639_3"): print(f"{lang} supported")
            # else: raise ValueError("Language not supported, Please choose from the following: ", DATASET_FULL.unique("iso_639_3"))
            if lang not in flores_lang_map.keys(): raise ValueError("Language not supported in lang_map")
            DATASET_ENGLISH_LANG = DATASET_FULL.filter(lambda x: x["iso_639_3"] == "eng" and x["iso_15924"] == "Latn")
            DATASET_TARGET_LANG = DATASET_FULL.filter(lambda x: x["iso_639_3"] == lang.split("_")[0] and x["iso_15924"] == lang.split("_")[1])
            assert(len(DATASET_ENGLISH_LANG) == len(DATASET_TARGET_LANG)) # quick sanity check if ids are in the same order as the list entities
            for i in range(len(DATASET_ENGLISH_LANG)): assert(DATASET_ENGLISH_LANG[i]["id"] == DATASET_TARGET_LANG[i]["id"])
            data = [{"input": DATASET_ENGLISH_LANG[i]["text"], "output": DATASET_TARGET_LANG[i]["text"]} for i in range(len(DATASET_ENGLISH_LANG))]
            return data
        elif task == "xquad" or task == "xquad_2.1" or task == "xquad_2.2":
            if lang in ["or", "te", "hi", "mr", "en", "pa", "bn", "ml", "kn", "gu"]:
                with open(f"datasets/xquad_in/{lang}/xquad_{lang}_test.json") as f:
                    data = json.load(f)["examples"]
                print("Data Length: ", len(data))
                return data
            elif lang in ["es", "vi", "de", "zh", "th", "tr", "ru"]:
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
                raise ValueError(f"Language {lang} not supported")
        elif task == "xsum":
            if lang in ["or", "bn", "en", "hi"]:
                with open(f"datasets/xsum_in/{lang}/crosssum_english-{lang}_test.json") as f:
                    data = json.load(f)["examples"]
                print("Data length: ", len(data))
                return data
        else:
            raise NotImplementedError


    # To be used for or/en/te/mr/hi
    def create_input(dataitem, task, lang = None):
        if task == "xquad" or task == "xquad_2.1" or task == "xquad_2.2":
            if dataitem["answers"][0]["answer_start"] != -1:
                print(f"Data ID: {dataitem['id']} has answer_start != -1")
            if len(dataitem["answers"]) != 1:
                print(f"Data ID: {dataitem['id']} has more than one answer") 
            prompt = f"{dataitem['context']}\n\nQ: {dataitem['question']}\nA:"
            return prompt, dataitem["answers"][0]["text"]
        elif task == "xsum":
            prompt = f"Text: {dataitem['text']}\n\nQ: Summarize the above text in {lang_map[lang]}.\nSummary:"
            return prompt, dataitem["summary"]
        elif task == "flores":
            # Prompt type normal
            prompt = f"Translate the following text to {flores_lang_map[lang]}\nQ:{dataitem['input']}\nA:"
            # Prompt type I0_specific
            # prompt = f"{dataitem['input']}\nQ: Translate the above text to {flores_lang_map[lang]}\nA:"
            return prompt, dataitem["output"]
        else: 
            raise NotImplementedError
        

    with open(f"logit_lens/tokenid_to_lang/{model_type}/languages_to_vector_idx.json") as f:
        languages_to_idx = json.load(f)   

    if task == "xquad":
        with open(f"logit_lens/tokenid_to_lang/{model_type}/{model_type}2.npy", "rb") as f:
            tokenid_to_lang = np.load(f)
    elif task == "xquad_2.1" or task == "xsum" or task == "xquad_2.2" or task == "flores":
        with open(f"logit_lens/tokenid_to_lang/{model_type}/{model_type}2.1.npy", "rb") as f:
            tokenid_to_lang = np.load(f)

    # Convert to torch tensor
    tokenid_to_lang = torch.tensor(tokenid_to_lang).to("cuda")

    # Sanity check
    assert(tokenid_to_lang.shape[1] == len(languages_to_idx))

    data = load_dataset(task, task_lang)
    if task == "xquad_2.2":
        if task_type == "normally_incorrect" or task_type == "normally_correct":
            generations_file_path = f"generations/xquad_{task_lang}_test/bloomz_base.json"
            with open(generations_file_path, "r") as f:
                data_ = json.load(f)
            correct_ids_, incorrect_ids_ = [], []
            for id_ in data_.keys():
                prediction = data_[id_]["prediction_text"]
                correct_answer = data_[id_]["references"][0]["text"]
                if prediction == correct_answer:
                    correct_ids_.append(id_)
                else:
                    incorrect_ids_.append(id_)
            if task_type == "normally_incorrect": intersection_ids = incorrect_ids_
            elif task_type == "normally_correct": intersection_ids = correct_ids_
        elif task_type == "incorrect" or task_type == "correct" or task_type == "normal":   
            assert(checkpoint == 0 or checkpoint == 60) # HAHA
            file_path = f"generations/xquad_{task_lang}_test/bloomz_{model_name_for_the_probing_new_plots}_ckpt-{60}.json"
            with open(file_path, "r") as f:
                data_ = json.load(f)
            correct_ids1 = []
            for id_ in data_.keys():
                prediction = data_[id_]["prediction_text"]
                correct_answer = data_[id_]["references"][0]["text"]
                if prediction == correct_answer:
                    correct_ids1.append(id_)
            file_path = f"generations/xquad_{task_lang}_test/bloomz_base.json"
            with open(file_path, "r") as f:
                data_ = json.load(f)
            correct_ids2 = []
            for id_ in data_.keys():
                prediction = data_[id_]["prediction_text"]
                correct_answer = data_[id_]["references"][0]["text"]
                if prediction == correct_answer:
                    correct_ids2.append(id_)

            if task_type == "normal":
                # intersection of correct_ids1 and correct_ids2
                intersection_ids = list(set(correct_ids1).intersection(set(correct_ids2)))
            elif task_type == "incorrect":
                # corrected_incorrect.npy -- intersection of earlier correct predictions of bloomz_base which are now incorrect
                intersection_ids = list(set(correct_ids2).difference(set(correct_ids1)))
            elif task_type == "correct":
                # corrected_correct.npy -- intersection of earlier incorrect predictions of bloomz_base which are now correct
                intersection_ids = list(set(correct_ids1).difference(set(correct_ids2)))
        else: 
            raise NotImplementedError
        
        # filter out from data the ones with ids in intersection_ids
        data = [dataitem for dataitem in data if dataitem["id"] in intersection_ids]
        print(f"Number of data items after filtering: {len(data)}")    
        dtst_sz = len(data)
            

    def update_mean(mean, new_val, n):
        if n == 0:
            return new_val
        else: return (mean*n + new_val) / (n+1)
    mean = 0

    torch.autograd.set_grad_enabled(False)
    random.seed(seed)
    random.shuffle(data)
    num_considered = 0
    dtst_sz = min(dtst_sz, len(data))
    for num_ in tqdm(range(dtst_sz)):
        torch.cuda.empty_cache()
        dataitem = data[num_]
        prompt, answer = create_input(dataitem, task, task_lang)
        if task == "xsum":
            if len(tokenizer.encode(prompt)) > 1000: 
                print("Rejecting")
                continue
        if task == "xquad" or task == "xquad_2.1" or task == "xquad_2.2":
            # print(len(tokenizer.encode(prompt)))
            if len(tokenizer.encode(prompt)) > 2000: 
                print("Rejecting")
                continue
        num_considered += 1
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model(**inputs)
        if model_type == "bloomz" or model_type == "polylm":
            all_logits = [torch.nn.functional.softmax(model.lm_head(model.transformer.ln_f(outputs.hidden_states[i])), dim=-1) for i in range(len(outputs.hidden_states)-1)] + [torch.nn.functional.softmax(model.lm_head(outputs.hidden_states[-1]), dim=-1)]
        elif model_type == "gemma" or model_type == "mistral":
            all_logits = [torch.nn.functional.softmax(model.lm_head(model.model.norm(outputs.hidden_states[i])), dim=-1) for i in range(len(outputs.hidden_states)-1)] + [torch.nn.functional.softmax(model.lm_head(outputs.hidden_states[-1]), dim=-1)]
        elif model_type == "llama":
            all_logits = [torch.nn.functional.softmax(model.lm_head(model.model.norm(outputs.hidden_states[i])), dim=-1) for i in range(len(outputs.hidden_states)-1)] + [torch.nn.functional.softmax(model.lm_head(outputs.hidden_states[-1]), dim=-1)]
        else: raise NotImplementedError
        last_token_logits = [logits[0, -1, :] for logits in all_logits]
        layer_lang_probs = np.zeros((len(last_token_logits), len(languages_to_idx)))
        for i_, logits in enumerate(last_token_logits):        
            prod = last_token_logits[i_].reshape(-1,1)*tokenid_to_lang
            assert(prod.shape[1] == len(languages_to_idx))
            assert(prod.shape[0] == len(tokenid_to_lang))
            layer_lang_probs[i_] = torch.sum(prod, dim=0).cpu().numpy()

        mean = update_mean(mean, layer_lang_probs, num_)

    with open(npy_file_to_save, "wb") as f:
        np.save(f, mean)

    print(f"Number of data items considered: {num_considered}")