import os 
import torch
import json
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
from evaluate import load 
import tensorflow_datasets as tfds
import numpy as np
import langid 
import string
from tqdm import tqdm
import argparse 

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def classified_punct_or_num(k):
    for c in k:
        if c in string.punctuation or c in string.digits:
            continue
        else:
            return False 
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Logit lens token id to language mapping")
    parser.add_argument("--model_name", type=str, default="bigscience/bloomz-7b1", help="Model name")
    parser.add_argument("--model_type", type=str, default="bloomz", help="Model type")
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face token")
    args = parser.parse_args()

    model_name = args.model_name
    model_type = args.model_type

    os.system(f"mkdir -p logit_lens/tokenid_to_lang/{model_type}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=args.hf_token)
    vocab = tokenizer.get_vocab()

    output = langid.rank("ମୋ")
    languages = [lang for lang, score in output]
    languages_to_idx = {lang: idx for idx, lang in enumerate(languages)}
    with open(f"logit_lens/tokenid_to_lang/{model_type}/languages_to_vector_idx.json", "w") as f:
        json.dump(languages_to_idx, f)

    token_id_to_lang_bloomz = np.zeros((len(vocab), len(languages)))
    token_id_to_lang_bloomz.shape


    count_classified_as_punct_or_num = 0
    for k,v in tqdm(vocab.items()):
        output = tokenizer.decode([v]).strip()
        vector = [0] * len(languages)
        if classified_punct_or_num(output):
            count_classified_as_punct_or_num += 1
        else:        
            output = langid.rank(output)
            for lang, score in output:
                vector[languages_to_idx[lang]] = score
            vector = np.array(vector)
            vector = softmax(vector)
        token_id_to_lang_bloomz[v] = vector


    print("Number of tokens classified as punctuation or number: ", count_classified_as_punct_or_num)

    with open(f"logit_lens/tokenid_to_lang/{model_type}/{model_type}.npy", "wb") as f:
        np.save(f, token_id_to_lang_bloomz)