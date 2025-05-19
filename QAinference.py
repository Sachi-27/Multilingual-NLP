import os 
os.environ['CUDA_VISIBLE_DEVICES'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import json
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
from evaluate import load 
import tensorflow_datasets as tfds
from tqdm import tqdm
import argparse 

# set seeds
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device_map = {"": 0} 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="bloomz", help="Model type")
    parser.add_argument("--name", type=str, default="bloomz_or_RA_DS=103_lora_qkv", help="Model name which is to be evaluated")
    parser.add_argument("--checkpoint", type=int, default=0, help="Checkpoint number")
    parser.add_argument("--lang", type=str, default="or", help="Language to be evaluated")
    parser.add_argument("--data_source", type=str, default="indicgenbench", help="Data source to be evaluated", choices=["indicgenbench", "ai4bharat", "tydiqa", "tfds"])
    parser.add_argument("--prompt_type", type=int, default=0, help="Prompt type to be used", choices=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--kshot", type=int, default=0, help="Number of examples to be used for few-shot learning")
    parser.add_argument("--hf_token", type=str, required=True, help="Huggingface token")
    args = parser.parse_args()

    MODEL_TYPE = args.model_type
    NAME = args.name
    CHECKPOINT = args.checkpoint
    lang = args.lang
    data_source = args.data_source
    prompt_type = args.prompt_type
    KSHOT = args.kshot
    HF_TOKEN = args.hf_token
    task = "xquad"

    model_name = f'models/{MODEL_TYPE}/xquad/{NAME}/checkpoint-{CHECKPOINT}'
    if KSHOT == 0: file_name_to_save = f"{MODEL_TYPE}_{NAME}_ckpt-{CHECKPOINT}"
    else: file_name_to_save = f"{MODEL_TYPE}_{NAME}_ckpt-{CHECKPOINT}_{KSHOT}shot"

    if CHECKPOINT == 0:
        model_map = {
            "bloomz": "bigscience/bloomz-7b1",
            "bloomz3": "bigscience/bloomz-3b",
            "bloomz1": "bigscience/bloomz-1b1",
            "gemma": "google/gemma-7b",
            "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
            "llama": "meta-llama/Llama-3.1-8B-Instruct"
        }
        model_name = model_map[MODEL_TYPE]
        if KSHOT == 0: file_name_to_save = f"{MODEL_TYPE}_base"
        else: file_name_to_save = f"{MODEL_TYPE}_base_{KSHOT}shot"

    prompt_type = int(sys.argv[5])
    if prompt_type != 0:
        file_name_to_save += f"_I{prompt_type}_infer"
    prompt_version = {
        0: "normal",
        1: "instruction1",
        2: "instruction2",
        3: "instruction3",
        4: "instruction4",
        5: "instruction5"
    }

    print(f"Prompt Version: {prompt_version[prompt_type]}")
    print(model_name)
    print(lang)
    print(file_name_to_save)
    print(data_source)

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
            token=HF_TOKEN
        )

    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer = tokenizer, 
        torch_dtype=torch.bfloat16, 
        device_map=device_map
    )

    if task == "xquad":
        if data_source == "tydiqa":
            if lang in ["ar", "bn", "en", "fi", "id", "ja", "ko", "ru", "sw", "te", "th"]:
                dataset_path = f"datasets/tydiqa/{lang}.dev.jsonl"
                data = []
                with open(dataset_path) as f:
                    for line in f:
                        dataitem = json.loads(line)
                        unique_answers = set()
                        for x in dataitem["answers"]:
                            unique_answers.add(x["text"])
                        data.append({
                            "context": dataitem["passage_text"],
                            "question": dataitem["question_text"],
                            "id": dataitem["id"],
                            "answers": [{
                                "answer_start": -1,
                                "text": x
                            } for x in unique_answers]
                        })
                        
                print("Data Length: ", len(data))
            else:
                raise ValueError("Language not supported by TyDiQA")
        elif lang in ["or", "te", "hi", "mr", "en", "ml", "pa", "ta", "kn", "as", "ur", "bn", "gu"]:
            if data_source == "indicgenbench": 
                dataset_path = f"datasets/xquad_in/{lang}/xquad_{lang}_test.json"
            elif data_source == "ai4bharat":
                dataset_path = f"datasets/indicQA/indicqa_{lang}_test.json"
            else: raise ValueError("Data source not supported")
            with open(dataset_path) as f:
                data = json.load(f)["examples"]
            print("Data Length: ", len(data))
        elif lang in ["es", "de", "vi", "zh", "ru", "el", "th", "tr"]:
            if data_source != "tfds": raise ValueError("Invalid source for this language")
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

    def create_input(dataitem, task):
        if task == "xquad":
            prompt = ""
            if prompt_type == 0:
                for i in range(KSHOT):
                    prompt += f"{data[0]['context']}\n\nQ: {data[0]['question']}\nA: {data[0]['answers'][0]['text']}\n\n"
                prompt += f"{dataitem['context']}\n\nQ: {dataitem['question']}\nA:"
            elif prompt_type == 4:
                for i in range(KSHOT):
                    prompt += f"{data[0]['context']}\n\nNow answer the following question in a short span. Please do not hallucinate, and answer based on the above provided context.\nQ: {data[0]['question']}\nA: {data[0]['answers'][0]['text']}\n\n"
                prompt += f"{dataitem['context']}\n\nNow answer the following question in a short span. Please do not hallucinate, and answer based on the above provided context.\nQ: {dataitem['question']}\nA:"
            elif prompt_type == 5:
                for i in range(KSHOT):
                    prompt += f"Instruction: Answer the question based on the context.\nInput: Context: {data[0]['context']} Question: {data[0]['question']}\nOutput: {data[0]['answers'][0]['text']}\n\n"
                prompt += f"Instruction: Answer the question based on the context.\nInput: Context: {dataitem['context']} Question: {dataitem['question']}\Output:"
            else: raise ValueError("Prompt type not supported")

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

    # set padding token id to eos token id
    if MODEL_TYPE == "mistral": tokenizer.pad_token_id = tokenizer.eos_token_id

    squad_metric = load("squad_v2")
    for i in tqdm(range(KSHOT, len(data))):
        prompt = create_input(data[i], task)
        output, true_answer = check(i,temperature, num_tokens, prompt)
        output_answer = (" ".join(output.split(prompt)[1:])).strip()
        output_answer = output_answer.split("\n")[0]
        if task == "xquad":
            predictions.append({
                "prediction_text" : output_answer,
                "id" : str(data[i]["id"]),
                'no_answer_probability': 0
            })
            references.append({
                "answers" : data[i]["answers"],
                "id" : str(data[i]["id"])
            })  

    results = squad_metric.compute(predictions=predictions, references=references)
    print(results)

    # Save the results
    to_save = {}
    for i in range(len(references)):
        to_save[references[i]["id"]] = {
            "prediction_text" : predictions[i]["prediction_text"],
            "references" : references[i]["answers"]
        }


    if data_source == "indicgenbench" or data_source == "tfds":
        folder_name = f"xquad_{lang}_test"
    elif data_source == "ai4bharat":
        folder_name = f"indicqa_{lang}_test"
    elif data_source == "tydiqa":
        folder_name = f"tydiqa_{lang}_dev"
    else: raise ValueError("Data source not supported")
    print(folder_name)
    os.system(f"mkdir -p generations/{folder_name}")

    with open(f"generations/{folder_name}/{file_name_to_save}.json", "w") as f:
        json.dump(to_save, f, indent=4, ensure_ascii=False)

    with open(f"generations/{folder_name}/{file_name_to_save}.csv", "w") as f:
        f.write("metric, value\n")
        for key in results:
            f.write(f"{key}, {results[key]}\n")   