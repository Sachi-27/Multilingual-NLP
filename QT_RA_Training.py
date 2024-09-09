# This script adapts the following order
# 1. Question Translation
# 2. Response Alignment

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="5,7"
os.environ["WANDB_DISABLED"] = "true" 
# report_to = 'wandb' also in training arguments

from dataclasses import dataclass, field
from typing import Optional
from datasets.arrow_dataset import Dataset
import torch
import random
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig
from peft import AutoPeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
    set_seed as hf_set_seed,
)
import tensorflow_datasets as tfds
import numpy as np
import random
import json 
import trl
from trl import SFTTrainer
from huggingface_hub import login
login(token = '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')

# torch.manual_seed(42)
# random.seed(42)
hf_set_seed(42)

MODEL_NAME = "bigscience/bloomz-7b1"
# MODEL_NAME = "google/gemma-7b-it"
# MODEL_NAME = "sarvamai/sarvam-2b-v0.5"
MODEL_TYPE = "bloomz"

# TRANSLATION_DATASET_PATH = "./datasets/flores_in/en_or/flores_en_or_dev.json"
# TRANSLATION_DATASET_PATH = "ai4bharat/samanantar"
# TRANSLATION_DATASET_PATH = "facebook/flores"
# LANGUAGE = "te"
# LANGUAGE = "jpn_Jpan-eng_Latn"
# USE_HUGGINGFACE_TRANSLATION_DATASET = True
# TRANSLATION_TEST_DATASET_PATH = "./datasets/flores_in/en_or/flores_en_or_test.json"
# ALIGNMENT_TRAIN_DATASET_PATH = "./datasets/xquad_in/or/xquad_or_train.json"

# ALIGNMENT_TRAIN_DATASET_PATH = "xquad/de"
# USE_TFDS_ALIGNMENT_TRAIN_DATASET = True
# ALIGNMENT_DEV_DATASET_PATH = "./datasets/xquad_in/en/xquad_en_dev.json"

# SPLIT_TO_USE = "translate-train"
# SPLIT_TO_USE = "dev"
# DATASET_SIZE = 1000

# RA on xquad/de
TRANSLATION_DATASET_PATH = ""
LANGUAGE = ""
TRANSLATION_TEST_DATASET_PATH = ""
ALIGNMENT_TRAIN_DATASET_PATH = "./datasets/xquad_in/or/xquad_or_train.json"
USE_TFDS_ALIGNMENT_TRAIN_DATASET = False
ALIGNMENT_DEV_DATASET_PATH = "./datasets/xquad_in/en/xquad_en_dev.json"
# DATASET_SIZE = 1000
SPLIT_TO_USE = "translate-train"

STAGE = "RA"
# QT_TYPE = "te->en"
NUM_TRAIN_EPOCHS = 10
DATASET_NAME = "NULL"
USE_4BIT = False
USE_NESTED_QUANT = False
BNB_4BIT_COMPUTE_DTYPE = "float16"
BNB_4BIT_QUANT_TYPE = "nf4"
FP16 = False
BF16 = True
PACKING = False
GRADIENT_CHECKPOINTING = True
OPTIM = "paged_adamw_32bit"
LR_SCHEDULER_TYPE = "constant"
OPTIM_STEPS = -1
WARMUP_RATIO = 0.03
GROUP_BY_LENGTH = True
SAVE_STEPS = 10
LOGGING_STEPS = 10
MERGE_AND_PUSH = False
SEED = 42
NAME = "bloomz_or_RA_DS=103_lora_all_modules"
OUTPUT_DIR = f"./models/{MODEL_TYPE}/{NAME}"
CONTEXT_WINDOW = 256
# TOTAL_SAMPLES = 10
# TOTAL_SAMPLES = 51597

# PROMPT_DICT = {
#     "inst_prompt_input": (
#         "Below is an instruction that describes a task, paired with an input that provides further context. "
#         "Write a response that appropriately completes the request.\n\n"
#         "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
#     ),
#     "inst_prompt_no_input": (
#         "Below is an instruction that describes a task. "
#         "Write a response that appropriately completes the request.\n\n"
#         "### Instruction:\n{instruction}\n\n### Response:\n{output}"
#     ),
# }

@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})

    per_device_train_batch_size: Optional[int] = field(default=4)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[float] = field(default=2e-4)
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[int] = field(default=0.001)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    max_seq_length: Optional[int] = field(default=1024)
    model_name: Optional[str] = field(
        default=MODEL_NAME,
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        }
    )
    model_type: Optional[str] = field(
        default=MODEL_TYPE,
        metadata={"help": "The type of model to train."},
    )
    # dataset_path: Optional[str] = field(
    #     default=DATASET_PATH,
    #     metadata={"help": "The path to the dataset."},
    # )
    # en_dataset_path: Optional[str] = field(
    #     default=EN_DATASET_PATH,
    #     metadata={"help": "The path to the English dataset."},
    # )
    translation_dataset_path : Optional[str] = field(
        default=TRANSLATION_DATASET_PATH,
        metadata={"help": "The list of paths to the translation dataset."},
    )
    alignment_train_dataset_path: Optional[str] = field(
        default=ALIGNMENT_TRAIN_DATASET_PATH,
        metadata={"help": "The path to the alignment train dataset."},
    )
    alignment_dev_dataset_path: Optional[str] = field(
        default=ALIGNMENT_DEV_DATASET_PATH,
        metadata={"help": "The path to the alignment dev dataset."},
    )
    dataset_name: Optional[str] = field(
        default=DATASET_NAME,
        metadata={"help": "The preference dataset to use."},
    )
    context_window: Optional[int] = field(
        default=CONTEXT_WINDOW,
        metadata={"help": "The context window size."},
    )
    stage: Optional[str] = field(
        default=STAGE,
        metadata={"help": "The stage of the training."},
    )
    use_4bit: Optional[bool] = field(
        default=USE_4BIT,
        metadata={"help": "Activate 4bit precision base model loading"},
    )
    use_nested_quant: Optional[bool] = field(
        default=USE_NESTED_QUANT,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default=BNB_4BIT_COMPUTE_DTYPE,
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default=BNB_4BIT_QUANT_TYPE,
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    num_train_epochs: Optional[int] = field(
        default=NUM_TRAIN_EPOCHS,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    fp16: Optional[bool] = field(
        default=FP16,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=BF16,
        metadata={"help": "Enables bf16 training."},
    )
    packing: Optional[bool] = field(
        default=PACKING,
        metadata={"help": "Use packing dataset creating."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=GRADIENT_CHECKPOINTING,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default=OPTIM,
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        default=LR_SCHEDULER_TYPE,
        metadata={"help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"},
    )
    max_steps: int = field(default=OPTIM_STEPS, metadata={"help": "How many optimizer update steps to take"})
    warmup_ratio: float = field(default=WARMUP_RATIO, metadata={"help": "Fraction of steps to do a warmup for"})
    group_by_length: bool = field(
        default=GROUP_BY_LENGTH,
        metadata={
            "help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."
        },
    )
    save_steps: int = field(default=SAVE_STEPS, metadata={"help": "Save checkpoint every X updates steps."})
    logging_steps: int = field(default=LOGGING_STEPS, metadata={"help": "Log every X updates steps."})
    merge_and_push: Optional[bool] = field(
        default=MERGE_AND_PUSH,
        metadata={"help": "Merge and push weights after training"},
    )
    output_dir: str = field(
        default=OUTPUT_DIR,
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

def gen_batches_train():
    if script_args.stage == "QT":
        print("Loading train data for Question Translation")
        print(script_args.translation_dataset_path)
        if not USE_HUGGINGFACE_TRANSLATION_DATASET:
            translation_data = json.load(open(script_args.translation_dataset_path))["examples"]
            trans_data = []
            for data in translation_data:
                    trans_data.append([data["source"], data["target"]])
        else:
            if TRANSLATION_DATASET_PATH == "ai4bharat/samanantar":
                translation_data = load_dataset(TRANSLATION_DATASET_PATH, LANGUAGE)[SPLIT_TO_USE]
                trans_data = []
                random.seed(SEED)
                for i in random.sample(range(len(translation_data)), DATASET_SIZE):
                    trans_data.append([translation_data[i]["src"], translation_data[i]["tgt"]])
            elif TRANSLATION_DATASET_PATH == "facebook/flores":
                translation_data = load_dataset(TRANSLATION_DATASET_PATH, LANGUAGE)[SPLIT_TO_USE]
                trans_data = []
                for i in range(len(translation_data)):
                    trans_data.append([translation_data[i]["sentence_eng_Latn"], translation_data[i]["sentence_jpn_Jpan"]])
            else:
                raise ValueError("Invalid TRANSLATION_DATASET_PATH")
        total_samples = len(trans_data)
        train_limit = total_samples
        counter = 0
        # Yielding Question Translation Instructions
        for sample in iter(trans_data):
            if counter >= train_limit:
                break
            if QT_TYPE == "en->or":
                new_text_format = f'<s>{sample[0]}\n\nQ: Translate the above text from English to Odia.\nA: {sample[1]}</s>'
            elif QT_TYPE == "or->en":
                new_text_format = f'<s>{sample[1]}\n\nQ: Translate the above text from Odia to English.\nA: {sample[0]}</s>'
            elif QT_TYPE == "en->te":
                new_text_format = f'<s>{sample[0]}\n\nQ: Translate the above text from English to Telugu.\nA: {sample[1]}</s>'
            elif QT_TYPE == "te->en":
                new_text_format = f'<s>{sample[1]}\n\nQ: Translate the above text from Telugu to English.\nA: {sample[0]}</s>'
            elif QT_TYPE == "jpn->en":
                new_text_format = f'<s>{sample[1]}\n\nQ: Translate the above text from Japanese to English.\nA: {sample[0]}</s>'
            elif QT_TYPE == "en->jpn":
                new_text_format = f'<s>{sample[0]}\n\nQ: Translate the above text from English to Japanese.\nA: {sample[1]}</s>'
            else:
                raise ValueError("Invalid QT_TYPE")
            yield {'text': new_text_format}
            counter += 1
    elif script_args.stage == "RA":
        print("Loading train data for Response Alignment")
        print(script_args.alignment_train_dataset_path)

        if not USE_TFDS_ALIGNMENT_TRAIN_DATASET:
            alignment_data = json.load(open(script_args.alignment_train_dataset_path))["examples"]
            train_limit = len(alignment_data)
            counter = 0

            # Yielding Response Alignment Instructions
            for sample in iter(alignment_data):
                if counter >= train_limit:
                    break
                    
                if len(sample["answers"]) != 1:
                    raise ValueError(f"Data ID: {sample['id']} has more than one answer")
                if sample["answers"][0]["answer_start"] != -1:
                    raise ValueError(f"Data ID: {sample['id']} has answer_start != -1")
                new_text_format = f"<s>{sample['context']}\n\nQ: {sample['question']}\nA: {sample['answers'][0]['text']}</s>"

                # print only for first case -- debugging
                if counter == 0:
                    print(new_text_format)

                yield {'text': new_text_format}
                counter += 1
        else:
            alignment_data = tfds.load(script_args.alignment_train_dataset_path, split=SPLIT_TO_USE)
            train_limit = DATASET_SIZE
            len_data = len(alignment_data)
            indices = []
            random.seed(SEED)
            while len(indices) < train_limit:
                index = random.randint(0, len_data-1)
                if index not in indices:
                    indices.append(index)
            assert(len(indices) == train_limit)

            counter = 0

            # Yielding response alignment instructions
            for sample in iter(alignment_data):
                if counter in indices:
                    assert(len(sample["answers"]["text"]) == 1)
                    context = sample["context"].numpy().decode("utf-8")
                    question = sample["question"].numpy().decode("utf-8")
                    answer = sample["answers"]["text"][0].numpy().decode("utf-8")
                    new_text_format = f"<s>{context}\n\nQ: {question}\nA: {answer}</s>"
                    if counter == 0:
                        print(new_text_format)
                    yield {'text': new_text_format}
                counter += 1

# This entire function doesn't matter
def gen_batches_val():
    if script_args.stage == "QT":
        print("Loading val data for Question Translation")
        print(script_args.translation_dataset_path)
        # Doesn't matter
        translation_data = json.load(open("./datasets/flores_in/en_or/flores_en_or_dev.json"))["examples"]
        trans_data = []
        for data in translation_data:
                trans_data.append([data["source"], data["target"]])
        total_samples = len(trans_data)
        train_limit = 10
        counter = 0

        # Yielding Question Translation Instructions -- Not to care about
        for sample in iter(trans_data):
            if counter >= train_limit:
                break

            new_text_format = f'<s>{sample[0]}\n\nQ: Translate the above text from English to Odia.\nA: {sample[1]}</s>'
            yield {'text': new_text_format}
            counter += 1
    elif script_args.stage == "RA":
        print("Loading val data for Response Alignment")
        print(script_args.alignment_dev_dataset_path)

        alignment_data = json.load(open(script_args.alignment_dev_dataset_path))["examples"]
        train_limit = 10 ### Doesn't Matter
        counter = 0

        # Yielding Response Alignment Instructions
        for sample in iter(alignment_data):
            if counter >= train_limit:
                break
                
            if len(sample["answers"]) != 1:
                raise ValueError(f"Data ID: {sample['id']} has more than one answer")
            if sample["answers"][0]["answer_start"] != -1:
                raise ValueError(f"Data ID: {sample['id']} has answer_start != -1")
            new_text_format = f"<s>{sample['context']}\n\nQ: {sample['question']}\nA: {sample['answers'][0]['text']}</s>"

            yield {'text': new_text_format}
            counter += 1

def create_and_prepare_model(args):
    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)

    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=args.use_4bit,
    #     bnb_4bit_quant_type=args.bnb_4bit_quant_type,
    #     bnb_4bit_compute_dtype=compute_dtype,
    #     bnb_4bit_use_double_quant=args.use_nested_quant,
    # )

    if compute_dtype == torch.float16 and args.use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
            print("=" * 80)


    # Load the entire model on the GPU 0
    # switch to `device_map = "auto"` for multi-GPU
    
    # The following line means that 0th GPU out of the visible ones will be used
    # device_map = {"": 0}
    device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        # quantization_config=bnb_config, 
        device_map=device_map, 
        # use_auth_token=True,
        token='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',
        # revision="refs/pr/35" 
    )

    def get_target_modules(model_type):
        if model_type == "gemma":
            return [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj"
            ]
        elif model_type == "bloomz":
            return [
                "query_key_value",
                "dense",
                "dense_h_to_4h",
                "dense_4h_to_h"
            ]
    
    #### LLAMA STUFF 
    # check: https://github.com/huggingface/transformers/pull/24906
    model.config.pretraining_tp = 1 
    # model.config.
    #### LLAMA STUFF 
    model.config.window = script_args.context_window 
    peft_config = LoraConfig(
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        r=script_args.lora_r,
        bias="none",
        task_type="CAUSAL_LM", 
        target_modules=get_target_modules(script_args.model_type),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name, 
        trust_remote_code=True, 
        token='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')
    tokenizer.pad_token = tokenizer.eos_token
    return model, peft_config, tokenizer

training_arguments = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optim=script_args.optim,
    save_steps=script_args.save_steps,
    logging_steps=script_args.logging_steps,
    learning_rate=script_args.learning_rate,
    fp16=script_args.fp16,
    bf16=script_args.bf16,
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=script_args.num_train_epochs,
    max_grad_norm=script_args.max_grad_norm,
    max_steps=script_args.max_steps,
    warmup_ratio=script_args.warmup_ratio,
    group_by_length=script_args.group_by_length,
    lr_scheduler_type=script_args.lr_scheduler_type,
    # report_to='wandb',
    # run_name=NAME,
    report_to=[],
)


model, peft_config, tokenizer = create_and_prepare_model(script_args)
model.config.use_cache = False

train_gen = Dataset.from_generator(gen_batches_train)
val_gen = Dataset.from_generator(gen_batches_val)

# Fix weird overflow issue with fp16 training
tokenizer.padding_side = "right"
trainer = SFTTrainer(
    model=model,
    train_dataset=train_gen,
    eval_dataset=val_gen, 
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=script_args.max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=script_args.packing,
)

trainer.train()