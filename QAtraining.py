from dataclasses import dataclass, field
from typing import Optional
from datasets.arrow_dataset import Dataset
import torch
import random
import os
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
    set_seed as hf_set_seed,
)
import tensorflow_datasets as tfds
import random
import json 
import argparse
from trl import SFTTrainer
from huggingface_hub import login

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["WANDB_DISABLED"] = "true" 

hf_set_seed(42)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Instruction Tuning script")
    parser.add_argument("--model_name", type=str, default="bigscience/bloomz-7b1", help="Name of the model to be trained")
    parser.add_argument("--model_type", type=str, default="bloomz", help="Type of the model to be trained")
    parser.add_argument("--train_dataset_path", type=str, default="./datasets/xquad_in/or/xquad_or_train.json", help="Path to the training dataset")
    parser.add_argument("--output_model_name", type=str, default="bloomz_or_RA_DS=103_lora_qkv", help="Name of the output model")
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face token")
    args = parser.parse_args()

    login(token = args.hf_token)

    MODEL_NAME = args.model_name 
    MODEL_TYPE = args.model_type
    ALIGNMENT_TRAIN_DATASET_PATH = args.train_dataset_path
    OUTPUT_MODEL_NAME = args.output_model_name
    ALIGNMENT_DEV_DATASET_PATH = "./datasets/xquad_in/en/xquad_en_dev.json" # This doesn't matter, just ensure this is a some valid dataset path

    # Following hyperparameters are set
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
    OUTPUT_DIR = f"./models/{MODEL_TYPE}/{OUTPUT_MODEL_NAME}"
    CONTEXT_WINDOW = 256

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
        '''
        This function generates instances of training data for the model.
        It reads the training data from the specified path and yields batches of instances of training prompts.
        '''
        print("Loading train data for Response Alignment")
        print(script_args.alignment_train_dataset_path)

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


    def gen_batches_val():
        '''
        This function generates instances of validation data for the model.
        It reads the validation data from the specified path and yields batches of instances of validation prompts.
        This function is not to be bothered about much, as it is is no way going to affect the training.
        '''
        print("Loading val data for Response Alignment")
        print(script_args.alignment_dev_dataset_path)

        alignment_data = json.load(open(script_args.alignment_dev_dataset_path))["examples"]
        train_limit = 10 ### Setting it small to get through this step quickly
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
        """
        This function creates and prepares the model for training.
        It loads the model from the Hugging Face hub, sets the device map, and prepares the model for training.

        Returns: the model, the PEFT config, and the tokenizer.
        """
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)

        if compute_dtype == torch.float16 and args.use_4bit:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
                print("=" * 80)

        device_map = "auto"

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,  
            device_map=device_map
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
        
        model.config.pretraining_tp = 1 
        model.config.window = script_args.context_window 

        # Setting LoRA hyperparameters
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
        report_to=[],
    )

    model, peft_config, tokenizer = create_and_prepare_model(script_args)
    model.config.use_cache = False

    # Loading the training dataset
    train_gen = Dataset.from_generator(gen_batches_train)
    val_gen = Dataset.from_generator(gen_batches_val)

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