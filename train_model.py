# train_tinyllama.py
# Version avec le modèle le plus léger possible

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

def train_tinyllama():
    print("🚀 Entraînement avec TinyLlama (1.1B paramètres)")
    
    # Charger TinyLlama
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True
    )
    
    # LoRA minimal
    lora_config = LoraConfig(
        r=4,  # Très petit rang
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    # Dataset
    dataset = load_dataset('json', data_files="training_dataset.jsonl", split='train')
    
    def format_and_tokenize(example):
        text = f"### Instruction:\n{example['instruction']}\n\n### Réponse:\n{example['output']}"
        return tokenizer(text, padding="max_length", truncation=True, max_length=512)
    
    dataset = dataset.map(format_and_tokenize, remove_columns=dataset.column_names)
    dataset = dataset.train_test_split(test_size=0.1)
    
    # Configuration minimale
    training_args = TrainingArguments(
        output_dir="./tinyllama-form-generator",
        num_train_epochs=1,  # 1 seul epoch
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-4,
        logging_steps=5,
        save_strategy="no",  # Pas de sauvegarde intermédiaire
        fp16=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )
    
    print("\n📊 Entraînement en cours...")
    trainer.train()
    
    print("\n💾 Sauvegarde...")
    trainer.save_model("./tinyllama-form-generator")
    tokenizer.save_pretrained("./tinyllama-form-generator")
    
    print("\n✅ Terminé!")

if __name__ == "__main__":
    train_tinyllama()
