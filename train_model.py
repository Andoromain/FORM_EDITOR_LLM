# train_model.py
# Script d'entraînement avec Llama 3.2 3B
# Compatible avec Google Colab et environnement local

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import os

def train_llama3():
    print("🚀 Entraînement avec Llama 3.2 3B")

    # Détection automatique de l'environnement
    is_colab = 'COLAB_GPU' in os.environ or os.path.exists('/content')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Environment: {'Google Colab' if is_colab else 'Local'}")
    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Charger Llama 3.2 3B Instruct
    model_name = "meta-llama/Llama-3.2-3B-Instruct"

    print(f"\n📥 Chargement du tokenizer depuis {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    # Llama 3.2 utilise un pad_token différent
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"\n📥 Chargement du modèle depuis {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    # Préparer le modèle pour l'entraînement avec LoRA
    model = prepare_model_for_kbit_training(model)

    # Configuration LoRA optimisée pour Llama 3.2 3B
    print("\n🔧 Configuration LoRA...")
    lora_config = LoraConfig(
        r=16,  # Rang plus élevé pour un modèle plus grand
        lora_alpha=32,  # Alpha = 2 * r (recommandé)
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Charger le dataset
    print("\n📚 Chargement du dataset...")
    dataset_path = "training_dataset.jsonl"
    if is_colab and not os.path.exists(dataset_path):
        dataset_path = "/content/training_dataset.jsonl"

    dataset = load_dataset('json', data_files=dataset_path, split='train')
    print(f"Dataset size: {len(dataset)} exemples")

    # Template de prompt pour Llama 3.2 Instruct
    def format_and_tokenize(example):
        # Format spécial pour Llama 3.2 Instruct
        text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Tu es un assistant spécialisé dans la génération de structures de formulaires JSON.<|eot_id|><|start_header_id|>user<|end_header_id|>

{example['instruction']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{example['output']}<|eot_id|>"""

        return tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=2048  # Llama 3.2 peut gérer des contextes plus longs
        )

    print("\n🔄 Tokenisation du dataset...")
    dataset = dataset.map(
        format_and_tokenize,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )
    dataset = dataset.train_test_split(test_size=0.1)

    print(f"Train size: {len(dataset['train'])}")
    print(f"Test size: {len(dataset['test'])}")

    # Configuration d'entraînement optimisée pour Colab
    output_dir = "/content/llama3-form-generator" if is_colab else "./llama3-form-generator"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,  # Plus d'epochs pour un meilleur apprentissage
        per_device_train_batch_size=2 if device == "cuda" else 1,
        per_device_eval_batch_size=2 if device == "cuda" else 1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,  # Économise la mémoire
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        fp16=device == "cuda",  # Utiliser fp16 uniquement sur GPU
        bf16=False,
        optim="paged_adamw_32bit" if device == "cuda" else "adamw_torch",
        dataloader_num_workers=2 if device == "cuda" else 0,
        dataloader_pin_memory=True if device == "cuda" else False,
        report_to="none",  # Désactiver wandb/tensorboard par défaut
        load_best_model_at_end=True,
        metric_for_best_model="loss",
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Trainer
    print("\n🏋️ Initialisation du Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator
    )

    print("\n📊 Début de l'entraînement...")
    print("=" * 60)
    trainer.train()

    print("\n💾 Sauvegarde du modèle...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n✅ Entraînement terminé!")
    print(f"📁 Modèle sauvegardé dans: {output_dir}")

    # Si sur Colab, suggérer de télécharger ou sauvegarder sur Drive
    if is_colab:
        print("\n💡 Pour sauvegarder sur Google Drive:")
        print("   from google.colab import drive")
        print("   drive.mount('/content/drive')")
        print(f"   !cp -r {output_dir} /content/drive/MyDrive/")

if __name__ == "__main__":
    train_llama3()
