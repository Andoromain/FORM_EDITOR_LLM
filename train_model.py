# train_model.py
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
import json

class FormGeneratorTrainer:
    def __init__(self, model_name="microsoft/phi-2"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_and_prepare_model(self):
        """
        Charge et prépare le modèle avec LoRA
        """
        print(f"Chargement du modèle {self.model_name}...")
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Modèle
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Configuration LoRA
        lora_config = LoraConfig(
            r=16,  # Rang de la matrice LoRA
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],  # Adapter selon le modèle
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Appliquer LoRA
        model = prepare_model_for_kbit_training(model)
        self.model = get_peft_model(model, lora_config)
        
        print(f"Modèle chargé avec {self.model.num_parameters()} paramètres")
        print(f"Paramètres entraînables: {self.model.num_parameters(only_trainable=True)}")
        
    def prepare_dataset(self, dataset_path="training_dataset.jsonl"):
        """
        Prépare le dataset pour l'entraînement
        """
        print("Chargement du dataset...")
        dataset = load_dataset('json', data_files=dataset_path, split='train')
        
        def format_prompt(example):
            """Format le prompt pour l'entraînement"""
            prompt = f"""### Instruction:
{example['instruction']}

### Réponse:
{example['output']}"""
            return {"text": prompt}
        
        dataset = dataset.map(format_prompt)
        
        # Split train/validation
        dataset = dataset.train_test_split(test_size=0.1)
        
        return dataset
    
    def tokenize_dataset(self, dataset):
        """
        Tokenize le dataset
        """
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=2048
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        
        return tokenized_dataset
    
    def train(self, output_dir="./form-generator-model"):
        """
        Lance l'entraînement
        """
        # Préparer les données
        dataset = self.prepare_dataset()
        tokenized_dataset = self.tokenize_dataset(dataset)
        
        # Configuration d'entraînement
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            fp16=True if self.device == "cuda" else False,
            logging_steps=10,
            save_steps=100,
            eval_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True,
            report_to="none",
            warmup_steps=100,
            optim="adamw_torch"
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            data_collator=data_collator
        )
        
        # Entraînement
        print("Début de l'entraînement...")
        trainer.train()
        
        # Sauvegarde
        print(f"Sauvegarde du modèle dans {output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        return trainer

def main():
    trainer = FormGeneratorTrainer(model_name="microsoft/phi-2")
    trainer.load_and_prepare_model()
    trainer.train()

if __name__ == "__main__":
    main()