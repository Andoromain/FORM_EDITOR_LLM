# train_model.py (version compatible avec toutes les versions de transformers)
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
import transformers

class FormGeneratorTrainer:
    def __init__(self, model_name="microsoft/phi-2"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Transformers version: {transformers.__version__}")
        
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
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Appliquer LoRA
        model = prepare_model_for_kbit_training(model)
        self.model = get_peft_model(model, lora_config)
        
        print(f"Modèle chargé avec {self.model.num_parameters()} paramètres")
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Paramètres entraînables: {trainable_params:,}")
        
    def prepare_dataset(self, dataset_path="training_dataset.jsonl"):
        """
        Prépare le dataset pour l'entraînement
        """
        print("Chargement du dataset...")
        dataset = load_dataset('json', data_files=dataset_path, split='train')
        
        print(f"Dataset chargé: {len(dataset)} exemples")
        
        def format_prompt(example):
            """Format le prompt pour l'entraînement"""
            prompt = f"""### Instruction:
{example['instruction']}

### Réponse:
{example['output']}"""
            return {"text": prompt}
        
        dataset = dataset.map(format_prompt)
        
        # Split train/validation
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
        
        print(f"Train: {len(dataset['train'])} exemples")
        print(f"Validation: {len(dataset['test'])} exemples")
        
        return dataset
    
    def tokenize_dataset(self, dataset):
        """
        Tokenize le dataset
        """
        print("Tokenization du dataset...")
        
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
            remove_columns=dataset["train"].column_names,
            # Désactiver le cache pour éviter les warnings de hashing
            load_from_cache_file=False,
            desc="Tokenization"
        )
        
        print("Tokenization terminée")
        
        return tokenized_dataset
    
    def train(self, output_dir="./form-generator-model", num_epochs=3):
        """
        Lance l'entraînement
        """
        # Préparer les données
        dataset = self.prepare_dataset()
        tokenized_dataset = self.tokenize_dataset(dataset)
        
        # Détecter la version de transformers pour utiliser les bons paramètres
        transformers_version = tuple(int(x) for x in transformers.__version__.split('.')[:2])
        use_new_api = transformers_version >= (4, 30)
        
        print(f"Utilisation de l'API: {'nouvelle' if use_new_api else 'ancienne'}")
        
        # Configuration d'entraînement - Compatible toutes versions
        training_args_dict = {
            "output_dir": output_dir,
            "num_train_epochs": num_epochs,
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "fp16": True if self.device == "cuda" else False,
            
            # Stratégies - avec le bon nom selon la version
            "save_strategy": "steps",
            "save_steps": 100,
            
            "logging_steps": 10,
            "save_total_limit": 3,
            "load_best_model_at_end": True,
            "metric_for_best_model": "loss",
            "greater_is_better": False,
            
            "report_to": "none",
            "warmup_steps": 100,
            "optim": "adamw_torch",
            
            "logging_dir": f"{output_dir}/logs",
            "push_to_hub": False,
        }
        
        # Ajouter le paramètre d'évaluation avec le bon nom
        if use_new_api:
            training_args_dict["eval_strategy"] = "steps"  # Nouvelle API
            training_args_dict["eval_steps"] = 100
        else:
            training_args_dict["evaluation_strategy"] = "steps"  # Ancienne API
            training_args_dict["eval_steps"] = 100
        
        training_args = TrainingArguments(**training_args_dict)
        
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
        print("\n" + "="*50)
        print("Début de l'entraînement...")
        print("="*50 + "\n")
        
        trainer.train()
        
        # Sauvegarde finale
        print(f"\nSauvegarde du modèle dans {output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Sauvegarder aussi la configuration LoRA
        self.model.save_pretrained(output_dir)
        
        print("\n" + "="*50)
        print("Entraînement terminé avec succès!")
        print("="*50 + "\n")
        
        return trainer

def main():
    # Choix du modèle
    models = {
        "phi-2": "microsoft/phi-2",
        "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "mistral": "mistralai/Mistral-7B-v0.1",
    }
    
    # Sélectionner le modèle
    selected_model = "phi-2"
    
    print(f"Utilisation du modèle: {models[selected_model]}")
    
    trainer = FormGeneratorTrainer(model_name=models[selected_model])
    trainer.load_and_prepare_model()
    trainer.train(num_epochs=3)

if __name__ == "__main__":
    main()
