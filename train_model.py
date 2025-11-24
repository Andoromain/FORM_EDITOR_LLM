# train_model_light.py
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
import transformers
import gc

class FormGeneratorTrainer:
    def __init__(self, model_name="microsoft/phi-2"):
        self.model_name = model_name
        self.device = "cpu"  # Forcer CPU
        print(f"Device: {self.device}")
        print(f"Transformers version: {transformers.__version__}")
        
    def load_and_prepare_model(self):
        """Charge et prépare le modèle avec LoRA - Version légère"""
        print(f"\nChargement du modèle {self.model_name}...")
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Modèle en CPU avec optimisations mémoire
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,  # CPU nécessite float32
            device_map={"": "cpu"},  # Tout sur CPU
            trust_remote_code=True,
            low_cpu_mem_usage=True  # Optimisation mémoire
        )
        
        # Configuration LoRA minimale
        lora_config = LoraConfig(
            r=8,  # Réduit de 16 à 8
            lora_alpha=16,  # Réduit de 32 à 16
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(model, lora_config)
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = self.model.num_parameters()
        print(f"Paramètres totaux: {total_params:,}")
        print(f"Paramètres entraînables: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
    def prepare_dataset(self, dataset_path="training_dataset.jsonl"):
        """Prépare le dataset"""
        print(f"\nChargement du dataset: {dataset_path}")
        dataset = load_dataset('json', data_files=dataset_path, split='train')
        print(f"Total exemples: {len(dataset)}")
        
        def format_prompt(example):
            return {
                "text": f"""### Instruction:
{example['instruction']}

### Réponse:
{example['output']}"""
            }
        
        dataset = dataset.map(format_prompt, load_from_cache_file=False)
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
        
        print(f"Train: {len(dataset['train'])} | Validation: {len(dataset['test'])}")
        return dataset
    
    def tokenize_dataset(self, dataset):
        """Tokenize le dataset avec longueur réduite"""
        print("\nTokenization en cours...")
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=1024  # Réduit de 2048 à 1024
            )
        
        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=False,
            desc="Tokenizing"
        )
        
        print("✓ Tokenization terminée")
        return tokenized
    
    def train(self, output_dir="./form-generator-model", num_epochs=2):
        """Lance l'entraînement - Configuration ultra-légère"""
        dataset = self.prepare_dataset()
        tokenized_dataset = self.tokenize_dataset(dataset)
        
        # Libérer la mémoire
        gc.collect()
        
        # Configuration ULTRA-LÉGÈRE
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            
            # Batch minimal
            per_device_train_batch_size=1,  # ✅ Batch size = 1
            gradient_accumulation_steps=16,  # ✅ Accumuler pour compenser
            
            # Optimisation
            learning_rate=3e-4,
            warmup_steps=10,
            weight_decay=0.01,
            
            # Logging
            logging_dir=f"{output_dir}/logs",
            logging_steps=5,
            
            # Sauvegarde
            save_strategy="epoch",
            save_total_limit=2,
            
            # Optimisations CPU
            fp16=False,  # ✅ Pas de FP16 sur CPU
            dataloader_num_workers=0,  # ✅ Pas de workers parallèles
            dataloader_pin_memory=False,  # ✅ Désactiver pin_memory
            gradient_checkpointing=True,  # ✅ Économie mémoire
            
            # Autres
            report_to="none",
            push_to_hub=False,
            disable_tqdm=False,
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            data_collator=data_collator
        )
        
        print("\n" + "="*60)
        print("🚀 DÉBUT DE L'ENTRAÎNEMENT")
        print("="*60)
        print(f"⚠️  Mode CPU - L'entraînement sera lent")
        print(f"Epochs: {num_epochs} | Batch size: 1 | Grad accumulation: 16")
        print("="*60 + "\n")
        
        try:
            trainer.train()
            
            print("\n" + "="*60)
            print("💾 SAUVEGARDE DU MODÈLE")
            print("="*60)
            
            trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            self.model.save_pretrained(output_dir)
            
            print(f"\n✅ Modèle sauvegardé dans: {output_dir}")
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"\n❌ Erreur pendant l'entraînement: {e}")
            raise
        
        return trainer

def main():
    print("="*60)
    print("ENTRAÎNEMENT FORM GENERATOR - MODE LÉGER")
    print("="*60 + "\n")
    
    # Utiliser TinyLlama au lieu de Phi-2 (plus léger)
    # Commenter/décommenter selon vos besoins
    
    # Option 1: Phi-2 (2.7B paramètres)
    trainer = FormGeneratorTrainer(model_name="microsoft/phi-2")
    
    # Option 2: TinyLlama (1.1B paramètres - RECOMMANDÉ pour CPU)
    # trainer = FormGeneratorTrainer(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    trainer.load_and_prepare_model()
    trainer.train(num_epochs=2)  # Seulement 2 epochs

if __name__ == "__main__":
    main()
