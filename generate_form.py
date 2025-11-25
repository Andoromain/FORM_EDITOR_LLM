# generate_form.py
# Générateur de formulaires avec Llama 3.2 3B
# Compatible avec Google Colab et environnement local

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import os

class FormGenerator:
    def __init__(self, model_path=None):
        """
        Initialise le générateur de formulaires.

        Args:
            model_path: Chemin vers le modèle entraîné.
                       Si None, détecte automatiquement (Colab ou local)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Détection automatique du chemin du modèle
        if model_path is None:
            is_colab = 'COLAB_GPU' in os.environ or os.path.exists('/content')
            if is_colab:
                model_path = "/content/llama3-form-generator"
                # Essayer aussi sur Drive
                if not os.path.exists(model_path):
                    model_path = "/content/drive/MyDrive/llama3-form-generator"
            else:
                model_path = "./llama3-form-generator"

        self.model_path = model_path

        print(f"Device: {self.device}")
        print(f"Chargement du modèle depuis: {model_path}")

        # Charger le tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Charger le modèle
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        self.model.eval()
        print("✅ Modèle chargé avec succès!")

    def generate_form(self, description: str, max_new_tokens=1024, temperature=0.7, top_p=0.9):
        """
        Génère une structure de formulaire à partir d'une description.

        Args:
            description: Description du formulaire à générer
            max_new_tokens: Nombre maximum de tokens à générer
            temperature: Température de génération (0.0-1.0)
            top_p: Top-p sampling

        Returns:
            dict: Structure de formulaire JSON ou dict avec raw_output en cas d'erreur
        """
        # Format de prompt pour Llama 3.2 Instruct
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Tu es un assistant spécialisé dans la génération de structures de formulaires JSON.<|eot_id|><|start_header_id|>user<|end_header_id|>

{description}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extraire uniquement la réponse de l'assistant
        if "assistant" in generated_text:
            response = generated_text.split("assistant")[1].strip()
        else:
            response = generated_text.strip()

        # Nettoyer la réponse (retirer les balises EOT si présentes)
        response = response.replace("<|eot_id|>", "").strip()

        # Parser le JSON
        try:
            form_structure = json.loads(response)
            return form_structure
        except json.JSONDecodeError as e:
            print(f"⚠️ Erreur de parsing JSON: {e}")
            print(f"Réponse brute: {response[:200]}...")
            return {"raw_output": response, "error": str(e)}

    def generate_from_template(self, form_type: str, fields: list):
        """
        Génère un formulaire à partir d'un template.

        Args:
            form_type: Type de formulaire (ex: "contact", "inscription")
            fields: Liste des champs requis

        Returns:
            dict: Structure de formulaire JSON
        """
        description = f"Crée un formulaire de type {form_type} avec les champs suivants : "
        description += ", ".join(fields)

        return self.generate_form(description)

    def generate_batch(self, descriptions: list, **kwargs):
        """
        Génère plusieurs formulaires en batch.

        Args:
            descriptions: Liste de descriptions
            **kwargs: Arguments pour generate_form

        Returns:
            list: Liste de structures de formulaires
        """
        results = []
        for desc in descriptions:
            result = self.generate_form(desc, **kwargs)
            results.append(result)
        return results

# Exemple d'utilisation
if __name__ == "__main__":
    print("=" * 60)
    print("Générateur de Formulaires - Llama 3.2 3B")
    print("=" * 60)

    # Initialiser le générateur
    generator = FormGenerator()

    print("\n" + "=" * 60)
    print("Exemple 1: Description libre")
    print("=" * 60)

    # Exemple 1: Description libre
    form1 = generator.generate_form(
        "Crée un formulaire d'inscription avec nom, prénom, email, téléphone et adresse"
    )
    print("\n📋 Résultat:")
    print(json.dumps(form1, indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    print("Exemple 2: Template")
    print("=" * 60)

    # Exemple 2: Template
    form2 = generator.generate_from_template(
        "contact",
        ["nom", "email", "sujet", "message"]
    )
    print("\n📋 Résultat:")
    print(json.dumps(form2, indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    print("Exemple 3: Batch generation")
    print("=" * 60)

    # Exemple 3: Batch
    descriptions = [
        "Crée un formulaire de commande avec produit, quantité et adresse",
        "Crée un formulaire de feedback avec note et commentaire"
    ]
    forms = generator.generate_batch(descriptions)

    for i, form in enumerate(forms, 1):
        print(f"\n📋 Formulaire {i}:")
        print(json.dumps(form, indent=2, ensure_ascii=False))
