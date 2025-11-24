# generate_form.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json

class FormGenerator:
    def __init__(self, model_path="./form-generator-model"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print("Chargement du modèle...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
        
        self.model = base_model
        self.model.eval()
    
    def generate_form(self, description: str, max_length=4096):
        """
        Génère une structure de formulaire à partir d'une description
        """
        prompt = f"""### Instruction:
{description}

### Réponse:
"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extraire uniquement la réponse
        response = generated_text.split("### Réponse:")[-1].strip()
        
        # Parser le JSON
        try:
            form_structure = json.loads(response)
            return form_structure
        except json.JSONDecodeError:
            print("Erreur de parsing JSON, retour du texte brut")
            return {"raw_output": response}
    
    def generate_from_template(self, form_type: str, fields: list):
        """
        Génère un formulaire à partir d'un template
        """
        description = f"Crée un formulaire de type {form_type} avec les champs suivants : "
        description += ", ".join(fields)
        
        return self.generate_form(description)

# Exemple d'utilisation
if __name__ == "__main__":
    generator = FormGenerator()
    
    # Exemple 1: Description libre
    form1 = generator.generate_form(
        "Crée un formulaire d'inscription avec nom, prénom, email, téléphone et adresse"
    )
    print(json.dumps(form1, indent=2, ensure_ascii=False))
    
    # Exemple 2: Template
    form2 = generator.generate_from_template(
        "contact",
        ["nom", "email", "sujet", "message"]
    )
    print(json.dumps(form2, indent=2, ensure_ascii=False))