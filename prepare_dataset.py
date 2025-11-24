# prepare_dataset.py
import json
import random
from typing import List, Dict

class FormDatasetGenerator:
    def __init__(self):
        self.field_types = [
            "text", "textarea", "date", "time", "select", 
            "multiselect", "radio", "checkbox", "switch", 
            "file", "number", "email", "titre"
        ]
        
        self.example_contexts = [
            "création d'entreprise",
            "modification de société",
            "déclaration fiscale",
            "demande de permis",
            "inscription en ligne",
            "formulaire de contact",
            "demande de crédit",
            "dossier médical",
            "réservation",
            "candidature emploi"
        ]
    
    def generate_training_pair(self, form_structure: Dict) -> Dict:
        """
        Génère une paire instruction/réponse pour l'entraînement
        """
        # Extraire les informations clés
        label = form_structure.get('label', '')
        key_word = form_structure.get('key_word', '')
        options = form_structure.get('options', [])
        
        # Créer une description en langage naturel
        instruction = self.create_natural_description(label, options)
        
        # La réponse est la structure JSON
        response = json.dumps(form_structure, indent=2, ensure_ascii=False)
        
        return {
            "instruction": instruction,
            "input": "",
            "output": response
        }
    
    def create_natural_description(self, label: str, options: List[Dict]) -> str:
        """
        Crée une description en langage naturel du formulaire
        """
        descriptions = []
        descriptions.append(f"Crée un formulaire pour : {label}")
        
        # Analyser les champs
        field_descriptions = []
        for option in options[:10]:  # Limiter pour la description
            field_type = option.get('type', '')
            field_label = option.get('label', '')
            is_required = option.get('is_required', False)
            
            desc = f"- {field_label}"
            if field_type == 'select':
                values = option.get('values', [])
                if values:
                    desc += f" (choix parmi: {', '.join([v.get('label', '') for v in values[:3]])})"
            if is_required:
                desc += " [obligatoire]"
            
            field_descriptions.append(desc)
        
        if field_descriptions:
            descriptions.append("\nChamps nécessaires:")
            descriptions.extend(field_descriptions[:8])
        
        # Ajouter des détails sur les dépendances
        dependent_fields = [o for o in options if o.get('has_dependency')]
        if dependent_fields:
            descriptions.append(f"\n{len(dependent_fields)} champs avec des dépendances conditionnelles")
        
        return "\n".join(descriptions)
    
    def augment_instruction(self, base_instruction: str) -> List[str]:
        """
        Crée des variations de l'instruction pour enrichir le dataset
        """
        variations = [
            base_instruction,
            f"Je veux {base_instruction.lower()}",
            f"Peux-tu {base_instruction.lower()}",
            f"Génère {base_instruction[5:].lower()}" if base_instruction.startswith("Crée") else base_instruction,
            f"J'ai besoin d'{base_instruction[5:].lower()}" if base_instruction.startswith("Crée") else base_instruction,
        ]
        return variations
    
    def generate_synthetic_examples(self, num_examples: int = 100) -> List[Dict]:
        """
        Génère des exemples synthétiques de formulaires
        """
        examples = []
        
        templates = {
            "contact": {
                "fields": ["nom", "prénom", "email", "téléphone", "message"],
                "types": ["text", "text", "email", "text", "textarea"]
            },
            "inscription": {
                "fields": ["nom", "prénom", "date de naissance", "adresse", "ville", "code postal"],
                "types": ["text", "text", "date", "text", "text", "text"]
            },
            "commande": {
                "fields": ["produit", "quantité", "adresse de livraison", "mode de paiement"],
                "types": ["select", "number", "textarea", "radio"]
            }
        }
        
        for i in range(num_examples):
            template_name = random.choice(list(templates.keys()))
            template = templates[template_name]
            
            form_structure = self.create_synthetic_form(
                f"Formulaire de {template_name} #{i}",
                template["fields"],
                template["types"]
            )
            
            examples.append(self.generate_training_pair(form_structure))
        
        return examples
    
    def create_synthetic_form(self, label: str, fields: List[str], types: List[str]) -> Dict:
        """
        Crée une structure de formulaire synthétique
        """
        options = []
        for idx, (field_name, field_type) in enumerate(zip(fields, types)):
            option = {
                "step": 1,
                "type": field_type,
                "champ": field_name.lower().replace(" ", "_"),
                "label": field_name.capitalize(),
                "ordre": idx + 1,
                "is_required": random.choice([True, False])
            }
            
            if field_type in ["select", "radio"]:
                option["values"] = [
                    {"label": f"Option {j+1}", "value": f"option_{j+1}"}
                    for j in range(random.randint(2, 5))
                ]
            
            options.append(option)
        
        return {
            "code": "1",
            "label": label,
            "key_word": label.lower().replace(" ", "_"),
            "options": options
        }

def main():
    generator = FormDatasetGenerator()
    
    # Charger votre formulaire existant
    with open('form_structure.json', 'r', encoding='utf-8') as f:
        existing_forms = json.load(f)
    
    # Générer le dataset
    dataset = []
    
    # Ajouter les formulaires existants
    for form in existing_forms:
        pair = generator.generate_training_pair(form)
        
        # Créer des variations
        for instruction_variant in generator.augment_instruction(pair["instruction"]):
            dataset.append({
                "instruction": instruction_variant,
                "input": "",
                "output": pair["output"]
            })
    
    # Ajouter des exemples synthétiques
    synthetic_examples = generator.generate_synthetic_examples(100)
    dataset.extend(synthetic_examples)
    
    # Sauvegarder le dataset
    with open('training_dataset.jsonl', 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Dataset créé avec {len(dataset)} exemples")
    print(f"Sauvegardé dans: training_dataset.jsonl")

if __name__ == "__main__":
    main()