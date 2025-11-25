# 🚀 Guide d'utilisation - Llama 3.2 3B avec Google Colab

Ce guide explique comment entraîner et utiliser Llama 3.2 3B pour générer des structures de formulaires JSON, avec support complet pour Google Colab.

## 📋 Table des matières

- [Vue d'ensemble](#vue-densemble)
- [Prérequis](#prérequis)
- [Option 1: Google Colab (Recommandé)](#option-1-google-colab-recommandé)
- [Option 2: Environnement local](#option-2-environnement-local)
- [Structure du projet](#structure-du-projet)
- [FAQ et dépannage](#faq-et-dépannage)

## 🎯 Vue d'ensemble

Ce projet vous permet de:
- ✅ Entraîner un modèle Llama 3.2 3B (fine-tuning avec LoRA)
- ✅ Générer des structures de formulaires JSON à partir de descriptions en langage naturel
- ✅ Utiliser Google Colab pour l'entraînement et l'inférence (GPU gratuit)
- ✅ Exécuter localement sur CPU ou GPU

**Modèle utilisé:** `meta-llama/Llama-3.2-3B-Instruct`

**Technique d'entraînement:** LoRA (Low-Rank Adaptation) pour un entraînement efficace avec peu de ressources

## 🔧 Prérequis

### Pour Google Colab
1. **Compte Google** avec Google Drive
2. **Token Hugging Face**
   - Créez un compte sur [Hugging Face](https://huggingface.co/)
   - Acceptez la licence pour Llama 3.2: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
   - Créez un token: https://huggingface.co/settings/tokens
   - ⚠️ **Important:** Le token doit avoir les permissions de lecture

3. **GPU Colab** (gratuit)
   - Runtime > Change runtime type > Hardware accelerator > GPU
   - GPU recommandés: T4 (gratuit), V100, A100

### Pour environnement local
```bash
# Python 3.8+
pip install torch transformers datasets accelerate peft bitsandbytes sentencepiece
```

## 🌐 Option 1: Google Colab (Recommandé)

### 📝 Étape 1: Préparation des données

#### Option A: Cloner le repository
```python
# Dans Colab
!git clone https://github.com/VOTRE_USERNAME/FORM_EDITOR_LLM.git
%cd FORM_EDITOR_LLM
```

#### Option B: Uploader vos fichiers
1. Préparez votre dataset localement avec `prepare_dataset.py`
2. Uploadez `training_dataset.jsonl` dans Colab

### 🏋️ Étape 2: Entraînement

1. **Ouvrez le notebook d'entraînement**
   - Uploadez `train_colab.ipynb` dans Google Colab
   - Ou ouvrez directement depuis GitHub: [Lien à ajouter]

2. **Configuration**
   ```python
   # Dans le notebook
   # Runtime > Change runtime type > GPU
   ```

3. **Exécutez les cellules dans l'ordre**
   - Installation des dépendances (2-3 min)
   - Montage de Google Drive
   - Authentification Hugging Face (entrez votre token)
   - Upload du dataset
   - Entraînement (1-3 heures selon GPU)
   - Sauvegarde sur Google Drive

4. **Temps estimés**
   - T4 (gratuit): ~2-3 heures
   - V100: ~1-1.5 heures
   - A100: ~45-60 minutes

### 🎯 Étape 3: Génération de formulaires

1. **Ouvrez le notebook d'inférence**
   - Uploadez `inference_colab.ipynb` dans Google Colab

2. **Exécutez les cellules**
   - Installation des dépendances
   - Montage de Google Drive
   - Chargement du modèle entraîné
   - Génération de formulaires

3. **Exemples d'utilisation**
   ```python
   # Génération simple
   form = generate_form(
       "Crée un formulaire d'inscription avec nom, prénom, email et téléphone"
   )

   # Génération avec paramètres
   form = generate_form(
       "Crée un formulaire de contact",
       temperature=0.7,  # Créativité
       top_p=0.9,        # Diversité
       max_new_tokens=1024
   )
   ```

### 💾 Sauvegarde et téléchargement

**Le modèle est automatiquement sauvegardé sur Google Drive:**
```
/content/drive/MyDrive/llama3-form-generator/
```

**Pour télécharger localement:**
```python
# Dans Colab
!zip -r llama3-form-generator.zip /content/drive/MyDrive/llama3-form-generator
from google.colab import files
files.download('llama3-form-generator.zip')
```

## 💻 Option 2: Environnement local

### Installation

```bash
# Cloner le repository
git clone https://github.com/VOTRE_USERNAME/FORM_EDITOR_LLM.git
cd FORM_EDITOR_LLM

# Installer les dépendances
pip install -r requirements.txt

# Authentification Hugging Face
huggingface-cli login
```

### Préparation des données

```bash
# Générer le dataset
python prepare_dataset.py

# Vérifier que training_dataset.jsonl est créé
ls -lh training_dataset.jsonl
```

### Entraînement

```bash
# Lancer l'entraînement
python train_model.py

# Le modèle sera sauvegardé dans ./llama3-form-generator/
```

**Configuration système recommandée:**
- GPU: NVIDIA avec 16+ GB VRAM (RTX 3090, RTX 4090, A100)
- RAM: 32+ GB
- Stockage: 20+ GB libre

**Sans GPU:**
- L'entraînement fonctionnera mais sera très lent (10-20x plus lent)
- Préférez Google Colab avec GPU gratuit

### Génération

```python
from generate_form import FormGenerator

# Initialiser
generator = FormGenerator()

# Générer
form = generator.generate_form(
    "Crée un formulaire d'inscription avec nom, email et téléphone"
)

print(form)
```

## 📁 Structure du projet

```
FORM_EDITOR_LLM/
├── train_model.py              # Script d'entraînement (local + Colab)
├── generate_form.py            # Générateur de formulaires (local + Colab)
├── prepare_dataset.py          # Préparation du dataset
├── train_colab.ipynb          # 📓 Notebook Colab pour l'entraînement
├── inference_colab.ipynb      # 📓 Notebook Colab pour l'inférence
├── README_LLAMA3_COLAB.md     # Ce fichier
├── form_structure.json         # Structures de formulaires existantes
├── training_dataset.jsonl      # Dataset d'entraînement (généré)
└── llama3-form-generator/     # Modèle entraîné (après training)
    ├── adapter_config.json
    ├── adapter_model.bin
    ├── tokenizer_config.json
    └── ...
```

## 🎓 Guide détaillé

### Comprendre les paramètres d'entraînement

```python
# Dans train_model.py
TrainingArguments(
    num_train_epochs=3,              # Nombre de passages sur le dataset
    per_device_train_batch_size=2,   # Taille du batch (ajuster selon GPU)
    gradient_accumulation_steps=4,    # Accumulation de gradient
    learning_rate=2e-4,              # Taux d'apprentissage
    fp16=True,                       # Précision mixte (économise la mémoire)
)
```

**Ajustements selon votre GPU:**

| GPU | VRAM | batch_size | gradient_accumulation |
|-----|------|------------|-----------------------|
| T4 | 16GB | 1-2 | 8 |
| V100 | 32GB | 2-4 | 4 |
| A100 | 40GB | 4-8 | 2 |

### Configuration LoRA

```python
LoraConfig(
    r=16,              # Rang de la décomposition (plus grand = plus de paramètres)
    lora_alpha=32,     # Scaling factor (généralement 2*r)
    target_modules=[   # Modules à adapter
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05, # Dropout pour régularisation
)
```

**Paramètres entraînables:** ~21M paramètres (0.7% du modèle complet)

### Format de prompt Llama 3.2 Instruct

```python
prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Tu es un assistant spécialisé dans la génération de structures de formulaires JSON.<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{output}<|eot_id|>"""
```

⚠️ **Important:** Respectez exactement ce format pour de meilleurs résultats.

### Paramètres de génération

```python
generate_form(
    description="...",
    max_new_tokens=1024,  # Longueur de la génération
    temperature=0.7,      # Créativité (0.0-1.0)
    top_p=0.9,           # Diversité (0.0-1.0)
)
```

**Recommandations:**
- **temperature=0.3-0.5**: Résultats déterministes et cohérents
- **temperature=0.7-0.8**: Bon équilibre créativité/cohérence
- **temperature=0.9-1.0**: Très créatif mais moins prévisible

## ❓ FAQ et dépannage

### Erreurs courantes

#### ❌ "RuntimeError: CUDA out of memory"
**Solutions:**
```python
# 1. Réduire le batch size
per_device_train_batch_size=1

# 2. Augmenter gradient_accumulation_steps
gradient_accumulation_steps=8

# 3. Activer gradient checkpointing (déjà activé par défaut)
gradient_checkpointing=True

# 4. Réduire max_length
max_length=1024  # au lieu de 2048
```

#### ❌ "Token is not valid" (Hugging Face)
1. Vérifiez que vous avez accepté la licence Llama 3.2
2. Créez un nouveau token avec permissions de lecture
3. Reconnectez-vous: `huggingface-cli login`

#### ❌ "JSONDecodeError" lors de la génération
**Solutions:**
```python
# 1. Réduire la température
temperature=0.3

# 2. Augmenter max_new_tokens
max_new_tokens=2048

# 3. Vérifier le format du prompt

# 4. Améliorer le dataset d'entraînement
```

#### ❌ Le modèle génère toujours la même chose
**Solutions:**
```python
# Augmenter la température
temperature=0.8

# Augmenter top_p
top_p=0.95

# Activer do_sample
do_sample=True
```

### Performances

**Entraînement (3 epochs, 500 exemples):**
- T4 (16GB): ~2-3h
- V100 (32GB): ~1-1.5h
- A100 (40GB): ~45-60min
- CPU: ~20-30h (non recommandé)

**Inférence (génération d'un formulaire):**
- GPU: 2-5 secondes
- CPU: 10-30 secondes

### Optimisations

#### Pour accélérer l'entraînement:
```python
# Utiliser moins d'epochs
num_train_epochs=1

# Augmenter le batch size (si mémoire suffisante)
per_device_train_batch_size=4

# Réduire les eval_steps
eval_steps=100  # au lieu de 50
```

#### Pour améliorer la qualité:
```python
# Plus d'epochs
num_train_epochs=5

# Plus de données d'entraînement
# Modifier prepare_dataset.py pour générer plus d'exemples

# Augmenter le rang LoRA
r=32  # au lieu de 16
lora_alpha=64
```

## 🔗 Ressources utiles

- [Documentation Llama 3.2](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- [Guide LoRA](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)
- [Documentation Transformers](https://huggingface.co/docs/transformers/)
- [Google Colab GPU](https://colab.research.google.com/notebooks/gpu.ipynb)
- [Hugging Face Hub](https://huggingface.co/docs/hub/index)

## 📊 Comparaison avec TinyLlama

| Caractéristique | TinyLlama 1.1B | Llama 3.2 3B |
|----------------|----------------|--------------|
| Paramètres | 1.1B | 3B |
| Qualité | Basique | Excellente |
| Vitesse training | Rapide | Moyen |
| Mémoire GPU | 6-8 GB | 12-16 GB |
| Contexte | 2048 | 8192 |
| Format prompt | Simple | Instruct |

**Recommandation:** Utilisez Llama 3.2 3B pour de meilleurs résultats. La différence de qualité justifie largement les ressources supplémentaires.

## 🎉 Prochaines étapes

Après avoir entraîné votre modèle:

1. **Testez différents types de formulaires**
   - Formulaires d'inscription
   - Formulaires de contact
   - Formulaires de commande
   - Formulaires de feedback

2. **Expérimentez avec les paramètres**
   - Ajustez temperature et top_p
   - Testez différentes descriptions
   - Créez vos propres templates

3. **Intégrez dans votre application**
   - API REST avec FastAPI
   - Application web
   - Service backend

4. **Améliorez le modèle**
   - Ajoutez plus de données d'entraînement
   - Fine-tunez avec vos propres formulaires
   - Augmentez le nombre d'epochs

## 💡 Astuces

### Pour de meilleurs résultats:
1. ✅ Soyez spécifique dans vos descriptions
2. ✅ Listez tous les champs nécessaires
3. ✅ Mentionnez les types de champs (select, checkbox, etc.)
4. ✅ Indiquez les champs obligatoires
5. ✅ Donnez du contexte sur l'usage du formulaire

### Exemples de bonnes descriptions:
```
✅ "Crée un formulaire d'inscription à une conférence avec nom complet,
   email (obligatoire), entreprise, poste, régime alimentaire
   (végétarien/vegan/aucun) et questions pour les speakers"

✅ "Crée un formulaire de commande e-commerce avec sélection de produit,
   quantité (min 1, max 10), mode de livraison (express/standard/retrait),
   adresse de livraison complète et code promo optionnel"

❌ "Fais un formulaire" (trop vague)
❌ "Formulaire inscription" (manque de détails)
```

## 📝 License

Ce projet utilise Llama 3.2 qui est sous licence [Llama 3.2 Community License](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct).

## 🤝 Contribution

Les contributions sont les bienvenues! N'hésitez pas à:
- Signaler des bugs
- Proposer des améliorations
- Partager vos résultats
- Ajouter des exemples

## 📧 Support

Pour toute question:
- Ouvrez une issue sur GitHub
- Consultez la FAQ ci-dessus
- Vérifiez les ressources utiles

---

**Bon entraînement! 🚀**
