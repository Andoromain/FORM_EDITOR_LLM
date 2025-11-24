# monitor_training.py
import psutil
import time
import subprocess
import sys

def get_memory_usage():
    """Récupère l'utilisation mémoire"""
    mem = psutil.virtual_memory()
    return {
        'total': mem.total / (1024**3),  # GB
        'used': mem.used / (1024**3),
        'free': mem.available / (1024**3),
        'percent': mem.percent
    }

def monitor_training(script_name="train_model_light.py"):
    """Lance l'entraînement et monitore la mémoire"""
    
    print("="*60)
    print("MONITORING MÉMOIRE")
    print("="*60)
    
    # Afficher la mémoire initiale
    mem = get_memory_usage()
    print(f"\n📊 Mémoire initiale:")
    print(f"   Total: {mem['total']:.2f} GB")
    print(f"   Utilisée: {mem['used']:.2f} GB")
    print(f"   Libre: {mem['free']:.2f} GB")
    print(f"   Pourcentage: {mem['percent']:.1f}%")
    
    if mem['free'] < 4:
        print(f"\n⚠️  ATTENTION: Seulement {mem['free']:.2f} GB de RAM libre!")
        print("   L'entraînement risque d'échouer.")
        response = input("\nContinuer quand même? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    print(f"\n🚀 Lancement de {script_name}...\n")
    print("="*60 + "\n")
    
    # Lancer le script
    process = subprocess.Popen(
        [sys.executable, script_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Monitorer en temps réel
    max_mem = 0
    try:
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
            
            # Vérifier la mémoire toutes les lignes
            mem = get_memory_usage()
            max_mem = max(max_mem, mem['used'])
            
            if mem['percent'] > 95:
                print(f"\n⚠️  ALERTE MÉMOIRE: {mem['percent']:.1f}% utilisée!")
                
    except KeyboardInterrupt:
        print("\n\n⚠️  Entraînement interrompu par l'utilisateur")
        process.terminate()
    
    process.wait()
    
    print("\n" + "="*60)
    print("RÉSUMÉ")
    print("="*60)
    print(f"Mémoire max utilisée: {max_mem:.2f} GB")
    print(f"Code de sortie: {process.returncode}")
    
    if process.returncode == -9 or process.returncode == 137:
        print("\n❌ Processus tué par OOM Killer (manque de RAM)")
        print("\n💡 Solutions:")
        print("   1. Réduire batch_size à 1")
        print("   2. Réduire max_length à 512")
        print("   3. Utiliser TinyLlama au lieu de Phi-2")
        print("   4. Ajouter de la SWAP")
    elif process.returncode == 0:
        print("\n✅ Entraînement terminé avec succès!")

if __name__ == "__main__":
    monitor_training("train_model_light.py")
