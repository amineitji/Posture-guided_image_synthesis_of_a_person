import sys
import os
import time

# =============================================================================
# CONFIGURATION DU CHEMIN PYTHON
# =============================================================================
# Les fichiers Python sont dans le dossier 'code/'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))

try:
    from VideoSkeleton import VideoSkeleton
    from DanceDemo import DanceDemo
    from GenVanillaNN import GenVanillaNN
    from GenGAN import GenGAN
    try:
        from GenNearest import GenNeirest
    except ImportError:
        GenNeirest = None
        
except ImportError as e:
    print("\n/!\\ ERREUR CRITIQUE D'IMPORT /!\\")
    print(f"Détail : {e}")
    print("Vérifiez que tous les fichiers Python sont bien dans le dossier 'code/'")
    print("Fichiers requis: code/VideoSkeleton.py, code/DanceDemo.py, etc.")
    print(f"\nChemin actuel : {os.getcwd()}")
    print(f"Dossier code existe : {os.path.exists('code')}")
    sys.exit(1)

# =============================================================================
# UTILITAIRES D'INTERFACE
# =============================================================================

def clear():
    """Nettoie l'écran (Windows/Linux/Mac)"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_title(title):
    print("\n" + "="*60)
    print(f" {title.center(58)}")
    print("="*60 + "\n")

def get_files(directory, extension):
    """Récupère la liste des fichiers avec une extension donnée"""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        return []
    return [f for f in os.listdir(directory) if f.endswith(extension)]

def select_file_menu(directory, extension, prompt_text):
    """Affiche un menu pour choisir un fichier"""
    files = get_files(directory, extension)
    if not files:
        print(f"   [!] Aucun fichier '{extension}' trouvé dans {directory}.")
        return None
    
    print(f"--- {prompt_text} ---")
    for i, f in enumerate(files):
        print(f"   {i+1}. {f}")
    
    while True:
        choice = input("\n> Votre choix (numéro) : ")
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                return os.path.join(directory, files[idx])
            print("   [!] Numéro invalide.")
        except ValueError:
            print("   [!] Veuillez entrer un nombre.")

# =============================================================================
# FONCTIONS PRINCIPALES (ETAPES)
# =============================================================================

def step_1_extraction():
    print_title("ETAPE 1 : EXTRACTION DES SQUELETTES (Data Prep)")
    print("Cette étape va :")
    print("1. Lire une vidéo brute (.mp4)")
    print("2. Détecter les squelettes frame par frame")
    print("3. Créer un dataset (.pkl) prêt pour l'entraînement")
    print("-" * 60)

    video_path = select_file_menu("data/raw", ".mp4", "Choisissez la vidéo à traiter")
    if not video_path: 
        print("\n[CONSEIL] Placez vos fichiers .mp4 dans le dossier 'data/raw/'")
        return

    print(f"\n[SELECTION] Vous avez choisi : {os.path.basename(video_path)}")
    mod_frame = input("> Garder 1 image sur X (défaut 5 pour qualité, 50 pour test rapide) : ") or "5"
    try:
        mod_frame = int(mod_frame)
    except:
        mod_frame = 5
        print(f"   [!] Valeur invalide, utilisation de {mod_frame} par défaut")
    
    print("\n" + "="*60)
    print("[INFO] Démarrage de l'extraction...")
    print("[INFO] Cela peut prendre plusieurs minutes selon la longueur de la vidéo")
    print("="*60)
    
    try:
        # ForceCompute=True est important pour écraser les données potentiellement corrompues
        start_time = time.time()
        vs = VideoSkeleton(video_path, forceCompute=True, modFrame=mod_frame)
        elapsed = time.time() - start_time
        
        print("\n" + "="*60)
        print("[✓ SUCCES] Extraction terminée!")
        print(f"   - Squelettes extraits : {vs.skeCount()}")
        print(f"   - Temps écoulé : {elapsed:.1f} secondes")
        print(f"   - Dataset sauvegardé dans 'data/processed/'")
        print("="*60)
    except Exception as e:
        print("\n" + "="*60)
        print(f"[✗ ERREUR] L'extraction a échoué : {e}")
        print("="*60)
        import traceback
        traceback.print_exc()

def step_2_training():
    print_title("ETAPE 2 : ENTRAÎNEMENT DU MODÈLE (Training)")
    print("Cette étape va :")
    print("1. Charger un dataset préparé (.pkl)")
    print("2. Entraîner un réseau de neurones à générer le personnage")
    print("-" * 60)

    pkl_path = select_file_menu("data/processed", ".pkl", "Choisissez le dataset cible (Personnage)")
    if not pkl_path: 
        print("\n[CONSEIL] Faites d'abord l'étape 1 pour créer un dataset !")
        return

    print("\n--- Choix du Modèle ---")
    print("   1. Vanilla NN (Vecteur -> Image) : Rapide, résultats corrects")
    print("   2. Vanilla NN (Image -> Image)   : Plus lent, meilleurs détails")
    print("   3. GAN (Generative Adversarial)  : Le plus long, meilleur réalisme")
    
    model_choice = input("\n> Votre choix (1-3) : ")
    epochs = input("> Nombre d'epochs (défaut 50, recommandé: 100-200) : ") or "50"
    try:
        epochs = int(epochs)
    except:
        epochs = 50
        print(f"   [!] Valeur invalide, utilisation de {epochs} epochs")

    print("\n" + "="*60)
    print(f"[INFO] Chargement du dataset: {os.path.basename(pkl_path)}")
    print("="*60)
    
    try:
        vs = VideoSkeleton(pkl_path, forceCompute=False)
        if vs.skeCount() == 0:
            print("\n[✗ ERREUR] Ce dataset est vide !")
            print("[CONSEIL] Refaites l'étape 1 sur cette vidéo avec un modFrame plus faible")
            return
        print(f"[✓] Dataset chargé : {vs.skeCount()} squelettes disponibles")
    except Exception as e:
        print(f"\n[✗ ERREUR] Impossible de charger le dataset : {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "="*60)
    print(f"[INFO] Démarrage de l'entraînement ({epochs} epochs)")
    print("[INFO] Cela va prendre du temps... Soyez patient !")
    print("="*60 + "\n")
    
    try:
        start_time = time.time()
        
        if model_choice == '1':
            print("[MODELE] Vanilla NN - Mode Vecteur vers Image")
            gen = GenVanillaNN(vs, loadFromFile=False, optSkeOrImage=1)
            gen.train(n_epochs=epochs)
        elif model_choice == '2':
            print("[MODELE] Vanilla NN - Mode Image vers Image")
            gen = GenVanillaNN(vs, loadFromFile=False, optSkeOrImage=2)
            gen.train(n_epochs=epochs)
        elif model_choice == '3':
            print("[MODELE] GAN - Generative Adversarial Network")
            gen = GenGAN(vs, loadFromFile=False)
            gen.train(n_epochs=epochs)
        else:
            print("\n[!] Choix invalide. Retour au menu.")
            return
        
        elapsed = time.time() - start_time
        print("\n" + "="*60)
        print("[✓ SUCCES] Entraînement terminé!")
        print(f"   - Temps total : {elapsed/60:.1f} minutes")
        print(f"   - Modèle sauvegardé dans 'models/'")
        print("="*60)
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"[✗ ERREUR] L'entraînement a planté : {e}")
        print("="*60)
        import traceback
        traceback.print_exc()

def step_3_demo():
    print_title("ETAPE 3 : DÉMONSTRATION (Inference)")
    print("Cette étape va :")
    print("1. Prendre le mouvement d'une vidéo SOURCE")
    print("2. L'appliquer sur le personnage CIBLE (appris à l'étape 2)")
    print("-" * 60)

    print("\n---> 1. LA SOURCE DU MOUVEMENT")
    src_path = select_file_menu("data/raw", ".mp4", "Choisissez la vidéo Source (pour le mouvement)")
    if not src_path: 
        print("\n[CONSEIL] Placez une vidéo .mp4 dans 'data/raw/'")
        return

    print("\n---> 2. LE PERSONNAGE CIBLE")
    tgt_path = select_file_menu("data/processed", ".pkl", "Choisissez le dataset Cible (personnage entraîné)")
    if not tgt_path: 
        print("\n[CONSEIL] Faites d'abord l'étape 1 et 2 sur une vidéo !")
        return

    print("\n---> 3. LE GÉNÉRATEUR")
    print("   1. Nearest Neighbor (Pas d'IA, colle l'image la plus proche)")
    print("   2. Vanilla NN (Vecteur -> Image)")
    print("   3. Vanilla NN (Image -> Image)")
    print("   4. GAN")
    gen_choice = input("\n> Quel modèle utiliser ? (celui entraîné à l'étape 2) : ")
    try:
        gen_type = int(gen_choice)
    except:
        gen_type = 1
        print(f"   [!] Valeur invalide, utilisation de Nearest Neighbor")

    print("\n" + "="*60)
    print("[INFO] Configuration de la démo :")
    print(f"   - Mouvement source : {os.path.basename(src_path)}")
    print(f"   - Personnage cible : {os.path.basename(tgt_path)}")
    print(f"   - Générateur : Type {gen_type}")
    print("="*60)
    print("\n[DEMO] Lancement...")
    print("       Appuyez sur 'q' dans la fenêtre vidéo pour quitter.")
    print("-" * 60)
    
    try:
        # On passe explicitement le fichier cible pour éviter que DanceDemo ne charge un défaut
        dd = DanceDemo(src_path, gen_type, filename_tgt=tgt_path)
        dd.draw(skip_frames=8, wait_ms=1)
        print("\n[INFO] Démo terminée.")
    except Exception as e:
        print("\n" + "="*60)
        print(f"[✗ ERREUR] La démo a planté : {e}")
        print("="*60)
        print("\n[DIAGNOSTIC]")
        print(" - Si 'index out of bounds' → le dataset cible est vide")
        print(" - Si 'model not found' → il faut entraîner le modèle (étape 2)")
        print(" - Vérifiez que vous avez bien fait l'étape 1 ET 2 pour ce personnage")
        print("-" * 60)
        import traceback
        traceback.print_exc()

# =============================================================================
# MENU PRINCIPAL
# =============================================================================

def main():
    while True:
        print_title("PROJET DANCE GENERATION")
        print(" 1. [PREPARATION]   Extraire les squelettes d'une vidéo")
        print(" 2. [APPRENTISSAGE] Entraîner un réseau de neurones")
        print(" 3. [GENERATION]    Générer une vidéo de danse (Transfert)")
        print(" 4. Quitter")
        
        choice = input("\n> Que voulez-vous faire ? ")
        
        if choice == '1':
            step_1_extraction()
        elif choice == '2':
            step_2_training()
        elif choice == '3':
            step_3_demo()
        elif choice == '4':
            print("\n" + "="*60)
            print("Au revoir ! Bon développement 🚀")
            print("="*60)
            break
        else:
            print("\n[!] Option inconnue. Choisissez entre 1 et 4.")
        
        input("\n>>> Appuyez sur Entrée pour revenir au menu principal...")
        clear()

if __name__ == '__main__':
    # Vérification basique de la structure
    if not os.path.exists("data/raw"):
        os.makedirs("data/raw")
        print("[Setup] Dossier 'data/raw' créé. Mettez vos vidéos dedans !")
    if not os.path.exists("data/processed"):
        os.makedirs("data/processed")
    if not os.path.exists("models"):
        os.makedirs("models")
        print("[Setup] Dossier 'models' créé pour sauvegarder les modèles.")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Programme interrompu par l'utilisateur (Ctrl+C)")
        print("Au revoir !")
        sys.exit(0)