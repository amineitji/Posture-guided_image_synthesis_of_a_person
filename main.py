import sys
import os
import time
import glob

# =============================================================================
# CONFIGURATION
# =============================================================================
# On s'assure que Python trouve les fichiers dans le dossier 'code'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))

try:
    from VideoSkeleton import VideoSkeleton
    from DanceDemo import DanceDemo
    from GenVanillaNN import GenVanillaNN
    from GenGAN import GenGAN
    # Optionnel
    try:
        from GenNearest import GenNeirest
    except ImportError:
        GenNeirest = None
        
except ImportError as e:
    print("\n" + "!"*60)
    print(" ERREUR CRITIQUE : Fichiers manquants")
    print(f" Détail : {e}")
    print(" Vérifiez que le dossier 'code/' contient bien tous les scripts.")
    print("!"*60)
    sys.exit(1)

# =============================================================================
# OUTILS D'INTERFACE
# =============================================================================

def clear():
    """Nettoie la console"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(title, subtitle=None):
    """Affiche un joli header"""
    print("\n" + "═"*70)
    print(f" {title.center(68)}")
    if subtitle:
        print(f" {subtitle.center(68)}")
    print("═"*70 + "\n")

def print_info(msg):
    print(f" [INFO] {msg}")

def print_success(msg):
    print(f" [OK]   {msg}")

def print_warn(msg):
    print(f" [!]    {msg}")

def print_error(msg):
    print(f" [X]    ERREUR: {msg}")

def wait_enter():
    input("\nAppuyez sur [Entrée] pour continuer...")

def get_file_selection(directory, extension, description):
    """Menu générique pour choisir un fichier"""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print_warn(f"Le dossier '{directory}' n'existait pas, je viens de le créer.")
        return None

    files = glob.glob(os.path.join(directory, f"*{extension}"))
    # Tri par date de modification (plus récent en premier)
    files.sort(key=os.path.getmtime, reverse=True)

    if not files:
        print_warn(f"Aucun fichier '{extension}' trouvé dans '{directory}'.")
        return None

    print(f"--- Sélectionnez {description} ---")
    for i, f in enumerate(files):
        filename = os.path.basename(f)
        size = os.path.getsize(f) / (1024*1024)
        print(f"   {i+1}. {filename:<30} ({size:.1f} MB)")
    
    print("   0. Retour")

    while True:
        try:
            choice = input("\n> Votre choix : ")
            if choice == '0': return None
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                return files[idx]
        except ValueError:
            pass
        print("   > Choix invalide.")

# =============================================================================
# ETAPE 1 : EXTRACTION
# =============================================================================
def step_extract():
    print_header("ETAPE 1 : EXTRACTION", "Vidéo MP4 -> Dataset PKL")
    
    print("Cette étape transforme une vidéo en une liste de squelettes.")
    print("Placez vos vidéos dans le dossier 'data/raw/'.\n")

    video_path = get_file_selection("data/raw", ".mp4", "une vidéo source")
    if not video_path: return

    print("\n[CONFIGURATION]")
    print("modFrame : Garder 1 image sur X.")
    print("   * 1 = Qualité maximale (Lourd, extraction lente)")
    print("   * 3 = Bon compromis (Recommandé)")
    print("   * 10 = Rapide (Pour tester)")
    
    mod = input("> modFrame (défaut 3) : ") or "3"
    
    try:
        mod = int(mod)
        print_header("EXTRACTION EN COURS...")
        print_info(f"Traitement de : {os.path.basename(video_path)}")
        print_info(f"Mode : 1 frame sur {5}")
        print_info("Cela peut prendre quelques minutes selon la durée de la vidéo...")
        
        start_time = time.time()
        
        # Lancement du calcul
        vs = VideoSkeleton(video_path, forceCompute=True, modFrame=5,cropRatio=1.3,newVideoWidth=128)
        
        duration = time.time() - start_time
        
        print_success(f"Terminé en {duration:.1f} secondes.")
        print_success(f"Squelettes extraits : {vs.skeCount()}")
        print_success(f"Fichier sauvegardé dans 'data/processed/'")
        
    except Exception as e:
        print_error(str(e))
        import traceback
        traceback.print_exc()
    
    wait_enter()

# =============================================================================
# ETAPE 2 : ENTRAINEMENT
# =============================================================================
def step_train():
    print_header("ETAPE 2 : ENTRAINEMENT", "Apprendre à générer le personnage")

    pkl_path = get_file_selection("data/processed", ".pkl", "le dataset du personnage")
    if not pkl_path: return

    print("\n[CHOIX DU MODELE]")
    print(" 1. Vanilla NN (Vecteur) : Rapide, mais résultats flous/abstraits.")
    print(" 2. Vanilla NN (U-Net)   : Meilleur que le vecteur, traite les images.")
    print(" 3. WGAN-GP (Pix2Pix)    : [RECOMMANDÉ] Le plus réaliste et net.")
    
    type_model = input("\n> Modèle (1-3) : ")
    epochs = input("> Nombre d'epochs (défaut 200, recomm. 500+) : ") or "200"
    
    try:
        n_epochs = int(epochs)
        print_header("DÉMARRAGE DE L'ENTRAINEMENT")
        print_info(f"Chargement du dataset : {os.path.basename(pkl_path)}")
        
        # Chargement dataset
        vs = VideoSkeleton(pkl_path)
        print_info(f"{vs.skeCount()} images disponibles pour l'entraînement.")

        start_time = time.time()
        gen = None

        if type_model == '1':
            print_info("Initialisation : Vanilla NN (Vector -> Image)")
            gen = GenVanillaNN(vs, loadFromFile=False, optSkeOrImage=1)
        elif type_model == '2':
            print_info("Initialisation : Vanilla NN (Image -> Image)")
            gen = GenVanillaNN(vs, loadFromFile=False, optSkeOrImage=2)
        elif type_model == '3':
            print_info("Initialisation : WGAN-GP (Architecture GAN Stable)")
            gen = GenGAN(vs, loadFromFile=False)
        else:
            print_warn("Choix invalide.")
            return

        print_info(f"Lancement pour {n_epochs} epochs. Patientez...")
        
        # Lancement Train
        gen.train(n_epochs=n_epochs)
        
        duration = (time.time() - start_time) / 60
        print_success(f"Entraînement terminé en {duration:.1f} minutes.")
        print_success("Le modèle a été sauvegardé dans le dossier 'models/'.")
        
    except Exception as e:
        print_error(str(e))
    
    wait_enter()

# =============================================================================
# ETAPE 3 : DEMO
# =============================================================================
def step_demo():
    print_header("ETAPE 3 : GENERATION (DEMO)", "Transfert de mouvement")

    print("1. Choisissez la vidéo qui donne le MOUVEMENT (Source)")
    src_path = get_file_selection("data/raw", ".mp4", "Vidéo Source")
    if not src_path: return

    print("\n2. Choisissez le PERSONNAGE entraîné (Target)")
    tgt_path = get_file_selection("data/processed", ".pkl", "Dataset Cible")
    if not tgt_path: return

    print("\n[TYPE DE GENERATEUR]")
    print("Attention : Choisissez le même type que celui utilisé lors de l'entraînement !")
    print(" 1. Nearest Neighbor (Pas d'IA, copie l'image la plus proche)")
    print(" 2. Vanilla NN (Vecteur)")
    print(" 3. Vanilla NN (U-Net)")
    print(" 4. WGAN-GP (Si vous avez entraîné l'option 3)")
    
    gen_type = input("\n> Choix (1-4, défaut 4) : ") or "4"

    try:
        print_header("LANCEMENT DE LA DEMO")
        print_info("Chargement des modèles...")
        print_info("Commandes : Appuyez sur 'q' pour quitter la fenêtre vidéo.")
        
        dd = DanceDemo(src_path, int(gen_type), filename_tgt=tgt_path)
        dd.draw()
        
        print_success("Démo terminée correctement.")

    except Exception as e:
        print_error(str(e))
        import traceback
        traceback.print_exc()
        print("\n[ASTUCE] Si l'erreur est 'Size Mismatch' ou 'KeyError', vous n'utilisez pas le bon générateur pour ce modèle.")
    
    wait_enter()

# =============================================================================
# MENU PRINCIPAL
# =============================================================================
def main():
    while True:
        clear()
        print_header("DEEP DANCE TRANSFER", "Projet Apprentissage & Image")
        print(" 1. [PREPARATION]   Extraire les squelettes d'une vidéo")
        print(" 2. [APPRENTISSAGE] Entraîner le réseau de neurones")
        print(" 3. [GENERATION]    Lancer la démo de transfert")
        print(" 4. Quitter")
        
        choice = input("\n> Votre choix : ")
        
        if choice == '1':
            step_extract()
        elif choice == '2':
            step_train()
        elif choice == '3':
            step_demo()
        elif choice == '4':
            print("\nAu revoir !")
            sys.exit(0)
        else:
            print_warn("Option inconnue.")
            time.sleep(1)

if __name__ == "__main__":
    # Vérification des dossiers
    for d in ["data/raw", "data/processed", "models"]:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"[INIT] Dossier créé : {d}")
    
    main()