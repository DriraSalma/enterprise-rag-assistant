import os
import json
from pathlib import Path
#call venv\Scripts\activate
#uvicorn api:app --reload
def nettoyer_fichiers(dossier_source, dossier_destination):
    chemin_source = Path(dossier_source)
    chemin_destination = Path(dossier_destination)
    
    # Créer le dossier de destination s'il n'existe pas
    chemin_destination.mkdir(parents=True, exist_ok=True)
    
    # Lister tous les fichiers .json et .txt du dossier raw
    fichiers = [f for f in chemin_source.iterdir() if f.suffix in ['.json', '.txt']]
    
    if not fichiers:
        print(f"[-] Aucun fichier .json ou .txt trouvé dans '{dossier_source}'.")
        print("Veuillez y déposer quelques fichiers originaux pour essayer.")
        return
    
    print(f"[+] Début du nettoyage de {len(fichiers)} fichiers...")
    fichiers_nettoyes = 0
    
    for chemin_fichier in fichiers:
        try:
            # 1. Lire le fichier d'origine (encodage UTF-8 recommandé)
            with open(chemin_fichier, 'r', encoding='utf-8') as f:
                donnees = json.load(f)
            
            # 2. Supprimer la clé 'imageUrl' selon la structure du JSON
            if isinstance(donnees, list):
                # Si le fichier contient une liste d'objets (comme votre exemple)
                for element in donnees:
                    if isinstance(element, dict) and 'imageUrl' in element:
                        del element['imageUrl']
            elif isinstance(donnees, dict):
                # Si le fichier ne contient qu'un seul objet
                if 'imageUrl' in donnees:
                    del donnees['imageUrl']
            
            # 3. Enregistrer le fichier nettoyé dans le dossier de destination
            chemin_sortie = chemin_destination / chemin_fichier.name
            with open(chemin_sortie, 'w', encoding='utf-8') as f:
                # ensure_ascii=False permet de conserver les accents français correctement
                json.dump(donnees, f, ensure_ascii=False, indent=2)
            
            fichiers_nettoyes += 1
            
        except json.JSONDecodeError:
            print(f"[!] Erreur de format JSON dans le fichier : {chemin_fichier.name}")
        except Exception as e:
            print(f"[!] Impossible de traiter le fichier {chemin_fichier.name}. Erreur : {e}")
            
    print(f"\n[+] Nettoyage terminé !")
    print(f"[+] {fichiers_nettoyes}/{len(fichiers)} fichiers ont été nettoyés avec succès.")
    print(f"[+] Les fichiers propres sont disponibles dans : '{dossier_destination}'")

if __name__ == "__main__":
    # Définition des dossiers source et destination
    DOSSIER_RAW = "data/raw"
    DOSSIER_CLEANED = "data/cleaned"
    
    nettoyer_fichiers(DOSSIER_RAW, DOSSIER_CLEANED)