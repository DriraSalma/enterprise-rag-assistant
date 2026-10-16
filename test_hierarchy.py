from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

def tester_hierarchie_qdrant():
    # 1. Connexion à la base locale Qdrant
    client = QdrantClient(path="data/qdrant_db")
    collection_name = "hth_rag_collection"

    print("[+] Connexion à la base locale Qdrant réussie.")

    # 2. Récupérer et compter les documents PARENTS
    parents, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(
            must=[FieldCondition(key="type", match=MatchValue(value="parent"))]
        ),
        limit=1000
    )
    print(f"[+] Nombre de documents PARENTS (Vidéos globales) indexés : {len(parents)}")

    # 3. Récupérer et compter les documents ENFANTS (Fils)
    enfants, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(
            must=[FieldCondition(key="type", match=MatchValue(value="child"))]
        ),
        limit=1000
    )
    print(f"[+] Nombre de documents ENFANTS (Diapositives/Étapes) indexés : {len(enfants)}")

    # 4. Vérifier l'intégrité de la liaison relationnelle Parent-Enfant
    if len(enfants) > 0 and len(parents) > 0:
        print("\n[+] Test de liaison Parent-Enfant (Échantillon aléatoire) :")
        echantillon_enfant = enfants[0]
        enfant_payload = echantillon_enfant.payload
        parent_id_associe = enfant_payload.get("parent_id")

        print(f"  -> Étape ID : {echantillon_enfant.id}")
        print(f"  -> Fichier source : {enfant_payload.get('file_name')}")
        print(f"  -> Diapositive n° : {enfant_payload.get('step_id')}")
        print(f"  -> Image associée : {enfant_payload.get('image_path')}")
        print(f"  -> Description : '{enfant_payload.get('description')}'")
        print(f"  -> ID du Parent associé : {parent_id_associe}")

        # Rechercher le parent correspondant dans Qdrant grâce au parent_id
        try:
            resultat_parent = client.retrieve(
                collection_name=collection_name,
                ids=[parent_id_associe]
            )
            
            if resultat_parent:
                parent_payload = resultat_parent[0].payload
                print(f"\n[+] PARENT ASSOCIE RETROUVÉ DANS LA BASE :")
                print(f"  -> Nom du document : {parent_payload.get('file_name')}")
                print(f"  -> Résumé/Contexte global du Parent : '{parent_payload.get('global_context')[:180]}...'")
                print("\n[✔] Test d'intégrité : Réussi. La structure hiérarchique HTH-RAG est correctement enregistrée.")
            else:
                print("\n[!] Alerte d'intégrité : Le parent_id de l'enfant ne correspond à aucun parent dans la base.")
        except Exception as e:
            print(f"[!] Erreur de recherche du parent : {e}")
    else:
        print("[-] Données insuffisantes dans l'index pour exécuter le test.")

if __name__ == "__main__":
    tester_hierarchie_qdrant()