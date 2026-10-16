from qdrant_client import QdrantClient

try:
    # Tenter de se connecter au serveur Qdrant local
    client = QdrantClient(url="http://localhost:6333")
    
    # Interroger les collections existantes pour valider l'accès
    collections = client.get_collections()
    
    print("[✔] Connexion réussie ! Python arrive à dialoguer avec le serveur Qdrant.")
    print(f"[+] Collections actuellement détectées dans la base : {collections}")
    
except Exception as e:
    print(f"[!] Échec de la connexion. Le serveur Qdrant n'est pas actif ou inaccessible. Erreur : {e}")