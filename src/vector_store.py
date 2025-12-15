from qdrant_client import QdrantClient

qdrant_client = QdrantClient(
    url="https://a5a73dea-bb23-4c36-9113-56e407f4d1cb.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.0i2x2ZcuT9JNS6-ERm4xQeTCU52irdpwKFtGiMebsXA",
)

try:
    health = qdrant_client.get_collections()
    print("✅ Qdrant работает!")
except:
    print("❌ Qdrant не работает")
