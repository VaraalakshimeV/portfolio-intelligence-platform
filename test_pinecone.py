from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()

print("Testing Pinecone connection...")

try:
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    
    indexes = pc.list_indexes()
    print(f"✓ Connected to Pinecone!")
    print(f"✓ Indexes: {[idx.name for idx in indexes]}")
    
    index = pc.Index('fintech-rag')
    stats = index.describe_index_stats()
    print(f"✓ Index ready!")
    print(f"  Vectors: {stats['total_vector_count']}")
    
except Exception as e:
    print(f"✗ Error: {e}")