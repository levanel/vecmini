import os
import vecmini
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import time

DATA_FOLDER = "./data"
CHUNK_SIZE = 1000 
#you can use any SentenceTransformer as long as vectors are normalized 
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2') 
documents = []

# scan the data folder for pdfs
for filename in os.listdir(DATA_FOLDER):
    if filename.endswith(".pdf"):	
        filepath = os.path.join(DATA_FOLDER, filename)
        print(f"{filename}")
        reader = PdfReader(filepath)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + " "
        words = full_text.split()
        for i in range(0, len(words), CHUNK_SIZE):
            chunk = " ".join(words[i:i+CHUNK_SIZE])
            if chunk.strip():
                documents.append(f"[{filename}] {chunk}")

print(f"total memory chunks created: {len(documents)}")
embeddings = model.encode(documents).astype(np.float32)

d = embeddings.shape[1]
nb = len(documents)
db = vecmini.IndexIVF(d, nbucket=1)
db.train(nb, embeddings)
db.add(nb, embeddings)
while True:
    print("\n" + "="*50)
    user_query = input("ask a question: ")
    
    if user_query.lower() in ['exit', 'quit']:
        break
        
    start_embed = time.time()
    query_vector = model.encode([user_query]).astype(np.float32)
    end_embed = time.time()
    
    start_search = time.time()
    distances, labels = db.search(1, query_vector, 10)
    end_search = time.time()
    
    print(f"\n{(end_embed - start_embed)*1000:.2f} ms / vecmini Search: {(end_search - start_search)*1000:.2f} ms]")
    print("Top retrieved chunks:\n")

    for i in range(10):
        idx = labels[i]
        if idx != -1:
            print(f"[{i+1}] Distance: {distances[i]:.2f}")
            print(f"{documents[idx]}\n")
