import ollama
import psycopg2
#from psycopg2.extras import register_vector
import numpy as np
import sys

# --- Configuration ---
OLLAMA_MODEL = "nomic-embed-text:v1.5"  # Or your chosen Ollama embedding model
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "rag_db"
DB_USER = "rag_user"
DB_PASSWORD = "rag_password"
TABLE_NAME = "public.data_nbs_doc_md_contextual_rag"
EMBEDDING_DIM = 768  # Dimension of embeddings from nomic-embed-text

# Construct DATABASE_URL
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# --- Database Connection ---
def get_db_connection():
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    #register_vector(conn)  # Enable pgvector support for psycopg2
    return conn

# --- Embedding Generation ---
def generate_embedding(text):
    response = ollama.embeddings(model=OLLAMA_MODEL, prompt=text)
    return response['embedding']

# --- Database Operations ---
def setup_database(conn):
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                id SERIAL PRIMARY KEY,
                content TEXT,
                embedding VECTOR({EMBEDDING_DIM})
            );
        """)
        conn.commit()

def insert_document(conn, content):
    embedding = generate_embedding(content)
    with conn.cursor() as cur:
        cur.execute(f"""
            INSERT INTO {TABLE_NAME} (content, embedding)
            VALUES (%s, %s);
        """, (content, np.array(embedding)))
        conn.commit()

def find_similar_documents(conn, query_text, limit=5):
    query_embedding = generate_embedding(query_text)
    with conn.cursor() as cur:

        cur.execute(f"""
            SELECT id, text
            FROM {TABLE_NAME}
            ORDER BY embedding <-> %s
            LIMIT %s;
        """, (str(query_embedding), limit))

        results = cur.fetchall()
    return [(row[0],row[1]) for row in results]

# --- Main Execution ---
if __name__ == "__main__":
    conn = None
    try:

		# Check if a query was passed as a command-line argument
        if len(sys.argv) > 1:
            user_query = " ".join(sys.argv[1:])
        else:
            # Prompt the user for a query if none was provided
            user_query = input("Please enter your text for PGVector Similarity Search :")

        conn = get_db_connection()
        
        #setup_database(conn)

        # Insert some example documents
        #print("Inserting documents...")
        #insert_document(conn, "The quick brown fox jumps over the lazy dog.")
        #insert_document(conn, "Artificial intelligence is transforming industries.")
        #insert_document(conn, "Machine learning algorithms are at the core of AI.")
        #insert_document(conn, "Dogs are known for their loyalty and companionship.")
        #print("Documents inserted.")

        # Perform similarity search
        #search_query = "Tell me about AI and ML."

        #search_query = "Who is Ali Rashed Al-Ketabi"
        #search_query = "Ali Rashed Al-Ketabi"

        #print(f"\nFinding documents similar to: '{search_query}'")
        print(f"\nFinding documents similar to: '{user_query}'")

        #similar_docs = find_similar_documents(conn, search_query)
        similar_docs = find_similar_documents(conn, user_query)

        print("\nSimilar Documents:")
        for doc in similar_docs:
            print(f"- {doc}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if conn:
            conn.close()
