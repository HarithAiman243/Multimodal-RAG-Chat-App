import os
import json
import yaml
from dotenv import load_dotenv

import openai
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings

def ingest_data_script():
    """
    Main function for the simplified data ingestion script.
    """
    load_dotenv()

    # --- Configuration Loading ---
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    INPUT_FILE = os.path.join("data", "dataset.json")
    PINECONE_INDEX_NAME = config['pinecone']['index_name']

    # --- Client Initialization ---
    try:
        openai_client = openai.OpenAI()
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
    except Exception as e:
        print(f"Error initializing clients: {e}")
        return

    # --- Helper Functions ---
    def generate_image_caption(image_url: str) -> str:
        """Uses GPT-4o to generate a text description of an ad creative."""
        if not image_url:
            return "No visual creative provided."
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this ad creative in detail for a marketing analysis. What is the subject, setting, color palette, and overall mood? What is the likely target audience?"},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                max_tokens=150,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Image analysis failed: {e}"

    def create_text_chunk(ad: dict) -> str:
        """Transforms a single ad's data into a descriptive text chunk for RAG."""
        creative = ad.get('creative', {})
        campaign_name = ad.get('campaign', {}).get('name', 'N/A')
        
        # Get multimodal description
        visual_description = generate_image_caption(creative.get('image_url'))

        # Assemble the final text chunk without GSheet data
        chunk = (
            f"Ad Report for '{ad.get('name', 'N/A')}' (ID: {ad.get('id')}):\n"
            f"- Status: {ad.get('status', 'N/A')}\n"
            f"- Hierarchy: Campaign '{campaign_name}' > Ad Set '{ad.get('adset', {}).get('name', 'N/A')}'.\n"
            f"- Ad Creative Text:\n"
            f"  - Title: {creative.get('title', 'N/A')}\n"
            f"  - Body: {creative.get('body', 'N/A')}\n"
            f"- Creative Visual Analysis: {visual_description}"
        )
        return chunk

    # --- Main Ingestion Logic ---
    print("Starting simplified data ingestion process...")
    
    # 1. Load source data
    try:
        with open(INPUT_FILE, 'r') as f:
            ads_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found. Please run get_data.py first.")
        return

    # 2. Connect to or create Pinecone Index
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")
        pc.create_index(name=PINECONE_INDEX_NAME, dimension=1536, metric='cosine',
                        spec=ServerlessSpec(
                            cloud='aws',
                            region='us-east-1'
                        ))
    index = pc.Index(PINECONE_INDEX_NAME)

    # 3. Transform and Prepare Vectors
    vectors_to_upsert = []
    print(f"\nProcessing {len(ads_data)} ads. This may take some time due to image captioning...")
    for i, ad in enumerate(ads_data):
        print(f"  - Processing ad {i+1}/{len(ads_data)}: {ad.get('name', 'N/A')}")
        text_chunk = create_text_chunk(ad)
        vector = embeddings.embed_query(text_chunk)
        vectors_to_upsert.append({
            "id": ad['id'],
            "values": vector,
            "metadata": {"text": text_chunk}
        })

    # 4. Upsert to Pinecone in batches
    if vectors_to_upsert:
        print("\nUpserting vectors to Pinecone...")
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            index.upsert(vectors=batch)
            print(f"  - Upserted batch {i//batch_size + 1}")
        
        stats = index.describe_index_stats()
        print(f"\nSuccess: Ingestion complete. Pinecone index '{PINECONE_INDEX_NAME}' now contains {stats['total_vector_count']} vectors.")
    else:
        print("No vectors were generated to upsert.")

if __name__ == "__main__":
    ingest_data_script()