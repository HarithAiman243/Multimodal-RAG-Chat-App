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
            # Be more specific about potential URL errors
            if "HTTP status code 400" in str(e) or "invalid image format" in str(e).lower():
                return "Image analysis failed: Invalid or inaccessible image URL."
            return f"Image analysis failed: {e}"
        
    def safe_get(data_dict, key_path, default="N/A"):
        """Safely gets a nested key from a dictionary."""
        keys = key_path.split('.')
        val = data_dict
        try:
            for key in keys:
                val = val[key]
            return val if val else default
        except (KeyError, TypeError, IndexError):
            return default

    def create_text_chunk(ad: dict) -> str:
        """Transforms a single ad's data into a descriptive text chunk for RAG, using richer data."""
        creative = ad.get('creative', {})
        campaign_name = safe_get(ad, 'campaign.name')
        adset_name = safe_get(ad, 'adset.name')
        
        # --- Extract Creative Details with Fallbacks ---
        title = creative.get('title') or safe_get(creative, 'object_story_spec.link_data.name') or safe_get(creative, 'object_story_spec.video_data.title', 'N/A')
        body = creative.get('body') or safe_get(creative, 'object_story_spec.link_data.message') or safe_get(creative, 'object_story_spec.text_data.message', 'N/A')
        description = safe_get(creative, 'object_story_spec.link_data.description', 'N/A')
        link = safe_get(creative, 'object_story_spec.link_data.link', 'N/A')
        cta_type = safe_get(creative, 'object_story_spec.video_data.call_to_action.type', 'N/A')
        
        # Determine media type
        media_type = "Unknown"
        primary_image_url = creative.get('image_url') # Use the primary image for captioning
        
        if safe_get(creative, 'asset_feed_spec.videos', []):
            media_type = "Video"
            # Try to get a video thumbnail if primary image is missing
            if not primary_image_url:
                 primary_image_url = safe_get(creative, 'object_story_spec.video_data.image_url') or creative.get('thumbnail_url')
        elif safe_get(creative, 'asset_feed_spec.images', []):
             media_type = "Image Feed"
        elif primary_image_url:
             media_type = "Image"
        elif safe_get(creative, 'object_story_spec.link_data.child_attachments', []):
            media_type = "Carousel"
            # Try to get first carousel image if primary is missing
            if not primary_image_url:
                first_child = safe_get(creative, 'object_story_spec.link_data.child_attachments.0', {})
                # Need logic here to potentially fetch image URL from image_hash if needed, complex.
                # Sticking to thumbnail_url as simpler fallback for now.
                primary_image_url = creative.get('thumbnail_url') # Fallback for carousel preview

        # Get multimodal description using the best available image URL
        visual_description = generate_image_caption(primary_image_url)
        # Get the URL we actually want to embed for preview (might be different from caption source)
        image_url_to_save = creative.get('image_url') or creative.get('thumbnail_url', 'N/A')

        # Assemble the final text chunk
        chunk_lines = [
            f"Ad Report for '{safe_get(ad, 'name')}' (ID: {safe_get(ad, 'id')}):",
            f"- Status: {safe_get(ad, 'status')}",
            f"- Hierarchy: Campaign '{campaign_name}' > Ad Set '{adset_name}'.",
            f"- Media Type: {media_type}",
            f"- Ad Creative Text:",
            f"  - Title: {title}",
            f"  - Body: {body}",
            f"  - Description (Link): {description}",
            f"  - Destination Link: {link}",
            f"  - Call To Action: {cta_type}",
            f"- Creative Visual Analysis: {visual_description}",
            # Embed the preview URL tag
            f"##IMAGE_PREVIEW_URL##{image_url_to_save}##"
        ]
        chunk = "\n".join(chunk_lines)
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
        try:
            text_chunk = create_text_chunk(ad)
            vector = embeddings.embed_query(text_chunk)
            vectors_to_upsert.append({
                "id": ad['id'],
                "values": vector,
                "metadata": {"text": text_chunk}
            })
        except Exception as e:
            print(f"    ⚠️ Error processing ad {ad.get('id', 'N/A')}: {e}")
            # Optionally skip this ad or handle the error differently
            continue

    # 4. Upsert to Pinecone in batches
    if vectors_to_upsert:
        print("\nUpserting vectors to Pinecone...")
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            try:
                index.upsert(vectors=batch)
                print(f"  - Upserted batch {i//batch_size + 1}")
            except Exception as e:
                 print(f"    ⚠️ Error upserting batch {i//batch_size + 1}: {e}")
        
        try:
            stats = index.describe_index_stats()
            print(f"\nSuccess: Ingestion complete. Pinecone index '{PINECONE_INDEX_NAME}' now contains {stats['total_vector_count']} vectors.")
        except Exception as e:
            print(f"\nIngestion finished, but failed to get final index stats: {e}")
            print("Please check the Pinecone console for the vector count.")
            
    else:
        print("No vectors were generated to upsert.")

if __name__ == "__main__":
    ingest_data_script()