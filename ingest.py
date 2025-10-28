import os
import json
import time
import boto3
import yaml
from io import BytesIO
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

# Load .env
load_dotenv()

# ============ CONFIG SETUP ============
CONFIG_PATH = "config/config.yaml"

def load_config():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Missing config file at {CONFIG_PATH}")
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

config = load_config()
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", config["pinecone"]["index_name"])
EMBEDDING_MODEL = config["embedding_model"]["model_name"]

# ============ CLIENT SETUP ============
print("Initializing OpenAI, Pinecone, and AWS clients...")

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)

bucket_name = os.getenv("AWS_S3_BUCKET_NAME")
s3_key = os.getenv("S3_KEY", "data/dataset.json")
local_data_path = "data/dataset.json"

# ============ LOAD DATA ============
print("\nStep 1: Loading dataset...")

def load_dataset():
    # Try to load from S3, fallback to local
    try:
        print(f"Attempting to load from S3: {bucket_name}/{s3_key}")
        response = s3.get_object(Bucket=bucket_name, Key=s3_key)
        data = json.load(response["Body"])
        print(f"✅ Loaded dataset from S3 ({len(data)} records)")
    except Exception as e:
        print(f"⚠️ S3 load failed ({e}). Falling back to local file.")
        if not os.path.exists(local_data_path):
            raise FileNotFoundError("No dataset found in S3 or local path.")
        with open(local_data_path, "r") as f:
            data = json.load(f)
        print(f"✅ Loaded dataset locally ({len(data)} records)")
    return data

dataset = load_dataset()

# ============ ENSURE INDEX EXISTS ============
print("\nStep 2: Verifying Pinecone index...")

if INDEX_NAME not in pinecone_client.list_indexes().names():
    print(f"⚠️ Index '{INDEX_NAME}' not found. Creating new index...")
    pinecone_client.create_index(
        name=INDEX_NAME,
        dimension=1536,  # text-embedding-3-small dimension
        metric="cosine"
    )
else:
    print(f"✅ Pinecone index '{INDEX_NAME}' found.")

index = pinecone_client.Index(INDEX_NAME)

# ============ EMBEDDING FUNCTION ============
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

# ============ IMAGE CAPTIONING (GPT-4o) ============
def generate_caption(image_url):
    """
    Uses GPT-4o to generate a short caption or description of an image.
    Returns a concise textual summary to enrich the RAG context.
    """
    try:
        prompt = f"Describe this marketing image in one short sentence (focus on emotion, color, and message): {image_url}"
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a creative advertising assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=50
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ Failed to generate caption for {image_url}: {e}")
        return ""

# ============ BUILD DOCUMENTS ============
print("\nStep 3: Preparing documents for embedding...")

documents = []
for ad in tqdm(dataset, desc="Processing Ads"):
    ad_text = (
        f"Ad Name: {ad.get('name', '')}\n"
        f"Primary Text: {ad.get('primary_text', '')}\n"
        f"Headline: {ad.get('headline', '')}\n"
        f"Description: {ad.get('description', '')}\n"
        f"CTA: {ad.get('call_to_action_type', '')}\n"
    )

    # Optional image caption
    image_caption = ""
    if ad.get("image_url"):
        image_caption = generate_caption(ad["image_url"])

    combined_text = f"{ad_text}\nImage Summary: {image_caption}"

    documents.append({
        "id": ad.get("id", str(time.time())),
        "text": combined_text,
        "metadata": {
            "ad_id": ad.get("id", ""),
            "ad_name": ad.get("name", ""),
            "image_url": ad.get("image_url", ""),
            "video_url": ad.get("video_url", ""),
            "caption": image_caption
        }
    })

print(f"✅ Prepared {len(documents)} documents for embedding.")

# ============ EMBEDDING + UPSERT ============
print("\nStep 4: Generating embeddings and upserting to Pinecone...")

batch_size = 50
for i in range(0, len(documents), batch_size):
    batch = documents[i:i + batch_size]
    texts = [doc["text"] for doc in batch]
    metadatas = [doc["metadata"] for doc in batch]
    ids = [doc["id"] for doc in batch]

    try:
        vectors = embeddings.embed_documents(texts)
        index.upsert(vectors=zip(ids, vectors, metadatas))
    except Exception as e:
        print(f"⚠️ Batch {i // batch_size + 1} failed: {e}")

print("✅ Ingestion complete.")
print(f"💾 Pinecone index '{INDEX_NAME}' now contains {len(documents)} vectors.")
