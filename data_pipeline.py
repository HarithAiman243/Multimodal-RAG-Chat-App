import os
import json
import boto3
from datetime import datetime
from dotenv import load_dotenv

# Meta/Facebook specific imports
from facebook_business.api import FacebookAdsApi
from facebook_business.adobjects.adaccount import AdAccount

# Google Sheets specific imports
import gspread
from google.oauth2.service_account import Credentials

# Pinecone and LangChain imports
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

# --- CONFIGURATION ---
load_dotenv()

# Meta/Facebook API Config
FB_APP_ID = os.environ.get('FB_APP_ID')
FB_APP_SECRET = os.environ.get('FB_APP_SECRET')
FB_ACCESS_TOKEN = os.environ.get('FB_ACCESS_TOKEN')
AD_ACCOUNT_ID = os.environ.get('AD_ACCOUNT_ID')

# AWS S3 Config (for storing raw data snapshots)
S3_BUCKET_NAME = 'your-s3-bucket-for-ad-data' # <-- IMPORTANT: Change this!

# --- NEW: Google Sheets Config ---
GSHEET_CREDENTIALS_FILE = 'your-downloaded-credentials-file.json' # <-- The JSON key file
GSHEET_NAME = 'Your Google Sheet Name' # <-- The exact name of your Google Sheet
WORKSHEET_NAME = 'Sheet1' # <-- The name of the worksheet/tab with your data

# Pinecone Config
PINECONE_INDEX_NAME = 'multimodal-rag-chatbot' # Must match your config.yaml

# Initialize clients (do this once)
s3_client = boto3.client('s3')
pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

# --- 1. EXTRACT ---
def fetch_ads_data():
    """
    Connects to the Facebook Marketing API and pulls key data for all active ads.
    """
    print("🚀 Starting data extraction from Meta Ads API...")
    try:
        FacebookAdsApi.init(FB_APP_ID, FB_APP_SECRET, FB_ACCESS_TOKEN)
        account = AdAccount(AD_ACCOUNT_ID)

        # Define the fields you want from the API
        # We are pulling data from the 'ad' level, which includes campaign and ad set info
        ad_fields = [
            'name',
            'status',
            'campaign{name,objective}',
            'adset{name,status,daily_budget}',
        ]
        
        # Define the performance metrics you want
        insight_fields = [
            'spend',
            'impressions',
            'clicks',
            'ctr',
            'cpc',
            'actions', # This field contains conversions like 'purchase'
            'cost_per_action_type'
        ]

        # Get all active ads and their insights
        ads = account.get_ads(
            fields=ad_fields,
            params={
                'date_preset': 'today',
                'filtering': [{'field': 'ad.effective_status', 'operator': 'IN', 'value': ['ACTIVE']}]
            }
        )
        
        insights = account.get_insights(
            fields=insight_fields,
            params={
                'level': 'ad',
                'date_preset': 'today',
                'filtering': [{'field': 'ad.effective_status', 'operator': 'IN', 'value': ['ACTIVE']}]
            }
        )

        # Map insights to ads using ad_id for easy lookup
        insights_map = {insight['ad_id']: insight for insight in insights}

        # Combine ad data with its corresponding insights
        combined_data = []
        for ad in ads:
            ad_id = ad['id']
            if ad_id in insights_map:
                ad_data = ad.export_all_data()
                insight_data = insights_map[ad_id].export_all_data()
                ad_data['insights'] = insight_data
                combined_data.append(ad_data)
        
        print(f"✅ Successfully extracted data for {len(combined_data)} active ads.")
        return combined_data

    except Exception as e:
        print(f"❌ Error during data extraction: {e}")
        return None
    
def fetch_gsheet_data():
    """
    Connects to Google Sheets API and pulls conversion rate data.
    """
    print("🚀 Starting data extraction from Google Sheets...")
    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_file(GSHEET_CREDENTIALS_FILE, scopes=scopes)
        client = gspread.authorize(creds)
        
        sheet = client.open(GSHEET_NAME).worksheet(WORKSHEET_NAME)
        # .get_all_records() assumes the first row is the header
        gsheet_data = sheet.get_all_records()
        print(f"✅ Successfully extracted {len(gsheet_data)} rows from Google Sheets.")
        return gsheet_data
    except Exception as e:
        print(f"❌ Error during Google Sheets extraction: {e}")
        return []

# --- 2. TRANSFORM ---
def transform_data_for_rag(ad_data_list, gsheet_data):
    """
    Converts the raw JSON data from the API into natural language text chunks.
    """
    print("\n🔄 Transforming raw data into text chunks for RAG...")
    gsheet_map = {row['Campaign Name']: row for row in gsheet_data}

    text_chunks = []
    for ad in ad_data_list:
        try:
            # Safely get nested data using .get() to avoid errors if a key is missing
            campaign = ad.get('campaign', {})
            adset = ad.get('adset', {})
            insights = ad.get('insights', {})

            # Handle conversions (actions)
            purchases = 0

            # --- NEW: Merge GSheet Data ---
            campaign_name = campaign.get('name')
            # Look up the conversion rate from our map; default to "N/A" if not found.
            conversion_rate_str = "N/A"
            if campaign_name in gsheet_map:
                conversion_rate = gsheet_map[campaign_name].get('conversion_rate', 'N/A')
                # Format it nicely as a percentage if it's a number
                if isinstance(conversion_rate, (int, float)):
                    conversion_rate_str = f"{conversion_rate:.2%}"

            cost_per_purchase = "N/A"
            if 'actions' in insights:
                for action in insights['actions']:
                    if action['action_type'] == 'purchase':
                        purchases = int(action['value'])
            
            if 'cost_per_action_type' in insights:
                 for cost_action in insights['cost_per_action_type']:
                     if cost_action['action_type'] == 'purchase':
                         cost_per_purchase = f"${float(cost_action['value']):.2f}"

            # Create the descriptive text chunk
            chunk = (
                f"Ad Name: '{ad.get('name', 'N/A')}' (Ad ID: {ad.get('id')}) is currently {ad.get('status', 'N/A')}.\n"
                f"It belongs to Ad Set '{adset.get('name', 'N/A')}' which is {adset.get('status', 'N/A')} and has a daily budget of ${adset.get('daily_budget', '0')}.\n"
                f"This ad set is part of the '{campaign.get('name', 'N/A')}' campaign, whose objective is {campaign.get('objective', 'N/A')}.\n"
                f"Today's performance: It has spent ${float(insights.get('spend', 0)):.2f}, received {insights.get('impressions', 0)} impressions, and {insights.get('clicks', 0)} clicks.\n"
                f"The Click-Through Rate (CTR) is {float(insights.get('ctr', 0)):.2f}% and the Cost Per Click (CPC) is ${float(insights.get('cpc', 0)):.2f}.\n"
                f"It has generated {purchases} purchases at a cost of {cost_per_purchase} per purchase.\n"
                f"The overall campaign conversion rate from our internal tracking is {conversion_rate_str}." # <-- Added the new data!
            )
            
            # We also store the original ad_id as metadata with the chunk
            text_chunks.append({'id': ad.get('id'), 'text': chunk})

        except Exception as e:
            print(f"⚠️ Warning: Could not process ad {ad.get('id')}. Error: {e}")
            continue
    
    print(f"✅ Successfully transformed {len(text_chunks)} ads into text chunks.")
    return text_chunks

# --- 3. LOAD ---
def load_data_to_pinecone(text_chunks):
    """
    Takes the text chunks, embeds them, and upserts them into Pinecone.
    """
    print("\n💾 Loading and vectorizing data into Pinecone...")
    if not text_chunks:
        print("No data to load. Skipping Pinecone update.")
        return

    try:
        index = pc.Index(PINECONE_INDEX_NAME)

        # Prepare vectors for upsert
        vectors_to_upsert = []
        for chunk in text_chunks:
            # The 'id' becomes the Pinecone vector ID, 'text' is what we embed
            vector = embeddings.embed_query(chunk['text'])
            vectors_to_upsert.append({
                "id": chunk['id'], 
                "values": vector,
                "metadata": {"text": chunk['text']} # Store original text in metadata
            })

        # Upsert in batches (good practice for large datasets)
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            index.upsert(vectors=batch)
            print(f"  - Upserted batch {i//batch_size + 1}...")

        stats = index.describe_index_stats()
        print(f"✅ Successfully loaded data. Pinecone index now contains {stats['total_vector_count']} vectors.")

    except Exception as e:
        print(f"❌ Error during Pinecone loading: {e}")

# --- MAIN EXECUTION ---
def main():
    """
    The main function to run the entire ETL pipeline.
    """
    print("--- Starting Meta Ads Data Pipeline ---")
    
    # 1. Extract
    raw_data = fetch_ads_data()
    raw_gsheet_data = fetch_gsheet_data()
    
    if raw_data:
        # Save a snapshot of the raw data to S3 for archiving/debugging
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        filename = f"ads-data-snapshot-{timestamp}.json"
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=filename,
            Body=json.dumps(raw_data, indent=2)
        )
        print(f"\n📦 Raw data snapshot saved to S3 bucket '{S3_BUCKET_NAME}' as '{filename}'.")
        
        # 2. Transform
        chunks = transform_data_for_rag(raw_data, raw_gsheet_data)
        
        # 3. Load
        load_data_to_pinecone(chunks)
    
    print("\n--- Pipeline execution finished. ---")


if __name__ == "__main__":
    main()