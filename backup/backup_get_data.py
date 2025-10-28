import os
import json
import requests
from dotenv import load_dotenv
import boto3

# --- Helper Function for Safe Nested Dictionary Access ---
def safe_get(data_dict, key_path, default=None):
    """Safely gets a nested key from a dictionary."""
    keys = key_path.split('.')
    val = data_dict
    try:
        for key in keys:
            # Simple handling for list indices
            if isinstance(val, list):
                try:
                    key = int(key)
                except ValueError:
                    return default
            val = val[key]
        return val
    except (KeyError, TypeError, IndexError):
        return default

# --- Data Determination Functions ---

def determine_format_category(ad_creative):
    """Analyzes the creative data to determine the ad format."""
    if not ad_creative:
        return "Unknown"

    # Check for video first
    if safe_get(ad_creative, 'asset_feed_spec.videos') or safe_get(ad_creative, 'object_story_spec.video_data.video_id'):
        return "Video/Reel"

    # Check for Carousel
    if safe_get(ad_creative, 'object_story_spec.link_data.child_attachments'):
        return "Carousel"

    # Check for Static Image (Includes checking for top-level image/hash, thumbnail, or photo_data)
    if safe_get(ad_creative, 'image_url') or safe_get(ad_creative, 'asset_feed_spec.images') or safe_get(ad_creative, 'image_hash') or safe_get(ad_creative, 'thumbnail_url') or safe_get(ad_creative, 'object_story_spec.photo_data'):
        return "Static Image" 

    return "Unknown"

# ----------------------------------------------------------------------
# --- ASSET RESOLUTION FUNCTIONS (Step 2) ---
# ----------------------------------------------------------------------

def fetch_image_urls(hashes, access_token, ad_account_id, api_version):
    """Fetches the image URL for all provided hashes using the /adimages endpoint."""
    if not hashes:
        return {}
    
    print(f"Step 2: Resolving {len(hashes)} unique image hashes to URLs...")
    hash_url_map = {}
    hashes_str = json.dumps(list(hashes))

    adimages_params = {
        'fields': 'url,hash',
        'hashes': hashes_str,
        'access_token': access_token
    }
    
    BASE_URL = f"https://graph.facebook.com/{api_version}/"
    url = f"{BASE_URL}{ad_account_id}/adimages"
    
    try:
        response = requests.get(url, params=adimages_params)
        response.raise_for_status()
        data = response.json()
        
        for image_data in data.get('data', []):
            hash_url_map[image_data['hash']] = image_data['url']
            
    except requests.exceptions.RequestException as e:
        # If API error, print it but allow function to return partial map
        print(f"Error fetching image URLs: {e}") 

    print(f"Resolved {len(hash_url_map)} image URLs.")
    return hash_url_map

def fetch_video_urls(video_ids, access_token, api_version):
    """Fetches the permanent source URL for all provided video IDs."""
    if not video_ids:
        return {}
    
    video_url_map = {}
    BASE_URL = f"https://graph.facebook.com/{api_version}/"
    #BATCH_SIZE = 50 # Limit the number of IDs per request
    
    # CORRECTED LOGIC: Query IDs individually
    for video_id in video_ids:
        # Correct URL format: /<video_id>?fields=source
        url = f"{BASE_URL}{video_id}" 
        params = {
            'fields': 'source',
            'access_token': access_token
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if isinstance(data, dict) and 'source' in data:
                video_url_map[video_id] = data['source']
            
        except requests.exceptions.RequestException as e:
            # Print error for the specific ID that failed
            print(f"Skipping video ID {video_id} due to HTTP Error: {e}") 
            
    return video_url_map

# ----------------------------------------------------------------------
# --- NEW FUNCTION: S3 UPLOAD ---
# ----------------------------------------------------------------------

def upload_to_s3(json_content, bucket_name, key, aws_access_key_id, aws_secret_access_key):
    """Uploads the JSON string content directly to an S3 bucket."""
    try:
        s3 = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        
        s3.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=json_content.encode('utf-8'),
            ContentType='application/json'
        )
        return True
    except Exception as e:
        print(f"FATAL S3 UPLOAD ERROR: {e}")
        return False
    

# ----------------------------------------------------------------------
# --- MAIN SCRIPT EXECUTION (Steps 1 & 3) ---
# ----------------------------------------------------------------------

def get_data_script():
    """
    Main function for the data acquisition script, fetching ads, resolving assets, 
    and saving data.
    """
    load_dotenv()

    # --- Configuration ---
    ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")
    AD_ACCOUNT_ID = os.getenv("META_AD_ACCOUNT_ID")
    API_VERSION = "v24.0"
    
    # S3 Configuration
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME")
    S3_KEY = os.getenv("S3_KEY", "data/dataset.json") # Use default key if not set

    #OUTPUT_FILE = os.path.join("data", "dataset.json") 
    
    #if not all([ACCESS_TOKEN, AD_ACCOUNT_ID]):
    #    print("Error: META_ACCESS_TOKEN and META_AD_ACCOUNT_ID must be set in the .env file.")
    #    return

    if not all([ACCESS_TOKEN, AD_ACCOUNT_ID, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_BUCKET_NAME]):
        print("Error: Required environment variables (META_ACCESS_TOKEN, AD_ACCOUNT_ID, AWS keys, S3_BUCKET_NAME) must be set.")
        return

    # --- API Fields: Comprehensive selection for all ad types ---
    FIELDS = (
        "id,name,status,"
        "campaign{id,name},"
        "adset{id,name},"
        "creative{"
            "title,body,image_url,thumbnail_url,"
            "image_hash," 
            "asset_feed_spec{videos{video_id},images{url}},"
            "object_story_spec{"
                "text_data{message},"
                "link_data{link,name,description,caption,child_attachments{link,name,description,image_hash,video_id,picture},picture},"
                "video_data{video_id,image_url,title,call_to_action{type,value}},"
                "photo_data{image_hash,url}" 
            "}"
        "}"
    )
    
    def fetch_all_ads():
        """
        Fetches all active ads and collects all image hashes and video IDs for later resolution.
        Returns (list_of_ads, set_of_hashes, set_of_video_ids).
        """
        all_ads = []
        all_hashes = set()
        all_video_ids = set()
        url = f"https://graph.facebook.com/{API_VERSION}/{AD_ACCOUNT_ID}/ads"
        params = {
            'access_token': ACCESS_TOKEN,
            'fields': FIELDS,
            'filtering': "[{'field':'effective_status','operator':'IN','value':['ACTIVE']}]",
            'limit': 100
        }

        print("Step 1: Fetching ad creative data from Meta API...")
        page_num = 1
        while url:
            try:
                if page_num > 1:
                    response = requests.get(url)
                    params = {}
                else:
                    response = requests.get(url, params=params)

                response.encoding = 'utf-8'
                response.raise_for_status() 
                data = response.json()

                if 'data' in data and data['data']:
                    
                    for ad in data['data']:
                        creative = ad.get('creative', {})
                        ad['format_category'] = determine_format_category(creative)
                        
                        # --- HASH COLLECTION LOGIC ---
                        if safe_get(creative, 'image_hash'):
                            all_hashes.add(creative['image_hash'])
                            
                        for attachment in safe_get(creative, 'object_story_spec.link_data.child_attachments', []):
                            if 'image_hash' in attachment:
                                all_hashes.add(attachment['image_hash'])

                        photo_data = safe_get(creative, 'object_story_spec.photo_data', {})
                        if photo_data.get('image_hash') and not photo_data.get('url'):
                            all_hashes.add(photo_data['image_hash'])

                        # --- VIDEO ID COLLECTION LOGIC ---
                        for video_asset in safe_get(creative, 'asset_feed_spec.videos', []):
                            if 'video_id' in video_asset:
                                all_video_ids.add(video_asset['video_id'])

                        if safe_get(creative, 'object_story_spec.video_data.video_id'):
                            all_video_ids.add(creative['object_story_spec']['video_data']['video_id'])
                            
                        all_ads.append(ad)
                        
                    print(f"  - Fetched page {page_num} ({len(data['data'])} ads)")
                    
                    url = data.get('paging', {}).get('next')
                    page_num += 1
                else:
                    break
            except requests.exceptions.RequestException as e:
                # Print API error, but continue if possible
                print(f"API Request failed: {e}")
                if 'response' in locals() and response.text:
                    print(f"Response content: {response.text}")
                break
        
        return all_ads, all_hashes, all_video_ids

    print("Starting full data acquisition and asset resolution...")
    ads_data, unique_hashes, unique_video_ids = fetch_all_ads()
    
    # Step 2A & 2B: Resolve assets
    hash_url_map = fetch_image_urls(unique_hashes, ACCESS_TOKEN, AD_ACCOUNT_ID, API_VERSION)
    video_url_map = fetch_video_urls(unique_video_ids, ACCESS_TOKEN, API_VERSION)

    # Step 3: Inject resolved URLs back into the creative data
    print("Step 3: Appending resolved image/video URLs back to ad data...")
    
    processed_ads = []
    
    for ad in ads_data:
        creative = ad.get('creative', {})
        
        # --- IMAGE RESOLUTION ---
        
        # 1. Top-level hash 
        top_hash = safe_get(creative, 'image_hash')
        if top_hash in hash_url_map:
            creative['image_url'] = hash_url_map[top_hash] 

        # 2. Carousel hashes
        if ad['format_category'] == 'Carousel':
            children = safe_get(creative, 'object_story_spec.link_data.child_attachments', [])
            for child in children:
                child_hash = child.get('image_hash')
                if child_hash in hash_url_map:
                    child['image_url'] = hash_url_map[child_hash]

        # 3. Photo_data hash
        photo_data = safe_get(ad, 'creative.object_story_spec.photo_data')
        if photo_data and 'image_hash' in photo_data:
            photo_hash = photo_data['image_hash']
            if photo_hash in hash_url_map:
                photo_data['url'] = hash_url_map[photo_hash]
        
        # --- VIDEO RESOLUTION ---

        # 4. Inject resolved video URLs (asset_feed_spec)
        for video_asset in safe_get(creative, 'asset_feed_spec.videos', []):
            video_id = video_asset.get('video_id')
            if video_id in video_url_map:
                # Use a new field 'source_url' for assets in the feed spec
                video_asset['source_url'] = video_url_map[video_id]

        # 5. Inject resolved video URL (video_data)
        video_data = safe_get(creative, 'object_story_spec.video_data')
        if video_data and 'video_id' in video_data:
            video_id = video_data['video_id']
            if video_id in video_url_map:
                # INJECTION FIX: Map the resolved URL to 'video_url'
                video_data['video_url'] = video_url_map[video_id]
                    
        processed_ads.append(ad)

    # --- Local File Storage ---
    if processed_ads:
        json_content = json.dumps(processed_ads, indent=2, ensure_ascii=False)
        
    #    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    #    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    #        f.write(json_content)
    #    print(f"Success: Fetched and processed a total of {len(processed_ads)} ads and saved to {OUTPUT_FILE}")

    #else:
    #    print("Warning: No data was fetched or processed.")

        success = upload_to_s3(json_content, AWS_S3_BUCKET_NAME, S3_KEY, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
        
        if success:
            print(f"Success: Fetched and processed a total of {len(processed_ads)} ads and SAVED TO S3 ({AWS_S3_BUCKET_NAME}/{S3_KEY}).")
        else:
            # Fallback to local file save if S3 fails
            print("S3 upload failed. Falling back to local file save...")
            OUTPUT_FILE = os.path.join("data", "dataset.json") 
            os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                f.write(json_content)
            print(f"Warning: Data saved locally to {OUTPUT_FILE}.")

    else:
        print("Warning: No data was fetched or processed.")

if __name__ == "__main__":
    get_data_script()