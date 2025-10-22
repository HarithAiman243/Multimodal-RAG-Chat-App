import os
import json
import requests
from dotenv import load_dotenv

def get_data_script():
    """
    Main function for the data acquisition script.
    """
    load_dotenv()

    # --- Configuration ---
    ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")
    AD_ACCOUNT_ID = os.getenv("META_AD_ACCOUNT_ID")
    API_VERSION = "v19.0"
    OUTPUT_FILE = os.path.join("data", "dataset.json")

    if not all([ACCESS_TOKEN, AD_ACCOUNT_ID]):
        print("Error: META_ACCESS_TOKEN and META_AD_ACCOUNT_ID must be set in the .env file.")
        return

    # --- API Fields ---
    # This query defines the structure of the data you need. It pulls ad, ad set,
    # campaign, and creative details in a single nested request.
    FIELDS = (
        "id,name,status,"
        "campaign{id,name},"
        "adset{id,name},"
        "creative{title,body,image_url,thumbnail_url}"
    )
    
    def fetch_all_ads():
        """
        Fetches all active ads from the Meta Graph API, handling pagination automatically.
        """
        all_ads = []
        url = f"https://graph.facebook.com/{API_VERSION}/{AD_ACCOUNT_ID}/ads"
        params = {
            'access_token': ACCESS_TOKEN,
            'fields': FIELDS,
            'filtering': "[{'field':'effective_status','operator':'IN','value':['ACTIVE']}]",
            'limit': 100  # Request the maximum allowed limit per page
        }

        print("Fetching data from Meta API...")
        page_num = 1
        while url:
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
                data = response.json()

                if 'data' in data and data['data']:
                    all_ads.extend(data['data'])
                    print(f"  - Fetched page {page_num} ({len(data['data'])} ads)")
                    # The 'next' URL provided by the API includes all necessary parameters
                    url = data.get('paging', {}).get('next')
                    params = {}  # Clear params as they are included in the 'next' URL
                    page_num += 1
                else:
                    # No more data or an unexpected response format
                    break
            except requests.exceptions.RequestException as e:
                print(f"API Request failed: {e}")
                print(f"Response content: {response.text}")
                break
        
        return all_ads

    print("Starting data acquisition from Meta Ads API...")
    ads_data = fetch_all_ads()
    
    if ads_data:
        # Ensure the data directory exists before writing the file
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(ads_data, f, indent=2)
        print(f"\nSuccess: Fetched a total of {len(ads_data)} ads and saved to {OUTPUT_FILE}")
    else:
        print("\nWarning: No data was fetched from the API.")

if __name__ == "__main__":
    get_data_script()