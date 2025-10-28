import os
import json
import requests
from dotenv import load_dotenv
import pandas as pd

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------
def safe_get(data_dict, key_path, default=None):
    """Safely get nested dictionary or list values using dot notation."""
    keys = key_path.split('.')
    val = data_dict
    try:
        for key in keys:
            if isinstance(val, list):
                try:
                    key = int(key)
                except ValueError:
                    return default
            val = val[key]
        return val
    except (KeyError, TypeError, IndexError):
        return default


def determine_format_category(ad_creative):
    """Classify ad creative into Video, Carousel, or Static Image."""
    if not ad_creative:
        return "Unknown"
    if safe_get(ad_creative, 'asset_feed_spec.videos') or safe_get(ad_creative, 'object_story_spec.video_data.video_id'):
        return "Video/Reel"
    if safe_get(ad_creative, 'object_story_spec.link_data.child_attachments'):
        return "Carousel"
    if safe_get(ad_creative, 'image_url') or safe_get(ad_creative, 'asset_feed_spec.images') or safe_get(ad_creative, 'image_hash') or safe_get(ad_creative, 'thumbnail_url') or safe_get(ad_creative, 'object_story_spec.photo_data'):
        return "Static Image"
    return "Unknown"


def fetch_image_urls(hashes, access_token, ad_account_id, api_version):
    """Resolve image hashes into actual image URLs."""
    if not hashes:
        return {}
    print(f"Step 2: Resolving {len(hashes)} image hashes...")
    hash_url_map = {}
    BASE_URL = f"https://graph.facebook.com/{api_version}/"
    url = f"{BASE_URL}{ad_account_id}/adimages"
    params = {'fields': 'url,hash', 'hashes': json.dumps(list(hashes)), 'access_token': access_token}
    try:
        res = requests.get(url, params=params)
        res.raise_for_status()
        for item in res.json().get('data', []):
            hash_url_map[item['hash']] = item['url']
    except Exception as e:
        print("Error fetching image URLs:", e)
    return hash_url_map


def fetch_video_urls(video_ids, access_token, api_version):
    """Resolve video IDs into direct source URLs."""
    if not video_ids:
        return {}
    print(f"Step 2B: Resolving {len(video_ids)} video IDs...")
    video_url_map = {}
    BASE_URL = f"https://graph.facebook.com/{api_version}/"
    for vid in video_ids:
        url = f"{BASE_URL}{vid}"
        params = {'fields': 'source', 'access_token': access_token}
        try:
            res = requests.get(url, params=params)
            res.raise_for_status()
            data = res.json()
            if 'source' in data:
                video_url_map[vid] = data['source']
        except Exception as e:
            print(f"Skipping {vid}:", e)
    return video_url_map

# -------------------------------------------------------------------
# Main Data Fetch Script
# -------------------------------------------------------------------
def get_data_script():
    load_dotenv()

    ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")
    AD_ACCOUNT_ID = os.getenv("META_AD_ACCOUNT_ID")
    API_VERSION = "v21.0"

    if not all([ACCESS_TOKEN, AD_ACCOUNT_ID]):
        print("❌ Missing Meta Ads credentials in .env file.")
        return

    # -------------------------------------------------------------
    # Step 1: Fetch all ads and creatives
    # -------------------------------------------------------------
    FIELDS = (
        "id,name,status,"
        "campaign{id,name,objective,status,daily_budget,start_time,stop_time},"
        "adset{id,name,optimization_goal,status,daily_budget,"
        "targeting{geo_locations,countries,age_min,age_max,genders,"
        "publisher_platforms,facebook_positions,instagram_positions}},"
        "creative{title,body,image_url,thumbnail_url,image_hash,"
        "asset_feed_spec{videos{video_id},images{url}},"
        "object_story_spec{"
        "text_data{message},"
        "link_data{link,name,description,caption,picture,"
        "child_attachments{link,name,description,image_hash,video_id,picture}},"
        "video_data{video_id,image_url,title,call_to_action{type,value}},"
        "photo_data{image_hash,url}"
        "}},"
        "insights.time_range({'since':'2025-10-01','until':'2025-10-28'})"
        "{spend,impressions,clicks,ctr,cpc,cpm,actions,results,cost_per_action_type,purchase_roas}"
    )

    def fetch_all_ads():
        url = f"https://graph.facebook.com/{API_VERSION}/{AD_ACCOUNT_ID}/ads"
        params = {
            'access_token': ACCESS_TOKEN,
            'fields': FIELDS,
            'filtering': "[{'field':'effective_status','operator':'IN','value':['ACTIVE']}]",
            'limit': 100
        }

        ads, hashes, vids = [], set(), set()
        print("Fetching ads...")
        page = 1
        while url:
            res = requests.get(url, params=params if page == 1 else {})
            res.raise_for_status()
            data = res.json()
            for ad in data.get("data", []):
                creative = ad.get("creative", {})
                ad["format_category"] = determine_format_category(creative)

                # Collect image hashes
                if safe_get(creative, "image_hash"):
                    hashes.add(creative["image_hash"])
                for att in safe_get(creative, "object_story_spec.link_data.child_attachments", []):
                    if "image_hash" in att:
                        hashes.add(att["image_hash"])

                # Collect video IDs
                for vid in safe_get(creative, "asset_feed_spec.videos", []):
                    if "video_id" in vid:
                        vids.add(vid["video_id"])
                if safe_get(creative, "object_story_spec.video_data.video_id"):
                    vids.add(creative["object_story_spec"]["video_data"]["video_id"])
                ads.append(ad)

            print(f"Fetched page {page} ({len(data.get('data', []))} ads)")
            url = data.get("paging", {}).get("next")
            page += 1
        return ads, hashes, vids

    ads_data, unique_hashes, unique_video_ids = fetch_all_ads()

    # -------------------------------------------------------------
    # Step 2: Resolve media URLs
    # -------------------------------------------------------------
    hash_url_map = fetch_image_urls(unique_hashes, ACCESS_TOKEN, AD_ACCOUNT_ID, API_VERSION)
    video_url_map = fetch_video_urls(unique_video_ids, ACCESS_TOKEN, API_VERSION)

    for ad in ads_data:
        creative = ad.get("creative", {})
        top_hash = safe_get(creative, "image_hash")
        if top_hash in hash_url_map:
            creative["image_url"] = hash_url_map[top_hash]

        for v in safe_get(creative, "asset_feed_spec.videos", []):
            vid = v.get("video_id")
            if vid in video_url_map:
                v["source_url"] = video_url_map[vid]

        video_data = safe_get(creative, "object_story_spec.video_data")
        if video_data and "video_id" in video_data:
            vid = video_data["video_id"]
            if vid in video_url_map:
                video_data["video_url"] = video_url_map[vid]

    # -------------------------------------------------------------
    # Step 3: Save raw JSON locally
    # -------------------------------------------------------------
    os.makedirs("data", exist_ok=True)
    json_path = "data/dataset.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ads_data, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved raw dataset to {json_path}")

    # -------------------------------------------------------------
    # Step 4: Flatten into CSV
    # -------------------------------------------------------------
    records = []
    for ad in ads_data:
        campaign = ad.get("campaign", {})
        adset = ad.get("adset", {})
        creative = ad.get("creative", {})

        # ✅ Unwrap nested insights
        ins_data = None
        if isinstance(ad.get("insights"), dict):
            ins_list = ad["insights"].get("data", [])
            if isinstance(ins_list, list) and len(ins_list) > 0:
                ins_data = ins_list[0]
        ins = ins_data or {}

        # ✅ Extract metrics safely
        spend = float(ins.get("spend", 0))
        impressions = int(ins.get("impressions", 0))
        clicks = int(ins.get("clicks", 0))
        ctr = float(ins.get("ctr", 0))
        cpc = float(ins.get("cpc", 0))
        cpm = float(ins.get("cpm", 0))

        # Handle purchase_roas (list or dict)
        roas_val = 0
        roas_field = ins.get("purchase_roas")
        if isinstance(roas_field, list) and len(roas_field) > 0:
            val = roas_field[0].get("value")
            if val:
                roas_val = float(val)

        # Derive conversions and conversion rate
        actions = ins.get("actions", [])
        conversions = sum(int(a.get("value", 0)) for a in actions if a.get("action_type") in ["offsite_conversion", "purchase"])
        conversion_rate = (conversions / clicks * 100) if clicks > 0 else 0

        rec = {
            "ad_id": ad.get("id"),
            "ad_name": ad.get("name"),
            "ad_status": ad.get("status"),
            "format_category": ad.get("format_category"),

            "campaign_id": campaign.get("id"),
            "campaign_name": campaign.get("name"),
            "campaign_objective": campaign.get("objective"),

            "adset_id": adset.get("id"),
            "adset_name": adset.get("name"),
            "optimization_goal": adset.get("optimization_goal"),

            "creative_title": creative.get("title"),
            "creative_body": creative.get("body"),
            "creative_image_url": creative.get("image_url"),
            "creative_thumbnail_url": creative.get("thumbnail_url"),

            "copy_text": safe_get(creative, "object_story_spec.text_data.message"),
            "link_url": safe_get(creative, "object_story_spec.link_data.link"),

            "spend": spend,
            "impressions": impressions,
            "clicks": clicks,
            "ctr": ctr,
            "cpc": cpc,
            "cpm": cpm,
            "roas": roas_val,
            "conversions": conversions,
            "conversion_rate": conversion_rate,
        }
        records.append(rec)

    df = pd.DataFrame(records)
    csv_path = "data/meta_ads_data.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ Exported {len(df)} ads to {csv_path}")

# -------------------------------------------------------------------
if __name__ == "__main__":
    get_data_script()
