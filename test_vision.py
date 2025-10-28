import os
import base64
import json # Import for JSON processing
from dotenv import load_dotenv
import openai

def encode_image_to_base64(image_path):
    """Encodes a local image file to a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: Local image file not found at path: {image_path}")
        return None

def test_image_reading():
    load_dotenv()
    
    try:
        client = openai.OpenAI()
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        print("Please ensure your OPENAI_API_KEY is set correctly in the .env file.")
        return

    # Updated function to extract structured marketing data
    def extract_marketing_data(base64_image: str, mime_type: str = "image/jpeg") -> str:
        if not base64_image:
            return "No image data provided."
        
        data_uri = f"data:{mime_type};base64,{base64_image}"
        
        print("Sending Base64 encoded image data for structured marketing analysis.")
        
        # --- Structured Prompting for Marketing Data ---
        
        system_prompt = "You are a Digital Marketing Analyst AI. Your task is to analyze the provided image, " \
        "which is an advertisement, and extract all relevant marketing data points. You must identify the product," \
        "offer, pricing, and all visible text."
        
        user_prompt = "Analyze this ad visual. Extract the Campaign Name, main Offer/Promotion, all specific Prices," \
        "the Primary Call-to-Action (CTA) if visible, the Target Audience inferred from the content, and a descriptive " \
        "label for the Visual Type (e.g., product photo, lifestyle ad, infographic)."

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {"url": data_uri}},
                        ],
                    }
                ],
                max_tokens=512, # Increased tokens for detailed JSON output
                response_format={"type": "json_object"} # Enforce JSON output
            )
            # The output is now a clean JSON string
            json_string = response.choices[0].message.content
            return json_string
        except Exception as e:
            # Added error handling to show the API-specific error message
            return f"Error: Image analysis failed. Details: {e}"

    # --- TEST EXECUTION ---
    LOCAL_IMAGE_PATH = "testburger1.jpg"
    base64_string = encode_image_to_base64(LOCAL_IMAGE_PATH)
    
    print("\n--- Starting Structured Marketing Data Extraction ---")
    
    json_output = extract_marketing_data(base64_string)
    
    print("\n--- Result (JSON Output for RAG Indexing) ---")
    
    # Attempt to pretty-print the JSON output
    try:
        data = json.loads(json_output)
        # Use json.dumps for structured, readable output
        print(json.dumps(data, indent=4))
    except json.JSONDecodeError:
        # Fallback if the model did not return valid JSON
        print("Error: Could not decode JSON. Raw output:")
        print(json_output)

if __name__ == "__main__":
    test_image_reading()