import requests
import base64
import os
import time
from tqdm import tqdm

# Use this function to convert an image file from the filesystem to base64
def image_file_to_base64(image_path):
    with open(image_path, 'rb') as f:
        image_data = f.read()
    return base64.b64encode(image_data).decode('utf-8')

# Use this function to fetch an image from a URL and convert it to base64
def image_url_to_base64(image_url):
    response = requests.get(image_url)
    image_data = response.content
    return base64.b64encode(image_data).decode('utf-8')

# Use this function to convert a list of image URLs to base64
def image_urls_to_base64(image_urls):
    return [image_url_to_base64(url) for url in image_urls]

def process_images(input_dir, output_dir):
    api_key = os.environ.get("SEGMIND_API_KEY")
    if not api_key:
        raise RuntimeError("Set the SEGMIND_API_KEY environment variable to call the Segmind API.")
    url = "https://api.segmind.com/v1/seedream-4"
    headers = {'x-api-key': api_key}
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, "fat"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "thin"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "muscle"), exist_ok=True)
    
    # Get list of image files
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Process each image
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(input_dir, image_file)
        
        # Define prompts for each transformation
        prompts = {
            "fat": "Make the person fatter",
            "thin": "Make the person thin",
            "muscle": "Make the person muscular"
        }
        
        # Process each transformation
        for transformation, prompt in prompts.items():
            # Request payload
            data = {
                "size": "custom",
                "width": 960,
                "height": 1280,
                "prompt": prompt,
                "max_images": 1,
                "image_input": [image_path],
                "aspect_ratio": "match_input_image",
                "sequential_image_generation": "disabled"
            }
            
            try:
                response = requests.post(url, json=data, headers=headers)
                if response.status_code == 200:
                    # Save the generated image
                    output_path = os.path.join(output_dir, transformation, image_file)
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                    print(f"Saved {transformation} version of {image_file}")
                else:
                    print(f"Error processing {image_file} for {transformation}: {response.status_code}, {response.text}")
                
                # Add a small delay to avoid API rate limits
                time.sleep(1)
                
            except Exception as e:
                print(f"Exception while processing {image_file} for {transformation}: {str(e)}")

if __name__ == "__main__":
    input_directory = "input_images"
    output_directory = "seedream_output"
    
    process_images(input_directory, output_directory)