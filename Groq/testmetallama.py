import os
import time
import base64
from pathlib import Path
from groq import Groq
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# --- CONFIGURATION ---
SCRAMBLED_FOLDER = "C://Users//HP//Documents//Projects//CAPTCHA_SOLVER//Scrambled"       # <--- Update this
UNSCRAMBLED_FOLDER = "C://Users//HP//Documents//Projects//CAPTCHA_SOLVER//Unscambled"   # <--- Update this
API_KEY = api_key   # or put your string directly
MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"
OUTPUT_CSV = "groq_scrambling_results.csv"
# ----------------------

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

# Encode an image to base64
def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# Build classification prompt
def get_prompt():
    return (
        "Look at this image and classify it as either 'scrambled' or 'unscrambled'.\n"
        "Use the word 'scrambled' if it appears jumbled or disordered.\n"
        "Use 'unscrambled' if it appears coherent and normal.\n"
        "Respond with ONLY one word: scrambled or unscrambled."
    )

# Classify a single image
def classify_image(client, image_path):
    try:
        base64_img = encode_image(image_path)
        prompt = get_prompt()

        start_time = time.time()

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_img}"
                            },
                        },
                    ],
                }
            ],
        )
        end_time = time.time()
        duration = end_time - start_time

        output = response.choices[0].message.content.strip().lower()

        # Normalize output
        if "scrambled" in output and "unscrambled" not in output:
            return "scrambled", duration
        elif "unscrambled" in output:
            return "unscrambled", duration
        else:
            return "unknown", duration

    except Exception as e:
        print(f"Error classifying {image_path}: {e}")
        return "error", 0

# Collect all images with labels
def load_images(scrambled_folder, unscrambled_folder):
    dataset = []
    for label, folder in [("scrambled", scrambled_folder), ("unscrambled", unscrambled_folder)]:
        for img_path in Path(folder).iterdir():
            if img_path.suffix.lower() in SUPPORTED_FORMATS:
                dataset.append({
                    "image_name": img_path.name,
                    "image_path": str(img_path),
                    "actual_label": label
                })
    return dataset

# Main pipeline
def main():
    client = Groq(api_key=API_KEY)
    dataset = load_images(SCRAMBLED_FOLDER, UNSCRAMBLED_FOLDER)

    print(f"Found {len(dataset)} images to process...")
    results = []

    for i, data in enumerate(dataset, 1):
        print(f"[{i}/{len(dataset)}] Processing: {data['image_name']}")
        pred, resp_time = classify_image(client, data["image_path"])
        results.append({
            "image_name": data["image_name"],
            "actual_label": data["actual_label"],
            "predicted_label": pred,
            "response_time_sec": resp_time,
            "timestamp": datetime.now().isoformat()
        })

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Results saved to: {OUTPUT_CSV}")

    # Optional summary
    correct = df[df.actual_label == df.predicted_label]
    accuracy = len(correct) / len(df)
    print(f"\n📊 Accuracy: {accuracy:.2%}")
    print(df["predicted_label"].value_counts())

if __name__ == "__main__":
    main()
