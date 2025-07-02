
import os
import time
import base64
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import json
from datetime import datetime
from groq import Groq

from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

class ImageScramblingDetectorGroq:
    def __init__(self, api_key, scrambled_folder, unscrambled_folder):
        self.api_key = api_key
        self.scrambled_folder = Path(scrambled_folder)
        self.unscrambled_folder = Path(unscrambled_folder)
        self.results = []
        self.supported_formats = {'.jpg', '.jpeg', '.png'}
        self.client = Groq(api_key=self.api_key)

    def encode_image_to_base64(self, image_path):
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None

    def classify_image_with_groq(self, image_path):
        start_time = time.time()
        base64_image = self.encode_image_to_base64(image_path)
        if not base64_image:
            return None, 0

        try:
            response = self.client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                # model = "meta-llama/llama-4-maverick-17b-128e-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": """
                             Examine the image and classify it based on its visual structure.
                                Respond with exactly one of the following words:

                                "scrambled" — if the image has jumbled, rearranged, or mixed-up segments that make it look distorted or disordered.

                                "unscrambled" — if the image appears coherent, orderly, and visually normal.

                                Respond with ONLY the single word: "scrambled" or "unscrambled". No explanations or extra text.
                             """},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
            )
            end_time = time.time()
            prediction = response.choices[0].message.content.strip().lower()
            if "scrambled" in prediction and "unscrambled" not in prediction:
                prediction = "scrambled"
            elif "unscrambled" in prediction:
                prediction = "unscrambled"
            else:
                prediction = "unscrambled"
            return prediction, end_time - start_time
        except Exception as e:
            print(f"Error during API call for {image_path}: {e}")
            return None, time.time() - start_time

    def collect_images(self):
        dataset = []
        for folder, label in [(self.scrambled_folder, "scrambled"), (self.unscrambled_folder, "unscrambled")]:
            if folder.exists():
                for img_path in folder.iterdir():
                    if img_path.suffix.lower() in self.supported_formats:
                        dataset.append({
                            "image_path": str(img_path),
                            "image_name": img_path.name,
                            "actual_label": label,
                            "folder": label
                        })
        return dataset

    def process_all_images(self):
        dataset = self.collect_images()
        print(f"Found {len(dataset)} images")
        for i, img_data in enumerate(dataset, 1):
            print(f"Processing {i}/{len(dataset)}: {img_data['image_name']}")
            prediction, response_time = self.classify_image_with_groq(img_data["image_path"])
            self.results.append({
                **img_data,
                "predicted_label": prediction,
                "response_time_seconds": response_time,
                "timestamp": datetime.now().isoformat()
            })
            time.sleep(1)
        return self.results

    def create_dataframe(self):
        if not self.results:
            print("No results to create DataFrame")
            return None
        df = pd.DataFrame(self.results)
        df["predicted_label"] = df["predicted_label"].fillna("unknown")
        return df

    def calculate_metrics(self, df):
        valid_df = df[df["predicted_label"].isin(["scrambled", "unscrambled"])]
        accuracy = accuracy_score(valid_df["actual_label"], valid_df["predicted_label"])
        report = classification_report(valid_df["actual_label"], valid_df["predicted_label"], output_dict=True)
        cm = confusion_matrix(valid_df["actual_label"], valid_df["predicted_label"], labels=["scrambled", "unscrambled"])
        timing = {
            "mean": valid_df["response_time_seconds"].mean(),
            "median": valid_df["response_time_seconds"].median(),
            "min": valid_df["response_time_seconds"].min(),
            "max": valid_df["response_time_seconds"].max(),
            "std": valid_df["response_time_seconds"].std()
        }
        return {"accuracy": accuracy, "report": report, "confusion_matrix": cm, "timing": timing}

    def visualize_results(self, df, metrics):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues", ax=axes[0],
                    xticklabels=["scrambled", "unscrambled"], yticklabels=["scrambled", "unscrambled"])
        axes[0].set_title("Confusion Matrix")
        df["response_time_seconds"].hist(bins=20, ax=axes[1], color="lightblue", edgecolor="black")
        axes[1].set_title("Response Time Distribution")
        plt.tight_layout()
        plt.savefig("vis_scout.png")
        plt.show()

def main():
    API_KEY = api_key
    SCRAMBLED_FOLDER = "C://Users//HP//Documents//Projects//CAPTCHA_SOLVER//Scrambled copy"
    UNSCRAMBLED_FOLDER = "C://Users//HP//Documents//Projects//CAPTCHA_SOLVER//Unscambled"

    detector = ImageScramblingDetectorGroq(API_KEY, SCRAMBLED_FOLDER, UNSCRAMBLED_FOLDER)
    detector.process_all_images()
    df = detector.create_dataframe()
    if df is not None:
        df.to_csv("scout.csv", index=False)
        metrics = detector.calculate_metrics(df)
        detector.visualize_results(df, metrics)

if __name__ == "__main__":
    main()
