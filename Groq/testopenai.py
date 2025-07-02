# Same imports as before + seaborn
import os
import time
import base64
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import openai  # OpenAI SDK (used for Groq)

from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

class ImageScramblingDetectorGroq:
    def __init__(self, api_key, scrambled_folder, unscrambled_folder, model="llama3-70b-8192"):
        self.api_key = api_key
        self.scrambled_folder = Path(scrambled_folder)
        self.unscrambled_folder = Path(unscrambled_folder)
        self.model = model
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        self.supported_formats = {'.jpg', '.jpeg', '.png'}
        self.results = []

    def encode_image(self, path):
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    def classify_image(self, image_path):
        base64_image = self.encode_image(image_path)
        image_data = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}"
            }
        }

        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            image_data,
                            {
                                "type": "text",
                                "text": """Examine the image and classify it based on its visual structure.
Respond with exactly one of the following words:

"scrambled" — if the image has jumbled, rearranged, or mixed-up segments that make it look distorted or disordered.
"unscrambled" — if the image appears coherent, orderly, and visually normal.

Respond with ONLY the single word: "scrambled" or "unscrambled". No explanations or extra text."""
                            }
                        ]
                    }
                ]
            )
            response_time = time.time() - start_time
            prediction = response.choices[0].message.content.strip().lower()

            if "scrambled" in prediction and "unscrambled" not in prediction:
                return "scrambled", response_time
            elif "unscrambled" in prediction:
                return "unscrambled", response_time
            else:
                return "unscrambled", response_time  # default fallback
        except Exception as e:
            print(f"Error classifying {image_path}: {e}")
            return None, time.time() - start_time

    def collect_images(self):
        dataset = []
        for folder, label in [(self.scrambled_folder, 'scrambled'), (self.unscrambled_folder, 'unscrambled')]:
            if folder.exists():
                for img in folder.iterdir():
                    if img.suffix.lower() in self.supported_formats:
                        dataset.append({
                            "image_path": str(img),
                            "image_name": img.name,
                            "actual_label": label
                        })
        return dataset

    def process_all(self):
        dataset = self.collect_images()
        for item in dataset:
            print(f"Processing: {item['image_name']}")
            prediction, rt = self.classify_image(item["image_path"])
            self.results.append({
                **item,
                "predicted_label": prediction,
                "response_time_seconds": rt,
                "timestamp": datetime.now().isoformat()
            })

    def to_dataframe(self):
        return pd.DataFrame(self.results)

    def save_results(self, filename="results.csv"):
        df = self.to_dataframe()
        df.to_csv(filename, index=False)

    def print_summary(self):
        df = self.to_dataframe()
        df = df[df['predicted_label'].isin(['scrambled', 'unscrambled'])]
        if df.empty:
            print("No valid results.")
            return

        print("\nAccuracy:", accuracy_score(df['actual_label'], df['predicted_label']))
        print("\nClassification Report:")
        print(classification_report(df['actual_label'], df['predicted_label']))
        print("\nConfusion Matrix:")
        print(confusion_matrix(df['actual_label'], df['predicted_label']))

    def visualize_results(self, filename="visualization.png"):
        df = self.to_dataframe()
        df = df[df['predicted_label'].isin(['scrambled', 'unscrambled'])]

        if df.empty:
            print("No valid predictions to visualize.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Image Classification (Scrambled vs Unscrambled)", fontsize=16)

        # 1. Confusion Matrix
        cm = confusion_matrix(df['actual_label'], df['predicted_label'], labels=['scrambled', 'unscrambled'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['scrambled', 'unscrambled'],
                    yticklabels=['scrambled', 'unscrambled'],
                    ax=axes[0, 0])
        axes[0, 0].set_title("Confusion Matrix")
        axes[0, 0].set_xlabel("Predicted")
        axes[0, 0].set_ylabel("Actual")

        # 2. Response Time Histogram
        axes[0, 1].hist(df['response_time_seconds'], bins=20, color='skyblue', edgecolor='black')
        axes[0, 1].set_title("Response Time Distribution")
        axes[0, 1].set_xlabel("Response Time (s)")
        axes[0, 1].set_ylabel("Count")

        # 3. Accuracy by Class
        class_accuracy = df.groupby('actual_label').apply(
            lambda x: (x['actual_label'] == x['predicted_label']).mean()
        )
        bars = axes[1, 0].bar(class_accuracy.index, class_accuracy.values, color=['tomato', 'mediumseagreen'])
        axes[1, 0].set_ylim(0, 1.05)
        axes[1, 0].set_title("Accuracy by Actual Label")
        axes[1, 0].set_ylabel("Accuracy")
        for bar, val in zip(bars, class_accuracy.values):
            axes[1, 0].text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}", ha='center')

        # 4. Response Time by Correctness
        df['correct'] = df['actual_label'] == df['predicted_label']
        axes[1, 1].hist(
            [df[df['correct']]['response_time_seconds'], df[~df['correct']]['response_time_seconds']],
            label=['Correct', 'Incorrect'],
            bins=15, stacked=True, color=['green', 'red'], edgecolor='black'
        )
        axes[1, 1].set_title("Response Time by Prediction Accuracy")
        axes[1, 1].set_xlabel("Response Time (s)")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.show()
        print(f"Visualization saved to: {filename}")
def main():
    API_KEY = os.getenv("GROQ_API_KEY")
    SCRAMBLED = "C://Users//HP//Documents//Projects//CAPTCHA_SOLVER//Scrambled"
    UNSCRAMBLED = "C://Users//HP//Documents//Projects//CAPTCHA_SOLVER//Unscambled"

    detector = ImageScramblingDetectorGroq(API_KEY, SCRAMBLED, UNSCRAMBLED)
    detector.process_all()
    detector.save_results("groq_image_results.csv")
    detector.print_summary()
    detector.visualize_results("groq_results_plot.png")

if __name__ == "__main__":
    main()
