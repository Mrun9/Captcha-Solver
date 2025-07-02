
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
    def __init__(self, api_key, scrambled_folder, unscrambled_folder, few_shot_examples=None):
        self.api_key = api_key
        self.scrambled_folder = Path(scrambled_folder)
        self.unscrambled_folder = Path(unscrambled_folder)
        self.results = []
        self.supported_formats = {'.jpg', '.jpeg', '.png'}
        self.client = Groq(api_key=self.api_key)
        self.few_shot_examples = few_shot_examples or []

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

        few_shot_content = []

        for example in self.few_shot_examples:
            encoded = self.encode_image_to_base64(example["path"])
            if encoded:
                few_shot_content.extend([
                    {"type": "text", "text": f'This is a {example["label"]} image.'},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}
                ])

        few_shot_content.append({"type": "text", "text": """
                                Now examine this new image and classify it based on the examples shown above. 
                                    
                                    Respond with exactly one of the following words: 
                                    "scrambled" — if the image has jumbled, rearranged, or mixed-up segments that make it look distorted or disordered, similar to the scrambled examples. 
                                    "unscrambled" — if the image appears coherent, orderly, and visually normal, similar to the unscrambled examples. 
                                    
                                    Respond with ONLY the single word: "scrambled" or "unscrambled". No explanations or extra text.
"""})
        few_shot_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        })

        try:
            response = self.client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{"role": "user", "content": few_shot_content}],
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
        class_report = classification_report(valid_df["actual_label"], valid_df["predicted_label"], output_dict=True)
        cm = confusion_matrix(valid_df["actual_label"], valid_df["predicted_label"], labels=["scrambled", "unscrambled"])
        timing = {
            "mean": valid_df["response_time_seconds"].mean(),
            "median": valid_df["response_time_seconds"].median(),
            "min": valid_df["response_time_seconds"].min(),
            "max": valid_df["response_time_seconds"].max(),
            "std": valid_df["response_time_seconds"].std()
        }
        return {"accuracy": accuracy, 
                "report": class_report,
                "confusion_matrix": cm, 
                "timing": timing,
                'valid_predictions': len(valid_df),
                'total_images': len(df)}

    def visualize_results(self, df, metrics, save_plots=True):
        """Create visualizations for the results"""
        if df is None or metrics is None:
            print("No data available for visualization")
            return
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Few-Shot Image Scrambling Detection Results', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix
        valid_df = df[df['predicted_label'].isin(['scrambled', 'unscrambled'])]
        cm = metrics['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['scrambled', 'unscrambled'],
                   yticklabels=['scrambled', 'unscrambled'],
                   ax=axes[0,0])
        axes[0,0].set_title('Confusion Matrix')
        axes[0,0].set_xlabel('Predicted Label')
        axes[0,0].set_ylabel('Actual Label')
        
        # 2. Response Time Distribution
        axes[0,1].hist(df['response_time_seconds'], bins=20, color='skyblue', alpha=0.7, edgecolor='black')
        axes[0,1].set_title('Response Time Distribution')
        axes[0,1].set_xlabel('Response Time (seconds)')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].axvline(metrics['timing']['mean_response_time'], 
                         color='red', linestyle='--', label=f"Mean: {metrics['timing_stats']['mean_response_time']:.2f}s")
        axes[0,1].legend()
        
        # 3. Accuracy by Category
        if not valid_df.empty:
            accuracy_by_category = valid_df.groupby('actual_label').apply(
                lambda x: (x['actual_label'] == x['predicted_label']).mean()
            )
            
            bars = axes[1,0].bar(accuracy_by_category.index, accuracy_by_category.values, 
                               color=['lightcoral', 'lightgreen'], alpha=0.7)
            axes[1,0].set_title('Accuracy by Category')
            axes[1,0].set_xlabel('Actual Label')
            axes[1,0].set_ylabel('Accuracy')
            axes[1,0].set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, accuracy_by_category.values):
                axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                              f'{value:.2f}', ha='center', va='bottom')
        
        # 4. Response Time by Result
        if not valid_df.empty:
            correct_predictions = valid_df[valid_df['actual_label'] == valid_df['predicted_label']]
            incorrect_predictions = valid_df[valid_df['actual_label'] != valid_df['predicted_label']]
            
            if not correct_predictions.empty:
                axes[1,1].hist(correct_predictions['response_time_seconds'], 
                              alpha=0.5, label='Correct', color='green', bins=15)
            if not incorrect_predictions.empty:
                axes[1,1].hist(incorrect_predictions['response_time_seconds'], 
                              alpha=0.5, label='Incorrect', color='red', bins=15)
            
            axes[1,1].set_title('Response Time by Prediction Accuracy')
            axes[1,1].set_xlabel('Response Time (seconds)')
            axes[1,1].set_ylabel('Frequency')
            axes[1,1].legend()
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('fewshot_vis_scout.png', dpi=300, bbox_inches='tight')
            print("Plots saved as 'few_shot_scrambling_detection_results.png'")
        
        plt.show()

def print_summary(self, metrics):
        """Print a summary of the results"""
        if metrics is None:
            print("No metrics available to summarize")
            return
        
        print("\n" + "="*60)
        print("FEW-SHOT IMAGE SCRAMBLING DETECTION SUMMARY")
        print("="*60)
        
        print(f"\nFew-Shot Configuration:")
        print(f"  Examples used: {len(self.few_shot_examples)}")
        scrambled_examples = sum(1 for ex in self.few_shot_examples if ex['label'] == 'scrambled')
        unscrambled_examples = sum(1 for ex in self.few_shot_examples if ex['label'] == 'unscrambled')
        print(f"  Scrambled examples: {scrambled_examples}")
        print(f"  Unscrambled examples: {unscrambled_examples}")
        
        print(f"\nDataset Information:")
        print(f"  Total images processed: {metrics['total_images']}")
        print(f"  Valid predictions: {metrics['valid_predictions']}")
        print(f"  Failed predictions: {metrics['total_images'] - metrics['valid_predictions']}")
        
        print(f"\nOverall Performance:")
        print(f"  Accuracy: {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
        
        print(f"\nTiming Statistics:")
        print(f"  Mean response time: {metrics['timing']['mean_response_time']:.2f} seconds")
        print(f"  Median response time: {metrics['timing']['median_response_time']:.2f} seconds")
        print(f"  Min response time: {metrics['timing']['min_response_time']:.2f} seconds")
        print(f"  Max response time: {metrics['timing']['max_response_time']:.2f} seconds")
        print(f"  Std deviation: {metrics['timing']['std_response_time']:.2f} seconds")
        
        print(f"\nDetailed Classification Report:")
        cr = metrics['classification_report']
        for label in ['scrambled', 'unscrambled']:
            if label in cr:
                print(f"  {label.capitalize()}:")
                print(f"    Precision: {cr[label]['precision']:.3f}")
                print(f"    Recall: {cr[label]['recall']:.3f}")
                print(f"    F1-score: {cr[label]['f1-score']:.3f}")
                print(f"    Support: {cr[label]['support']}")
        
        print("\n" + "="*60)

def main():
    API_KEY = api_key
    SCRAMBLED_FOLDER = "C://Users//HP//Documents//Projects//CAPTCHA_SOLVER//Scrambled copy"
    UNSCRAMBLED_FOLDER = "C://Users//HP//Documents//Projects//CAPTCHA_SOLVER//Unscambled"

    few_shots = [
        {"path": Path("C://Users//HP//Documents//Projects//CAPTCHA_SOLVER//FewShots//t4s.png"), "label": "scrambled"},
        {"path": Path("C://Users//HP//Documents//Projects//CAPTCHA_SOLVER//FewShots//t5s.png"), "label": "scrambled"},
        # {"path": Path("C://Users//HP//Documents//Projects//CAPTCHA_SOLVER//FewShots//t6s.png"), "label": "scrambled"},
        {"path": Path("C://Users//HP//Documents//Projects//CAPTCHA_SOLVER//FewShots//t1us.png"), "label": "unscrambled"},
        {"path": Path("C://Users//HP//Documents//Projects//CAPTCHA_SOLVER//FewShots//t2us.png"), "label": "unscrambled"},
        # {"path": Path("C://Users//HP//Documents//Projects//CAPTCHA_SOLVER//FewShots//t3us.png"), "label": "unscrambled"}
    ]

    detector = ImageScramblingDetectorGroq(API_KEY, SCRAMBLED_FOLDER, UNSCRAMBLED_FOLDER, few_shot_examples=few_shots)
    detector.process_all_images()
    df = detector.create_dataframe()
    if df is not None:
        df.to_csv("groq_scout_fewshots.csv", index=False)
        metrics = detector.calculate_metrics(df)
        detector.visualize_results(df, metrics)
        detector.print_summary(metrics)

if __name__ == "__main__":
    main()
