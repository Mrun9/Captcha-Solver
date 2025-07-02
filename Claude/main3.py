import os
import time
import base64
import pandas as pd
import requests
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import json
from datetime import datetime
import anthropic
from config import get_api_key 
import random
#client = anthropic.Anthropic(api_key = get_api_key())

class FewShotImageScramblingDetector:
    def __init__(self, api_key, scrambled_folder, unscrambled_folder, custom_examples=None):
        """
        Initialize the detector with API key and folder paths
        
        Args:
            api_key (str): Anthropic API key
            scrambled_folder (str): Path to folder containing scrambled images
            unscrambled_folder (str): Path to folder containing unscrambled images
            custom_examples (dict): Custom few-shot examples in format:
                {
                    'scrambled': ['path1.jpg', 'path2.png', ...],
                    'unscrambled': ['path3.jpg', 'path4.png', ...]
                }
        """
        self.api_key = api_key
        self.scrambled_folder = Path(scrambled_folder)
        self.unscrambled_folder = Path(unscrambled_folder)
        self.custom_examples = custom_examples
        self.results = []
        
        # Supported image formats
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        
        # API endpoint
        self.api_url = "https://api.anthropic.com/v1/messages"
        
        # Few-shot examples (will be populated)
        self.few_shot_examples = []
        
    def encode_image_to_base64(self, image_path):
        """Convert image to base64 string"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None
    
    def get_image_type(self, image_path):
        """Get MIME type based on file extension"""
        extension = Path(image_path).suffix.lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.tiff': 'image/tiff',
            '.webp': 'image/webp'
        }
        return mime_types.get(extension, 'image/jpeg')
    
    def prepare_few_shot_examples(self, num_examples_per_class=3):
        """
        Prepare few-shot examples either from custom specified examples or by random selection
        
        Args:
            num_examples_per_class (int): Number of examples per class (only used if no custom examples)
        """
        self.few_shot_examples = []
        
        if self.custom_examples:
            # Use custom specified examples
            print("Using custom few-shot examples...")
            
            # Process scrambled examples
            for filename in self.custom_examples.get('scrambled', []):
                img_path = self.scrambled_folder / filename
                if img_path.exists() and img_path.suffix.lower() in self.supported_formats:
                    base64_img = self.encode_image_to_base64(img_path)
                    if base64_img:
                        self.few_shot_examples.append({
                            'image_data': base64_img,
                            'image_type': self.get_image_type(img_path),
                            'label': 'scrambled',
                            'filename': filename,
                            'full_path': str(img_path)
                        })
                        print(f"Added scrambled example: {filename}")
                    else:
                        print(f"Warning: Could not encode scrambled image: {filename}")
                else:
                    print(f"Warning: Scrambled image not found or invalid format: {filename}")
            
            # Process unscrambled examples
            for filename in self.custom_examples.get('unscrambled', []):
                img_path = self.unscrambled_folder / filename
                if img_path.exists() and img_path.suffix.lower() in self.supported_formats:
                    base64_img = self.encode_image_to_base64(img_path)
                    if base64_img:
                        self.few_shot_examples.append({
                            'image_data': base64_img,
                            'image_type': self.get_image_type(img_path),
                            'label': 'unscrambled',
                            'filename': filename,
                            'full_path': str(img_path)
                        })
                        print(f"Added unscrambled example: {filename}")
                    else:
                        print(f"Warning: Could not encode unscrambled image: {filename}")
                else:
                    print(f"Warning: Unscrambled image not found or invalid format: {filename}")
            
        else:
            # Use random selection (original behavior)
            print("Using randomly selected few-shot examples...")
            
            # Get scrambled examples
            if self.scrambled_folder.exists():
                scrambled_images = [img for img in self.scrambled_folder.iterdir() 
                                  if img.suffix.lower() in self.supported_formats]
                scrambled_samples = random.sample(scrambled_images, 
                                                min(num_examples_per_class, len(scrambled_images)))
                
                for img_path in scrambled_samples:
                    base64_img = self.encode_image_to_base64(img_path)
                    if base64_img:
                        self.few_shot_examples.append({
                            'image_data': base64_img,
                            'image_type': self.get_image_type(img_path),
                            'label': 'scrambled',
                            'filename': img_path.name,
                            'full_path': str(img_path)
                        })
            
            # Get unscrambled examples
            if self.unscrambled_folder.exists():
                unscrambled_images = [img for img in self.unscrambled_folder.iterdir() 
                                    if img.suffix.lower() in self.supported_formats]
                unscrambled_samples = random.sample(unscrambled_images, 
                                                  min(num_examples_per_class, len(unscrambled_images)))
                
                for img_path in unscrambled_samples:
                    base64_img = self.encode_image_to_base64(img_path)
                    if base64_img:
                        self.few_shot_examples.append({
                            'image_data': base64_img,
                            'image_type': self.get_image_type(img_path),
                            'label': 'unscrambled',
                            'filename': img_path.name,
                            'full_path': str(img_path)
                        })
            
            # Shuffle examples to avoid bias
            random.shuffle(self.few_shot_examples)
        
        print(f"Prepared {len(self.few_shot_examples)} few-shot examples")
        
        # Print summary of examples
        scrambled_count = sum(1 for ex in self.few_shot_examples if ex['label'] == 'scrambled')
        unscrambled_count = sum(1 for ex in self.few_shot_examples if ex['label'] == 'unscrambled')
        print(f"Examples: {scrambled_count} scrambled, {unscrambled_count} unscrambled")
        
        # Print example filenames
        print("\nSelected examples:")
        for ex in self.few_shot_examples:
            print(f"  {ex['label']}: {ex['filename']}")
    
    def create_few_shot_messages(self):
        """
        Create the few-shot learning messages with examples
        
        Returns:
            list: List of message dictionaries for few-shot learning
        """
        messages = []
        
        # Add few-shot examples
        for i, example in enumerate(self.few_shot_examples):
            # User message with example image
            user_content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": example['image_type'],
                        "data": example['image_data']
                    }
                },
                {
                    "type": "text",
                    "text": f"Examine this image and classify it as either 'scrambled' or 'unscrambled'. Respond with only the single word."
                }
            ]
            
            messages.append({
                "role": "user",
                "content": user_content
            })
            
            # Assistant response with correct classification
            messages.append({
                "role": "assistant",
                "content": example['label']
            })
        
        return messages
    
    def classify_image_with_claude(self, image_path):
        """
        Send image to Claude AI for classification using few-shot learning
        
        Returns:
            tuple: (prediction, response_time)
        """
        start_time = time.time()
        
        # Encode image
        base64_image = self.encode_image_to_base64(image_path)
        if not base64_image:
            return None, 0
        
        # Prepare the request
        headers = {
            "x-api-key": self.api_key,
            "content-type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        # Create few-shot messages
        few_shot_messages = self.create_few_shot_messages()
        
        # Add the target image to classify
        target_message = {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": self.get_image_type(image_path),
                        "data": base64_image
                    }
                },
                {
                    "type": "text",
                    "text": """Now examine this new image and classify it based on the examples shown above.

Respond with exactly one of the following words:

"scrambled" — if the image has jumbled, rearranged, or mixed-up segments that make it look distorted or disordered, similar to the scrambled examples.

"unscrambled" — if the image appears coherent, orderly, and visually normal, similar to the unscrambled examples.

Respond with ONLY the single word: "scrambled" or "unscrambled". No explanations or extra text."""
                }
            ]
        }
        
        # Combine all messages
        all_messages = few_shot_messages + [target_message]
        
        # Create the message payload
        payload = {
            "model": "claude-opus-4-20250514",
            "max_tokens": 50,
            "temperature": 0.1,  # Lower temperature for more consistent responses
            "system": """You are an expert in image analysis and classification. You have been shown several examples of scrambled and unscrambled images. 
Use the examples provided to guide your classification of new images.""",
            "messages": all_messages
        }
        
        try:
            # Make the API call
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Parse response
            result = response.json()
            prediction = result['content'][0]['text'].strip().lower()
            
            # Clean up the prediction to ensure it's either 'scrambled' or 'unscrambled'
            if 'scrambled' in prediction and 'unscrambled' not in prediction:
                prediction = 'scrambled'
            elif 'unscrambled' in prediction:
                prediction = 'unscrambled'
            else:
                # If unclear, default to unscrambled (you can change this logic)
                prediction = 'unscrambled'
                print(f"Unclear response for {image_path}: {result['content'][0]['text'].strip()}")
            
            return prediction, response_time
            
        except requests.exceptions.RequestException as e:
            print(f"API request failed for {image_path}: {e}")
            return None, time.time() - start_time
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None, time.time() - start_time
    
    def collect_images(self, exclude_examples=True):
        """
        Collect all images from both folders and create dataset
        
        Args:
            exclude_examples (bool): Whether to exclude few-shot example images from the test set
        """
        dataset = []
        example_filenames = {ex['filename'] for ex in self.few_shot_examples} if exclude_examples else set()
        
        # Collect scrambled images
        if self.scrambled_folder.exists():
            for img_path in self.scrambled_folder.iterdir():
                if (img_path.suffix.lower() in self.supported_formats and 
                    (not exclude_examples or img_path.name not in example_filenames)):
                    dataset.append({
                        'image_path': str(img_path),
                        'image_name': img_path.name,
                        'actual_label': 'scrambled',
                        'folder': 'scrambled'
                    })
        
        # Collect unscrambled images
        if self.unscrambled_folder.exists():
            for img_path in self.unscrambled_folder.iterdir():
                if (img_path.suffix.lower() in self.supported_formats and 
                    (not exclude_examples or img_path.name not in example_filenames)):
                    dataset.append({
                        'image_path': str(img_path),
                        'image_name': img_path.name,
                        'actual_label': 'unscrambled',
                        'folder': 'unscrambled'
                    })
        
        return dataset
    
    def process_all_images(self, save_progress=True, progress_file='few_shot_progress.json', 
                          num_examples_per_class=3, exclude_examples=True):
        """
        Process all images and get predictions using few-shot learning
        
        Args:
            save_progress (bool): Whether to save progress periodically
            progress_file (str): File to save progress to
            num_examples_per_class (int): Number of few-shot examples per class
            exclude_examples (bool): Whether to exclude few-shot examples from test set
        """
        # Prepare few-shot examples
        print("Preparing few-shot examples...")
        self.prepare_few_shot_examples(num_examples_per_class)
        
        # Collect test images
        dataset = self.collect_images(exclude_examples)
        
        print(f"Found {len(dataset)} images to process")
        print(f"Scrambled folder: {len([d for d in dataset if d['actual_label'] == 'scrambled'])} images")
        print(f"Unscrambled folder: {len([d for d in dataset if d['actual_label'] == 'unscrambled'])} images")
        
        if exclude_examples:
            print(f"Excluded {len(self.few_shot_examples)} images used as few-shot examples")
        
        # Load previous progress if exists
        processed_images = set()
        if save_progress and os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
                    self.results = progress_data.get('results', [])
                    processed_images = set(progress_data.get('processed_images', []))
                print(f"Loaded {len(processed_images)} previously processed images")
            except Exception as e:
                print(f"Error loading progress: {e}")
        
        total_images = len(dataset)
        
        for i, img_data in enumerate(dataset, 1):
            image_path = img_data['image_path']
            
            # Skip if already processed
            if image_path in processed_images:
                print(f"Skipping already processed image {i}/{total_images}: {img_data['image_name']}")
                continue
            
            print(f"Processing image {i}/{total_images}: {img_data['image_name']}")
            
            # Get prediction from Claude with few-shot learning
            prediction, response_time = self.classify_image_with_claude(image_path)
            
            # Store result
            result = {
                'image_name': img_data['image_name'],
                'image_path': image_path,
                'actual_label': img_data['actual_label'],
                'predicted_label': prediction,
                'response_time_seconds': response_time,
                'folder': img_data['folder'],
                'timestamp': datetime.now().isoformat(),
                'few_shot_examples_used': len(self.few_shot_examples)
            }
            
            self.results.append(result)
            processed_images.add(image_path)
            
            # Save progress periodically
            if save_progress and i % 5 == 0:  # Save every 5 images
                progress_data = {
                    'results': self.results,
                    'processed_images': list(processed_images),
                    'few_shot_examples': [
                        {k: v for k, v in ex.items() if k != 'image_data'}  # Don't save base64 data
                        for ex in self.few_shot_examples
                    ],
                    'last_updated': datetime.now().isoformat()
                }
                with open(progress_file, 'w') as f:
                    json.dump(progress_data, f, indent=2)
                print(f"Progress saved after {i} images")
            
            # Small delay to be respectful to the API
            time.sleep(1)
        
        # Final save
        if save_progress:
            progress_data = {
                'results': self.results,
                'processed_images': list(processed_images),
                'few_shot_examples': [
                    {k: v for k, v in ex.items() if k != 'image_data'}  # Don't save base64 data
                    for ex in self.few_shot_examples
                ],
                'last_updated': datetime.now().isoformat()
            }
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
        
        print("All images processed!")
        return self.results
    
    def create_dataframe(self):
        """Convert results to pandas DataFrame"""
        if not self.results:
            print("No results to convert. Run process_all_images() first.")
            return None
        
        df = pd.DataFrame(self.results)
        
        # Clean up any None predictions
        df['predicted_label'] = df['predicted_label'].fillna('unknown')
        
        return df
    
    def calculate_metrics(self, df):
        """Calculate accuracy metrics and create confusion matrix"""
        if df is None or df.empty:
            print("No data available for metrics calculation")
            return None
        
        # Filter out any unknown predictions for metrics calculation
        valid_df = df[df['predicted_label'].isin(['scrambled', 'unscrambled'])].copy()
        
        if valid_df.empty:
            print("No valid predictions found")
            return None
        
        # Calculate basic metrics
        accuracy = accuracy_score(valid_df['actual_label'], valid_df['predicted_label'])
        
        # Generate classification report
        class_report = classification_report(valid_df['actual_label'], valid_df['predicted_label'], 
                                           target_names=['scrambled', 'unscrambled'],
                                           output_dict=True)
        
        # Create confusion matrix
        cm = confusion_matrix(valid_df['actual_label'], valid_df['predicted_label'], 
                            labels=['scrambled', 'unscrambled'])
        
        # Calculate timing statistics
        timing_stats = {
            'mean_response_time': valid_df['response_time_seconds'].mean(),
            'median_response_time': valid_df['response_time_seconds'].median(),
            'min_response_time': valid_df['response_time_seconds'].min(),
            'max_response_time': valid_df['response_time_seconds'].max(),
            'std_response_time': valid_df['response_time_seconds'].std()
        }
        
        return {
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'timing_stats': timing_stats,
            'valid_predictions': len(valid_df),
            'total_images': len(df)
        }
    
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
        axes[0,1].axvline(metrics['timing_stats']['mean_response_time'], 
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
            plt.savefig('few_shot_scrambling_detection_results.png', dpi=300, bbox_inches='tight')
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
        print(f"  Mean response time: {metrics['timing_stats']['mean_response_time']:.2f} seconds")
        print(f"  Median response time: {metrics['timing_stats']['median_response_time']:.2f} seconds")
        print(f"  Min response time: {metrics['timing_stats']['min_response_time']:.2f} seconds")
        print(f"  Max response time: {metrics['timing_stats']['max_response_time']:.2f} seconds")
        print(f"  Std deviation: {metrics['timing_stats']['std_response_time']:.2f} seconds")
        
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
    """Main function to run the few-shot image scrambling detection"""
    
    # Configuration - UPDATE THESE PATHS AND API KEY
    API_KEY = get_api_key()  # Replace with your actual API key
    SCRAMBLED_FOLDER = "C://Users//HP//Documents//Projects//CAPTCHA_SOLVER//Scrambled copy"  # Replace with your scrambled images folder
    UNSCRAMBLED_FOLDER = "C://Users//HP//Documents//Projects//CAPTCHA_SOLVER//Unscambled"  # Replace with your unscrambled images folder
    
    # Few-shot configuration
    NUM_EXAMPLES_PER_CLASS = 3  # Number of examples per class (only used if CUSTOM_EXAMPLES is None)
    EXCLUDE_EXAMPLES_FROM_TEST = True  # Whether to exclude few-shot examples from test set
    
    # CUSTOM FEW-SHOT EXAMPLES - Specify your own examples here
    # Set to None to use random selection, or specify exact filenames like below:
    CUSTOM_EXAMPLES = {
        'scrambled': [
            't4s.png',      # Replace with your actual scrambled image filenames
            't5s.png',      # These should be files in your SCRAMBLED_FOLDER
            't6s.png',
        ],
        'unscrambled': [
            't1us.png',         # Replace with your actual unscrambled image filenames  
            't2us.png',         # These should be files in your UNSCRAMBLED_FOLDER
            't3us.png',
        ]
    }
    
    # To use random selection instead, uncomment the line below:
    # CUSTOM_EXAMPLES = None
    
    # Validate inputs
    if not os.path.exists(SCRAMBLED_FOLDER):
        print(f"Error: Scrambled folder not found: {SCRAMBLED_FOLDER}")
        return
    
    if not os.path.exists(UNSCRAMBLED_FOLDER):
        print(f"Error: Unscrambled folder not found: {UNSCRAMBLED_FOLDER}")
        return
    
    # Validate custom examples if provided
    if CUSTOM_EXAMPLES:
        print("Validating custom examples...")
        scrambled_folder_path = Path(SCRAMBLED_FOLDER)
        unscrambled_folder_path = Path(UNSCRAMBLED_FOLDER)
        
        for filename in CUSTOM_EXAMPLES.get('scrambled', []):
            if not (scrambled_folder_path / filename).exists():
                print(f"Warning: Scrambled example not found: {filename}")
        
        for filename in CUSTOM_EXAMPLES.get('unscrambled', []):
            if not (unscrambled_folder_path / filename).exists():
                print(f"Warning: Unscrambled example not found: {filename}")
    
    # Initialize detector with custom examples
    detector = FewShotImageScramblingDetector(
        API_KEY, 
        SCRAMBLED_FOLDER, 
        UNSCRAMBLED_FOLDER,
        custom_examples=CUSTOM_EXAMPLES
    )
    
    # Process all images with few-shot learning
    print("Starting few-shot image processing...")
    results = detector.process_all_images(
        num_examples_per_class=NUM_EXAMPLES_PER_CLASS,
        exclude_examples=EXCLUDE_EXAMPLES_FROM_TEST
    )
    
    # Create DataFrame
    df = detector.create_dataframe()
    
    # Save results to CSV
    if df is not None:
        df.to_csv('few_shot_results.csv', index=False)
        print("Results saved to 'few_shot_results.csv'")
    
    # Calculate metrics
    metrics = detector.calculate_metrics(df)
    
    # Print summary
    detector.print_summary(metrics)
    
    # Create visualizations
    detector.visualize_results(df, metrics)
    
    print("\nFew-shot analysis complete!")

if __name__ == "__main__":
    main()