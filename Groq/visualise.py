import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def load_results(csv_path):
    return pd.read_csv(csv_path)

def calculate_metrics(df):
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
    return {
        "accuracy": accuracy,
        "classification_report": class_report,
        "confusion_matrix": cm,
        "timing": timing,
        "valid_predictions": len(valid_df),
        "total_images": len(df)
    }

def visualize_results(df, metrics, save_path="gpt4o_vis.png"):
    valid_df = df[df["predicted_label"].isin(["scrambled", "unscrambled"])]
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Few-Shot Image Scrambling Detection Results', fontsize=16, fontweight='bold')

    # Confusion Matrix
    sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                xticklabels=["scrambled", "unscrambled"],
                yticklabels=["scrambled", "unscrambled"],
                ax=axes[0, 0])
    axes[0, 0].set_title("Confusion Matrix")
    axes[0, 0].set_xlabel("Predicted")
    axes[0, 0].set_ylabel("Actual")

    # Response Time Distribution
    axes[0, 1].hist(df["response_time_seconds"], bins=20, color="skyblue", edgecolor="black", alpha=0.7)
    axes[0, 1].set_title("Response Time Distribution")
    axes[0, 1].set_xlabel("Seconds")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].axvline(metrics["timing"]["mean"], color='red', linestyle='--', label='Mean')
    axes[0, 1].legend()

    # Accuracy by Class
    accuracy_by_class = valid_df.groupby("actual_label").apply(
        lambda x: (x["actual_label"] == x["predicted_label"]).mean()
    )
    bars = axes[1, 0].bar(accuracy_by_class.index, accuracy_by_class.values, color=['coral', 'lightgreen'], alpha=0.7)
    axes[1, 0].set_title("Accuracy by Class")
    axes[1, 0].set_ylim(0, 1)
    for bar, acc in zip(bars, accuracy_by_class.values):
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{acc:.2f}", ha='center')

    # Response Time for Correct vs Incorrect
    correct = valid_df[valid_df["actual_label"] == valid_df["predicted_label"]]
    incorrect = valid_df[valid_df["actual_label"] != valid_df["predicted_label"]]
    if not correct.empty:
        axes[1, 1].hist(correct["response_time_seconds"], bins=15, alpha=0.5, label="Correct", color="green")
    if not incorrect.empty:
        axes[1, 1].hist(incorrect["response_time_seconds"], bins=15, alpha=0.5, label="Incorrect", color="red")
    axes[1, 1].set_title("Response Time by Accuracy")
    axes[1, 1].set_xlabel("Seconds")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Visualization saved as {save_path}")
    plt.show()

def print_summary(metrics):
    print("\n" + "="*60)
    print("IMAGE SCRAMBLING DETECTION SUMMARY")
    print("="*60)
    print(f"\nTotal Images: {metrics['total_images']}")
    print(f"Valid Predictions: {metrics['valid_predictions']}")
    print(f"Accuracy: {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
    
    print(f"\nTiming:")
    print(f"  Mean   : {metrics['timing']['mean']:.2f}s")
    print(f"  Median : {metrics['timing']['median']:.2f}s")
    print(f"  Min    : {metrics['timing']['min']:.2f}s")
    print(f"  Max    : {metrics['timing']['max']:.2f}s")
    print(f"  Std    : {metrics['timing']['std']:.2f}s")

    print(f"\nClassification Report:")
    for label in ['scrambled', 'unscrambled']:
        if label in metrics["classification_report"]:
            cr = metrics["classification_report"][label]
            print(f"  {label.capitalize()}:")
            print(f"    Precision: {cr['precision']:.3f}")
            print(f"    Recall   : {cr['recall']:.3f}")
            print(f"    F1-Score : {cr['f1-score']:.3f}")
            print(f"    Support  : {cr['support']}")

    print("="*60)

def main():
    csv_path = "gpt4o_results(0).csv"  # replace with your actual CSV file
    df = load_results(csv_path)
    metrics = calculate_metrics(df)
    print_summary(metrics)
    visualize_results(df, metrics)

if __name__ == "__main__":
    main()
