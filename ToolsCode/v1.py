import matplotlib.pyplot as plt

# Data
models = ["Claude 3", "Claude 3.5", "Claude 3.7", "Sonnet 4", "Opus 4", 
          "Scout", "Maverick", "gpt-4o", "gpt-4.1", "gpt-4.1-mini"]
fewshot_times = [1.55, 2.35, 2.15, 4.47, 3.02, 0.39, 0.71, 1.55, 2.93, 1.12]
fewshot_acc = [0.535, 0.710, 0.810, 0.720, 0.702, 0.527, 0.571, 0.667, 0.685, 0.607]

# Plot
fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
for i in range(len(models)):
    ax.scatter(fewshot_times[i], fewshot_acc[i], s=100, color="#E69F00")
    ax.text(fewshot_times[i] + 0.05, fewshot_acc[i], models[i], fontsize=8, color='black')

# Styling
ax.set_xlabel("Few-Shot Mean Inference Time (seconds)", color='black')
ax.set_ylabel("Accuracy", color='black')
ax.set_title("Inference Time vs Accuracy Trade-off (Few-Shot Configuration Only)", color='black')
ax.set_xlim(0, max(fewshot_times) + 0.5)
ax.set_ylim(0.45, 0.85)
ax.grid(True, linestyle='--', alpha=0.5)
ax.tick_params(colors='black')
plt.tight_layout()
plt.show()
