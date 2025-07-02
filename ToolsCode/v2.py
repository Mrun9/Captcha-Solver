import matplotlib.pyplot as plt

# Data
models = ["Claude 3", "Claude 3.5", "Claude 3.7", "Sonnet 4", "Opus 4", 
          "Scout", "Maverick", "gpt-4o", "gpt-4.1", "gpt-4.1-mini"]
prompt_failures = [0, 1, 0, 0, 86, 0, 0, 0, 0, 0]
fewshot_failures = [0] * 10
total_images = 200

# Convert to % of 200
prompt_percent = [f / total_images * 100 for f in prompt_failures]
fewshot_percent = [f / total_images * 100 for f in fewshot_failures]

# Y positions
y_pos = list(range(len(models)))

# Create plot
fig, ax = plt.subplots(figsize=(8, 5), dpi=200)

# Step 1: Add light gray dashed background lines
for y in y_pos:
    ax.axhline(y=y, color='lightgray', linestyle='--', linewidth=0.6, zorder=0)

# Step 2: Plot dots
ax.scatter(prompt_percent, y_pos, color="#56B4E9", s=100, marker='x', label="Prompt Only", zorder=2)
ax.scatter(fewshot_percent, y_pos, color="#E69F00", s=80, marker='+', label="Few-Shot", zorder=2)

# Step 3: Axes and labels
ax.set_yticks(y_pos)
ax.set_yticklabels(models, fontsize=9)
ax.set_xlabel("Failed Predictions (%)", color='black')
ax.set_title("Failed Predictions (as % of Total) – Prompt vs Few-Shot", color='black')
ax.set_xlim(-1, max(prompt_percent) + 10)
ax.tick_params(colors='black')
ax.grid(axis='x', linestyle='--', alpha=0.5)
ax.legend(loc="upper right")

# Step 4: Annotate failure values
for i in range(len(models)):
    if prompt_percent[i] > 0:
        ax.annotate(f"{prompt_percent[i]:.1f}%", (prompt_percent[i] + 1, y_pos[i]),
                    fontsize=8, color='black')
    if fewshot_percent[i] > 0:
        ax.annotate(f"{fewshot_percent[i]:.1f}%", (fewshot_percent[i] + 1, y_pos[i]),
                    fontsize=8, color='black')

# Final layout
plt.tight_layout(pad=1)
plt.show()
