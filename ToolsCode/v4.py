import matplotlib.pyplot as plt

# Data
models = ["Claude 3", "Claude 3.5", "Claude 3.7", "Sonnet 4", "Opus 4", 
          "Scout", "Maverick", "gpt-4o", "gpt-4.1", "gpt-4.1-mini"]
prompt_times = [0.97, 1.71, 1.41, 2.87, 2.94, 0.19, 0.45, 1.12, 2.38, 0.94]
few_shot_times = [1.55, 2.35, 2.15, 4.47, 3.02, 0.39, 0.71, 1.55, 2.93, 1.12]

# Reverse for top-down Y-axis
models = models[::-1]
prompt_times = prompt_times[::-1]
few_shot_times = few_shot_times[::-1]
y_pos = list(range(len(models)))

# Plot
fig, ax = plt.subplots(figsize=(8, 5), dpi=200)

# Step 1: Background dashed grey lines
for y in y_pos:
    ax.axhline(y=y, color='lightgray', linestyle='--', linewidth=0.6, zorder=0)

# Step 2: Solid black line between prompt and few-shot, X markers
for i in range(len(models)):
    ax.plot([prompt_times[i], few_shot_times[i]], [i, i], color='black', linestyle='-', linewidth=1.0, zorder=2)
    ax.scatter(prompt_times[i], i, color="#56B4E9", s=80, marker='x', label='Prompt Only' if i == 0 else "", zorder=3)
    ax.scatter(few_shot_times[i], i, color="#E69F00", s=80, marker='x', label='Few-Shot' if i == 0 else "", zorder=3)

# Style
ax.set_yticks(y_pos)
ax.set_yticklabels(models, fontsize=9)
ax.set_xlabel("Mean Inference Time (seconds)", color='black')
ax.set_title("Prompt vs Few-Shot Inference Time per Model", color='black')
ax.tick_params(colors='black')
ax.grid(axis='x', linestyle='--', alpha=0.4)
ax.legend(loc='lower right')

plt.tight_layout()
plt.show()
