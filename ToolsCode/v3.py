import matplotlib.pyplot as plt
import numpy as np

# Data
models = ["Claude 3", "Claude 3.5", "Claude 3.7", "Sonnet 4", "Opus 4", 
          "Scout", "Maverick", "gpt-4o", "gpt-4.1", "gpt-4.1-mini"]
scrambled_f1 = [0.574, 0.681, 0.833, 0.731, 0.721, 0.645, 0.674, 0.684, 0.747, 0.619]
unscrambled_f1 = [0.424, 0.554, 0.779, 0.708, 0.679, 0.210, 0.248, 0.503, 0.521, 0.347]

x = np.arange(len(models))
bar_width = 0.35

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)

# Bar plots
bars1 = ax.bar(x - bar_width/2, scrambled_f1, width=bar_width, color="#E69F00", label='Scrambled')
bars2 = ax.bar(x + bar_width/2, unscrambled_f1, width=bar_width, color="#56B4E9", label='Unscrambled')

# Labels & styling
ax.set_ylabel('F1 Score', color='black')
ax.set_xlabel('Model', color='black')
ax.set_title('F1 Score per Class for Each Model', color='black')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15)
ax.set_ylim(0, 1.0)
ax.tick_params(colors='black')
ax.legend()

# Annotate
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8, color='black')

plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()
