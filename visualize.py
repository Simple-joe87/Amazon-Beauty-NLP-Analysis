
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Locate analysis results
results_dir = Path("results")
topics = pd.read_csv(results_dir / "topics.csv")

# Create one chart for each topic model
for model in ["LDA", "NMF"]:
    data = topics[topics["model"] == model].copy()
    data = data.sort_values("topic")

    plt.figure(figsize=(10, 5))
    plt.bar(
        data["topic"].astype(str),
        data["percent"]
    )

    plt.title(f"{model} - Distribution of Review Topics")
    plt.xlabel("Topic Number")
    plt.ylabel("Percentage of Reviews (%)")
    plt.tight_layout()

    filename = results_dir / f"{model.lower()}_topics.png"
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Saved: {filename}")

print("All topic visualizations created successfully.")
