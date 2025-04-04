import numpy as np
from scipy.stats import pearsonr

# Ground truth CEAT scores
ground_truth_scores = np.array([-0.1274, 0.0428, 0.2301, -0.1664])

# Automated extraction CEAT scores
automated_scores = np.array([-0.1014, 0.0191, 0.2406, -0.1721])

# Calculate Pearson correlation coefficient
pearson_corr, _ = pearsonr(ground_truth_scores, automated_scores)

# Print the result
print(f"Pearson Correlation Coefficient: {pearson_corr:.4f}")
