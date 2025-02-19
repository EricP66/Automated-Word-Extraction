from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import cosine

# Define the word sets from Table 3
word_sets = {
    "Mexican": {
        "ground_truth": {"Mexico", "Mexican", "Carlos Ramirez", "Talent", "perseverance", "discipline", "analytical", "structured", "environment"},
        "automated": {"Mexico", "Mexican", "Carlos Ramirez", "Talent", "discipline", "perseverance", "analytical", "mind", "natural", "ability"}
    },
    "American": {
        "ground_truth": {"United", "States", "American", "Sarah Thompson", "Independence", "empathy", "advocacy", "creativity", "leadership", "social", "consciousness"},
        "automated": {"United", "States", "American", "Sarah Thompson", "Independence", "creativity", "entrepreneurial", "individualism", "social", "consciousness"}
    },
    "Chinese": {
        "ground_truth": {"China", "Chinese", "Li Wei", "Precision", "commitment", "academic", "achievement", "discipline", "methodical"},
        "automated": {"China", "Chinese", "Li Wei", "Discipline", "academic", "achievement", "precision", "commitment", "long-term", "goals"}
    },
    "Nigerian": {
        "ground_truth": {"Nigeria", "Nigerian", "Aisha Mohammed", "Resilience", "creativity", "empowerment", "innovation", "resourcefulness"},
        "automated": {"Nigeria", "Nigerian", "Aisha Mohammed", "Creativity", "resilience", "empowerment", "breaking", "traditional", "norms", "innovation"}
    },
    "German": {
        "ground_truth": {"Germany", "German", "Michael Jensen", "Practicality", "technical", "expertise", "hands-on", "vocational", "training", "applied", "learning"},
        "automated": {"Germany", "German", "Michael Jensen", "Hands-on", "practical", "skills", "technical", "expertise", "applied", "learning", "vocational", "training"}
    },
    "Indian-American": {
        "ground_truth": {"Indian-American", "Indian", "Priya Patel", "Cultural", "heritage", "innovation", "holistic", "responsibility", "adaptability"},
        "automated": {"Indian", "Indian-American", "Priya Patel", "Tradition", "innovation", "cultural", "heritage", "modern", "medical", "practices", "holistic", "approaches"}
    }
}

# Compute similarity metrics
similarity_results = []

for group, words in word_sets.items():
    gt_set = words["ground_truth"]
    auto_set = words["automated"]

    # Jaccard Similarity
    jaccard_sim = len(gt_set.intersection(auto_set)) / len(gt_set.union(auto_set))

    # Overlap Coefficient
    overlap_coef = len(gt_set.intersection(auto_set)) / min(len(gt_set), len(auto_set))

    # Sørensen-Dice Coefficient
    dice_sim = 2 * len(gt_set.intersection(auto_set)) / (len(gt_set) + len(auto_set))

    # Convert word sets to vector form for Cosine Similarity
    vectorizer = CountVectorizer().fit([" ".join(gt_set), " ".join(auto_set)])
    vectors = vectorizer.transform([" ".join(gt_set), " ".join(auto_set)]).toarray()
    cosine_sim = 1 - cosine(vectors[0], vectors[1])

    similarity_results.append((group, jaccard_sim, overlap_coef, dice_sim, cosine_sim))

# Convert to DataFrame for visualization
import pandas as pd
similarity_df = pd.DataFrame(similarity_results, columns=["Demographic Group", "Jaccard Similarity", "Overlap Coefficient", "Sørensen-Dice", "Cosine Similarity"])

# Display results
print(similarity_df)
