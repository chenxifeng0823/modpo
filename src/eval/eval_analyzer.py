import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from collections import defaultdict
import matplotlib.pyplot as plt
import re
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationStats:
    """Class to store statistics for evaluation metrics."""
    mean_reward: float
    std_reward: float
    mean_multi_scores: Dict[str, float]
    std_multi_scores: Dict[str, float]

class EvaluationAnalyzer:
    def __init__(self):
        """Initialize the evaluation analyzer."""
        self.colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
        self.color_index = 0

    def extract_model_name(self, file_path: str) -> str:
        """Extract model name from file path."""
        # Extract the part between 'lm/' and '/gen'
        match = re.search(r'lm/(.*?)/gen', file_path)
        if match:
            return match.group(1)
        return Path(file_path).stem

    def plot_helpfulness_safety(self, file_paths: List[str], output_path: str = "helpfulness_safety_plot.pdf", standardize: bool = False, normalize: bool = False):
        """
        Create a scatter plot comparing helpfulness and safety scores across different models.
        
        Args:
            file_paths: List of paths to evaluation files
            output_path: Path to save the plot
            standardize: Whether to standardize the scores (z-score normalization)
            normalize: Whether to normalize the scores to [0,1] range
        """
        plt.figure(figsize=(10, 8))
        
        # Collect all scores for standardization/normalization if needed
        all_helpfulness = []
        all_safety = []
        model_scores = {}  # Store scores by model for plotting
        
        for file_path in file_paths:
            results = self.load_eval_file(file_path)
            if not results:
                continue

            # Extract helpfulness and safety scores
            helpfulness_scores = []
            safety_scores = []
            
            for result in results:
                eval_data = result["evaluation"]
                multi_scores = eval_data["multi_reward_scores"]
                helpfulness_scores.append(multi_scores.get("helpfulness", 0))
                safety_scores.append(multi_scores.get("safety", 0))
            
            model_name = self.extract_model_name(file_path)
            model_scores[model_name] = {
                'helpfulness': helpfulness_scores,
                'safety': safety_scores
            }
            
            all_helpfulness.extend(helpfulness_scores)
            all_safety.extend(safety_scores)
        
        # Compute standardization/normalization parameters if needed
        if standardize:
            helpfulness_mean = np.mean(all_helpfulness)
            helpfulness_std = np.std(all_helpfulness)
            safety_mean = np.mean(all_safety)
            safety_std = np.std(all_safety)
        elif normalize:
            helpfulness_min = min(all_helpfulness)
            helpfulness_max = max(all_helpfulness)
            safety_min = min(all_safety)
            safety_max = max(all_safety)
        
        # Plot each model's data
        for model_name, scores in model_scores.items():
            helpfulness_scores = scores['helpfulness']
            safety_scores = scores['safety']
            
            if standardize:
                helpfulness_scores = [(x - helpfulness_mean) / helpfulness_std for x in helpfulness_scores]
                safety_scores = [(x - safety_mean) / safety_std for x in safety_scores]
            elif normalize:
                helpfulness_scores = [(x - helpfulness_min) / (helpfulness_max - helpfulness_min) for x in helpfulness_scores]
                safety_scores = [(x - safety_min) / (safety_max - safety_min) for x in safety_scores]
            
            color = self.colors[self.color_index % len(self.colors)]
            self.color_index += 1
            
            # Plot scatter points
            plt.scatter(
                helpfulness_scores,
                safety_scores,
                label=model_name,
                color=color,
                alpha=0.6
            )

            # Plot mean point
            mean_helpfulness = np.mean(helpfulness_scores)
            mean_safety = np.mean(safety_scores)
            plt.scatter(
                [mean_helpfulness],
                [mean_safety],
                color=color,
                marker='*',
                s=200,
                label=f"{model_name} (mean)"
            )

        # Set axis labels and title based on transformation type
        if standardize:
            xlabel = 'Helpfulness Score (Standardized)'
            ylabel = 'Safety Score (Standardized)'
            title = 'Helpfulness vs Safety Scores Across Models (Standardized)'
        elif normalize:
            xlabel = 'Helpfulness Score (Normalized)'
            ylabel = 'Safety Score (Normalized)'
            title = 'Helpfulness vs Safety Scores Across Models (Normalized)'
        else:
            xlabel = 'Helpfulness Score'
            ylabel = 'Safety Score'
            title = 'Helpfulness vs Safety Scores Across Models'

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(output_path, bbox_inches='tight', format='pdf')
        plt.close()
        logger.info(f"Plot saved to {output_path}")

    def load_eval_file(self, file_path: str) -> List[Dict]:
        """Load evaluation results from a JSON file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading evaluation file {file_path}: {str(e)}")
            return []

    def compute_stats(self, eval_results: List[Dict]) -> EvaluationStats:
        """
        Compute mean and standard deviation for all scores in the evaluation results.
        
        Args:
            eval_results: List of evaluation results
            
        Returns:
            EvaluationStats object containing mean and std for all metrics
        """
        if not eval_results:
            return EvaluationStats(0.0, 0.0, {}, {})

        # Extract all scores
        reward_scores = []
        multi_scores = defaultdict(list)
        
        for result in eval_results:
            eval_data = result["evaluation"]
            reward_scores.append(eval_data["reward_score"])
            
            for dimension, score in eval_data["multi_reward_scores"].items():
                multi_scores[dimension].append(score)

        # Compute statistics
        mean_reward = np.mean(reward_scores)
        std_reward = np.std(reward_scores)
        
        mean_multi_scores = {dim: np.mean(scores) for dim, scores in multi_scores.items()}
        std_multi_scores = {dim: np.std(scores) for dim, scores in multi_scores.items()}

        return EvaluationStats(
            mean_reward=mean_reward,
            std_reward=std_reward,
            mean_multi_scores=mean_multi_scores,
            std_multi_scores=std_multi_scores
        )

    def compare_files(self, base_file: str, test_file: str) -> Dict[str, Dict[str, float]]:
        """
        Compare two evaluation files and compute winning rates for each dimension.
        The rates show the probability that the test file performs better than the base file.
        
        Args:
            base_file: Path to base evaluation file (reference)
            test_file: Path to test evaluation file (to compare against base)
            
        Returns:
            Dictionary containing winning rates for each dimension
        """
        results1 = self.load_eval_file(base_file)
        results2 = self.load_eval_file(test_file)
        
        if not results1 or not results2:
            logger.error("Could not load one or both evaluation files")
            return {}

        # Create dictionaries mapping prompts to scores
        base_scores = {r["original_content"]["user_query"]: r["evaluation"] for r in results1}
        test_scores = {r["original_content"]["user_query"]: r["evaluation"] for r in results2}

        # Find common prompts
        common_prompts = set(base_scores.keys()) & set(test_scores.keys())
        
        if not common_prompts:
            logger.warning("No common prompts found between the two files")
            return {}

        # Initialize counters for each dimension
        dimensions = ["reward_score"] + list(next(iter(base_scores.values()))["multi_reward_scores"].keys())
        wins = {dim: {"test_better": 0, "base_better": 0, "tie": 0} for dim in dimensions}
        
        # Compare scores for each prompt
        for prompt in common_prompts:
            base_eval = base_scores[prompt]
            test_eval = test_scores[prompt]
            
            # Compare reward score
            if test_eval["reward_score"] > base_eval["reward_score"]:
                wins["reward_score"]["test_better"] += 1
            elif test_eval["reward_score"] < base_eval["reward_score"]:
                wins["reward_score"]["base_better"] += 1
            else:
                wins["reward_score"]["tie"] += 1
            
            # Compare multi-dimensional scores
            for dim in base_eval["multi_reward_scores"]:
                base_score = base_eval["multi_reward_scores"][dim]
                test_score = test_eval["multi_reward_scores"][dim]
                
                if test_score > base_score:
                    wins[dim]["test_better"] += 1
                elif test_score < base_score:
                    wins[dim]["base_better"] += 1
                else:
                    wins[dim]["tie"] += 1

        # Compute winning rates
        total = len(common_prompts)
        winning_rates = {}
        
        for dim in dimensions:
            winning_rates[dim] = {
                "test_better_rate": wins[dim]["test_better"] / total,
                "base_better_rate": wins[dim]["base_better"] / total,
                "tie_rate": wins[dim]["tie"] / total
            }

        return winning_rates

    def analyze_file(self, file_path: str) -> None:
        """
        Analyze a single evaluation file and print statistics.
        
        Args:
            file_path: Path to evaluation file
        """
        results = self.load_eval_file(file_path)
        if not results:
            return

        stats = self.compute_stats(results)
        
        print(f"\nAnalysis for {file_path}:")
        print(f"Number of samples: {len(results)}")
        print("\nReward Score:")
        print(f"  Mean: {stats.mean_reward:.4f}")
        print(f"  Std:  {stats.std_reward:.4f}")
        
        print("\nMulti-dimensional Scores:")
        for dim in stats.mean_multi_scores:
            print(f"\n{dim.upper()}:")
            print(f"  Mean: {stats.mean_multi_scores[dim]:.4f}")
            print(f"  Std:  {stats.std_multi_scores[dim]:.4f}")

    def compare_and_print(self, base_file: str, test_file: str) -> None:
        """
        Compare two evaluation files and print winning rates.
        Shows the probability that the test file performs better than the base file.
        
        Args:
            base_file: Path to base evaluation file (reference)
            test_file: Path to test evaluation file (to compare against base)
        """
        winning_rates = self.compare_files(base_file, test_file)
        if not winning_rates:
            return

        print(f"\nComparison between:")
        print(f"Base file: {base_file}")
        print(f"Test file: {test_file}")
        print(f"Number of common prompts: {len(self.load_eval_file(base_file))}")
        
        for dim, rates in winning_rates.items():
            print(f"\n{dim.upper()}:")
            print(f"  Test better rate: {rates['test_better_rate']:.2%}")
            print(f"  Base better rate: {rates['base_better_rate']:.2%}")
            print(f"  Tie rate: {rates['tie_rate']:.2%}")

    def plot_pca(self, file_paths: List[str], output_path: str = "helpfulness_safety_pca.pdf"):
        """
        Perform PCA on helpfulness and safety scores and create a scatter plot.
        
        Args:
            file_paths: List of paths to evaluation files
            output_path: Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        # Collect all scores
        all_scores = []
        model_scores = {}  # Store scores by model for plotting
        
        for file_path in file_paths:
            results = self.load_eval_file(file_path)
            if not results:
                continue

            # Extract helpfulness and safety scores
            scores = []
            for result in results:
                eval_data = result["evaluation"]
                multi_scores = eval_data["multi_reward_scores"]
                scores.append([
                    multi_scores.get("helpfulness", 0),
                    multi_scores.get("safety", 0)
                ])
            
            model_name = self.extract_model_name(file_path)
            model_scores[model_name] = np.array(scores)
            all_scores.extend(scores)
        
        # Convert to numpy array
        all_scores = np.array(all_scores)
        
        # Perform PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(all_scores)
        
        # Plot each model's data
        start_idx = 0
        for model_name, scores in model_scores.items():
            n_samples = len(scores)
            model_pca = pca_result[start_idx:start_idx + n_samples]
            start_idx += n_samples
            
            color = self.colors[self.color_index % len(self.colors)]
            self.color_index += 1
            
            # Plot scatter points
            plt.scatter(
                model_pca[:, 0],
                model_pca[:, 1],
                label=model_name,
                color=color,
                alpha=0.6
            )

            # Plot mean point
            mean_pca = np.mean(model_pca, axis=0)
            plt.scatter(
                [mean_pca[0]],
                [mean_pca[1]],
                color=color,
                marker='*',
                s=200,
                label=f"{model_name} (mean)"
            )

        # Add explained variance ratio to axis labels
        var_ratio = pca.explained_variance_ratio_
        plt.xlabel(f'PC1 ({var_ratio[0]:.1%} variance explained)')
        plt.ylabel(f'PC2 ({var_ratio[1]:.1%} variance explained)')
        plt.title('PCA of Helpfulness and Safety Scores')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(output_path, bbox_inches='tight', format='pdf')
        plt.close()
        logger.info(f"PCA plot saved to {output_path}")
        
        # Print PCA information
        print("\nPCA Information:")
        print(f"Total variance explained: {sum(var_ratio):.1%}")
        print("\nPrincipal Components:")
        for i, (var, comp) in enumerate(zip(var_ratio, pca.components_)):
            print(f"PC{i+1}:")
            print(f"  Variance explained: {var:.1%}")
            print(f"  Component weights: [helpfulness: {comp[0]:.3f}, safety: {comp[1]:.3f}]")

def main():
    """Example usage of the evaluation analyzer."""
    analyzer = EvaluationAnalyzer()
    
    # Example: Analyze multiple files and create scatter plots
    file_paths = [
        "/Users/wan/Desktop/CS224R/project/output/PKU-Alignment/PKU-SafeRLHF-10K/modpo/lm/(0.1)better+(1-0.1)safer/gen/00001-of-00001_eval.json",
        "/Users/wan/Desktop/CS224R/project/output/PKU-Alignment/PKU-SafeRLHF-10K/modpo/lm/(0.5)better+(1-0.5)safer/gen/00001-of-00001_eval.json",
        "/Users/wan/Desktop/CS224R/project/output/PKU-Alignment/PKU-SafeRLHF-10K/modpo/lm/(0.9)better+(1-0.9)safer/gen/00001-of-00001_eval.json"
    ]
    
    # Analyze each file
    for file_path in file_paths:
        analyzer.analyze_file(file_path)
    
    file_0_9_0_1 = "/Users/wan/Desktop/CS224R/project/output/PKU-Alignment/PKU-SafeRLHF-10K/modpo/lm/(0.9)better+(1-0.9)safer/gen/00001-of-00001_eval.json"
    file_0_5_0_5 = "/Users/wan/Desktop/CS224R/project/output/PKU-Alignment/PKU-SafeRLHF-10K/modpo/lm/(0.5)better+(1-0.5)safer/gen/00001-of-00001_eval.json"
    file_0_1_0_9 = "/Users/wan/Desktop/CS224R/project/output/PKU-Alignment/PKU-SafeRLHF-10K/modpo/lm/(0.1)better+(1-0.1)safer/gen/00001-of-00001_eval.json"
    analyzer.compare_and_print(file_0_5_0_5, file_0_9_0_1)
    analyzer.compare_and_print(file_0_5_0_5, file_0_1_0_9)

    # compare_and_print()

    # Create raw scores plot
    analyzer.plot_helpfulness_safety(
        file_paths,
        output_path="helpfulness_safety_comparison.pdf"
    )
    
    # Create standardized scores plot
    analyzer.plot_helpfulness_safety(
        file_paths,
        output_path="helpfulness_safety_comparison_standardized.pdf",
        standardize=True
    )
    
    # Create normalized scores plot
    analyzer.plot_helpfulness_safety(
        file_paths,
        output_path="helpfulness_safety_comparison_normalized.pdf",
        normalize=True
    )
    
    # Create PCA plot
    analyzer.plot_pca(
        file_paths,
        output_path="helpfulness_safety_pca.pdf"
    )

if __name__ == "__main__":
    main() 