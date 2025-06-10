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
import seaborn as sns
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationStats:
    """Class to store statistics for evaluation metrics."""
    mean_reward: float
    std_reward: float
    mean_multi_scores: Dict[str, float]
    std_multi_scores: Dict[str, float]
    mean_harmlessness: float = 0.0
    std_harmlessness: float = 0.0
    mean_helpfulness: float = 0.0
    std_helpfulness: float = 0.0
    win_rate_harmlessness: float = 0.0
    win_rate_helpfulness: float = 0.0

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
            return EvaluationStats(0.0, 0.0, {}, {}, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

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
            std_multi_scores=std_multi_scores,
            mean_harmlessness=0.0,
            std_harmlessness=0.0,
            mean_helpfulness=0.0,
            std_helpfulness=0.0,
            win_rate_harmlessness=0.0,
            win_rate_helpfulness=0.0
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

    def compute_pairwise_stats(self, eval_results: List[Dict]) -> EvaluationStats:
        """
        Compute statistics for direct harmlessness and helpfulness scores.
        
        Args:
            eval_results: List of evaluation results
            
        Returns:
            EvaluationStats object containing mean and std for harmlessness and helpfulness scores
        """
        if not eval_results:
            return EvaluationStats(0.0, 0.0, {}, {}, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # Extract direct scores
        harmlessness_scores = []
        helpfulness_scores = []

        for result in eval_results:
            eval_data = result["evaluation"]
            harmlessness_scores.append(eval_data["harmlessness_score"])
            helpfulness_scores.append(eval_data["helpfulness_score"])

        return EvaluationStats(
            mean_reward=0.0,
            std_reward=0.0,
            mean_multi_scores={},
            std_multi_scores={},
            mean_harmlessness=np.mean(harmlessness_scores),
            std_harmlessness=np.std(harmlessness_scores),
            mean_helpfulness=np.mean(helpfulness_scores),
            std_helpfulness=np.std(helpfulness_scores),
            win_rate_harmlessness=0.0,  # Will be computed in compare_pairwise_files
            win_rate_helpfulness=0.0    # Will be computed in compare_pairwise_files
        )

    def analyze_pairwise_file(self, file_path: str) -> None:
        """
        Analyze a single evaluation file and print statistics for harmlessness and helpfulness scores.
        
        Args:
            file_path: Path to evaluation file
        """
        results = self.load_eval_file(file_path)
        if not results:
            return

        stats = self.compute_pairwise_stats(results)
        
        print(f"\nAnalysis for {file_path}:")
        print(f"Number of samples: {len(results)}")
        
        print("\nHarmlessness Score:")
        print(f"  Mean: {stats.mean_harmlessness:.2f} ± {stats.std_harmlessness:.2f}")
        
        print("\nHelpfulness Score:")
        print(f"  Mean: {stats.mean_helpfulness:.2f} ± {stats.std_helpfulness:.2f}")

    def compare_pairwise_files(self, base_file: str, test_file: str) -> None:
        """
        Compare two evaluation files and print statistics for harmlessness and helpfulness scores.
        Also computes win rates by comparing scores between the two files.
        
        Args:
            base_file: Path to base evaluation file
            test_file: Path to test evaluation file
        """
        base_results = self.load_eval_file(base_file)
        test_results = self.load_eval_file(test_file)
        
        if not base_results or not test_results:
            logger.error("Could not load one or both evaluation files")
            return

        # Create dictionaries mapping queries to scores
        base_scores = {r["original_content"]["user_query"]: r["evaluation"] for r in base_results}
        test_scores = {r["original_content"]["user_query"]: r["evaluation"] for r in test_results}

        # Find common queries
        common_queries = set(base_scores.keys()) & set(test_scores.keys())
        
        if not common_queries:
            logger.warning("No common queries found between the two files")
            return

        # Initialize counters for win rates
        harmlessness_wins = 0
        harmlessness_ties = 0
        helpfulness_wins = 0
        helpfulness_ties = 0
        total_comparisons = 0

        # Compare scores for each query
        for query in common_queries:
            base_eval = base_scores[query]
            test_eval = test_scores[query]
            
            # Compare harmlessness scores
            if test_eval["harmlessness_score"] > base_eval["harmlessness_score"]:
                harmlessness_wins += 1
            elif test_eval["harmlessness_score"] < base_eval["harmlessness_score"]:
                pass  # Base wins
            else:
                harmlessness_ties += 1
            
            # Compare helpfulness scores
            if test_eval["helpfulness_score"] > base_eval["helpfulness_score"]:
                helpfulness_wins += 1
            elif test_eval["helpfulness_score"] < base_eval["helpfulness_score"]:
                pass  # Base wins
            else:
                helpfulness_ties += 1
            
            total_comparisons += 1

        # Compute win rates
        harmlessness_win_rate = harmlessness_wins / total_comparisons if total_comparisons > 0 else 0.0
        harmlessness_tie_rate = harmlessness_ties / total_comparisons if total_comparisons > 0 else 0.0
        helpfulness_win_rate = helpfulness_wins / total_comparisons if total_comparisons > 0 else 0.0
        helpfulness_tie_rate = helpfulness_ties / total_comparisons if total_comparisons > 0 else 0.0

        base_stats = self.compute_pairwise_stats(base_results)
        test_stats = self.compute_pairwise_stats(test_results)
        
        print(f"\nComparison between:")
        print(f"Base file: {base_file}")
        print(f"Test file: {test_file}")
        print(f"Number of common queries: {total_comparisons}")
        
        print("\nHarmlessness Score:")
        print(f"  Base: {base_stats.mean_harmlessness:.2f} ± {base_stats.std_harmlessness:.2f}")
        print(f"  Test: {test_stats.mean_harmlessness:.2f} ± {test_stats.std_harmlessness:.2f}")
        print(f"  Difference: {test_stats.mean_harmlessness - base_stats.mean_harmlessness:.2f}")
        print(f"  Test Win Rate: {harmlessness_win_rate:.2%}")
        print(f"  Tie Rate: {harmlessness_tie_rate:.2%}")
        
        print("\nHelpfulness Score:")
        print(f"  Base: {base_stats.mean_helpfulness:.2f} ± {base_stats.std_helpfulness:.2f}")
        print(f"  Test: {test_stats.mean_helpfulness:.2f} ± {test_stats.std_helpfulness:.2f}")
        print(f"  Difference: {test_stats.mean_helpfulness - base_stats.mean_helpfulness:.2f}")
        print(f"  Test Win Rate: {helpfulness_win_rate:.2%}")
        print(f"  Tie Rate: {helpfulness_tie_rate:.2%}")

    def plot_pairwise_comparison(
        self,
        file1: str,
        file2: str,
        output_dir: str = "plots",
        save_plot: bool = True,
        output_file_tag: str = ""
    ) -> None:
        """
        Create a scatter plot comparing harmlessness and helpfulness scores between two files.
        Each point represents a query with coordinates (harmlessness_score, helpfulness_score).
        Different colors are used for different files.
        
        Args:
            file1: Path to first evaluation file
            file2: Path to second evaluation file
            output_dir: Directory to save plots
            save_plot: Whether to save the plot to file
            output_file_tag: Additional tag for output filename
        """
        # Load evaluation results
        results1 = self.load_eval_file(file1)
        results2 = self.load_eval_file(file2)
        
        # Extract scores
        harmlessness1 = [r["evaluation"]["harmlessness_score"] for r in results1]
        helpfulness1 = [r["evaluation"]["helpfulness_score"] for r in results1]
        harmlessness2 = [r["evaluation"]["harmlessness_score"] for r in results2]
        helpfulness2 = [r["evaluation"]["helpfulness_score"] for r in results2]
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot points for file1
        plt.scatter(
            harmlessness1,
            helpfulness1,
            alpha=0.6,
            label=Path(file1).parent.parent.name,
            color='blue'
        )
        
        # Plot points for file2
        plt.scatter(
            harmlessness2,
            helpfulness2,
            alpha=0.6,
            label=Path(file2).parent.parent.name,
            color='red'
        )
        
        # Add diagonal line
        plt.plot([0, 10], [0, 10], 'k--', alpha=0.3, label='Diagonal')
        
        # Set labels and title
        plt.xlabel('Harmlessness Score')
        plt.ylabel('Helpfulness Score')
        plt.title('Harmlessness vs Helpfulness Scores')
        
        # Set axis limits
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Add legend
        plt.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        if save_plot:
            # Create output directory if it doesn't exist
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Generate output filename
            tag1 = Path(file1).parent.parent.name
            tag2 = Path(file2).parent.parent.name
            output_file = Path(output_dir) / f"{tag1}_vs_{tag2}_scores{output_file_tag}.png"
            
            # Save plot
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {output_file}")
        
        plt.close()

    def plot_all_comparisons(
        self,
        files: Dict[str, str],
        base_file: str,
        output_dir: str = "plots"
    ) -> None:
        """
        Create scatter plots comparing all files against a base file.
        Each plot shows harmlessness vs helpfulness scores for both files.
        
        Args:
            files: Dictionary mapping file names to file paths
            base_file: Name of the base file to compare against
            output_dir: Directory to save plots
        """
        if base_file not in files:
            logger.error(f"Base file {base_file} not found in files dictionary")
            return
            
        base_path = files[base_file]
        
        for name, path in tqdm(files.items(), desc="Creating comparison plots"):
            if name == base_file:
                continue
                
            self.plot_pairwise_comparison(
                file1=path,
                file2=base_path,
                output_dir=output_dir
            )

    def plot_2d_histogram(
        self,
        file_path: str,
        output_dir: str = "plots",
        save_plot: bool = True,
        plot_type: str = "hexbin",
        output_file_tag: str = ""
    ) -> None:
        """
        Create a 2D histogram with marginal distributions for a single file.
        
        Args:
            file_path: Path to evaluation file
            output_dir: Directory to save plots
            save_plot: Whether to save the plot to file
            plot_type: Type of 2D plot ('hexbin' or 'heatmap')
        """
        # Load evaluation results
        results = self.load_eval_file(file_path)
        if not results:
            return
            
        # Extract scores
        harmlessness = [r["evaluation"]["harmlessness_score"] for r in results]
        helpfulness = [r["evaluation"]["helpfulness_score"] for r in results]
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Create main plot
        if plot_type == "hexbin":
            plt.hexbin(
                harmlessness,
                helpfulness,
                gridsize=11,
                cmap='viridis',
                extent=[0, 10, 0, 10]
            )
        else:  # heatmap
            hist, xedges, yedges = np.histogram2d(
                harmlessness,
                helpfulness,
                bins=11,
                range=[[0, 10], [0, 10]]
            )
            plt.imshow(
                hist.T,
                origin='lower',
                extent=[0, 10, 0, 10],
                aspect='auto',
                cmap='viridis'
            )
        
        # Add colorbar
        plt.colorbar(label='Count')
        
        # Set labels and title
        plt.xlabel('Harmlessness Score')
        plt.ylabel('Helpfulness Score')
        model_name = Path(file_path).parent.parent.name
        plt.title(f'Distribution of Scores for {model_name}')
        
        # Set axis limits
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        if save_plot:
            # Create output directory if it doesn't exist
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Generate output filename
            output_file = Path(output_dir) / f"{model_name}_2d_histogram{output_file_tag}.png"
            
            # Save plot
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {output_file}")
        
        plt.close()

    def plot_joint_distribution(
        self,
        file_path: str,
        output_dir: str = "plots",
        save_plot: bool = True,
        output_file_tag: str="",
    ) -> None:
        """
        Create a joint distribution plot with marginal histograms using seaborn.
        
        Args:
            file_path: Path to evaluation file
            output_dir: Directory to save plots
            save_plot: Whether to save the plot to file
        """
        # Load evaluation results
        results = self.load_eval_file(file_path)
        if not results:
            return
            
        # Extract scores
        harmlessness = [r["evaluation"]["harmlessness_score"] for r in results]
        helpfulness = [r["evaluation"]["helpfulness_score"] for r in results]
        
        # Create joint plot
        g = sns.jointplot(
            x=harmlessness,
            y=helpfulness,
            kind="hist",
            bins=11,
            height=10,
            marginal_kws=dict(bins=11)
        )
        
        # Set labels and title
        g.set_axis_labels('Harmlessness Score', 'Helpfulness Score')
        model_name = Path(file_path).parent.parent.name
        g.fig.suptitle(f'Joint Distribution of Scores for {model_name}', y=1.02)
        
        # Set axis limits
        g.ax_joint.set_xlim(0, 10)
        g.ax_joint.set_ylim(0, 10)
        
        # Add grid
        g.ax_joint.grid(True, alpha=0.3)
        
        if save_plot:
            # Create output directory if it doesn't exist
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Generate output filename
            output_file = Path(output_dir) / f"{model_name}_joint_distribution{output_file_tag}.png"
            
            # Save plot
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {output_file}")
        
        plt.close()

    def plot_all_distributions(
        self,
        files: Dict[str, str],
        output_dir: str = "plots",
        plot_type: str = "hexbin",
        output_file_tag: str = "",
    ) -> None:
        """
        Create distribution plots for all files.
        
        Args:
            files: Dictionary mapping file names to file paths
            output_dir: Directory to save plots
            plot_type: Type of 2D plot ('hexbin' or 'heatmap')
        """
        for name, path in tqdm(files.items(), desc="Creating distribution plots"):
            # Create 2D histogram
            self.plot_2d_histogram(
                file_path=path,
                output_dir=output_dir,
                plot_type=plot_type,
                output_file_tag=output_file_tag
            )
            
            # Create joint distribution plot
            self.plot_joint_distribution(
                file_path=path,
                output_dir=output_dir,
                output_file_tag=output_file_tag
            )

def main():
    """Example usage of the evaluation analyzer."""
    analyzer = EvaluationAnalyzer()

    lm_folder = "/Users/wan/Desktop/CS224R/project/output/PKU-Alignment/PKU-SafeRLHF-10K/modpo/lm"
    # Example: Analyze multiple files and create scatter plots
    file_paths = [
        lm_folder + "/(0.1)better+(1-0.1)safer/gen/00001-of-00001_eval.json",
        lm_folder + "/(0.5)better+(1-0.5)safer/gen/00001-of-00001_eval.json",
        lm_folder + "/(0.9)better+(1-0.9)safer/gen/00001-of-00001_eval.json"
    ]
    
    # Analyze each file
    for file_path in file_paths:
        analyzer.analyze_file(file_path)
    
    file_0_9_0_1 = lm_folder + "/(0.9)better+(1-0.9)safer/gen/00001-of-00001_eval.json"
    file_0_5_0_5 = lm_folder + "/(0.5)better+(1-0.5)safer/gen/00001-of-00001_eval.json"
    file_0_1_0_9 = lm_folder + "/(0.1)better+(1-0.1)safer/gen/00001-of-00001_eval.json"
    # analyzer.compare_and_print(file_0_5_0_5, file_0_9_0_1)
    # analyzer.compare_and_print(file_0_5_0_5, file_0_1_0_9)
    # analyzer.compare_and_print(file_0_9_0_1, file_0_1_0_9)
    pairwise_file_09_01_vs_05_05 = lm_folder + "/(0.9)better+(1-0.9)safer/gen/00001-of-00001_vs_(0.5)better+(1-0.5)safer_pairwise_eval.json"
    pairwise_file_09_01_vs_01_09 = lm_folder + "/(0.9)better+(1-0.9)safer/gen/00001-of-00001_vs_(0.1)better+(1-0.1)safer_pairwise_eval.json"

    pairwise_file_05_05_vs_09_01 = lm_folder + "/(0.5)better+(1-0.5)safer/gen/00001-of-00001_vs_(0.9)better+(1-0.9)safer_pairwise_eval.json"
    pairwise_file_05_05_vs_01_09 = lm_folder + "/(0.5)better+(1-0.5)safer/gen/00001-of-00001_vs_(0.1)better+(1-0.1)safer_pairwise_eval.json"

    pairwise_file_01_09_vs_05_05 = lm_folder + "/(0.1)better+(1-0.1)safer/gen/00001-of-00001_vs_(0.5)better+(1-0.5)safer_pairwise_eval.json"
    pairwise_file_01_09_vs_09_01 = lm_folder + "/(0.1)better+(1-0.1)safer/gen/00001-of-00001_vs_(0.9)better+(1-0.9)safer_pairwise_eval.json"

    # analyzer.compare_pairwise_files(pairwise_file_09_01_vs_01_09, pairwise_file_01_09_vs_09_01)
    # compare_and_print(pairwise_file_09_01_vs_05_05, pairwise_file_05_05_vs_09_01)

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

    if False:
        # Example of using new pairwise analysis functions
        print("\n========================================================")
        print("\n========================================================")
        print("\nAnalyzing pairwise evaluation files 09_01 vs 01_09:")
        analyzer.analyze_pairwise_file(pairwise_file_09_01_vs_01_09)
        analyzer.analyze_pairwise_file(pairwise_file_01_09_vs_09_01)
        analyzer.compare_pairwise_files(pairwise_file_09_01_vs_01_09, pairwise_file_01_09_vs_09_01)

        print("\n========================================================")
        print("\n========================================================")
        print("\nAnalyzing pairwise evaluation files 09_01 vs 05_05:")
        analyzer.analyze_pairwise_file(pairwise_file_05_05_vs_09_01)
        analyzer.analyze_pairwise_file(pairwise_file_09_01_vs_05_05)
        analyzer.compare_pairwise_files(pairwise_file_05_05_vs_09_01, pairwise_file_09_01_vs_05_05)

        print("\n========================================================")
        print("\n========================================================")
        print("\nAnalyzing pairwise evaluation files 01_09 vs 05_05:")
        analyzer.analyze_pairwise_file(pairwise_file_05_05_vs_01_09)
        analyzer.analyze_pairwise_file(pairwise_file_01_09_vs_05_05)
        analyzer.compare_pairwise_files(pairwise_file_05_05_vs_01_09, pairwise_file_01_09_vs_05_05)

    advanced_modpo_folder = "/Users/wan/Desktop/CS224R/project/output/advanced_modpo_infer"
    eval_files_advanced_modpo = {
        "chebyshev_0.5": "/Users/wan/Desktop/CS224R/project/output/advanced_modpo_infer/chebyshev_w0.5/gen/chebyshev_0.5_vs_linear_w0.5_pairwise_eval.json",
        "exponential_risk0.5": "/Users/wan/Desktop/CS224R/project/output/advanced_modpo_infer/exponential_risk0.5/gen/exponential_risk0.5_vs_linear_w0.5_pairwise_eval.json",
        "exponential_risk2.0": "/Users/wan/Desktop/CS224R/project/output/advanced_modpo_infer/exponential_risk2.0/gen/exponential_risk2.0_vs_linear_w0.5_pairwise_eval.json",
        "exponential_w0.5": "/Users/wan/Desktop/CS224R/project/output/advanced_modpo_infer/exponential_w0.5/gen/exponential_w0.5_vs_linear_w0.5_pairwise_eval.json",
        # mark
        "linear_w0.5": "/Users/wan/Desktop/CS224R/project/output/advanced_modpo_infer/linear_w0.5/gen/linear_w0.5_vs_exponential_w0.5_pairwise_eval.json",
        "power_risk0.2": "/Users/wan/Desktop/CS224R/project/output/advanced_modpo_infer/power_risk0.2/gen/power_risk0.2_vs_linear_w0.5_pairwise_eval.json",
        "power_risk0.5": "/Users/wan/Desktop/CS224R/project/output/advanced_modpo_infer/power_risk0.5/gen/power_risk0.5_vs_linear_w0.5_pairwise_eval.json",
        "power_risk0.8": "/Users/wan/Desktop/CS224R/project/output/advanced_modpo_infer/power_risk0.8/gen/power_risk0.8_vs_linear_w0.5_pairwise_eval.json",
        "power_w0.5": "/Users/wan/Desktop/CS224R/project/output/advanced_modpo_infer/power_w0.5/gen/power_w0.5_vs_linear_w0.5_pairwise_eval.json",
    }


    eval_files_test_linear_w0_5 = {
    "chebyshev_0.5": "/Users/wan/Desktop/CS224R/project/output/advanced_modpo_infer/linear_w0.5/gen/linear_w0.5_vs_chebyshev_w0.5_pairwise_eval.json",
    "exponential_risk0.5": "/Users/wan/Desktop/CS224R/project/output/advanced_modpo_infer/linear_w0.5/gen/linear_w0.5_vs_exponential_risk0.5_pairwise_eval.json",
    "exponential_risk2.0": "/Users/wan/Desktop/CS224R/project/output/advanced_modpo_infer/linear_w0.5/gen/linear_w0.5_vs_exponential_risk2.0_pairwise_eval.json",
    "exponential_w0.5": "/Users/wan/Desktop/CS224R/project/output/advanced_modpo_infer/linear_w0.5/gen/linear_w0.5_vs_exponential_w0.5_pairwise_eval.json",
    "power_risk0.2": "/Users/wan/Desktop/CS224R/project/output/advanced_modpo_infer/linear_w0.5/gen/linear_w0.5_vs_power_risk0.2_pairwise_eval.json",
    "power_risk0.5": "/Users/wan/Desktop/CS224R/project/output/advanced_modpo_infer/linear_w0.5/gen/linear_w0.5_vs_power_risk0.5_pairwise_eval.json",
    "power_risk0.8": "/Users/wan/Desktop/CS224R/project/output/advanced_modpo_infer/linear_w0.5/gen/linear_w0.5_vs_power_risk0.8_pairwise_eval.json",
    "power_w0.5": "/Users/wan/Desktop/CS224R/project/output/advanced_modpo_infer/linear_w0.5/gen/linear_w0.5_vs_power_w0.5_pairwise_eval.json",
}
    
    eval_files_advanced_modpo_decimal = {
        "chebyshev_0.5": "/Users/wan/Desktop/CS224R/project/output/advanced_modpo_infer/chebyshev_w0.5/gen/chebyshev_0.5_vs_linear_w0.5_pairwise_eval_1.json",
        "exponential_risk0.5": "/Users/wan/Desktop/CS224R/project/output/advanced_modpo_infer/exponential_risk0.5/gen/exponential_risk0.5_vs_linear_w0.5_pairwise_eval_1.json",
        "exponential_risk2.0": "/Users/wan/Desktop/CS224R/project/output/advanced_modpo_infer/exponential_risk2.0/gen/exponential_risk2.0_vs_linear_w0.5_pairwise_eval_1.json",
        "exponential_w0.5": "/Users/wan/Desktop/CS224R/project/output/advanced_modpo_infer/exponential_w0.5/gen/exponential_w0.5_vs_linear_w0.5_pairwise_eval_1.json",
        # mark
        "linear_w0.5": "/Users/wan/Desktop/CS224R/project/output/advanced_modpo_infer/linear_w0.5/gen/linear_w0.5_vs_exponential_w0.5_pairwise_eval_1.json",
        "power_risk0.2": "/Users/wan/Desktop/CS224R/project/output/advanced_modpo_infer/power_risk0.2/gen/power_risk0.2_vs_linear_w0.5_pairwise_eval_1.json",
        "power_risk0.5": "/Users/wan/Desktop/CS224R/project/output/advanced_modpo_infer/power_risk0.5/gen/power_risk0.5_vs_linear_w0.5_pairwise_eval_1.json",
        "power_risk0.8": "/Users/wan/Desktop/CS224R/project/output/advanced_modpo_infer/power_risk0.8/gen/power_risk0.8_vs_linear_w0.5_pairwise_eval_1.json",
        "power_w0.5": "/Users/wan/Desktop/CS224R/project/output/advanced_modpo_infer/power_w0.5/gen/power_w0.5_vs_linear_w0.5_pairwise_eval_1.json",
    }


    eval_files_test_linear_w0_5_decimal = {
    "chebyshev_0.5": "/Users/wan/Desktop/CS224R/project/output/advanced_modpo_infer/linear_w0.5/gen/linear_w0.5_vs_chebyshev_w0.5_pairwise_eval_1.json",
    "exponential_risk0.5": "/Users/wan/Desktop/CS224R/project/output/advanced_modpo_infer/linear_w0.5/gen/linear_w0.5_vs_exponential_risk0.5_pairwise_eval_1.json",
    "exponential_risk2.0": "/Users/wan/Desktop/CS224R/project/output/advanced_modpo_infer/linear_w0.5/gen/linear_w0.5_vs_exponential_risk2.0_pairwise_eval_1.json",
    "exponential_w0.5": "/Users/wan/Desktop/CS224R/project/output/advanced_modpo_infer/linear_w0.5/gen/linear_w0.5_vs_exponential_w0.5_pairwise_eval_1.json",
    "power_risk0.2": "/Users/wan/Desktop/CS224R/project/output/advanced_modpo_infer/linear_w0.5/gen/linear_w0.5_vs_power_risk0.2_pairwise_eval_1.json",
    "power_risk0.5": "/Users/wan/Desktop/CS224R/project/output/advanced_modpo_infer/linear_w0.5/gen/linear_w0.5_vs_power_risk0.5_pairwise_eval_1.json",
    "power_risk0.8": "/Users/wan/Desktop/CS224R/project/output/advanced_modpo_infer/linear_w0.5/gen/linear_w0.5_vs_power_risk0.8_pairwise_eval_1.json",
    "power_w0.5": "/Users/wan/Desktop/CS224R/project/output/advanced_modpo_infer/linear_w0.5/gen/linear_w0.5_vs_power_w0.5_pairwise_eval_1.json",
}

    for name, eval_file_name in eval_files_advanced_modpo.items():
        analyzer.analyze_pairwise_file(eval_file_name)
    
    for name, eval_file_name in eval_files_advanced_modpo.items():
        if name == "linear_w0.5":
            continue
        analyzer.compare_pairwise_files(eval_files_test_linear_w0_5[name], eval_files_advanced_modpo[name])


    for name, eval_file_name in eval_files_advanced_modpo_decimal.items():
        analyzer.analyze_pairwise_file(eval_file_name)
    
    for name, eval_file_name in eval_files_advanced_modpo_decimal.items():
        if name == "linear_w0.5":
            continue
        analyzer.compare_pairwise_files(eval_files_test_linear_w0_5_decimal[name], eval_files_advanced_modpo_decimal[name])


    for name, file_path in eval_files_advanced_modpo.items():
        analyzer.plot_2d_histogram(
            file_path=eval_files_advanced_modpo[name],
            output_dir="/Users/wan/Desktop/CS224R/project/output/advanced_modpo_infer",
            plot_type="hexbin"
        )
        
        analyzer.plot_joint_distribution(
            file_path=eval_files_advanced_modpo[name],
            output_dir="/Users/wan/Desktop/CS224R/project/output/advanced_modpo_infer"
        )
    
    for name, file_path in eval_files_advanced_modpo_decimal.items():
        analyzer.plot_2d_histogram(
            file_path=eval_files_advanced_modpo_decimal[name],
            output_dir="/Users/wan/Desktop/CS224R/project/output/advanced_modpo_infer",
            plot_type="hexbin",
            output_file_tag="_decimal"
        )
        
        analyzer.plot_joint_distribution(
            file_path=eval_files_advanced_modpo_decimal[name],
            output_dir="/Users/wan/Desktop/CS224R/project/output/advanced_modpo_infer",
            output_file_tag="_decimal"
        )
    

    # analyzer.plot_pairwise_comparison(
    #     file1=eval_files_advanced_modpo_decimal["exponential_w0.5"],
    #     file2=eval_files_advanced_modpo_decimal["linear_w0.5"],
    #     output_dir="/Users/wan/Desktop/CS224R/project/output/advanced_modpo_infer",
    #     output_file_tag="_decimal"
    # )
    for name, file_path in eval_files_advanced_modpo_decimal.items():
        if name == "linear_w0.5":
            continue
        analyzer.plot_pairwise_comparison(
            file1=eval_files_advanced_modpo_decimal[name],
            file2=eval_files_advanced_modpo_decimal["linear_w0.5"],
            output_dir="/Users/wan/Desktop/CS224R/project/output/advanced_modpo_infer",
            output_file_tag="_decimal"
        )

if __name__ == "__main__":
    main() 