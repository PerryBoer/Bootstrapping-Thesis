from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import os


class SimulationPlotter:
    def __init__(self, raw_df, summary_df, save_dir="results/plots"):
        self.raw_df = raw_df
        self.summary_df = summary_df
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def generate_all_plots(self, beta_indices: List[int]):
        self.plot_coverage_vs_lambda()
        self.plot_support_size_vs_threshold()
        for beta_index in beta_indices:
            self.plot_beta_fidelity_scatter(beta_index)
            self.plot_ci_length_distribution(beta_index)

    def plot_coverage_vs_lambda(self):
        plt.figure()
        sns.lineplot(data=self.summary_df, x="lambda_val", y="mean_coverage", hue="method", marker="o")
        plt.title("Coverage vs Lambda")
        plt.xlabel("Lambda")
        plt.ylabel("Mean Coverage")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "coverage_vs_lambda.png"))
        plt.close()

    def plot_support_size_vs_threshold(self):
        plt.figure()
        sns.lineplot(data=self.summary_df, x="threshold_val", y="support_size", hue="method", marker="o")
        plt.title("Support Size vs Threshold Level")
        plt.xlabel("Threshold Level (a_n)")
        plt.ylabel("Support Size")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "support_size_vs_threshold.png"))
        plt.close()

    def plot_beta_fidelity_scatter(self, beta_index: int):
        beta_tilde_vals = []
        beta_true_vals = []
        for _, row in self.raw_df.iterrows():
            beta_tilde_vals.append(row["beta_tilde"][beta_index])
            beta_true_vals.append(row["beta_true"][beta_index])
        plt.figure()
        sns.scatterplot(x=beta_true_vals, y=beta_tilde_vals, alpha=0.6)
        plt.title(f"β̃ vs β True (index={beta_index})")
        plt.xlabel("β True")
        plt.ylabel("β̃")
        plt.axline((0, 0), slope=1, color="red", linestyle="--")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"beta_fidelity_scatter_index{beta_index}.png"))
        plt.close()

    def plot_ci_length_distribution(self, beta_index: int):
        ci_lengths = []
        for _, row in self.raw_df.iterrows():
            ci_lengths.append(row["ci_length"][beta_index])
        plt.figure()
        sns.histplot(ci_lengths, kde=True)
        plt.title(f"CI Length Distribution (index={beta_index})")
        plt.xlabel("CI Length")
        plt.ylabel("Density")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"ci_length_distribution_index{beta_index}.png"))
        plt.close()
