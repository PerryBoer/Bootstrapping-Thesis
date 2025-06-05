import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class SimulationPlotter:
    def __init__(self, raw_df: pd.DataFrame, summary_df: pd.DataFrame, save_dir: str = "results/plots"):
        self.raw_df = raw_df
        self.summary_df = summary_df
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def plot_lambda_distribution(self):
        plt.figure()
        sns.histplot(self.summary_df["lambda_star"], kde=True, bins=20)
        lambda_mean = self.summary_df["lambda_star"].mean()
        plt.axvline(lambda_mean, color='blue', linestyle='--', label=f"λ̄* = {lambda_mean:.5f}")
        plt.title("Distribution of λ* across MC draws")
        plt.xlabel("λ*")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "lambda_distribution.png"))
        plt.close()

    def plot_ci_length_distribution(self):
        ci_lengths = self.raw_df["ci_length"]
        active_lengths, inactive_lengths = [], []

        for i, row in self.raw_df.iterrows():
            support = set(row["support"])
            for idx, length in enumerate(row["ci_length"]):
                if idx in support:
                    active_lengths.append(length)
                else:
                    inactive_lengths.append(length)

        plt.figure()
        sns.histplot(active_lengths, kde=True, bins=30, color="green", label="Active")
        sns.histplot(inactive_lengths, kde=True, bins=30, color="red", label="Inactive")
        plt.title("CI Length Distribution: Active vs Inactive")
        plt.xlabel("CI Length")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "ci_length_active_vs_inactive.png"))
        plt.close()

    def plot_bootstrap_distributions(self, beta_indices=[5, 20]):
        for idx in beta_indices:
            beta_star_all = self.raw_df["beta_star"].to_list()
            combined_draws = np.vstack(beta_star_all)[:, idx]
            beta_true_val = self.raw_df["beta_true"].iloc[0][idx]
            beta_tilde_vals = [row["beta_tilde"][idx] for _, row in self.raw_df.iterrows()]

            plt.figure()
            sns.kdeplot(combined_draws, fill=True, label="Bootstrap β*")
            plt.axvline(beta_true_val, color='black', linestyle='--', label=f"True β_{idx} = {beta_true_val:.3f}")
            plt.axvline(np.mean(beta_tilde_vals), color='orange', linestyle='--', label=f"Mean β̃_{idx}")
            plt.title(f"Bootstrap Distribution for β*_{idx}")
            plt.xlabel(f"β*_{idx}")
            plt.ylabel("Density")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, f"bootstrap_beta_{idx}.png"))
            plt.close()

    def plot_support_size_distribution(self):
        plt.figure()
        sns.histplot(self.raw_df["support_size"], bins=20, kde=False)
        plt.title("Distribution of Selected Support Sizes")
        plt.xlabel("Support Size")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "support_size_distribution.png"))
        plt.close()

    def plot_jaccard_distribution(self):
        if "jaccard" in self.summary_df.columns:
            plt.figure()
            sns.histplot(self.summary_df["jaccard"], kde=True, bins=20)
            plt.title("Jaccard Index Distribution")
            plt.xlabel("Jaccard Index")
            plt.ylabel("Density")
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, "jaccard_distribution.png"))
            plt.close()

    def generate_all_plots(self, beta_indices=[5, 20]):
        self.plot_lambda_distribution()
        self.plot_ci_length_distribution()
        self.plot_bootstrap_distributions(beta_indices=beta_indices)
        self.plot_support_size_distribution()
        self.plot_jaccard_distribution()
