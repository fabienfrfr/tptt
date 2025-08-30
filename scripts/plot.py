# PAPER PLOT : Warning, it's a mess, needs refactoring
from huggingface_hub import hf_hub_download, list_repo_files
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib as mpl

mpl.rcParams["text.usetex"] = False

import warnings

warnings.filterwarnings("ignore")

repo_dict = {
    "Llama-3.2-1B": "ffurfaro/Titans-Llama-3.2-1B",
    "OLMo-1B-hf": "ffurfaro/Titans-OLMo-1B-hf",
    "OpenELM-1_1B": "ffurfaro/Titans-OpenELM-1_1B",
    "Qwen2.5-1.5B": "ffurfaro/Titans-Qwen2.5-1.5B",
    "Mistral-7B-v0.3": "ffurfaro/Titans-v2-Mistral-7B-v0.3",
    "OLMoE-1B-7B-0924": "ffurfaro/Titans-v2-OLMoE-1B-7B-0924",
    "gemma-3-270m": "ffurfaro/Titanesque-gemma-3-270m",
}

tag_lora_delta_rule = [
    {"model": "Llama-3.2-1B", "variant": "lora_delta_rule_m0.5_constant"},
    {"model": "Llama-3.2-1B", "variant": "lora_delta_rule_gelu_m0.5_constant"},
    {"model": "Llama-3.2-1B", "variant": "lora_delta_product_m0.5_constant"},
]

tag_lora_delta_rule_vkv = [
    {"model": "Llama-3.2-1B", "variant": "lora_delta_product_m0.5_constant"},
    {"model": "Llama-3.2-1B", "variant": "lora_delta_rule_m0.5_constant"},
    {"model": "Llama-3.2-1B", "variant": "lora_delta_rule_v_m0.5_constant"},
    {"model": "Llama-3.2-1B", "variant": "lora_delta_rule_kv_m0.5_constant"},
]

tag_lora_delta_product_variants = [
    {"model": "Llama-3.2-1B", "variant": "lora_delta_rule_m0.5_constant"},
    {"model": "Llama-3.2-1B", "variant": "lora_delta_product_m0.5_constant"},
    {"model": "Llama-3.2-1B", "variant": "lora_delta_product_r_m0.5_constant"},
    {"model": "Llama-3.2-1B", "variant": "lora_delta_product_c_m0.5_constant"},
]

tag_delta_product_experimental = [
    {"model": "Llama-3.2-1B", "variant": "lora_delta_product_m0.5_constant"},
    {"model": "Llama-3.2-1B", "variant": "delta_product_m0.5_constant"},
    {"model": "Llama-3.2-1B", "variant": "lora_cross_delta_product_m0.5_constant"},
]

tag_delta_product_callback = [
    {"model": "Llama-3.2-1B", "variant": "lora_delta_product_m0.5_constant"},
    {"model": "Llama-3.2-1B", "variant": "lora_delta_product_m0.5_gradual"},
    {"model": "Llama-3.2-1B", "variant": "lora_delta_product_m0.5_cyclic"},
]

tag_lora_delta_varied_m = [
    {"model": "Llama-3.2-1B", "variant": "lora_delta_rule_m0.125_constant"},
    {"model": "Llama-3.2-1B", "variant": "lora_delta_product_m0.125_constant"},
    {"model": "Llama-3.2-1B", "variant": "lora_delta_product_m0.5_constant"},
    {"model": "Llama-3.2-1B", "variant": "lora_delta_product_m1.0_constant"},
]

tag_model_lora_delta_product = [
    {"model": "Llama-3.2-1B", "variant": "lora_delta_product_m0.5_constant"},
    {"model": "OLMo-1B-hf", "variant": "lora_delta_product_m0.5_constant"},
    {"model": "OpenELM-1_1B", "variant": "lora_delta_product_m0.5_constant"},
    {"model": "Qwen2.5-1.5B", "variant": "lora_delta_product_m0.5_constant"},
    {"model": "gemma-3-270m", "variant": "lora_delta_product_m0.5_constant"},
]

tag_model_7b_lora = [
    {"model": "OLMo-1B-hf", "variant": "lora_delta_product_m0.5_constant"},
    {"model": "OLMoE-1B-7B-0924", "variant": "lora_delta_product_m0.5_constant"},
    {"model": "Mistral-7B-v0.3", "variant": "lora_delta_product_m0.5_constant"},
]


model_name_mapping = {
    "Llama-3.2-1B": "Llama",
    "OLMo-1B-hf": "OlMo",
    "OpenELM-1_1B": "OpenELM",
    "Qwen2.5-1.5B": "Qwen",
    "Mistral-7B-v0.3": "Mistral",
    "OLMoE-1B-7B-0924": "OlMoE",
    "gemma-3-270m": "Gemma",
}

variant_name_mapping = {
    "lora_delta_rule_m0.5_constant": r"$k, +, n_h=1, \alpha=0.5$",
    "lora_delta_rule_gelu_m0.5_constant": r"$\Delta, k, +, \mathrm{GELU}, n_h=1, \alpha=0.5$",
    "lora_delta_product_m0.5_constant": r"$\Delta, k, +, n_h=2, \alpha=0.5$",
    "lora_delta_rule_v_m0.5_constant": r"$v, +, n_h=1, \alpha=0.5$",
    "lora_delta_rule_kv_m0.5_constant": r"$(k,v), +, n_h=1, \alpha=0.5$",
    "lora_delta_product_r_m0.5_constant": r"$\Theta, k, +, n_h=2, \alpha=0.5$",
    "lora_delta_product_c_m0.5_constant": r"$(\Delta,\Theta), k, +, n_h=2, \alpha=0.5$",
    "lora_delta_product_c_m1.0_constant": r"$(\Delta,\Theta), k, +, n_h=2, \alpha=1.0$",
    "delta_product_m0.5_constant": r"$\Delta, k, +, n_h=2, \alpha=0.5, \mathrm{LoRA} -$",
    "lora_cross_delta_product_m0.5_constant": r"$\Delta, k, \odot, n_h=2, \alpha=0.5$",
    "lora_delta_product_m0.5_gradual": r"$\Delta, k, +, n_h=2, \alpha \in [0, 0.5]$",
    "lora_delta_product_m0.5_cyclic": r"$\Delta, k, +, n_h=2, \alpha \in \{0, 0.5, 1\}$",
    "lora_delta_rule_m0.125_constant": r"$ k, +, n_h=1, \alpha=0.125$",
    "lora_delta_product_m0.125_constant": r"$\Delta, k, +, n_h=2, \alpha=0.125$",
    "lora_delta_product_m1.0_constant": r"$\Delta, k, +, n_h=2, \alpha=1.0$",
}


class TBDataLoader:
    def __init__(self, repo_dict, variant_filter=None):
        self.repo_dict = repo_dict
        self.variant_filter = variant_filter
        self.raw_data = None

    def load_and_process(self, hf_list_repo_files, hf_hub_download):
        dfs = []
        for model_name, repo_id in self.repo_dict.items():
            files = hf_list_repo_files(repo_id)
            log_files = [f for f in files if "events.out" in f]
            for log_file in log_files:
                if self.variant_filter and self.variant_filter not in log_file:
                    continue
                local_path = hf_hub_download(repo_id=repo_id, filename=log_file)
                ea = EventAccumulator(local_path)
                ea.Reload()
                tags = ea.Tags().get("scalars", [])
                for tag in tags:
                    scalars = ea.Scalars(tag)
                    for scalar_event in scalars:
                        variant = log_file.split("/")[0]
                        order = (
                            "delta_rule" if "delta_rule" in variant else "delta_product"
                        )
                        dfs.append(
                            {
                                "modele": model_name,
                                "variant": variant,
                                "model_id": model_name_mapping[model_name]
                                + "+LiZA("
                                + variant_name_mapping[variant]
                                + ")",
                                "tag": tag,
                                "step": scalar_event.step,
                                "value": scalar_event.value,
                                "wall_time": scalar_event.wall_time,
                                "log_file": log_file,
                                "model_order": model_name + "::" + order,
                            }
                        )
        df = pd.DataFrame(dfs)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        self.raw_data = df
        return self.raw_data


class TBSmoother:
    def __init__(self, df):
        self.df = df.copy()
        self.smoothed_df = None

    def smooth(self, alpha=0.6, window=20):
        max_steps = self.df.groupby(["modele", "tag"])["step"].transform("max")
        self.df["step_norm"] = self.df["step"] / max_steps
        mask = self.df["tag"] == "train/grad_norm"
        self.df.loc[mask, "value"] = np.log(self.df.loc[mask, "value"])

        def exponential_smooth(group):
            group = group.sort_values("step_norm")
            group["value_smooth"] = (
                group["value"].ewm(alpha=(1 - alpha), adjust=False).mean()
            )
            group["sd"] = (
                group["value"]
                .rolling(window=window, min_periods=1, center=True)
                .std()
                .fillna(0)
            )
            return group

        self.smoothed_df = (
            self.df.groupby(["modele", "tag"])
            .apply(exponential_smooth)
            .reset_index(drop=True)
        )
        return self.smoothed_df


class TBExperiment:
    def __init__(self, df, experiment_tags):
        self.df = df.copy()
        self.experiment_tags = experiment_tags
        self.expanded_df = None

    def expand_rows_by_experiment_tag(self):
        # WIDE to LONG
        dfs_expanded = []
        for tag_name, tag_entries in self.experiment_tags.items():
            mask = self.df.apply(
                lambda row: any(
                    (
                        row["modele"] == entry["model"]
                        and row["variant"] == entry["variant"]
                    )
                    for entry in tag_entries
                ),
                axis=1,
            )
            df_tag = self.df[mask].copy()
            if not df_tag.empty:
                df_tag["experiment_tag"] = tag_name
                dfs_expanded.append(df_tag)
        if dfs_expanded:
            self.expanded_df = pd.concat(dfs_expanded, ignore_index=True)
        else:
            self.expanded_df = pd.DataFrame(
                columns=self.df.columns.tolist() + ["experiment_tag"]
            )
        return self.expanded_df


class TBPlotter:
    def __init__(self, df):
        self.df = df

    def plot(
        self,
        tag_filter=None,
        modele_filter=None,
        variant_filter=None,
        experiment_tag_filter=None,
        height=4,
        col_wrap=2,
        x_col="step_norm",
        y_col="value",
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        facet_col_rename=None,
        x_label=None,
        y_label=None,
        save=None,
    ):

        plot_data = self.df

        if tag_filter:
            plot_data = plot_data[
                plot_data["tag"].isin(
                    tag_filter if isinstance(tag_filter, list) else [tag_filter]
                )
            ]
        if modele_filter:
            plot_data = plot_data[
                plot_data["modele"].isin(
                    modele_filter
                    if isinstance(modele_filter, list)
                    else [modele_filter]
                )
            ]
        if variant_filter:
            plot_data = plot_data[plot_data["variant"] == variant_filter]
        if experiment_tag_filter:
            plot_data = plot_data[
                plot_data["experiment_tag"].isin(
                    experiment_tag_filter
                    if isinstance(experiment_tag_filter, list)
                    else [experiment_tag_filter]
                )
            ]
            facet_col = "experiment_tag"
        else:
            facet_col = "tag"

        if x_min is None:
            x_min = plot_data[x_col].min()
        if x_max is None:
            x_max = plot_data[x_col].max()
        if y_min is None:
            y_min = 0.5
        if y_max is None:
            y_max = plot_data[y_col].max()

        n_colors = len(plot_data["model_id"].unique())
        palette = sns.color_palette("muted", n_colors)

        g = sns.FacetGrid(
            plot_data, col=facet_col, col_wrap=col_wrap, height=height, sharey=True
        )
        g.map_dataframe(
            sns.lineplot,
            x=x_col,
            y=y_col,
            hue="model_id",
            palette=palette,
            err_style="band",
            errorbar="sd",
            estimator="mean",
        )

        if g._legend is not None:
            g._legend.remove()

        for ax in g.axes.flat:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, loc="best", fontsize="small", frameon=False)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

            current_title = ax.get_title()  # ex: "experiment_tag = lora_delta_rule"
            title_val = current_title.split(" = ")[-1]
            if facet_col_rename and title_val in facet_col_rename:
                ax.set_title(
                    facet_col_rename[title_val],
                    fontsize=12,
                )  # fontweight='bold')

        if x_label is not None:
            g.set_xlabels(x_label)
        if y_label is not None:
            g.set_ylabels(y_label)

        if save is not None:
            plt.savefig(save, bbox_inches="tight")

        plt.show()


if __name__ == "__main__":
    loader = TBDataLoader(repo_dict)
    df_raw = loader.load_and_process(list_repo_files, hf_hub_download)

    experiment_tags = {
        "tag_lora_delta_rule": tag_lora_delta_rule,
        "tag_lora_delta_rule_vkv": tag_lora_delta_rule_vkv,
        "tag_lora_delta_product_variants": tag_lora_delta_product_variants,
        "tag_delta_product_experimental": tag_delta_product_experimental,
        "tag_delta_product_callback": tag_delta_product_callback,
        "tag_lora_delta_varied_m": tag_lora_delta_varied_m,
        "tag_model_lora_delta_product": tag_model_lora_delta_product,
        "tag_model_7b_lora": tag_model_7b_lora,
    }

    exp = TBExperiment(df_raw, experiment_tags)
    df_expanded = exp.expand_rows_by_experiment_tag()

    smoother = TBSmoother(df_expanded)
    df_smoothed = smoother.smooth()
    try:
        display(df_smoothed)
    except:
        print(df_smoothed)

    facet_renames = {
        "tag_lora_delta_rule": "LoRA Delta Rule",
        "tag_lora_delta_rule_vkv": "LoRA Delta Rule Gating",
        "tag_lora_delta_product_variants": "LoRA Delta Product Trick",
        "tag_delta_product_experimental": "LiZA Experimental",
        "tag_delta_product_callback": "Delta Product Callback",
        "tag_lora_delta_varied_m": "LoRA Delta Varied MaG",
        "tag_model_lora_delta_product": "Model LoRA Delta Product",
        "tag_model_7b_lora": "Model 7B LoRA Delta Product",
    }

    plotter = TBPlotter(df_smoothed)

    plotter.plot(
        tag_filter=["train/loss"],
        experiment_tag_filter=list(experiment_tags.keys()),
        x_min=0,
        x_max=1,
        y_min=0,
        y_max=15,
        facet_col_rename=facet_renames,
        x_label="Epoch",
        y_label="Loss (train)",
        col_wrap=4,
        save="train_loss_plot.pdf",
    )
