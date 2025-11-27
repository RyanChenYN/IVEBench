import pandas as pd
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate the average score based on the scores of each video")
    parser.add_argument("--input", "-i", required=True, help="path to each video score csv file")
    parser.add_argument("--output", "-o", required=True, help="path to save average score csv file")
    parser.add_argument("--quality_weight", type=float, default=1.0, help="Weight for quality dimension")
    parser.add_argument("--compliance_weight", type=float, default=1.0, help="Weight for compliance dimension")
    parser.add_argument("--fidelity_weight", type=float, default=1.0, help="Weight for fidelity dimension")
    args = parser.parse_args()
    return args.input, args.output


def main():
    input_file, output_file, q_weight, c_weight, f_weight = parse_args()

    df = pd.read_csv(input_file)

    exclude_cols = ["category", "frame_count", "subcategory", "video_id", "video_name"]

    metric_cols = [
        col for col in df.columns
        if col not in exclude_cols and not col.endswith("_error")
    ]

    df = df[~df["category"].isna() & (df["category"].astype(str).str.strip() != "")]

    df[metric_cols] = df[metric_cols].apply(pd.to_numeric, errors="coerce")

    df[metric_cols] = df[metric_cols].mask(df[metric_cols] <= -1, np.nan)

    means = df[metric_cols].mean()

    metric_category_map = {
        "subject_consistency_score": "quality",
        "background_consistency_score": "quality",
        "temporal_flickering_score": "quality",
        "motion_smoothness_score": "quality",
        "vtss_score": "quality",
        "overall_semantic_consistency_score": "compliance",
        "phrase_semantic_consistency_score": "compliance",
        "instruction_satisfaction_score": "compliance",
        "quantity_accuracy_score": "compliance",
        "semantic_fidelity_score": "fidelity",
        "motion_fidelity_score": "fidelity",
        "content_fidelity_score": "fidelity",
    }

    metric_weights = {
        "instruction_satisfaction_score": 3,
        "content_fidelity_score": 3,
        "vtss_score": 4,
    }

    metric_ranges = {
        "instruction_satisfaction_score": (1, 5),
        "content_fidelity_score": (1, 5),
        "vtss_score": (-0.05, 0.1),
    }

    def normalize_value(value, vmin, vmax):
        return (value - vmin) / (vmax - vmin)

    normalized_means = means.copy()
    for col, (vmin, vmax) in metric_ranges.items():
        if col in normalized_means.index:
            normalized_means[col] = normalize_value(normalized_means[col], vmin, vmax)

    dimension_scores = {}
    for category in ["quality", "compliance", "fidelity"]:
        cols = [col for col, cat in metric_category_map.items() if cat == category]

        weighted_sum = 0
        total_weight = 0
        for col in cols:
            if col in normalized_means.index and not pd.isna(normalized_means[col]):
                weight = metric_weights.get(col, 1)
                weighted_sum += normalized_means[col] * weight
                total_weight += weight

        if total_weight > 0:
            dimension_scores[category] = weighted_sum / total_weight
        else:
            dimension_scores[category] = np.nan

    dim_weights = {
        "quality": q_weight,
        "compliance": c_weight,
        "fidelity": f_weight
    }

    final_weighted_sum = 0
    final_total_weight = 0

    for dim, score in dimension_scores.items():
        if not pd.isna(score):
            w = dim_weights.get(dim, 1.0)
            final_weighted_sum += score * w
            final_total_weight += w

    if final_total_weight > 0:
        total_score = final_weighted_sum / final_total_weight
    else:
        total_score = np.nan

    result = {}

    result["total_score"] = total_score
    result["quality"] = dimension_scores["quality"]
    result["compliance"] = dimension_scores["compliance"]
    result["fidelity"] = dimension_scores["fidelity"]

    for metric_name in metric_category_map.keys():
        result[metric_name] = means[metric_name] if metric_name in means.index else np.nan

    result_df = pd.DataFrame([result])

    result_df.to_csv(output_file, index=False, encoding="utf-8-sig", float_format="%.6f")

    print(f"average score saved to {output_file}")


if __name__ == "__main__":
    main()