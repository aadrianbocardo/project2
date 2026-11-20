"""Prepare and analyze state-year health indicator transactions.

Run from any directory: python scripts/Thirumurugan_Bocardo_part2_alzheimer.py
Add --profile to also generate the exploratory HTML report.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save figures without requiring a graphical display.
import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "datasets" / "Alzheimer's_Disease_and_Healthy_Aging_Data_20261116.csv"
OUTPUTS = PROJECT_DIR / "outputs"
KEY_COLUMNS = [
    "YearStart", "YearEnd", "LocationID", "QuestionID",
    "StratificationID1", "StratificationID2",
]
STATES_DC = set(
    "AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN "
    "MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA "
    "WA WV WI WY DC".split()
)
SUPPORT_VALUES = (0.15, 0.20, 0.30)
CONFIDENCE_VALUES = (0.60, 0.70, 0.80)
CHOSEN_SUPPORT = 0.15
CHOSEN_CONFIDENCE = 0.70


def examine_data(data):
    """Print the missingness, duplicate, category and numeric summaries."""
    missing = pd.DataFrame({
        "Missing Values": data.isna().sum(),
        "Missing Percentage": (data.isna().mean() * 100).round(2),
    })
    print("Missing values:\n", missing)
    print("Duplicate full rows:", data.duplicated().sum())
    print("Unique RowId values:", data["RowId"].nunique())
    print("Duplicate RowId rows:", data.duplicated("RowId").sum())
    print("Duplicate composite keys:", data.duplicated(KEY_COLUMNS).sum())
    for column in data.select_dtypes(include=["object", "string", "category"]):
        print(f"\n{column}:\n{data[column].value_counts(dropna=False)}")
    print("\nNumeric summary:\n", data.select_dtypes(include="number").describe())
    print("YearStart values:", sorted(data["YearStart"].unique()))
    print("YearEnd values:", sorted(data["YearEnd"].unique()))
    print("Records spanning multiple years:", (data["YearStart"] != data["YearEnd"]).sum())


def prepare_data(data):
    """Select records and assign each indicator a relative category."""
    data = data.copy()
    data["RecordID"] = data[KEY_COLUMNS].astype(str).agg("_".join, axis=1)
    selected = data[
        data["Stratification1"].eq("65 years or older")
        & data["Stratification2"].eq("Female")
        & data["LocationAbbr"].isin(STATES_DC)
    ].copy()
    print("Target population rows:", len(selected))
    clean = selected[
        selected["YearStart"].eq(selected["YearEnd"])
        & selected["DataValueTypeID"].eq("PRCTG")
        & selected["Data_Value"].notna()
    ].copy()
    print("Usable measurements:", len(clean))

    # Rank within each question across all selected state-years, as in the notebook.
    clean["PercentileRank"] = clean.groupby("QuestionID")["Data_Value"].rank(pct=True)
    clean["Category"] = pd.cut(
        clean["PercentileRank"], [0, 1 / 3, 2 / 3, 1],
        labels=["Low", "Medium", "High"], include_lowest=True,
    )
    clean["Item"] = clean["QuestionID"].astype(str) + ": " + clean["Category"].astype(str)
    clean["TransactionID"] = clean["LocationAbbr"].astype(str) + "_" + clean["YearStart"].astype(str)
    return clean


def build_basket(clean):
    """Create a Boolean state-year basket and its long-form reconstruction."""
    wide = clean.pivot(index="TransactionID", columns="QuestionID", values="Category")
    basket = pd.get_dummies(wide, prefix_sep=": ").astype(bool)
    long_items = basket.reset_index().melt(
        id_vars="TransactionID", var_name="Item", value_name="Present"
    )
    long_items = long_items.loc[long_items["Present"], ["TransactionID", "Item"]]
    rebuilt = pd.crosstab(long_items["TransactionID"], long_items["Item"])
    rebuilt = rebuilt.reindex(index=basket.index, columns=basket.columns, fill_value=0).astype(bool)
    print("Basket shape:", basket.shape, "| long items:", len(long_items))
    return basket, long_items


def generate_rules(frequent, basket, confidence):
    """Generate rules when there are itemsets large enough to yield them."""
    if frequent.empty or frequent["itemsets"].map(len).max() < 2:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])
    return association_rules(
        frequent, num_itemsets=len(basket), metric="confidence", min_threshold=confidence
    )


def mine_rules(basket):
    """Compare thresholds and return the chosen configuration's rules."""
    results = []
    chosen = None
    for support in SUPPORT_VALUES:
        frequent = apriori(basket, min_support=support, use_colnames=True)
        for confidence in CONFIDENCE_VALUES:
            rules = generate_rules(frequent, basket, confidence)
            results.append({"min_support": support, "min_confidence": confidence,
                            "frequent_itemsets": len(frequent), "rules": len(rules)})
            if support == CHOSEN_SUPPORT and confidence == CHOSEN_CONFIDENCE:
                chosen = rules
    return pd.DataFrame(results), chosen


def save_figures(basket, rules, directory):
    """Save item frequency, rule scatter, and top-rule lift heatmap charts."""
    directory.mkdir(parents=True, exist_ok=True)
    support = basket.mean().sort_values()
    fig, ax = plt.subplots(figsize=(10, max(8, len(support) * 0.25)))
    support.plot.barh(ax=ax, color="steelblue")
    ax.axvline(CHOSEN_SUPPORT, color="darkorange", linestyle="--", label=f"Min support = {CHOSEN_SUPPORT:.2f}")
    ax.set(title="Frequency of health-indicator items", xlabel="Support: proportion of state-year transactions", ylabel="Item")
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "part2_item_frequency.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    if rules.empty:
        print("No rules at chosen thresholds; skipping rule charts.")
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(rules["support"], rules["confidence"], c=rules["lift"], cmap="viridis", s=70, alpha=0.75)
    fig.colorbar(scatter, ax=ax, label="Lift")
    ax.set(title="Association rules: support and confidence", xlabel="Support", ylabel="Confidence")
    fig.tight_layout()
    fig.savefig(directory / "part2_rules_scatter.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    top = rules.sort_values(["lift", "support"], ascending=False).head(10).copy()
    top["antecedent_label"] = top["antecedents"].apply(lambda items: ", ".join(sorted(items)))
    top["consequent_label"] = top["consequents"].apply(lambda items: ", ".join(sorted(items)))
    table = top.pivot_table(index="antecedent_label", columns="consequent_label", values="lift", aggfunc="max")
    fig, ax = plt.subplots(figsize=(10, max(5, len(table) * 0.6)))
    heatmap = ax.imshow(table, cmap="YlOrRd", aspect="auto")
    fig.colorbar(heatmap, ax=ax, label="Lift")
    ax.set_xticks(range(len(table.columns)), table.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(table.index)), table.index)
    ax.set(title="Lift of the top 10 association rules", xlabel="Consequent", ylabel="Antecedent")
    fig.tight_layout()
    fig.savefig(directory / "part2_lift_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="store_true", help="Generate the slower HTML profiling report")
    args = parser.parse_args()
    raw = pd.read_csv(DATA_PATH)
    examine_data(raw)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    if args.profile:
        from data_profiling import ProfileReport
        ProfileReport(raw, title="Alzheimer's Disease and Healthy Aging EDA Report").to_file(
            OUTPUTS / "Thirumurugan_Bocardo_part2_alzheimer_EDA.html"
        )
    clean = prepare_data(raw)
    basket, _ = build_basket(clean)
    counts, rules = mine_rules(basket)
    tables = OUTPUTS / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    counts.to_csv(tables / "part2_rule_counts.csv", index=False)
    print("\nThreshold comparison:\n", counts.to_string(index=False))
    print("\nTop rules:\n", rules.sort_values(["lift", "support"], ascending=False)[
        ["antecedents", "consequents", "support", "confidence", "lift"]].head(10).to_string(index=False))
    check = pd.crosstab(basket["Q46: High"], basket["Q32: High"],
                        rownames=["High disability"], colnames=["High fair-or-poor health"])
    print("\nDisability and fair-or-poor health:\n", check)
    save_figures(basket, rules, OUTPUTS / "figures")
    print("Saved results to", OUTPUTS)


if __name__ == "__main__":
    main()
