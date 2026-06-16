import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("Task_3_and_4_Loan_Data.csv")
fico = df["fico_score"].values
default = df["default"].values

def log_likelihood(defaults, total):
    if total == 0:
        return 0
    p = defaults / total
    if p == 0 or p == 1:
        return 0
    return defaults * np.log(p) + (total - defaults) * np.log(1 - p)

def dp_optimal_buckets(fico, default, n_buckets):
    min_score = int(fico.min())
    max_score = int(fico.max())
    scores = np.arange(min_score, max_score + 1)
    n = len(scores)

    total_per_score = np.zeros(n)
    default_per_score = np.zeros(n)
    for s, d in zip(fico, default):
        idx = int(s) - min_score
        total_per_score[idx] += 1
        default_per_score[idx] += d

    cum_total = np.cumsum(total_per_score)
    cum_default = np.cumsum(default_per_score)

    def range_ll(i, j):
        total = cum_total[j] - (cum_total[i-1] if i > 0 else 0)
        defs = cum_default[j] - (cum_default[i-1] if i > 0 else 0)
        return log_likelihood(defs, total)

    INF = float("-inf")
    dp = [[INF] * n for _ in range(n_buckets + 1)]
    split = [[0] * n for _ in range(n_buckets + 1)]

    for j in range(n):
        dp[1][j] = range_ll(0, j)

    for b in range(2, n_buckets + 1):
        for j in range(b - 1, n):
            for i in range(b - 1, j + 1):
                val = dp[b-1][i-1] + range_ll(i, j) if i > 0 else INF
                if val > dp[b][j]:
                    dp[b][j] = val
                    split[b][j] = i

    boundaries = []
    j = n - 1
    for b in range(n_buckets, 1, -1):
        i = split[b][j]
        boundaries.append(scores[i] - 1)
        j = i - 1
    boundaries.reverse()
    boundaries = [min_score - 1] + boundaries + [max_score]
    return boundaries

def equal_width_buckets(fico, n_buckets):
    min_score, max_score = int(fico.min()), int(fico.max())
    step = (max_score - min_score) / n_buckets
    boundaries = [min_score - 1] + [int(min_score + step * i) for i in range(1, n_buckets)] + [max_score]
    return boundaries

def assign_buckets(fico, default, boundaries):
    n_buckets = len(boundaries) - 1
    labels, pds, counts = [], [], []
    for i in range(n_buckets):
        lo, hi = boundaries[i], boundaries[i+1]
        mask = (fico > lo) & (fico <= hi)
        n = mask.sum()
        k = default[mask].sum()
        pd_val = k / n if n > 0 else 0
        labels.append(f"{lo+1}-{hi}")
        pds.append(pd_val)
        counts.append(n)
    return labels, pds, counts

N_BUCKETS = 5

dp_bounds = dp_optimal_buckets(fico, default, N_BUCKETS)
ew_bounds = equal_width_buckets(fico, N_BUCKETS)

dp_labels, dp_pds, dp_counts = assign_buckets(fico, default, dp_bounds)
ew_labels, ew_pds, ew_counts = assign_buckets(fico, default, ew_bounds)

def predict_pd(fico_score):
    for i in range(len(dp_bounds) - 1):
        if dp_bounds[i] < fico_score <= dp_bounds[i+1]:
            print(f"  FICO Score : {fico_score}")
            print(f"  Bucket     : {dp_labels[i]}")
            print(f"  PD         : {dp_pds[i]*100:.2f}%")
            return dp_pds[i]
    print("Out of range")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("FICO Score Bucketing: DP Optimal vs Equal Width", fontsize=14, fontweight="bold")

x = np.arange(N_BUCKETS)

axes[0, 0].bar(x, dp_pds, color="steelblue", width=0.5)
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(dp_labels, rotation=20, fontsize=8)
axes[0, 0].set_title("DP Optimal — PD per Bucket")
axes[0, 0].set_ylabel("Probability of Default")

axes[0, 1].bar(x, ew_pds, color="tomato", width=0.5)
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(ew_labels, rotation=20, fontsize=8)
axes[0, 1].set_title("Equal Width — PD per Bucket")
axes[0, 1].set_ylabel("Probability of Default")

axes[1, 0].bar(x, dp_counts, color="steelblue", width=0.5)
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(dp_labels, rotation=20, fontsize=8)
axes[1, 0].set_title("DP Optimal — Borrowers per Bucket")
axes[1, 0].set_ylabel("Count")

axes[1, 1].bar(x, ew_counts, color="tomato", width=0.5)
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(ew_labels, rotation=20, fontsize=8)
axes[1, 1].set_title("Equal Width — Borrowers per Bucket")
axes[1, 1].set_ylabel("Count")

plt.tight_layout()
plt.savefig("fico_bucketing.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n" + "=" * 50)
print("   DP OPTIMAL BUCKET BOUNDARIES & PD TABLE")
print("=" * 50)
print(f"  {'Bucket':<12} {'Count':>8} {'PD':>10}")
print("-" * 50)
for label, pd_val, count in zip(dp_labels, dp_pds, dp_counts):
    print(f"  {label:<12} {count:>8} {pd_val*100:>9.2f}%")

print("\n--- SAMPLE PREDICTIONS ---")
for score in [480, 550, 620, 680, 750]:
    print()
    predict_pd(score)