import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

df = pd.read_csv("Nat_Gas.csv")
df["Dates"] = pd.to_datetime(df["Dates"])
df = df.sort_values("Dates").reset_index(drop=True)

timestamps = np.array([d.timestamp() for d in df["Dates"]])
prices = df["Prices"].values

cs = CubicSpline(timestamps, prices)

last_date = df["Dates"].iloc[-1]
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=12, freq="ME")

months = np.array([d.month for d in df["Dates"]])
seasonal_avg = {}
for m in range(1, 13):
    seasonal_avg[m] = prices[months == m].mean()
overall_avg = prices.mean()
seasonal_factors = {m: seasonal_avg[m] / overall_avg for m in range(1, 13)}

trend_coef = np.polyfit(timestamps, prices, 1)
trend = np.poly1d(trend_coef)

future_prices = []
for d in future_dates:
    trend_projection = trend(d.timestamp())
    seasonal_bump = (seasonal_factors[d.month] - 1) * overall_avg
    projected = trend_projection + seasonal_bump * 0.5
    future_prices.append(projected)

all_dates = list(df["Dates"]) + list(future_dates)
all_timestamps = np.array([d.timestamp() for d in all_dates])
all_prices = list(prices) + future_prices

extended_spline = CubicSpline(all_timestamps, all_prices)

def estimate_price(input_date):
    if isinstance(input_date, str):
        input_date = datetime.strptime(input_date, "%Y-%m-%d")
    t = input_date.timestamp()
    if t < all_timestamps[0] or t > all_timestamps[-1]:
        return f"Date out of range."
    return round(float(extended_spline(t)), 4)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Natural Gas Price Analysis", fontsize=15, fontweight="bold")

ax1 = axes[0, 0]
smooth_range = pd.date_range(df["Dates"].iloc[0], df["Dates"].iloc[-1], periods=500)
smooth_ts = np.array([d.timestamp() for d in smooth_range])
ax1.scatter(df["Dates"], prices, color="steelblue", zorder=5, label="Actual Prices")
ax1.plot(smooth_range, cs(smooth_ts), color="navy", linewidth=1.5, label="Cubic Spline Fit")
ax1.set_title("Historical Prices with Spline Fit")
ax1.set_xlabel("Date")
ax1.set_ylabel("Price ($/MMBtu)")
ax1.legend()
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax1.tick_params(axis="x", rotation=30)

ax2 = axes[0, 1]
month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
month_vals = [seasonal_factors[m] * overall_avg for m in range(1, 13)]
ax2.bar(month_labels, month_vals, color=["#e74c3c" if v > overall_avg else "#3498db" for v in month_vals])
ax2.axhline(overall_avg, color="black", linestyle="--", linewidth=1, label="Overall Average")
ax2.set_title("Average Price by Month (Seasonal Pattern)")
ax2.set_ylabel("Avg Price ($/MMBtu)")
ax2.legend()

ax3 = axes[1, 0]
full_smooth = pd.date_range(all_dates[0], all_dates[-1], periods=800)
full_ts = np.array([d.timestamp() for d in full_smooth])
ax3.plot(df["Dates"], prices, "o-", color="steelblue", markersize=4, label="Historical")
ax3.plot(future_dates, future_prices, "s--", color="tomato", markersize=5, label="Extrapolated (1yr)")
ax3.axvline(last_date, color="gray", linestyle=":", linewidth=1.5, label="Forecast Start")
ax3.plot(full_smooth, extended_spline(full_ts), color="purple", alpha=0.5, linewidth=1, label="Extended Spline")
ax3.set_title("Price History + 1-Year Extrapolation")
ax3.set_xlabel("Date")
ax3.set_ylabel("Price ($/MMBtu)")
ax3.legend(fontsize=8)
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax3.tick_params(axis="x", rotation=30)

ax4 = axes[1, 1]
ax4.scatter(df["Dates"], prices, color="steelblue", zorder=5, label="Actual")
trend_line = trend(timestamps)
ax4.plot(df["Dates"], trend_line, color="red", linewidth=2, linestyle="--", label=f"Trend")
ax4.set_title("Long-Term Upward Trend")
ax4.set_xlabel("Year")
ax4.set_ylabel("Price ($/MMBtu)")
ax4.legend()
ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax4.tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig("nat_gas_analysis.png", dpi=150, bbox_inches="tight")
plt.show()

print("Date            | Estimated Price")
print("-" * 35)
test_dates = ["2021-06-15", "2023-03-01", "2024-08-20", "2025-03-31", "2025-09-15"]
for d in test_dates:
    print(f"{d}  |  ${estimate_price(d)}")