import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from datetime import datetime

df = pd.read_csv("Nat_Gas.csv")
df["Dates"] = pd.to_datetime(df["Dates"])
df = df.sort_values("Dates").reset_index(drop=True)

timestamps = np.array([d.timestamp() for d in df["Dates"]])
prices = df["Prices"].values

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
    tp = trend(d.timestamp()) + (seasonal_factors[d.month] - 1) * overall_avg * 0.5
    future_prices.append(tp)

all_dates = list(df["Dates"]) + list(future_dates)
all_timestamps = np.array([d.timestamp() for d in all_dates])
all_prices = list(prices) + future_prices

extended_spline = CubicSpline(all_timestamps, all_prices)

def estimate_price(date):
    if isinstance(date, str):
        date = datetime.strptime(date, "%Y-%m-%d")
    t = date.timestamp()
    if t < all_timestamps[0] or t > all_timestamps[-1]:
        raise ValueError(f"Date out of range.")
    return round(float(extended_spline(t)), 4)

def price_storage_contract(
    injection_dates,
    withdrawal_dates,
    injection_rate_mmbtu_per_day,
    withdrawal_rate_mmbtu_per_day,
    max_storage_volume,
    monthly_storage_cost
):
    injection_dates = [datetime.strptime(d, "%Y-%m-%d") if isinstance(d, str) else d for d in injection_dates]
    withdrawal_dates = [datetime.strptime(d, "%Y-%m-%d") if isinstance(d, str) else d for d in withdrawal_dates]

    all_events = []
    for d in injection_dates:
        all_events.append((d, "inject"))
    for d in withdrawal_dates:
        all_events.append((d, "withdraw"))
    all_events.sort(key=lambda x: x[0])

    current_volume = 0.0
    total_injection_cost = 0.0
    total_withdrawal_revenue = 0.0
    volume_log = []

    for date, action in all_events:
        price = estimate_price(date)
        if action == "inject":
            space_left = max_storage_volume - current_volume
            volume_to_inject = min(injection_rate_mmbtu_per_day, space_left)
            if volume_to_inject <= 0:
                print(f"  [!] {date.date()} - Storage full, cannot inject.")
                continue
            cost = volume_to_inject * price
            current_volume += volume_to_inject
            total_injection_cost += cost
            volume_log.append((date, "INJECT", volume_to_inject, price, current_volume))
        elif action == "withdraw":
            volume_to_withdraw = min(withdrawal_rate_mmbtu_per_day, current_volume)
            if volume_to_withdraw <= 0:
                print(f"  [!] {date.date()} - Storage empty, cannot withdraw.")
                continue
            revenue = volume_to_withdraw * price
            current_volume -= volume_to_withdraw
            total_withdrawal_revenue += revenue
            volume_log.append((date, "WITHDRAW", volume_to_withdraw, price, current_volume))

    first_event = volume_log[0][0]
    last_event = volume_log[-1][0]
    months_active = max(1, (last_event.year - first_event.year) * 12 + (last_event.month - first_event.month))
    total_storage_cost = monthly_storage_cost * months_active

    contract_value = total_withdrawal_revenue - total_injection_cost - total_storage_cost

    print("=" * 55)
    print("       NATURAL GAS STORAGE CONTRACT PRICER")
    print("=" * 55)
    print(f"  {'Date':<14} {'Action':<10} {'Volume':>12} {'Price':>8} {'Storage':>12}")
    print("-" * 55)
    for date, action, vol, price, storage in volume_log:
        print(f"  {str(date.date()):<14} {action:<10} {vol:>12,.1f} {price:>8.4f} {storage:>12,.1f}")
    print("-" * 55)
    print(f"  Months Active         : {months_active}")
    print(f"  Max Storage Volume    : {max_storage_volume:>15,.1f} MMBtu")
    print("-" * 55)
    print(f"  Total Revenue         : ${total_withdrawal_revenue:>15,.2f}")
    print(f"  Total Purchase Cost   : ${total_injection_cost:>15,.2f}")
    print(f"  Total Storage Cost    : ${total_storage_cost:>15,.2f}")
    print("=" * 55)
    print(f"  CONTRACT VALUE        : ${contract_value:>15,.2f}")
    print("=" * 55)

    return contract_value

print("\n--- TEST 1: Single inject, single withdraw ---")
price_storage_contract(
    injection_dates=["2024-06-30"],
    withdrawal_dates=["2024-12-31"],
    injection_rate_mmbtu_per_day=500000,
    withdrawal_rate_mmbtu_per_day=500000,
    max_storage_volume=2000000,
    monthly_storage_cost=100000
)

print("\n--- TEST 2: Two injections, two withdrawals ---")
price_storage_contract(
    injection_dates=["2024-04-30", "2024-06-30"],
    withdrawal_dates=["2024-11-30", "2024-12-31"],
    injection_rate_mmbtu_per_day=400000,
    withdrawal_rate_mmbtu_per_day=400000,
    max_storage_volume=1000000,
    monthly_storage_cost=80000
)