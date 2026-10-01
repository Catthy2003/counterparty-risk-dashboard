import numpy as np, pandas as pd

rng = np.random.default_rng(42)
TODAY = pd.Timestamp("2026-09-28")
N_CP, N_CONTRACTS = 50, 2000

# 1) Counterparties
cp = pd.DataFrame({
    "counterparty_id": range(1, N_CP + 1),
    "name": [f"Counterparty_{i:02d}" for i in range(1, N_CP + 1)],
    "type": rng.choice(["Buyer", "Supplier"], N_CP, p=[.6, .4]),
    "country": rng.choice(["Vietnam", "Brazil", "Uganda", "Indonesia", "Germany", "Switzerland"], N_CP),
    "kyc_status": rng.choice(["Approved", "Pending", "Expired"], N_CP, p=[.85, .08, .07]),
    "last_review_date": TODAY - pd.to_timedelta(rng.integers(30, 600, N_CP), unit="D"),
})

# 2) Limits
limits = pd.DataFrame({
    "counterparty_id": cp["counterparty_id"],
    "credit_limit_usd": rng.choice([200_000, 500_000, 1_000_000, 2_000_000], N_CP),
    "max_forward_days": rng.choice([60, 90, 120], N_CP),
    "max_fixation_days": rng.choice([30, 45, 60], N_CP),
    "payment_terms_days": rng.choice([0, 30, 45], N_CP),   # 0 = in trust / thanh toán ngay
    "insurance_covered": rng.choice(["Y", "N"], N_CP, p=[.6, .4]),
})

# 3) Market prices (random walk, USD/MT)
dates = pd.date_range(TODAY - pd.Timedelta(days=365), TODAY)
def walk(start, vol):
    return start * np.exp(np.cumsum(rng.normal(0, vol, len(dates))))
prices = pd.concat([
    pd.DataFrame({"date": dates, "commodity": "Robusta", "price_usd_mt": walk(4200, .012)}),
    pd.DataFrame({"date": dates, "commodity": "Arabica", "price_usd_mt": walk(7000, .014)}),
])

# 4) Contracts
k = pd.DataFrame({
    "contract_id": range(1, N_CONTRACTS + 1),
    "counterparty_id": rng.integers(1, N_CP + 1, N_CONTRACTS),
    "commodity": rng.choice(["Robusta", "Arabica"], N_CONTRACTS, p=[.6, .4]),
    "qty_mt": rng.choice([19, 38, 57, 96], N_CONTRACTS),
    "contract_date": TODAY - pd.to_timedelta(rng.integers(0, 300, N_CONTRACTS), unit="D"),
})
k = k.merge(prices, left_on=["contract_date", "commodity"], right_on=["date", "commodity"]).drop(columns="date")
k["price_usd_mt"] = (k["price_usd_mt"] * rng.normal(1, .03, len(k))).round(2)
k["contract_value_usd"] = (k["qty_mt"] * k["price_usd_mt"]).round(2)

k = k.merge(limits[["counterparty_id", "payment_terms_days"]], on="counterparty_id")
k["shipment_date"] = k["contract_date"] + pd.to_timedelta(rng.integers(30, 150, len(k)), unit="D")
k["fixation_date"] = k["contract_date"] + pd.to_timedelta(rng.integers(15, 80, len(k)), unit="D")
k["payment_due_date"] = k["shipment_date"] + pd.to_timedelta(k["payment_terms_days"], unit="D")
k["prepayment_usd"] = np.where(rng.random(len(k)) < .10, (k["contract_value_usd"] * rng.uniform(.1, .2, len(k))).round(2), 0)
k["status"] = np.where(k["shipment_date"] >= TODAY, "Open", "Shipped")
k = k.drop(columns="payment_terms_days")

# 5) Invoices (chỉ cho hợp đồng đã giao)
inv = k.loc[k["status"] == "Shipped", ["contract_id", "counterparty_id", "contract_value_usd", "payment_due_date"]].copy()
inv.columns = ["contract_id", "counterparty_id", "amount_usd", "due_date"]
inv.insert(0, "invoice_id", range(1, len(inv) + 1))
overdue = inv["due_date"] < TODAY
paid = overdue & (rng.random(len(inv)) < .85)
delay = pd.to_timedelta(rng.integers(0, 25, len(inv)), unit="D")
inv["paid_date"] = pd.NaT
inv.loc[paid, "paid_date"] = (inv.loc[paid, "due_date"] + delay[paid]).clip(upper=TODAY)

# Xuất file
for name, df in {"counterparties": cp, "limits": limits, "market_prices": prices,
                 "contracts": k, "invoices": inv}.items():
    df.to_csv(f"data/{name}.csv", index=False)
    print(name, df.shape)