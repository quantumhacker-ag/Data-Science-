"""COVID-19 India EDA — script version of notebooks/MiniProject_COVID19_India_EDA.ipynb.
Run from project root:  python src/covid_eda.py
"""
import matplotlib
matplotlib.use("Agg") if __name__ == "__main__" else None


# %% [markdown]
# # Mini Project — Exploratory Data Analysis on India's COVID-19 Data
#
# **Goal:** Uncover state-wise trends, recovery rates, and vaccination progress in India's COVID-19 pandemic using NumPy, Pandas and Matplotlib/Seaborn.
#
# **Data sources**
# - **Case data:** `imdevskp/covid-19-india-data` — daily state-wise confirmed / recovered / death counts (30 Jan 2020 – 6 Aug 2020).
# - **Vaccination data:** Our World in Data (OWID) — India national vaccination totals (from 15 Jan 2021).
#
# Data is read from the local `data/` folder; if missing, it is downloaded from the public GitHub URLs automatically.
#
# **Workflow:** Load → Inspect → Clean → Explore (groupby / pivot) → Visualise (12 charts) → Summarise.

# %% [markdown]
# ## 1. Setup

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
pd.set_option("display.max_columns", 20)

# Project folders (works from notebooks/ or project root)
ROOT = Path.cwd().parent if Path.cwd().name in ("notebooks", "src") else Path.cwd()
DATA_DIR = ROOT / "data"
IMG_DIR = ROOT / "images"
DATA_DIR.mkdir(exist_ok=True)
IMG_DIR.mkdir(exist_ok=True)

CASES_URL = "https://raw.githubusercontent.com/imdevskp/covid-19-india-data/master/complete.csv"
VAX_URL = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/country_data/India.csv"


def load_csv(filename, url):
    """Read CSV from data/ if present, otherwise download it and cache it."""
    path = DATA_DIR / filename
    if path.exists():
        return pd.read_csv(path)
    df = pd.read_csv(url)
    df.to_csv(path, index=False)
    return df


def finish(fig, name):
    """Tidy layout, save the chart to images/, and show it."""
    fig.tight_layout()
    fig.savefig(IMG_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig) if matplotlib.get_backend().lower()=='agg' else plt.show()


print("Ready.")

# %% [markdown]
# ## 2. Load the Data

# %%
cases = load_csv("complete.csv", CASES_URL)
vax = load_csv("India_vaccinations.csv", VAX_URL)

print("Cases shape:", cases.shape)
print("Vaccination shape:", vax.shape)
print(cases.head())

# %% [markdown]
# ## 3. Inspect Before Touching Anything

# %%
print(cases.dtypes)

# %%
print(cases.isna().sum())

# %%
print("Number of unique state/UT names:", cases["Name of State / UT"].nunique())
print(sorted(cases["Name of State / UT"].unique()))

# %% [markdown]
# Problems found:
#
# 1. **`Death` is stored as text**, not a number (stray characters).
# 2. **Same state, several spellings**: `Telangana`, `Telengana`, `Telangana***`; some UTs also appear with a `"Union Territory of ..."` prefix.

# %% [markdown]
# ## 4. Clean the Data

# %%
def clean_cases(df):
    df = df.copy()
    df.columns = ["date", "state", "lat", "long", "confirmed",
                  "deaths", "cured", "new_cases", "new_deaths", "new_recovered"]

    df["date"] = pd.to_datetime(df["date"])
    df["deaths"] = pd.to_numeric(df["deaths"], errors="coerce").fillna(0).astype(int)
    df["confirmed"] = df["confirmed"].astype(int)
    df["cured"] = df["cured"].astype(int)

    name_fix = {
        "Telangana***": "Telangana",
        "Telengana": "Telangana",
        "Union Territory of Jammu and Kashmir": "Jammu and Kashmir",
        "Union Territory of Ladakh": "Ladakh",
        "Union Territory of Chandigarh": "Chandigarh",
    }
    df["state"] = df["state"].replace(name_fix)

    df = (df.drop_duplicates(subset=["date", "state"])
            .sort_values(["state", "date"])
            .reset_index(drop=True))

    df["active"] = df["confirmed"] - df["deaths"] - df["cured"]
    df["month"] = df["date"].dt.strftime("%Y-%m")
    return df


cases = clean_cases(cases)
print("Clean state count:", cases["state"].nunique())
print("Missing values   :", cases.isna().sum().sum())
cases.dtypes

# %% [markdown]
# ## 5. National-Level Summary (groupby)

# %%
national = cases.groupby("date", as_index=False)[["confirmed", "deaths", "cured"]].sum()
national["recovery_rate"] = (national["cured"] / national["confirmed"] * 100).round(2)
national["death_rate"] = (national["deaths"] / national["confirmed"] * 100).round(2)
national["new_confirmed"] = national["confirmed"].diff().fillna(national["confirmed"])
national["new_7d_avg"] = national["new_confirmed"].rolling(7, min_periods=1).mean()
national["month"] = national["date"].dt.strftime("%Y-%m")

print(national.tail())

# %%
latest_date = cases["date"].max()
latest_row = national.iloc[-1]

print(f"As of {latest_date.date()}:")
print(f"  Total confirmed : {int(latest_row['confirmed']):,}")
print(f"  Total recovered : {int(latest_row['cured']):,}")
print(f"  Total deaths    : {int(latest_row['deaths']):,}")
print(f"  Recovery rate   : {latest_row['recovery_rate']}%")
print(f"  Death rate      : {latest_row['death_rate']}%")

# %% [markdown]
# ## 6. State-Wise Snapshot (latest date)

# %%
latest = cases[cases["date"] == latest_date].copy()
latest["recovery_rate"] = (latest["cured"] / latest["confirmed"] * 100).round(2)
latest["death_rate"] = (latest["deaths"] / latest["confirmed"] * 100).round(2)
latest = latest.sort_values("confirmed", ascending=False).reset_index(drop=True)

top10 = latest.head(10)
top5_states = top10["state"].head(5).tolist()
print(top10[["state", "confirmed", "deaths", "cured", "recovery_rate", "death_rate"]])

# %% [markdown]
# **pivot_table**: new cases by state and month for the top 5 states.

# %%
month_pivot = (cases[cases["state"].isin(top5_states)]
               .pivot_table(values="new_cases", index="state", columns="month",
                            aggfunc="sum", fill_value=0))
print(month_pivot)

# %% [markdown]
# ## 7. Visualisations (12 charts)
#
# All charts are also saved as PNGs in `images/`.

# %% [markdown]
# ### Chart 1 — National cumulative trend

# %%
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(national["date"], national["confirmed"], label="Confirmed", color="#2563eb", lw=2)
ax.plot(national["date"], national["cured"], label="Recovered", color="#16a34a", lw=2)
ax.plot(national["date"], national["deaths"], label="Deaths", color="#dc2626", lw=2)
ax.set_title("India: Cumulative COVID-19 Cases Over Time")
ax.set_xlabel("Date"); ax.set_ylabel("People")
ax.legend()
fig.autofmt_xdate()
finish(fig, "01_national_cumulative")

# %% [markdown]
# ### Chart 2 — Daily new confirmed cases (with 7-day average)

# %%
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(national["date"], national["new_confirmed"], color="#93c5fd", width=1.0, label="Daily")
ax.plot(national["date"], national["new_7d_avg"], color="#1d4ed8", lw=2, label="7-day average")
ax.set_title("India: Daily New Confirmed Cases")
ax.set_xlabel("Date"); ax.set_ylabel("New cases")
ax.legend()
fig.autofmt_xdate()
finish(fig, "02_daily_new_cases")

# %% [markdown]
# ### Chart 3 — Top 10 states by total confirmed cases

# %%
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=top10, y="state", x="confirmed", hue="state", palette="Blues_r", legend=False, ax=ax)
ax.set_title(f"Top 10 States by Total Confirmed Cases (as of {latest_date.date()})")
ax.set_xlabel("Confirmed cases"); ax.set_ylabel("")
finish(fig, "03_top10_states")

# %% [markdown]
# ### Chart 4 — Recovery rate of the 10 worst-hit states

# %%
top10_sorted = top10.sort_values("recovery_rate")
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=top10_sorted, y="state", x="recovery_rate", hue="state", palette="Greens", legend=False, ax=ax)
ax.set_title("Recovery Rate (%) — Top 10 Worst-Hit States")
ax.set_xlabel("Recovery rate (%)"); ax.set_ylabel("")
finish(fig, "04_recovery_rate_top10")

# %% [markdown]
# ### Chart 5 — Confirmed-case trend for the top 5 states

# %%
fig, ax = plt.subplots(figsize=(8, 4.5))
for s in top5_states:
    sub = cases[cases["state"] == s]
    ax.plot(sub["date"], sub["confirmed"], label=s, lw=2)
ax.set_title("Confirmed Cases Over Time — Top 5 States")
ax.set_xlabel("Date"); ax.set_ylabel("Confirmed cases")
ax.legend(fontsize=8)
fig.autofmt_xdate()
finish(fig, "05_top5_trend")

# %% [markdown]
# ### Chart 6 — Share of national cases by state (pie)

# %%
top6 = latest.nlargest(6, "confirmed")[["state", "confirmed"]]
others = latest["confirmed"].sum() - top6["confirmed"].sum()
pie_labels = top6["state"].tolist() + ["Rest of India"]
pie_values = top6["confirmed"].tolist() + [others]

fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.pie(pie_values, labels=pie_labels, autopct="%1.1f%%", startangle=90,
       colors=sns.color_palette("Blues_r", 7))
ax.set_title("Share of Total Confirmed Cases by State")
finish(fig, "06_share_pie")

# %% [markdown]
# ### Chart 7 — National recovery-rate trend

# %%
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(national["date"], national["recovery_rate"], color="#16a34a", lw=2)
ax.set_title("India: National Recovery Rate Over Time")
ax.set_xlabel("Date"); ax.set_ylabel("Recovery rate (%)")
fig.autofmt_xdate()
finish(fig, "07_recovery_trend")

# %% [markdown]
# ### Chart 8 — Recovery rate vs death rate, by state (bubble = case count)

# %%
sizeable = latest[latest["confirmed"] >= 1000]

fig, ax = plt.subplots(figsize=(7, 5.5))
sns.scatterplot(data=sizeable, x="recovery_rate", y="death_rate", size="confirmed",
                sizes=(40, 600), hue="confirmed", palette="Blues", legend=False, ax=ax)
for _, r in sizeable.nlargest(6, "confirmed").iterrows():
    ax.annotate(r["state"], (r["recovery_rate"], r["death_rate"]), fontsize=7,
                xytext=(3, 3), textcoords="offset points")
ax.set_title("Recovery Rate vs Death Rate by State (bubble = case count)")
ax.set_xlabel("Recovery rate (%)"); ax.set_ylabel("Death rate (%)")
finish(fig, "08_recovery_vs_death")

# %% [markdown]
# ### Chart 9 — Correlation heatmap of national metrics

# %%
corr_cols = ["confirmed", "deaths", "cured", "new_confirmed", "recovery_rate", "death_rate"]
corr = national[corr_cols].corr()

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation Between National COVID Metrics")
finish(fig, "09_correlation_heatmap")

# %% [markdown]
# ### Chart 10 — Spread of daily new cases, by month (boxplot)

# %%
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.boxplot(data=national, x="month", y="new_confirmed", hue="month", palette="Blues", legend=False, ax=ax)
ax.set_title("Spread of Daily New Cases by Month")
ax.set_xlabel("Month"); ax.set_ylabel("Daily new cases")
finish(fig, "10_monthly_boxplot")

# %% [markdown]
# ### Chart 11 — India's vaccination progress

# %%
vax["date"] = pd.to_datetime(vax["date"])
vax_years = f"{vax['date'].min().year}-{vax['date'].max().year}"

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(vax["date"], vax["total_vaccinations"] / 1e9, label="Total doses (Bn)", color="#7c3aed", lw=2)
ax.plot(vax["date"], vax["people_fully_vaccinated"] / 1e9, label="Fully vaccinated (Bn)", color="#0d9488", lw=2)
ax.set_title(f"India: COVID-19 Vaccination Progress ({vax_years})")
ax.set_xlabel("Date"); ax.set_ylabel("People / doses (in billions)")
ax.legend()
fig.autofmt_xdate()
finish(fig, "11_vaccination_progress")

# %% [markdown]
# ### Chart 12 — The 10 least-affected states/UTs

# %%
bottom10 = latest[latest["confirmed"] > 0].nsmallest(10, "confirmed")

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=bottom10, y="state", x="confirmed", hue="state", palette="Oranges", legend=False, ax=ax)
ax.set_title(f"10 Least-Affected States/UTs by Confirmed Cases (as of {latest_date.date()})")
ax.set_xlabel("Confirmed cases"); ax.set_ylabel("")
finish(fig, "12_least_affected")

# %% [markdown]
# ## 8. Save the Cleaned Dataset

# %%
out = DATA_DIR / "cleaned_covid_india.csv"
cases.to_csv(out, index=False)
print(f"Saved {out.name} — {cases.shape[0]} rows.")

# %% [markdown]
# ## 9. Key Findings (computed from the data)

# %%
peak = national.loc[national["new_confirmed"].idxmax()]
top_state = latest.iloc[0]
big = latest[latest["confirmed"] >= 10000]
best_rec = big.loc[big["recovery_rate"].idxmax()]
worst_rec = big.loc[big["recovery_rate"].idxmin()]
worst_death = big.loc[big["death_rate"].idxmax()]
top6_share = top6["confirmed"].sum() / latest["confirmed"].sum() * 100
last_vax = vax.dropna(subset=["total_vaccinations"]).iloc[-1]

print(f"Coverage     : {cases['state'].nunique()} states/UTs, {national.shape[0]} days "
      f"({cases['date'].min().date()} to {latest_date.date()})")
print(f"National     : {int(latest_row['confirmed']):,} confirmed, {int(latest_row['cured']):,} recovered, "
      f"{int(latest_row['deaths']):,} deaths")
print(f"Rates        : recovery {latest_row['recovery_rate']}%, death {latest_row['death_rate']}%")
print(f"Peak day     : {int(peak['new_confirmed']):,} new cases on {peak['date'].date()}")
print(f"Top state    : {top_state['state']} = {top_state['confirmed'] / latest['confirmed'].sum() * 100:.1f}% of all cases")
print(f"Top 6 share  : {top6_share:.1f}% of all cases")
print(f"Best recovery (>=10k cases): {best_rec['state']} {best_rec['recovery_rate']}%")
print(f"Worst recovery (>=10k)     : {worst_rec['state']} {worst_rec['recovery_rate']}%")
print(f"Highest death rate (>=10k) : {worst_death['state']} {worst_death['death_rate']}%")
print(f"Vaccination  : {int(last_vax['total_vaccinations']):,} doses by {last_vax['date'].date()}")

# %% [markdown]
# **Caveats.** State-bulletin data inherits reporting inconsistencies (hence the spelling / text-`Death` fixes). Recovery and death rates are simple ratios of cumulative totals, not adjusted for reporting lag — read them as directional trends. The case data ends Aug 2020 (first wave, pre-vaccine); the vaccination data starts Jan 2021.
