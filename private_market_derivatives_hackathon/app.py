
import math
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(
    page_title="PrivIndex AI",
    page_icon="📈",
    layout="wide",
)

SEED = 42
rng = np.random.default_rng(SEED)


@dataclass
class Contract:
    contract_type: str
    direction: str
    entry_level: float
    notional: float
    strike: float | None
    maturity_step: int


def generate_private_company_data(periods: int = 36) -> pd.DataFrame:
    """Generate synthetic private-company fundamentals for a hackathon demo."""
    companies = [
        ("Nexa Health", "Healthcare"),
        ("Sahab Logistics", "Logistics"),
        ("Qimam AI", "Technology"),
        ("Rimal Energy", "Energy"),
        ("Madar Foods", "Consumer"),
        ("Wasl Fintech", "Fintech"),
    ]

    rows = []
    start = date.today() - timedelta(days=30 * periods)

    for company_id, (name, sector) in enumerate(companies):
        revenue = rng.uniform(18, 70)
        margin = rng.uniform(0.08, 0.25)
        cash_runway = rng.uniform(12, 32)
        customer_growth = rng.uniform(0.03, 0.12)
        debt_ratio = rng.uniform(0.05, 0.55)
        delivery_score = rng.uniform(0.70, 0.95)
        governance = rng.uniform(0.60, 0.95)

        for t in range(periods):
            shock = rng.normal(0, 0.035)
            revenue *= 1 + customer_growth / 4 + shock
            margin = np.clip(margin + rng.normal(0, 0.012), -0.05, 0.40)
            cash_runway = np.clip(cash_runway + rng.normal(-0.10, 0.7), 3, 40)
            customer_growth = np.clip(customer_growth + rng.normal(0, 0.008), -0.05, 0.25)
            debt_ratio = np.clip(debt_ratio + rng.normal(0, 0.018), 0, 0.90)
            delivery_score = np.clip(delivery_score + rng.normal(0, 0.018), 0.35, 1.0)
            governance = np.clip(governance + rng.normal(0, 0.010), 0.35, 1.0)

            # Inject a visible stress event into one company.
            if name == "Sahab Logistics" and t == periods - 8:
                margin -= 0.10
                debt_ratio += 0.18
                delivery_score -= 0.18

            rows.append(
                {
                    "date": start + timedelta(days=30 * t),
                    "company": name,
                    "sector": sector,
                    "revenue_m": revenue,
                    "ebitda_margin": margin,
                    "cash_runway_months": cash_runway,
                    "customer_growth": customer_growth,
                    "debt_ratio": debt_ratio,
                    "delivery_score": delivery_score,
                    "governance_score": governance,
                }
            )

    return pd.DataFrame(rows)


@st.cache_data
def load_data() -> pd.DataFrame:
    return generate_private_company_data()


def score_companies(df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Convert heterogeneous private-market metrics into normalized 0-100 scores."""
    work = df.copy()

    positive = [
        "revenue_m",
        "ebitda_margin",
        "cash_runway_months",
        "customer_growth",
        "delivery_score",
        "governance_score",
    ]
    negative = ["debt_ratio"]

    for col in positive:
        values = work[[col]].fillna(work[col].median())
        work[f"{col}_score"] = MinMaxScaler().fit_transform(values)[:, 0] * 100

    for col in negative:
        values = work[[col]].fillna(work[col].median())
        work[f"{col}_score"] = (1 - MinMaxScaler().fit_transform(values)[:, 0]) * 100

    feature_cols = [
        "revenue_m",
        "ebitda_margin",
        "cash_runway_months",
        "customer_growth",
        "debt_ratio",
        "delivery_score",
        "governance_score",
    ]
    model = IsolationForest(contamination=0.06, random_state=SEED)
    model.fit(work[feature_cols])
    raw_anomaly = -model.score_samples(work[feature_cols])
    work["anomaly_risk"] = MinMaxScaler().fit_transform(raw_anomaly.reshape(-1, 1))[:, 0] * 100

    total_weight = max(sum(weights.values()), 1e-9)
    weighted = np.zeros(len(work))
    for metric, weight in weights.items():
        weighted += work[f"{metric}_score"].to_numpy() * (weight / total_weight)

    # AI layer: penalize unusual observations, but not so much that one anomaly destroys the index.
    work["ai_company_score"] = np.clip(weighted - 0.18 * work["anomaly_risk"], 0, 100)
    return work


def build_index(scored: pd.DataFrame, company_weights: pd.Series) -> pd.DataFrame:
    """Build a rebalanced composite index with base value 100."""
    latest_names = list(company_weights.index)
    filtered = scored[scored["company"].isin(latest_names)].copy()

    daily = (
        filtered.pivot(index="date", columns="company", values="ai_company_score")
        .sort_index()
        .ffill()
    )
    weights = company_weights.reindex(daily.columns).fillna(0).to_numpy()
    weights = weights / weights.sum()

    raw = daily.to_numpy() @ weights
    index_level = 100 * raw / raw[0]

    result = pd.DataFrame(
        {
            "date": daily.index,
            "index_level": index_level,
        }
    )
    result["return"] = result["index_level"].pct_change().fillna(0)
    result["volatility_annualized"] = (
        result["return"].rolling(6, min_periods=3).std() * math.sqrt(12) * 100
    )
    result["momentum_3m"] = result["index_level"].pct_change(3) * 100
    return result


def settle_contract(contract: Contract, final_level: float) -> float:
    multiplier = contract.notional / max(contract.entry_level, 1e-9)

    if contract.contract_type == "Index Future":
        pnl = (final_level - contract.entry_level) * multiplier
        return pnl if contract.direction == "Long" else -pnl

    if contract.contract_type == "Call Option":
        intrinsic = max(final_level - float(contract.strike), 0)
        return intrinsic * multiplier

    if contract.contract_type == "Put Option":
        intrinsic = max(float(contract.strike) - final_level, 0)
        return intrinsic * multiplier

    if contract.contract_type == "Volatility Swap":
        # Entry level is interpreted as the volatility strike here.
        realized_vol = final_level
        return (realized_vol - contract.entry_level) * contract.notional / 100

    return 0.0


def forecast_paths(index_df: pd.DataFrame, months: int = 6, paths: int = 1200) -> np.ndarray:
    returns = index_df["return"].dropna()
    mu = float(returns.mean())
    sigma = float(max(returns.std(), 0.01))
    last = float(index_df["index_level"].iloc[-1])

    simulations = np.zeros((paths, months + 1))
    simulations[:, 0] = last

    for t in range(1, months + 1):
        z = rng.normal(0, 1, paths)
        simulations[:, t] = simulations[:, t - 1] * np.exp(
            (mu - 0.5 * sigma**2) + sigma * z
        )
    return simulations


def ai_explanation(latest: pd.DataFrame, previous: pd.DataFrame) -> list[str]:
    merged = latest.merge(previous, on="company", suffixes=("_now", "_prev"))
    merged["score_change"] = merged["ai_company_score_now"] - merged["ai_company_score_prev"]
    strongest = merged.sort_values("score_change", ascending=False).iloc[0]
    weakest = merged.sort_values("score_change").iloc[0]

    messages = [
        f"أقوى مساهم إيجابي: **{strongest['company']}**، تحسن تقييمها بمقدار "
        f"{strongest['score_change']:.1f} نقطة.",
        f"أكبر ضغط سلبي: **{weakest['company']}**، انخفض تقييمها بمقدار "
        f"{abs(weakest['score_change']):.1f} نقطة.",
    ]

    stressed = latest.sort_values("anomaly_risk", ascending=False).iloc[0]
    messages.append(
        f"أعلى إشارة شذوذ: **{stressed['company']}** بدرجة مخاطر "
        f"{stressed['anomaly_risk']:.0f}/100؛ يلزم تحقق بشري من البيانات."
    )
    return messages


st.title("PrivIndex AI")
st.caption(
    "نموذج هاكاثون: مؤشر تقييم حي لشركات خاصة + مشتقات افتراضية مسوّاة نقدًا. "
    "لا ينفذ تداولًا حقيقيًا ولا يمثل منتجًا استثماريًا مرخصًا."
)

data = load_data()

with st.sidebar:
    st.header("إعداد المؤشر")
    st.write("عدّل أهمية كل عامل:")
    weights = {
        "revenue_m": st.slider("الإيرادات", 0, 30, 18),
        "ebitda_margin": st.slider("هامش EBITDA", 0, 30, 20),
        "cash_runway_months": st.slider("السيولة المتاحة", 0, 30, 14),
        "customer_growth": st.slider("نمو العملاء", 0, 30, 16),
        "debt_ratio": st.slider("سلامة المديونية", 0, 30, 12),
        "delivery_score": st.slider("تنفيذ المشاريع", 0, 30, 10),
        "governance_score": st.slider("الحوكمة", 0, 30, 10),
    }

scored = score_companies(data, weights)
companies = sorted(scored["company"].unique())

selected = st.multiselect(
    "الشركات الداخلة في السلة",
    companies,
    default=companies,
)

if not selected:
    st.warning("اختر شركة واحدة على الأقل.")
    st.stop()

default_weight = 1 / len(selected)
weight_editor = pd.DataFrame(
    {"company": selected, "weight": [default_weight] * len(selected)}
)
edited = st.data_editor(
    weight_editor,
    hide_index=True,
    use_container_width=True,
    column_config={
        "company": st.column_config.TextColumn("الشركة", disabled=True),
        "weight": st.column_config.NumberColumn(
            "الوزن", min_value=0.0, max_value=1.0, step=0.05, format="%.2f"
        ),
    },
)
company_weights = edited.set_index("company")["weight"]
if company_weights.sum() <= 0:
    st.error("مجموع الأوزان يجب أن يكون أكبر من صفر.")
    st.stop()

index_df = build_index(scored, company_weights)
latest_date = scored["date"].max()
previous_date = sorted(scored["date"].unique())[-2]
latest = scored[(scored["date"] == latest_date) & scored["company"].isin(selected)]
previous = scored[(scored["date"] == previous_date) & scored["company"].isin(selected)]

current_level = float(index_df["index_level"].iloc[-1])
monthly_change = float(index_df["return"].iloc[-1] * 100)
vol = float(index_df["volatility_annualized"].iloc[-1])

m1, m2, m3, m4 = st.columns(4)
m1.metric("مستوى المؤشر", f"{current_level:.2f}", f"{monthly_change:+.2f}%")
m2.metric("التقلب السنوي", f"{vol:.1f}%")
m3.metric("عدد الشركات", len(selected))
m4.metric("آخر تحديث", str(latest_date))

tab1, tab2, tab3, tab4 = st.tabs(
    ["المؤشر الحي", "الشركات", "مختبر المشتقات", "منهجية المؤشر"]
)

with tab1:
    fig = px.line(index_df, x="date", y="index_level", title="PrivIndex Composite")
    fig.update_layout(yaxis_title="Index level", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("تفسير حركة المؤشر")
    for message in ai_explanation(latest, previous):
        st.markdown(f"- {message}")

    simulations = forecast_paths(index_df, months=6)
    percentiles = np.percentile(simulations, [10, 50, 90], axis=0)
    forecast_df = pd.DataFrame(
        {
            "month": range(7),
            "P10": percentiles[0],
            "Median": percentiles[1],
            "P90": percentiles[2],
        }
    )
    forecast_long = forecast_df.melt("month", var_name="scenario", value_name="level")
    fig2 = px.line(
        forecast_long,
        x="month",
        y="level",
        color="scenario",
        title="محاكاة احتمالية لمدة 6 أشهر",
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("المحاكاة تعليمية، مبنية على عوائد المؤشر التجريبية وليست توقعًا استثماريًا.")

with tab2:
    display_cols = [
        "company",
        "sector",
        "ai_company_score",
        "anomaly_risk",
        "revenue_m",
        "ebitda_margin",
        "cash_runway_months",
        "customer_growth",
        "debt_ratio",
        "delivery_score",
        "governance_score",
    ]
    table = latest[display_cols].copy().sort_values("ai_company_score", ascending=False)
    table.columns = [
        "الشركة",
        "القطاع",
        "تقييم AI",
        "مخاطر الشذوذ",
        "الإيرادات (مليون)",
        "هامش EBITDA",
        "أشهر السيولة",
        "نمو العملاء",
        "نسبة الدين",
        "تنفيذ المشاريع",
        "الحوكمة",
    ]
    st.dataframe(table, use_container_width=True, hide_index=True)

    score_fig = px.bar(
        latest.sort_values("ai_company_score"),
        x="ai_company_score",
        y="company",
        orientation="h",
        title="التقييم المركب للشركات",
    )
    st.plotly_chart(score_fig, use_container_width=True)

with tab3:
    st.subheader("إنشاء عقد افتراضي مسوّى نقدًا")
    c1, c2, c3 = st.columns(3)
    with c1:
        contract_type = st.selectbox(
            "نوع العقد",
            ["Index Future", "Call Option", "Put Option", "Volatility Swap"],
        )
        direction = st.selectbox("المركز", ["Long", "Short"])
    with c2:
        notional = st.number_input(
            "القيمة الاسمية (ريال)", min_value=1000.0, value=100000.0, step=5000.0
        )
        maturity = st.slider("الاستحقاق بعد كم شهر؟", 1, 6, 3)
    with c3:
        strike = st.number_input(
            "سعر التنفيذ",
            min_value=1.0,
            value=float(round(current_level, 2)),
            disabled=contract_type not in ["Call Option", "Put Option"],
        )
        vol_strike = st.number_input(
            "تقلب العقد %",
            min_value=1.0,
            value=float(round(vol if not math.isnan(vol) else 20, 2)),
            disabled=contract_type != "Volatility Swap",
        )

    scenario_level = st.slider(
        "افترض مستوى المؤشر عند الاستحقاق",
        min_value=float(max(30, current_level * 0.55)),
        max_value=float(current_level * 1.55),
        value=float(current_level),
        step=0.5,
    )
    scenario_vol = st.slider(
        "افترض التقلب المحقق %",
        min_value=1.0,
        max_value=80.0,
        value=float(round(vol if not math.isnan(vol) else 20, 1)),
        step=0.5,
    )

    entry = vol_strike if contract_type == "Volatility Swap" else current_level
    final = scenario_vol if contract_type == "Volatility Swap" else scenario_level
    contract = Contract(
        contract_type=contract_type,
        direction=direction,
        entry_level=entry,
        notional=notional,
        strike=strike if contract_type in ["Call Option", "Put Option"] else None,
        maturity_step=maturity,
    )
    pnl = settle_contract(contract, final)
    if contract_type in ["Call Option", "Put Option"] and direction == "Short":
        pnl = -pnl

    st.metric("التسوية النقدية المتوقعة", f"{pnl:,.0f} ريال")
    st.progress(min(abs(pnl) / max(notional, 1), 1.0))
    st.write(
        {
            "نوع العقد": contract_type,
            "المركز": direction,
            "المستوى الابتدائي": round(entry, 2),
            "القيمة عند التسوية": round(final, 2),
            "القيمة الاسمية": notional,
            "الربح/الخسارة": round(pnl, 2),
        }
    )

    levels = np.linspace(current_level * 0.6, current_level * 1.4, 120)
    payoff = []
    for level in levels:
        payoff_contract = Contract(
            contract_type=contract_type,
            direction=direction,
            entry_level=entry,
            notional=notional,
            strike=strike if contract_type in ["Call Option", "Put Option"] else None,
            maturity_step=maturity,
        )
        value = settle_contract(
            payoff_contract,
            scenario_vol if contract_type == "Volatility Swap" else level,
        )
        if contract_type in ["Call Option", "Put Option"] and direction == "Short":
            value = -value
        payoff.append(value)

    payoff_df = pd.DataFrame({"index_level": levels, "cash_settlement": payoff})
    payoff_fig = px.line(
        payoff_df,
        x="index_level",
        y="cash_settlement",
        title="منحنى التسوية النقدية",
    )
    st.plotly_chart(payoff_fig, use_container_width=True)

with tab4:
    st.markdown(
        """
### طريقة الحساب

1. **توحيد المقاييس:** تحويل كل مؤشر مالي وتشغيلي إلى درجة من 0 إلى 100.
2. **التقييم المركب:** حساب المتوسط المرجح حسب أوزان المستخدم.
3. **طبقة AI:** استخدام Isolation Forest لرصد القراءات غير المعتادة وخصم مخاطرة محدودة.
4. **بناء المؤشر:** دمج درجات الشركات حسب وزن كل شركة، ثم تثبيت نقطة البداية عند 100.
5. **التسوية النقدية:** العقد لا يمنح ملكية في الشركات؛ يحسب فرقًا نقديًا وفق حركة المؤشر أو تقلبه.

### ضوابط مهمة للنسخة الحقيقية

- مصدر بيانات مستقل وقابل للتدقيق.
- سجل زمني غير قابل للتلاعب.
- منهجية منشورة وثابتة لإعادة الموازنة.
- لجنة حوكمة للمؤشر.
- آلية اعتراض وتصحيح للبيانات.
- فصل جهة حساب المؤشر عن جهة إصدار العقد.
- مراجعة قانونية وتنظيمية قبل أي تداول حقيقي.
"""
    )
