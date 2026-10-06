"""
Kenya MFI alternative-data credit scorecard
===========================================

Transparent, points-based (expert) scorecard built on M-Pesa behaviour plus a few
non-statement signals (SIM age, Chama, HELB, prior MFI/CRB history).

IMPORTANT - read before going live
  * Every point value below is an EXPERT PRIOR, not a fitted coefficient. Before it
    decides real loans: back-test on your own repayment data, recalibrate points
    (see `fit_score_to_pd`, `gini_ks`, `psi`), and set cut-offs to your risk appetite.
  * The regexes in KEYWORDS are best guesses at M-Pesa statement wording. Check them
    against 20-30 real statements before trusting any extracted feature.
  * Some features are NOT in a customer's M-Pesa statement (SIM age, Okoa Jahazi,
    HELB status, CRB/MFI history). They must come from other sources and are passed
    in via `ExternalInputs`; if absent they score 0 and lower the coverage figure.

Layout
  1. Config tables (points, bands, decision policy)
  2. Statement -> features      extract_mpesa_features()
  3. Features -> score          score_applicant()
  4. Calibration / validation   fit_score_to_pd(), gini_ks(), psi()
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# 1. CONFIG
# --------------------------------------------------------------------------- #

# Statement wording -> category. VERIFY against real statements.
KEYWORDS = {
    "fuliza_draw":  r"overdraft of credit party|fuliza.*(?:draw|loan|limit)",
    "fuliza_repay": r"od loan repayment|fuliza.*repay",
    "airtime":      r"airtime",
    "betting":      r"sportpesa|betika|odibets|betway|mozzart|1xbet|22bet|bet254|bangbet|shabiki|\bbet\b",
    "lock_savings": r"lock savings|m-shwari lock|kcb m-pesa lock|lock account",
    "agent_wd":     r"customer withdrawal at agent till\s*(\d+)",   # capturing group = agent id (used with .str.extract)
    "merchant":     r"merchant payment|pay ?bill|buy goods|small business|business payment|\btill\b",
    "p2p":          r"customer transfer|funds received|sent to|received from",
    "chama":        r"m-?changa|mchanga|chama|sacco|merry[- ]go",
    # Inflows that are not income (credit drawn, reversals, own savings coming back)
    "non_income_in": r"overdraft of credit party|reversal|m-shwari|kcb m-pesa|lock|loan disbursement|okoa",
}

REF_RENT_WINDOW_DAY = 10     # balance checked at end of this day of each month
MIN_BALANCE_AFTER_RENT = 2_000
LOCK_MIN_KES = 500
BETTING_RED_LINE = 0.05
FULIZA_REPAY_HOURS = 24
WINDOW_DAYS = 183            # ~6 months

INF = float("inf")


def band(value: float, table: list[tuple[float, float]]) -> float:
    """First (upper_bound, points) with value < upper_bound wins."""
    for upper, pts in table:
        if value < upper:
            return pts
    return table[-1][1]


T_INFLOW = [(5_000, -30), (15_000, 0), (30_000, 25), (60_000, 45), (INF, 60)]
T_RENT_BAL = [(0.5, -30), (0.8, 15), (1.01, 40)]
T_TILL = [(0.2, 0), (0.5, 15), (1.01, 30)]
T_FULIZA_REPAY = [(0.4, -60), (0.7, 0), (0.9, 40), (1.01, 70)]
T_FULIZA_FREQ = [(8, 0), (15, -10), (INF, -30)]
T_SIM = [(6, -40), (12, 0), (24, 30), (INF, 80)]        # >= 24 months = +80
T_AGENT = [(0.3, 0), (0.6, 10), (1.01, 20)]
T_CHAMA_REG = [(0.5, 0), (0.8, 15), (1.01, 30)]
T_OKOA = [(1, 0), (4, 0), (9, -15), (INF, -30)]
MFI_POINTS = {"repaid_2plus": 100, "repaid_1": 50, "none": 0, "late": -40, "default": -150}
HELB_POINTS = {"cleared": 25, "servicing": 10, "na": 0, "arrears": -40}
YOUTH_MAX_AGE = 34           # Kenya youth definition 18-34

# Decision policy (all placeholders - set from back-test and risk appetite)
BANDS = [("A", 740), ("B", 670), ("C", 590), ("D", 500), ("E", 0)]
LIMIT_MULTIPLE_OF_INFLOW = {"A": 0.60, "B": 0.40, "C": 0.25}
PRODUCT_CAP_KES = 100_000
MIN_HISTORY_MONTHS = 3
MIN_COVERAGE = 0.50
KNOCKOUT_ON_CRB_DEFAULT = True
SCORE_MIN, SCORE_MAX = 300, 850


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #

@dataclass
class ExternalInputs:
    """Signals that are not derivable from the M-Pesa statement."""
    age: Optional[int] = None
    sim_age_months: Optional[float] = None      # Safaricom / KYC provider
    okoa_per_month: Optional[float] = None      # Safaricom airtime-credit data
    mfi_history: Optional[str] = None           # repaid_2plus|repaid_1|none|late|default
    helb_status: Optional[str] = None           # cleared|servicing|arrears|na
    chama_member: Optional[bool] = None         # self-declared / KYC; else inferred
    kyc_name_matches_mpesa: Optional[bool] = None


@dataclass
class Features:
    avg_monthly_inflow: Optional[float] = None
    rent_balance_share: Optional[float] = None
    till_ratio: Optional[float] = None
    fuliza_draws_per_month: Optional[float] = None
    fuliza_repay_24h_rate: Optional[float] = None
    sim_age_months: Optional[float] = None
    max_lock_amount: Optional[float] = None
    agent_concentration: Optional[float] = None
    mfi_history: Optional[str] = None
    chama_member: Optional[bool] = None
    chama_regularity: Optional[float] = None
    helb_status: Optional[str] = None
    airtime_per_month: Optional[float] = None
    airtime_median_ticket: Optional[float] = None
    betting_share: Optional[float] = None
    okoa_per_month: Optional[float] = None
    history_months: Optional[float] = None
    age: Optional[int] = None
    kyc_name_matches_mpesa: Optional[bool] = None


# --------------------------------------------------------------------------- #
# 2. STATEMENT -> FEATURES
# --------------------------------------------------------------------------- #

def _fuliza_24h_rate(df: pd.DataFrame) -> tuple[Optional[float], Optional[float], float]:
    """FIFO-match Fuliza repayments to draws. Returns (rate, draws_per_month, n_draws)."""
    draws = df[df["is_fuliza_draw"] & (df["paid_in"] > 0)]
    reps = df[df["is_fuliza_repay"] & (df["withdrawn"] > 0)]
    if draws.empty:
        return None, None, 0
    events = sorted(
        [(t, "d", a) for t, a in zip(draws["time"], draws["paid_in"])]
        + [(t, "r", a) for t, a in zip(reps["time"], reps["withdrawn"])],
        key=lambda e: (e[0], e[1] != "d"),   # draw before repay on the same timestamp
    )
    queue: list[list] = []                    # [draw_time, remaining, cleared_at]
    done: list[list] = []
    for t, kind, amt in events:
        if kind == "d":
            queue.append([t, amt, None])
        else:
            left = amt
            while left > 1e-9 and queue:
                head = queue[0]
                pay = min(left, head[1])
                head[1] -= pay
                left -= pay
                if head[1] <= 1e-9:
                    head[2] = t
                    done.append(queue.pop(0))
    end = df["time"].max()
    window = pd.Timedelta(hours=FULIZA_REPAY_HOURS)
    ok = sum(1 for d in done if d[2] - d[0] <= window)
    late = sum(1 for d in done if d[2] - d[0] > window)
    # still open: a miss if the 24h window has already elapsed, otherwise unknown (excluded)
    late += sum(1 for d in queue if end - d[0] > window)
    judged = ok + late
    rate = ok / judged if judged else None
    return rate, None, len(draws)


def extract_mpesa_features(stmt: pd.DataFrame, ext: Optional[ExternalInputs] = None) -> Features:
    """
    stmt columns: time (datetime), details (str), paid_in (float), withdrawn (float), balance (float)
    Uses the last ~6 months of the statement.
    """
    ext = ext or ExternalInputs()
    df = stmt.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    df["paid_in"] = df["paid_in"].fillna(0.0)
    df["withdrawn"] = df["withdrawn"].abs().fillna(0.0)
    d = df["details"].fillna("").str.lower()

    def has(key):
        return d.str.contains(KEYWORDS[key], regex=True)   # patterns used here have no capture groups

    first, last = df["time"].min(), df["time"].max()
    history_months = (last - first).days / 30.4
    start = max(first, last - pd.Timedelta(days=WINDOW_DAYS))
    in_win = df["time"] >= start
    n_months = float(min(6, max(1, round((last - start).days / 30.4))))

    df["is_fuliza_draw"] = has("fuliza_draw")
    df["is_fuliza_repay"] = has("fuliza_repay")
    w = df[in_win]
    dw = d[in_win]

    f = Features(history_months=round(history_months, 1), age=ext.age,
                 kyc_name_matches_mpesa=ext.kyc_name_matches_mpesa)

    # --- capacity -----------------------------------------------------------
    income_mask = (w["paid_in"] > 0) & ~dw.str.contains(KEYWORDS["non_income_in"], regex=True)
    f.avg_monthly_inflow = float(w.loc[income_mask, "paid_in"].sum() / n_months)

    # Balance at end of day 10, carried forward, for each month fully inside the window
    bal_ok = []
    for ym, grp in df.groupby(df["time"].dt.to_period("M")):
        cutoff = pd.Timestamp(ym.year, ym.month, REF_RENT_WINDOW_DAY) + pd.Timedelta(days=1)
        if cutoff - pd.Timedelta(days=REF_RENT_WINDOW_DAY) < start or cutoff > last + pd.Timedelta(days=1):
            continue
        prior = df[df["time"] < cutoff]
        if prior.empty:
            continue
        bal_ok.append(prior["balance"].iloc[-1] > MIN_BALANCE_AFTER_RENT)
    f.rent_balance_share = float(np.mean(bal_ok)) if len(bal_ok) >= 2 else None

    merch = dw.str.contains(KEYWORDS["merchant"], regex=True)
    p2p = dw.str.contains(KEYWORDS["p2p"], regex=True) & ~merch
    amt = w["paid_in"] + w["withdrawn"]
    m_val, p_val = amt[merch].sum(), amt[p2p].sum()
    f.till_ratio = float(m_val / (m_val + p_val)) if (m_val + p_val) > 0 else None

    rate, _, n_draws = _fuliza_24h_rate(df[in_win])
    f.fuliza_repay_24h_rate = rate
    f.fuliza_draws_per_month = float(n_draws / n_months)

    # --- stability ----------------------------------------------------------
    f.sim_age_months = ext.sim_age_months
    if f.sim_age_months is None:
        f.sim_age_months = None   # statement length is only a LOWER bound; do not substitute
    lock = w[dw.str.contains(KEYWORDS["lock_savings"], regex=True)]
    f.max_lock_amount = float(max(lock["paid_in"].max(), lock["withdrawn"].max())) if len(lock) else 0.0

    agents = dw[w["withdrawn"] > 0].str.extract(KEYWORDS["agent_wd"], expand=False).dropna()
    f.agent_concentration = float(agents.value_counts(normalize=True).iloc[0]) if len(agents) >= 3 else None

    # --- character ----------------------------------------------------------
    f.mfi_history = ext.mfi_history
    f.helb_status = ext.helb_status
    chama = (w["withdrawn"] > 0) & dw.str.contains(KEYWORDS["chama"], regex=True)
    months_with = w.loc[chama, "time"].dt.to_period("M").nunique()
    f.chama_regularity = float(min(1.0, months_with / n_months)) if chama.any() else None
    f.chama_member = ext.chama_member if ext.chama_member is not None else (True if chama.any() else None)

    # --- willingness --------------------------------------------------------
    air = w[(w["withdrawn"] > 0) & dw.str.contains(KEYWORDS["airtime"], regex=True)]
    f.airtime_per_month = float(len(air) / n_months)
    f.airtime_median_ticket = float(air["withdrawn"].median()) if len(air) else None
    tot_out = w["withdrawn"].sum()
    bet_out = w.loc[(w["withdrawn"] > 0) & dw.str.contains(KEYWORDS["betting"], regex=True), "withdrawn"].sum()
    f.betting_share = float(bet_out / tot_out) if tot_out > 0 else None
    f.okoa_per_month = ext.okoa_per_month
    return f


# --------------------------------------------------------------------------- #
# 3. FEATURES -> SCORE
# --------------------------------------------------------------------------- #

@dataclass
class Rule:
    name: str
    group: str
    fn: Callable[[Features], Optional[float]]   # None = data missing
    lo: float
    hi: float
    text_neg: str


def _vals(table):
    return [p for _, p in table]


def _airtime_points(f: Features) -> Optional[float]:
    if f.airtime_per_month is None or f.airtime_median_ticket is None:
        return None
    if f.airtime_per_month >= 12 and f.airtime_median_ticket <= 100:
        return 25          # small, frequent top-ups (e.g. KES 20 daily) = steady low income
    if f.airtime_per_month <= 2 and f.airtime_median_ticket >= 400:
        return -25         # one big top-up a month = lumpy cash flow
    return 0


def _betting_points(f: Features) -> Optional[float]:
    s = f.betting_share
    if s is None:
        return None
    if s == 0:
        return 10
    if s <= BETTING_RED_LINE:
        return 0
    return -100 if s <= 0.15 else -140       # >15% tier is an extension of the brief: tune/remove


def _helb_points(f: Features) -> Optional[float]:
    if f.age is None or f.age > YOUTH_MAX_AGE:
        return 0           # not applicable, neutral
    if f.helb_status is None:
        return None
    return HELB_POINTS[f.helb_status]


RULES: list[Rule] = [
    # Capacity
    Rule("avg_monthly_inflow", "Capacity", lambda f: None if f.avg_monthly_inflow is None else band(f.avg_monthly_inflow, T_INFLOW),
         min(_vals(T_INFLOW)), max(_vals(T_INFLOW)), "Low average monthly M-Pesa income"),
    Rule("balance_after_rent", "Capacity", lambda f: None if f.rent_balance_share is None else band(f.rent_balance_share, T_RENT_BAL),
         min(_vals(T_RENT_BAL)), max(_vals(T_RENT_BAL)), "Balance often below KES 2,000 after rent period"),
    Rule("till_vs_p2p", "Capacity", lambda f: None if f.till_ratio is None else band(f.till_ratio, T_TILL),
         0, max(_vals(T_TILL)), "Little merchant/business activity"),
    Rule("fuliza_repayment", "Capacity",
         lambda f: 10 if (f.fuliza_draws_per_month == 0) else (None if f.fuliza_repay_24h_rate is None else band(f.fuliza_repay_24h_rate, T_FULIZA_REPAY)),
         min(_vals(T_FULIZA_REPAY)), max(_vals(T_FULIZA_REPAY)), "Fuliza not repaid within 24 hours"),
    Rule("fuliza_frequency", "Capacity",
         lambda f: None if f.fuliza_draws_per_month is None else (0 if f.fuliza_draws_per_month == 0 else band(f.fuliza_draws_per_month, T_FULIZA_FREQ)),
         min(_vals(T_FULIZA_FREQ)), 0, "Heavy Fuliza dependence"),
    # Stability
    Rule("sim_age", "Stability", lambda f: None if f.sim_age_months is None else band(f.sim_age_months, T_SIM),
         min(_vals(T_SIM)), max(_vals(T_SIM)), "Short Safaricom line history"),
    Rule("locked_savings", "Stability", lambda f: None if f.max_lock_amount is None else (30 if f.max_lock_amount >= LOCK_MIN_KES else 0),
         0, 30, "No locked savings history"),
    Rule("same_agent", "Stability", lambda f: None if f.agent_concentration is None else band(f.agent_concentration, T_AGENT),
         0, max(_vals(T_AGENT)), "Withdrawals spread across many agents"),
    # Character
    Rule("mfi_history", "Character", lambda f: None if f.mfi_history is None else MFI_POINTS[f.mfi_history],
         min(MFI_POINTS.values()), max(MFI_POINTS.values()), "Adverse previous loan repayment record"),
    Rule("chama_member", "Character", lambda f: None if f.chama_member is None else (20 if f.chama_member else 0),
         0, 20, "No Chama membership"),
    Rule("chama_regularity", "Character", lambda f: (0 if f.chama_member is False else (None if f.chama_regularity is None else band(f.chama_regularity, T_CHAMA_REG))),
         0, max(_vals(T_CHAMA_REG)), "Irregular Chama contributions"),
    Rule("helb", "Character", _helb_points, min(HELB_POINTS.values()), max(HELB_POINTS.values()), "HELB loan in arrears"),
    # Willingness
    Rule("airtime_pattern", "Willingness", _airtime_points, -25, 25, "Lumpy airtime purchasing"),
    Rule("betting", "Willingness", _betting_points, -140, 10, "Betting is a high share of spending"),
    Rule("okoa_jahazi", "Willingness", lambda f: None if f.okoa_per_month is None else band(f.okoa_per_month, T_OKOA),
         min(_vals(T_OKOA)), 0, "Frequent Okoa Jahazi borrowing"),
]

RAW_MIN = sum(r.lo for r in RULES)
RAW_MAX = sum(r.hi for r in RULES)


def scale(raw: float) -> float:
    s = SCORE_MIN + (SCORE_MAX - SCORE_MIN) * (raw - RAW_MIN) / (RAW_MAX - RAW_MIN)
    return float(min(SCORE_MAX, max(SCORE_MIN, s)))


@dataclass
class Result:
    score: float
    band: str
    decision: str                      # APPROVE | REFER | DECLINE
    limit_kes: Optional[float]
    raw_points: float
    coverage: float
    contributions: dict = field(default_factory=dict)
    reasons: list = field(default_factory=list)       # adverse reason codes
    flags: list = field(default_factory=list)
    missing: list = field(default_factory=list)


def score_applicant(f: Features) -> Result:
    contrib, missing, flags = {}, [], []
    seen_w = tot_w = 0.0
    for r in RULES:
        pts = r.fn(f)
        w = r.hi - r.lo
        tot_w += w
        if pts is None:
            missing.append(r.name)
            contrib[r.name] = 0.0
        else:
            contrib[r.name] = float(pts)
            seen_w += w
    raw = sum(contrib.values())
    coverage = seen_w / tot_w
    sc = scale(raw)
    b = next(name for name, floor in BANDS if sc >= floor)

    # Reason codes = rules that cost points. If there are fewer than 3 and the applicant is not
    # approved, add the biggest shortfalls vs. the maximum (skipping not-applicable HELB).
    by_rule = {r.name: r for r in RULES}
    neg = sorted((k for k, v in contrib.items() if v < 0), key=lambda k: contrib[k])
    reasons = [by_rule[k].text_neg for k in neg[:3]]
    shortfall_pool = [k for k in contrib if k not in neg and k not in missing
                      and not (k == "helb" and (f.age is None or f.age > YOUTH_MAX_AGE))
                      and contrib[k] < by_rule[k].hi]
    shortfall = sorted(shortfall_pool, key=lambda k: contrib[k] - by_rule[k].hi)

    # Policy overrides
    decision, limit = None, None
    if f.age is not None and f.age < 18:
        decision = "DECLINE"; flags.append("Under 18")
    elif KNOCKOUT_ON_CRB_DEFAULT and f.mfi_history == "default":
        decision = "DECLINE"; flags.append("Prior default / CRB-listed")
    elif f.kyc_name_matches_mpesa is False:
        decision = "REFER"; flags.append("M-Pesa registered name does not match KYC (possible fraud)")
    elif (f.history_months or 0) < MIN_HISTORY_MONTHS:
        decision = "REFER"; flags.append(f"Less than {MIN_HISTORY_MONTHS} months of statement history")
    elif coverage < MIN_COVERAGE:
        decision = "REFER"; flags.append(f"Only {coverage:.0%} of scorecard data available")
    if decision is None:
        if b in LIMIT_MULTIPLE_OF_INFLOW:
            decision = "APPROVE"
            limit = min(PRODUCT_CAP_KES, LIMIT_MULTIPLE_OF_INFLOW[b] * (f.avg_monthly_inflow or 0))
            limit = float(math.floor(limit / 500) * 500)
            if limit <= 0:
                decision, limit = "REFER", None
                flags.append("Zero affordability limit")
        elif b == "D":
            decision = "REFER"
        else:
            decision = "DECLINE"
    if decision != "APPROVE" and len(reasons) < 3:
        reasons += [by_rule[k].text_neg for k in shortfall[: 3 - len(reasons)]]
    if decision == "APPROVE":
        reasons = []
    return Result(round(sc, 1), b, decision, limit, raw, round(coverage, 2),
                  contrib, reasons, flags, missing)


# --------------------------------------------------------------------------- #
# 4. CALIBRATION / VALIDATION  (run on your own loan book)
# --------------------------------------------------------------------------- #

def gini_ks(scores, bad) -> dict:
    """Rank-based AUC/Gini and KS. `bad` = 1 for default. Higher score should mean lower risk."""
    s = pd.Series(np.asarray(scores, float)); y = np.asarray(bad, int)
    ranks = s.rank(method="average").to_numpy()
    n1, n0 = y.sum(), (1 - y).sum()
    auc = (ranks[y == 0].sum() - n0 * (n0 + 1) / 2) / (n0 * n1)   # P(good scores above bad)
    order = np.argsort(s.to_numpy())
    cb = np.cumsum(y[order]) / n1
    cg = np.cumsum(1 - y[order]) / n0
    return {"auc": float(auc), "gini": float(2 * auc - 1), "ks": float(np.max(np.abs(cb - cg)))}


def fit_score_to_pd(scores, bad, iters: int = 50) -> Callable[[float], float]:
    """Logistic map PD = 1 / (1 + exp(-(a + b*score))). Returns a callable. Newton-Raphson, numpy only."""
    x = (np.asarray(scores, float) - 500) / 100
    y = np.asarray(bad, float)
    X = np.c_[np.ones_like(x), x]
    beta = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ beta))
        W = p * (1 - p) + 1e-9
        step = np.linalg.solve(X.T @ (X * W[:, None]) + 1e-6 * np.eye(2), X.T @ (y - p))
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    return lambda s: float(1 / (1 + np.exp(-(beta[0] + beta[1] * (s - 500) / 100))))


def psi(expected, actual, bins: int = 10) -> float:
    """Population stability index between development and current score distributions (>0.25 = investigate)."""
    e, a = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e) + 1e-6
    pa = np.histogram(a, edges)[0] / len(a) + 1e-6
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def score_statement(stmt: pd.DataFrame, ext: Optional[ExternalInputs] = None) -> tuple[Features, Result]:
    f = extract_mpesa_features(stmt, ext)
    return f, score_applicant(f)


if __name__ == "__main__":
    print(f"Raw points range: {RAW_MIN} .. {RAW_MAX}   (maps to {SCORE_MIN}-{SCORE_MAX})")
    print(f"Neutral applicant (all signals missing/0) scores {scale(0):.0f}")
