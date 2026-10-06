"""Synthetic-data checks for kenya_mfi_scorecard. These verify wiring and direction of effects,
NOT predictive power - that needs your real repayment data."""
import numpy as np
import pandas as pd

import kenya_mfi_scorecard as sc


def make_statement(seed, inflow=40_000, fuliza_per_month=0, fuliza_fast=1.0, bet_share=0.0,
                   airtime=("daily", 20), agents=2, lock=True, chama=True, till_share=0.3,
                   rent_buffer=True, months=8):
    rng = np.random.default_rng(seed)
    rows, bal = [], 5_000.0
    day = pd.Timestamp("2026-01-01")
    end = day + pd.Timedelta(days=int(months * 30.4))
    agent_ids = [str(100000 + i) for i in range(agents)]

    def add(t, details, pin=0.0, out=0.0):
        nonlocal bal
        bal = bal + pin - out
        rows.append((t, details, pin, out, bal))

    while day < end:
        t = day + pd.Timedelta(hours=9)
        dom = day.day
        if dom in (1, 15, 28):                       # income arrives
            amt = inflow / 3 * rng.uniform(0.8, 1.2)
            add(t, "Funds received from - 0712345678 ACME LTD", pin=amt)
        if dom == 5:                                 # rent
            add(t + pd.Timedelta(hours=2), "Pay Bill to 400200 - Landlord rent", out=min(bal * 0.9, inflow * 0.25))
            if not rent_buffer:                      # spend down to near zero by day 10
                add(t + pd.Timedelta(hours=5), "Customer Transfer to - 0700000000", out=max(0, bal - 300))
        if airtime[0] == "daily":
            add(t + pd.Timedelta(hours=3), "Airtime Purchase", out=airtime[1])
        elif dom == 3:
            add(t + pd.Timedelta(hours=3), "Airtime Purchase", out=airtime[1])
        if rng.random() < till_share / 2:
            add(t + pd.Timedelta(hours=4), "Merchant Payment to 556677 - SHOP", out=min(bal * 0.3, 800))
        if dom in (7, 21):
            add(t + pd.Timedelta(hours=5), f"Customer Withdrawal At Agent Till {rng.choice(agent_ids)} - AGENT", out=min(bal * 0.2, 2_000))
        if chama and dom == 10:
            add(t + pd.Timedelta(hours=6), "Pay Bill to M-Changa - Family Chama", out=min(bal * 0.1, 1_000))
        if lock and dom == 12 and day.month == 3:
            add(t + pd.Timedelta(hours=6), "M-Shwari Lock Savings Deposit", out=min(bal * 0.1, 1_000))
        if bet_share > 0 and rng.random() < 0.3:
            add(t + pd.Timedelta(hours=7), "Pay Bill to 888880 - SportPesa", out=min(bal * 0.9, inflow * bet_share / 9))
        if fuliza_per_month and rng.random() < fuliza_per_month / 30:
            amt = 1_000.0
            add(t + pd.Timedelta(hours=8), "OverDraft of Credit Party", pin=amt)
            gap = pd.Timedelta(hours=6) if rng.random() < fuliza_fast else pd.Timedelta(days=4)
            add(t + pd.Timedelta(hours=8) + gap, "OD Loan Repayment to 232323 - M-PESA Overdraw", out=amt)
        day += pd.Timedelta(days=1)
    df = pd.DataFrame(rows, columns=["time", "details", "paid_in", "withdrawn", "balance"]).sort_values("time")
    # rebuild running balance in time order (generator appends slightly out of order)
    df["balance"] = 5_000.0 + (df["paid_in"] - df["withdrawn"]).cumsum()
    return df.reset_index(drop=True)


def test_fuliza_fifo():
    t0 = pd.Timestamp("2026-06-01 08:00")
    df = pd.DataFrame([
        (t0, "OverDraft of Credit Party", 1000, 0, 0),
        (t0 + pd.Timedelta(hours=2), "OD Loan Repayment to 232323", 0, 1000, 0),
        (t0 + pd.Timedelta(days=5), "OverDraft of Credit Party", 1000, 0, 0),
        (t0 + pd.Timedelta(days=8), "OD Loan Repayment to 232323", 0, 1000, 0),
        (t0 + pd.Timedelta(days=9), "OverDraft of Credit Party", 500, 0, 0),   # still open, 24h not elapsed at end? end = day 9 -> unknown
    ], columns=["time", "details", "paid_in", "withdrawn", "balance"])
    df["is_fuliza_draw"] = df["details"].str.lower().str.contains(sc.KEYWORDS["fuliza_draw"])
    df["is_fuliza_repay"] = df["details"].str.lower().str.contains(sc.KEYWORDS["fuliza_repay"])
    rate, _, n = sc._fuliza_24h_rate(df)
    assert n == 3 and abs(rate - 0.5) < 1e-9, (rate, n)      # 1 of 2 judged draws repaid in 24h; open one excluded
    # partial repayments carry across draws FIFO
    df2 = pd.DataFrame([
        (t0, "OverDraft of Credit Party", 1000, 0, 0),
        (t0 + pd.Timedelta(hours=1), "OverDraft of Credit Party", 1000, 0, 0),
        (t0 + pd.Timedelta(hours=3), "OD Loan Repayment to 232323", 0, 1500, 0),   # clears draw1, half of draw2
        (t0 + pd.Timedelta(days=3), "OD Loan Repayment to 232323", 0, 500, 0),     # draw2 cleared late
    ], columns=df.columns[:5])
    df2["is_fuliza_draw"] = df2["details"].str.lower().str.contains(sc.KEYWORDS["fuliza_draw"])
    df2["is_fuliza_repay"] = df2["details"].str.lower().str.contains(sc.KEYWORDS["fuliza_repay"])
    rate, _, n = sc._fuliza_24h_rate(df2)
    assert abs(rate - 0.5) < 1e-9, rate


def test_bands_and_rules():
    assert sc.band(24, sc.T_SIM) == 80 and sc.band(23.9, sc.T_SIM) == 30
    assert sc._betting_points(sc.Features(betting_share=0.05)) == 0
    assert sc._betting_points(sc.Features(betting_share=0.051)) == -100
    assert sc._betting_points(sc.Features(betting_share=0.0)) == 10
    assert sc._airtime_points(sc.Features(airtime_per_month=30, airtime_median_ticket=20)) == 25
    assert sc._airtime_points(sc.Features(airtime_per_month=1, airtime_median_ticket=500)) == -25
    assert sc.scale(sc.RAW_MAX) == 850 and sc.scale(sc.RAW_MIN) == 300


def run_profiles():
    ext_good = sc.ExternalInputs(age=38, sim_age_months=60, mfi_history="repaid_2plus", helb_status="na", okoa_per_month=0)
    ext_bad = sc.ExternalInputs(age=27, sim_age_months=5, mfi_history="late", helb_status="arrears", okoa_per_month=12)
    cases = {
        "steady salaried, Fuliza repaid fast": (make_statement(1, 55_000, 3, 1.0, 0, ("daily", 20), 1), ext_good),
        "average, some Fuliza, no extras": (make_statement(2, 22_000, 4, 0.7, 0.0, ("monthly", 500), 5, lock=False, chama=False), sc.ExternalInputs(age=33, sim_age_months=30)),
        "bettor, slow Fuliza, thin buffer": (make_statement(3, 14_000, 12, 0.2, 0.30, ("monthly", 500), 6, lock=False, chama=False, rent_buffer=False), ext_bad),
        "same, but statement only (no external)": (make_statement(3, 14_000, 12, 0.2, 0.30, ("monthly", 500), 6, lock=False, chama=False, rent_buffer=False), sc.ExternalInputs()),
    }
    out = {}
    for name, (st, ext) in cases.items():
        f, r = sc.score_statement(st, ext)
        out[name] = (f, r)
        print(f"\n== {name}\n  score {r.score}  band {r.band}  -> {r.decision}  limit {r.limit_kes}  coverage {r.coverage}")
        print(f"  inflow {f.avg_monthly_inflow:,.0f}  rent_bal_share {f.rent_balance_share}  till {f.till_ratio and round(f.till_ratio,2)}"
              f"  fuliza/mo {f.fuliza_draws_per_month:.1f}  fuliza24h {f.fuliza_repay_24h_rate}  bet {f.betting_share and round(f.betting_share,3)}"
              f"  agent {f.agent_concentration and round(f.agent_concentration,2)}  chama {f.chama_member}/{f.chama_regularity and round(f.chama_regularity,2)}"
              f"  lock {f.max_lock_amount}  airtime {f.airtime_per_month:.1f}x{f.airtime_median_ticket}")
        print("  reasons:", r.reasons, "| flags:", r.flags)
    return out


def test_profiles_order_and_decisions():
    out = run_profiles()
    names = list(out)
    s = [out[n][1].score for n in names]
    assert s[0] > s[1] > s[2], s
    assert out[names[0]][1].decision == "APPROVE" and out[names[0]][1].reasons == []
    assert out[names[2]][1].decision in ("DECLINE", "REFER")
    assert out[names[3]][1].coverage < out[names[2]][1].coverage          # missing externals lowers coverage
    assert out[names[2]][0].betting_share > 0.05


def test_knockouts():
    st = make_statement(1)
    f, r = sc.score_statement(st, sc.ExternalInputs(age=40, sim_age_months=60, mfi_history="default"))
    assert r.decision == "DECLINE"
    f, r = sc.score_statement(st, sc.ExternalInputs(age=40, sim_age_months=60, kyc_name_matches_mpesa=False))
    assert r.decision == "REFER"
    f, r = sc.score_statement(make_statement(1, months=2), sc.ExternalInputs(age=40, sim_age_months=60))
    assert r.decision == "REFER" and any("months" in x for x in r.flags)
    f, r = sc.score_statement(st, sc.ExternalInputs(age=17, sim_age_months=60))
    assert r.decision == "DECLINE"


def test_validation_helpers():
    rng = np.random.default_rng(0)
    n = 4000
    latent = rng.normal(size=n)
    scores = 560 + 60 * latent + rng.normal(0, 25, n)
    p_true = 1 / (1 + np.exp(-(-2.2 - 1.1 * latent)))
    bad = (rng.random(n) < p_true).astype(int)
    m = sc.gini_ks(scores, bad)
    assert 0.3 < m["gini"] < 0.8 and m["ks"] > 0.2, m
    pd_fn = sc.fit_score_to_pd(scores, bad)
    assert pd_fn(450) > pd_fn(560) > pd_fn(700)
    assert abs(np.mean([pd_fn(s) for s in scores]) - bad.mean()) < 0.01
    assert sc.psi(scores, scores) < 1e-3
    assert sc.psi(scores, scores - 80) > 0.25
    print("\nvalidation:", {k: round(v, 3) for k, v in m.items()},
          "PD@450/560/700:", [round(pd_fn(x), 3) for x in (450, 560, 700)])


if __name__ == "__main__":
    test_bands_and_rules(); test_fuliza_fifo(); test_profiles_order_and_decisions()
    test_knockouts(); test_validation_helpers()
    print("\nALL TESTS PASSED")
