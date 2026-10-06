# Kenya MFI Alternative-Data Credit Scorecard

A transparent, points-based credit scorecard prototype for Kenyan microfinance institutions. It scores borrowers from M-Pesa behaviour (Fuliza repayment, balances after rent date, betting share, airtime pattern and more) plus non-statement signals such as SIM age, Chama membership, HELB status and prior repayment history.

> **Status: prototype.** All point values are illustrative expert priors. They have **not** been validated on real repayment data and must not be used for real lending decisions until back-tested and recalibrated. The tests use synthetic data only.

## Quick start

```bash
pip install -r requirements.txt
python test_scorecard.py        # synthetic checks; should end with ALL TESTS PASSED
```

```python
import pandas as pd
import kenya_mfi_scorecard as sc

# columns: time, details, paid_in, withdrawn, balance
stmt = pd.read_csv("statement.csv", parse_dates=["time"])
ext = sc.ExternalInputs(age=34, sim_age_months=40, mfi_history="none")

features, result = sc.score_statement(stmt, ext)
print(result.score, result.band, result.decision, result.limit_kes)
print(result.reasons, result.flags)
```

## Data and privacy

Do not commit real statements, borrower lists or loan-book extracts to this repository (the `.gitignore` blocks common data file types). Processing real customers' M-Pesa data requires their explicit consent and compliance with the Kenya Data Protection Act 2019.

## Files

| File | Purpose |
|---|---|
| `kenya_mfi_scorecard.py` | Feature extraction, scoring, decisions, calibration and validation helpers |
| `test_scorecard.py` | Synthetic-data tests |
| `requirements.txt` | Dependencies (pandas, numpy) |

---

## Design and handover notes

`kenya_mfi_scorecard.py` is the engine. `test_scorecard.py` runs it on synthetic data (`python test_scorecard.py`).

## How it works
1. **Input:** a parsed M-Pesa statement (`time, details, paid_in, withdrawn, balance`) plus `ExternalInputs` for signals the statement cannot give.
2. **Features:** `extract_mpesa_features()` uses the last ~6 months.
3. **Points:** 16 rules across 4 groups (Capacity, Stability, Character, Willingness). Missing data scores 0 and lowers `coverage`.
4. **Score:** raw points are scaled linearly to 300–850 (raw range -575 to +540).
5. **Decision:** the band (A–E) plus overrides gives APPROVE, REFER or DECLINE. Approvals get a limit of band multiple × average monthly inflow, capped at the product cap. Adverse reason codes are returned for declines and refers.

## Points table (your priors, with additions marked)
| Group | Feature | Points |
|---|---|---|
| Capacity | Avg monthly inflow (6m, excluding Fuliza draws, reversals and savings returns) | <5k −30 · 5–15k 0 · 15–30k +25 · 30–60k +45 · ≥60k +60 |
| | Share of months with balance > KES 2,000 at end of day 10 | <50% −30 · 50–80% +15 · ≥80% +40 |
| | Till/Paybill vs P2P value share | <20% 0 · 20–50% +15 · ≥50% +30 |
| | **Fuliza repaid within 24h** (the #1 predictor, so the biggest capacity weight) | no Fuliza +10 · <40% −60 · 40–70% 0 · 70–90% +40 · ≥90% +70 |
| | Fuliza draws per month (added) | ≤8: 0 · 8–15: −10 · >15: −30 |
| Stability | SIM age | <6m −40 · 6–12m 0 · 12–24m +30 · **≥24m +80** |
| | Ever locked ≥ KES 500 (M-Shwari / KCB lock) | +30 |
| | Share of agent withdrawals at the top agent (min 3 withdrawals) | <30% 0 · 30–60% +10 · ≥60% +20 |
| Character | Previous MFI/CRB record | repaid 2+ cycles +100 · 1 cycle +50 · none 0 · late −40 · default −150 |
| | Chama member / contribution regularity | +20 / up to +30 |
| | HELB (age ≤ 34 only) | cleared +25 · servicing +10 · arrears −40 |
| Willingness | Airtime pattern | ≥12 small top-ups a month (≤KES 100 each) +25 · ≤2 large (≥KES 400) −25 |
| | **Betting share of M-Pesa out** | none +10 · ≤5% 0 · **>5% −100** · >15% −140 (my extension) |
| | Okoa Jahazi per month (assumed: more borrowing = more risk) | ≤3 0 · 4–8 −15 · ≥9 −30 |

Default policy thresholds are A ≥ 740, B ≥ 670, C ≥ 590, D ≥ 500. D is referred and E is declined. Limit multiples are A 0.6, B 0.4 and C 0.25. All of these are placeholders.

## Where each feature really comes from
| Feature | In the M-Pesa statement? | Otherwise |
|---|---|---|
| Inflow, rent-date balance, Till/P2P, Fuliza, agent, lock, airtime, betting, Chama/M-Changa | Yes, via keyword matching. **Verify the wording against real statements.** | |
| SIM age | No. Statement length is only a lower bound. | Safaricom or KYC provider, or customer-declared |
| Okoa Jahazi | Not reliably | Safaricom data, or customer-declared |
| MFI history / CRB | No | Your loan book and the CRB |
| HELB status | No | HELB clearance certificate or customer-declared |

Two practical points:
- **Daraja gives you your own shortcode's data**, such as repayments on your paybill. It does not give you a customer's personal statement. Personal statements come from a customer-supplied statement file (consented upload) or a licensed, consented data provider.
- **Statement parsing is the weakest link.** Build the parser against real samples, and keep `KEYWORDS` in config rather than hard-coded.

## Assumptions I made where your brief was ambiguous
- "Balance after rent date" is the balance at the end of day 10, carried forward.
- "Till vs P2P" is the merchant-channel share of total M-Pesa value, in either direction. If you meant only money received on the borrower's own till, change one line.
- Okoa Jahazi is penalised when frequent. HELB applies only when age ≤ 34.
- A borrower with no Fuliza history gets +10, so opting out of Fuliza is not penalised.
- A prior default or CRB listing is a knockout (`KNOCKOUT_ON_CRB_DEFAULT`). So are age under 18 and a KYC/M-Pesa name mismatch (referred, not declined).

## Before this decides real money
1. **Back-test.** Score your historical borrowers and run `gini_ks`. Treat roughly Gini 0.35–0.50 as realistic for an alternative-data scorecard, not 0.7+.
2. **Re-fit the points.** Bin the features, fit a logistic regression or WoE model on defaults, and compare coefficients with the priors above. Expect the betting and SIM-age points to change.
3. **Map score to PD** with `fit_score_to_pd`. Set cut-offs and limit multiples from PD, margin and loss-given-default, not from the placeholders.
4. **Monitor** with `psi` monthly, and re-run Gini quarterly.
5. **Fairness check.** Betting, airtime, agent and SIM-age signals can proxy for age, gender, region or religion. Test approval rates and default rates by those groups.
6. **Compliance (not legal advice; confirm with counsel).**
   - Kenya Data Protection Act 2019: lawful basis and explicit consent for statement access, a data protection impact assessment, ODPC registration, and the right to contest solely automated decisions.
   - CBK rules for your licence category.
   - CRB reporting duties.
   - Keep reason codes for adverse decisions.
