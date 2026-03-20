"""
Mutual Fund knowledge base for Kiara (Eazyhaina finance assistant).
Sourced from Eazyhaina's 130 Q&A guide and NISM exam material.
Injected into prompts when the user asks about mutual funds, SIP, ELSS, NAV, etc.
"""

# Topics this knowledge covers — used to detect when to inject.
MF_KEYWORDS: frozenset[str] = frozenset({
    "mutual fund", "mf", "sip", "lump sum", "lumpsum", "nav", "amc", "elss",
    "nifty 50 index", "index fund", "etf", "debt fund", "equity fund", "hybrid fund",
    "large cap", "mid cap", "small cap", "flexi cap", "balanced advantage",
    "cagr", "sharpe", "alpha", "beta", "expense ratio", "exit load",
    "swp", "stp", "folio", "nfo", "sebi", "amfi", "arn", "kyc", "idcw",
    "dividend", "growth option", "direct plan", "regular plan", "80c",
    "ltcg", "stcg", "redemption", "sectoral fund", "gilt fund", "liquid fund",
    "overnight fund", "corporate bond", "credit risk fund", "arbitrage fund",
    "rupee cost averaging", "step-up sip", "goal-based",
})

MF_KNOWLEDGE_BLOCK: str = """
## Mutual Fund Knowledge (Eazyhaina — ARN-274673)

### Basics
- A Mutual Fund pools money from many investors and a professional fund manager invests it in stocks, bonds, or other instruments.
- Regulated by SEBI; investor money is held with a custodian (not the AMC), so even if the AMC shuts down, money is safe.
- AMC (Asset Management Company) manages the fund; examples: HDFC AMC, SBI AMC, ICICI Prudential AMC.
- Minimum investment: SIP from ₹100; lump sum from ₹100–500 depending on fund.
- KYC (Know Your Customer) is mandatory — one-time process using PAN + Aadhaar.
- Invest easily through the Eazyhaina app: complete KYC, choose a fund, start SIP or lump sum. Connect with an Eazyhaina Advisor for fund selection help.

### Fund Types
- **Equity Funds**: Invest in stocks. High risk, high potential return (~12–15% long-term). Horizon: 5–7 years minimum.
- **Debt Funds**: Invest in bonds/fixed-income. Low risk, stable returns (~6–8%). Horizon: 1–3 years.
- **Hybrid Funds**: Mix of equity and debt. Moderate risk.
- **Liquid Funds**: Very short-term debt (≤91 days). Safest; best for emergency fund. Redeemable in 1–2 days.
- **Overnight Funds**: Lend for 1 day; practically zero risk; returns ~4–5%. Good for temporarily parking cash.
- **ELSS**: Equity Linked Savings Scheme — equity fund with Section 80C tax benefit; 3-year lock-in (shortest among 80C options).
- **Index Funds**: Passively copy an index (e.g., Nifty 50). Very low expense ratio (0.1–0.2%). Consistent market-level returns.
- **ETF**: Like index funds but traded on exchange. Requires demat account.
- **Balanced Advantage Fund (BAF)**: Dynamically shifts equity/debt based on market conditions. Good for stress-free investing.
- **Arbitrage Fund**: Profits from cash/futures price difference. Very low risk; gets equity taxation. Good for short-term tax-efficient parking.
- Open-ended: buy/redeem any business day. Close-ended: fixed maturity. Interval: transact only in specified windows.

### Market Cap Categories
- **Large Cap**: Top 100 companies (Reliance, TCS, HDFC Bank). Stable, lower risk, ~10–12% returns. Good for beginners.
- **Mid Cap**: Rank 101–250. Higher growth, more risk. Horizon: 5+ years.
- **Small Cap**: Rank 251+. Highest risk and potential return. Horizon: 7–10 years minimum.
- **Flexi Cap**: Fund manager allocates freely across all caps. Good all-in-one option.
- **Multi Cap**: SEBI mandates min 25% each in Large, Mid, Small Cap.
- Portfolio by age: At 30 → 50% Large, 30% Mid, 20% Small. At 50 → 70% Large, 20% Mid, 10% Small.

### SIP (Systematic Investment Plan)
- Invest a fixed amount every month (e.g., ₹500–₹1000). Like a recurring deposit into Mutual Funds.
- Benefit: Rupee Cost Averaging — more units when market is down, fewer when up. Average cost reduces over time.
- Minimum SIP: ₹100/month.
- Missing a SIP: no mutual fund penalty, but bank may charge ₹100–200 bounce fee. If 3 consecutive missed, mandate may cancel.
- Step-up SIP: Increase SIP by 10–15% each year as income grows.
- Multiple SIPs: Yes, run separate SIPs for different goals (retirement, child education, etc.).
- Best time to start SIP: Today. Starting at 25 vs 35 makes a massive difference compounded.
- ₹5,000/month SIP for 30 years at 12% CAGR ≈ ₹1.75 crore+.

### Lump Sum vs SIP
- SIP: Better for beginners; removes market timing worry; benefits from rupee cost averaging.
- Lump Sum: Better when market is down or you have a bonus. Use STP to move lump sum into equity gradually.

### Returns & Risk
- CAGR (Compound Annual Growth Rate): Average annual return. E.g., ₹1 lakh → ₹2 lakh in 5 years ≈ 15% CAGR.
- Absolute Return: Total return without time consideration (not comparable across different periods).
- Sharpe Ratio: Return earned per unit of risk. Higher is better; above 1 is good.
- Beta: Fund movement vs market. Beta 1.5 = if market falls 10%, fund may fall 15%. Conservative investors prefer low Beta.
- Alpha: Extra return over benchmark. Positive alpha = fund manager added value.
- Standard Deviation: How much returns fluctuate. Higher SD = more volatile.
- Risk by fund type: Small Cap > Mid Cap > Large Cap > Hybrid > Debt > Liquid > Overnight.
- Past returns are indicators, not guarantees. Check 5–7 year consistency vs benchmark.
- FD: guaranteed 6–7%. Equity MF: not guaranteed, but historically 12–15% over long term.

### Taxation
- **Equity funds**: Sold within 1 year → STCG at 20%. After 1 year → LTCG at 12.5% on profits above ₹1.25 lakh.
- **Debt funds**: Gains added to income and taxed at your income tax slab rate (irrespective of holding period, for investments from Apr 2023 onward).
- **ELSS**: Tax deduction up to ₹1.5 lakh under Section 80C. At withdrawal: LTCG applies (12.5% above ₹1.25 lakh).
- **SIP taxation**: Each installment treated as separate purchase. Older units may qualify as LTCG; newer as STCG.
- **Dividend (IDCW)**: Taxed as income at your slab rate. If >₹5,000/year from one fund house, 10% TDS deducted.
- **Growth option**: More tax-efficient than IDCW for long-term wealth. Recommended for most investors.
- **SWP**: Only the profit portion of each withdrawal is taxed, not the full amount.
- **STP**: Each transfer from source fund = taxable redemption from that fund.
- No TDS on equity/debt fund redemptions for resident investors (only on IDCW >₹5,000).

### ELSS & Section 80C
- ELSS: 3-year lock-in (shortest among 80C options). PPF = 15 years, NSC = 5 years, tax-saving FD = 5 years.
- Tax deduction up to ₹1.5 lakh. If in 30% bracket, saves up to ₹46,800 including cess.
- Can invest via SIP in ELSS. Each installment has its own 3-year lock-in from that date.
- 1–2 good ELSS funds are enough. Beyond ₹1.5 lakh investment, no additional 80C benefit.
- After 3-year lock-in, can stay invested (ELSS continues as equity fund) or redeem.
- ELSS vs PPF: PPF is risk-free at ~7%; ELSS has historically given 12–15% but with market risk.

### Expense Ratio, Exit Load & Direct Plans
- **Expense Ratio**: Annual fee auto-deducted from NAV. Not a separate charge.
  - Index funds: 0.1–0.3%. Active equity: 0.5–1%. Above 2% is high.
  - Direct plans: 0.5–1% lower expense ratio than Regular plans → significantly better long-term returns.
- **Exit Load**: Charged if you redeem early. Equity funds typically 1% if redeemed within 1 year.
  - Liquid funds: no exit load after 7 days. Overnight funds: no exit load. ELSS: no exit load (lock-in serves same purpose).
- **Direct vs Regular**: Direct plans → invest without a distributor → lower expense ratio → better returns long-term.
  - Switch from Regular to Direct = redemption + new purchase → may trigger capital gains tax.

### SWP, STP & Switch
- **SWP (Systematic Withdrawal Plan)**: Withdraw fixed amount monthly. Ideal for retirement income — like a pension from your investments.
- **STP (Systematic Transfer Plan)**: Auto-transfer fixed amount from one fund to another (e.g., Liquid Fund → Equity Fund monthly to reduce timing risk).
- **Switch**: Move money from one fund to another within the same AMC. Treated as redemption + purchase → taxable.
- SIP builds wealth; SWP generates income from wealth. Together they cover the full financial lifecycle.

### NAV
- NAV (Net Asset Value): Price per unit of the fund. Calculated at end of every business day.
- Lower NAV does NOT mean cheaper or better fund. What matters is percentage growth, not absolute price.
- ₹100 NAV growing 15% = same outcome as ₹10 NAV growing 15%.

### Market Crashes
- Don't stop SIP during crashes. More units at lower prices = bigger gains during recovery.
- Don't redeem on paper losses. Sell only if goal is achieved, not because market fell.
- Market at all-time high? Still invest via SIP. Long-term trend is always upward.
- Extra surplus during a crash: good lump sum opportunity. Don't use emergency fund.

### Goal-Based Investing
- Emergency fund: Liquid Fund or Overnight Fund (3–6 months expenses). Redeemable in 1–2 days.
- Short-term (1–3 years): Debt/Liquid Funds.
- Medium-term (3–5 years): Hybrid Funds.
- Long-term (5+ years): Equity Funds.
- Retirement: Start SIP in equity at 25–30. Invest 15–20% of income. Increase 10% yearly. Shift to debt at 55–60.
- Child education: SIP from birth. ₹5,000/month for 15 years at 12% ≈ ₹25 lakh. Shift to debt 2–3 years before goal.
- Sukanya Samriddhi vs MF for daughter: SSY gives ~8% guaranteed tax-free; MF gives 12–15% with flexibility. Invest in both.

### Regulatory & Operations (NISM)
- SEBI regulates mutual funds. AMFI is the industry association (not a regulator or SRO).
- ARN (AMFI Registration Number): Mandatory for mutual fund distributors. Eazyhaina's ARN: 274673.
- EUIN: Employee/relationship manager code on application forms (in addition to ARN).
- KYC: Maintained by KRA (KYC Registration Agency). Mandatory for all mutual fund investors.
- RTA (Registrar and Transfer Agent): Maintains investor records, processes transactions (purchase, redemption, switch, dividend), generates account statements.
- Folio: Investor account identifier with the mutual fund.
- NAV calculated by fund accounting team. Account statements generated by RTA.
- Units can be held in demat form (via Depository Participant) or folio form.
- Close-ended and interval fund units are listed on stock exchanges.
- Anti-money laundering requirements are linked with KYC compliance.
- Open-ended fund: buy/redeem any business day. Close-ended: fixed maturity, listed on exchange. Interval: transact only during specified windows (minimum 2 days per window per NISM).

### Common Myths
- "MFs are only for the rich" — FALSE. Start with ₹100 SIP.
- "Low NAV = better fund" — FALSE. NAV is just the unit price; % growth is what matters.
- "FDs are always safer" — Partially true. FD guarantees principal; MFs don't. But over 7+ years, equity MFs historically outperform FDs significantly.
- "High risk = guaranteed loss" — FALSE. High risk = more short-term volatility. Long-term equity has historically rewarded patience.
""".strip()
