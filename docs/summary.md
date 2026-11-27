# Project summary

**Question:** does order flow imbalance (OFI) predict NVDA's mid-price over
the next few milliseconds to seconds, and can that prediction be traded
after realistic costs?

**Answer:** yes, it predicts: modestly, but significantly and consistently.
No, we could not trade it profitably: the predicted moves are far smaller
than the cost of crossing the spread, and they don't stop a market maker's
quotes from being picked off.

## From raw data to a trading result

| Step | What we did | Key number | Details |
|---|---|---|---|
| **Data** | Nasdaq TotalView-ITCH captures, every message for NVDA, 61 trading days (3 Feb – 30 Apr 2025), parsed with a C++ parser (16× faster than Python, identical output) | 671M messages, 0 gaps, 0 crossed books | [E3](results.md#e3-real-data-quality-study-days), [E8](results.md#e8-c-vs-python-itch-parser) |
| **Order book** | Rebuilt the top 5 levels after every message | ~6.5M book changes per day | [E3](results.md#e3-real-data-quality-study-days) |
| **Features** | Every 100 ms: OFI at 5 levels over 4 windows (10 ms – 5 s), queue and depth imbalance, spread, trades, activity, volatility, time of day: 38 features. Labels: mid-price change over the next 10 ms – 5 s | 14.3M rows | [E9](results.md#e9-features-on-real-data-build-and-sanity-check) |
| **Split** | Train 3 Feb – 14 Mar, choose settings 17–31 Mar, test once on April (incl. the tariff selloff). Whole days only, never shuffled | — | [Guide](PROJECT_GUIDE.md) |
| **OFI explains the same-time move** | Regress the price change on OFI over the same interval | R² 30% (level 1), 44% (5 levels) at 1 s | [E4](results.md#e4-linear-ofi-baseline-on-real-nvda) |
| **Linear model predicts the next move** | Ridge regression on the 38 features | April R² 1.2–1.4% at 10–100 ms, 62–66% of moves called right | [E4](results.md#e4-linear-ofi-baseline-on-real-nvda) |
| **Machine learning predicts better** | LightGBM (gradient-boosted trees) on the same features | April R² **2.1%** at 10 ms, **68%** of moves called right; best feature: queue imbalance | [E5](results.md#e5-gradient-boosted-model-vs-linear) |
| **It holds up** | Significance tests, every test day, market regimes, sensitivity checks | Better than linear on 20–21 of 21 days up to 1 s; strongest at a 1-tick spread | [E6](results.md#e6-evaluation-and-robustness) |
| **Trade it: cross the spread** | Buy at the ask / sell at the bid on the strongest forecasts, exit after h ms; real fills, latency, fees | Loses on every day at every setting: the forecast covers ~1/10 of the cost | [E7a](results.md#e7a-crossing-the-spread) |
| **Simulate posted orders** | Order-by-order queue simulator: our place in line, cancels, trade-throughs, latency | Exact against the rebuilt book: 63,918 / 63,918 | [E7b](results.md#e7b-fill-simulator), [demo](fill-demo.md) |
| **Trade it: post quotes** | Quote both sides; with the signal, withdraw the side about to be picked off | No signal: −$15.4k/day. With signal: smaller loss only because it trades less; per share, no better | [E7c](results.md#e7c-quoting-with-vs-without-the-signal) |

## What we learned

1. **OFI matters for the next move, but the signal is small.** The best model
   explains about 2% of the variation in the price change 10 ms ahead, yet
   calls the direction right two times out of three, on every test day.
2. **Queue imbalance beats OFI.** Which side of the best quotes is thinner
   says more about the next move than recent order flow. OFI adds information
   only over the last ~10 ms.
3. **Nonlinear models help.** LightGBM's gain over linear is real
   (significant up to 1 s) and comes from interactions: the same imbalance
   means more in a quiet market.
4. **Predictable is not tradable.** Crossing the spread costs 1.6–3¢ per share
   round trip plus fees; the forecasts are worth 0.1–0.5¢. Posting quotes
   earns the spread but loses more to adverse selection (−0.4¢ per share within
   100 ms), and a 100 ms forecast reacts too late to avoid it.
5. **Most predictability sits right after book events.** Measured at book
   events, R² at 10 ms is 5× higher than on the 100 ms clock: the place to
   look next.

## Limitations

- One stock, one test month (with an unusual week: the April tariff shock).
- Backtests assume our 100-share orders don't change anyone's behaviour, and
  that we join the back of the queue.
- Fees at a standard taker rate and a mid-tier rebate.
- Hidden orders are invisible in the data; about 20% of trades have no known
  aggressor.

## What's next

- **Event-time direction model (E10):** the probability that the next mid
  change is up, at every book event. Code ready; run pending.
- **Event-time quoting:** use that probability to decide at each book event,
  not every 100 ms, which addresses why the signal didn't help in E7c.

Everything is reproducible from the repository; [Experiments](experiments.md)
has the exact commands, [Results](results.md) the full tables.
