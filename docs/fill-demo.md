# Fill simulator demo

How does the backtest decide whether one of **our** posted orders would have
been filled? Step through the scenarios below and watch our order (blue) join
the back of the queue, move up as orders ahead of it trade or cancel, and get
filled, or not.

<iframe src="../assets/fill-demo.html" title="Fill simulator, step by step"
        style="width:100%; height:1130px; border:1px solid #e3e7ee; border-radius:10px;"></iframe>

[Open the demo in its own tab](assets/fill-demo.html){ target=_blank }. Keyboard: ← → step, space play/pause.
A link can open a given moment, e.g. `assets/fill-demo.html#s=2&i=3` (scenario 3, event 4).

## The rules it shows

| Event | What happens to our order |
|---|---|
| We arrive | We join the **back** of the queue: every share already resting at our price is ahead of us |
| An order **ahead** of us trades | Shares ahead go down (trades hit the oldest orders first) |
| An order **ahead** of us cancels or is deleted | Shares ahead go down |
| An order **ahead** of us is replaced | A replace is a new order on Nasdaq: it goes behind us, so shares ahead go down |
| An order that joined **after** us cancels | No change |
| A trade at our price with **nobody** ahead of us | **We fill** (in the data it hit an order behind us; really it would have hit us) |
| The price trades **through** ours | **We fill** the rest: for a bid, a lower bid trades or an ask is posted at or below our price |
| A later order trades while orders are still ahead of us | Breaks first-come-first-served (some order types do): conservatively, no fill |
| Our cancel reaches the exchange | Removed; fills that happened before it arrived still count |

These are the rules of the backtest's simulator (`src/backtest/queue.py`,
[E7b](results.md#e7b-fill-simulator)). On 32 real days it matched the
rebuilt order book exactly: 63,918 of 63,918 queue sizes, no impossible fills.

## Where the scenarios come from

- **Five hand-built scenarios**, one per rule: waiting in line, a replace
  losing its place, the price moving through us, and a cancel arriving too
  late or in time.
- **Two moments from the synthetic day** (generated data, the same kind of
  messages as the real feed): one order filled by its turn in the queue, one
  by a trade-through.

The real NVDA data is licensed and this site is public, so it is never shown
here.

The demo's data is produced by `scripts/make_fill_demo.py` with a second,
deliberately simple implementation (an explicit first-in-first-out list per
price). For every scenario the script checks that it gives the same queue at
entry and the same fills at the same times as the backtest's simulator, and
fails otherwise. The test suite runs that check. Building it caught one
difference, which was in the simple version: it had filled an order when a
later order traded out of turn, where the simulator's documented rule is not
to.
