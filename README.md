# Order Flow Imbalance and Short-Term Price Prediction

## About the Project

This is a group project for FIN 556 at the University of Illinois Urbana-Champaign.

We study whether changes in buying and selling pressure in the order book can help predict stock price movements over the next few milliseconds to a few seconds. We will start with NVIDIA (NVDA), then consider extending the analysis to other stocks.

An order book shows outstanding buy and sell orders at different prices. Order flow imbalance measures changes in buying and selling pressure as orders are added, canceled, or executed.

## Research Question

Can order flow imbalance help predict short-term price changes? If so, does the predictive advantage remain after accounting for the bid-ask spread and trading fees?

Our price measure is the mid-price: the average of the best available buy price and sell price.

## Planned Approach

- Create synthetic order book data for one stock to test our code before using real market data.
- Process the market data in manageable chunks, starting with NVDA.
- Build order flow imbalance and trade-sign features to measure order activity and buying or selling direction.
- Start with a simple regression, then compare it with a gradient-boosted model.
- Test predictions over horizons ranging from milliseconds to a few seconds.
- Evaluate whether any predictive advantage survives the bid-ask spread and trading fees.

Finding that a predictive advantage disappears after trading costs is also a valid research result.

## Team Workflow

We use GitHub Issues and the project board to assign tasks, track progress, and review work.

For AI-assisted work, we will save prompts, outputs, and manual changes so that the process can be reviewed and reproduced.
