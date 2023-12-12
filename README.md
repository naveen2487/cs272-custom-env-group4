
# Stock Trading Reinforcement Learning Module
## Project Summary
<!--This project aims to develop a Reinforcement Learning module for stock trading simulation. The `StockTradingEnv` environment simulates a stock market where an agent can buy, sell, or hold stocks based on historical stock data. This module is beneficial for researchers and enthusiasts who want to experiment with RL in financial markets. The learning results will showcase how the agent learns to make trading decisions to maximize its net worth over time.-->

This project aims to address the complex challenge of decision-making in stock trading using Reinforcement Learning (RL). This RL module simulates a stock trading environment where an AI agent learns to buy, sell, or hold stocks based on historical data, aiming to maximize its financial gains. Targeted users include financial analysts, trading enthusiasts, and AI researchers, who are interested in exploring the application of RL in financial markets. The project's core is the `StockTradingEnv`, a custom Gymnasium environment that mirrors real-world stock market scenarios, providing a robust platform for experimenting and learning. The outcome of this project is a trained RL agent capable of making informed trading decisions, which is demonstrated through its performance over time in various market conditions.

## State Space
The state space in `StockTradingEnv` is a comprehensive representation of the stock market, including:
- Historical stock prices: Open, High, Low, Close, and Volume for each day.
- Agent's financial status: Current balance, number of shares held, average cost per share, total shares sold, and total value of sales.
- The data is normalized to aid in the learning process, scaling each value to a range between 0 and 1.

## Action Space
The action space of the environment is a 2D continuous space:
- Action Type: Represented by a continuous value where 0 indicates 'Buy', 1 indicates 'Sell', and values in between imply a 'Hold' strategy.
- Action Amount: A continuous value representing the percentage of the agent's balance to be used for buying or the proportion of held shares to be sold.

## Rewards
The reward strategy is designed to encourage profitable trades:
- The primary reward signal is the change in the agent's net worth, computed as the current balance plus the market value of held shares.
- A positive reward is given for an increase in net worth, incentivizing the agent to maximize returns.

## RL Algorithm 
This project utilizes the Proximal Policy Optimization (PPO) algorithm, a popular choice in RL for its balance between sample efficiency and ease of tuning. PPO's robustness makes it suitable for the complexities of the stock market environment.

## Starting State
The environment begins with:
- The agent having a predefined initial balance.
- No stock shares held initially.
- The first set of stock data based on the `n_days` parameter, providing the agent with initial market information.

## Episode End
An episode in the environment concludes when:
- The agent reaches the end of the stock data, signifying the completion of a trading period.
- This allows the agent to experience a full cycle of market conditions within an episode.

## Results
