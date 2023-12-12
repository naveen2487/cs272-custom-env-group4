import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pandas as pd
class StockTradingEnv(gym.Env):
    def __init__(self, csv_file, initial_balance=10000):
        super(StockTradingEnv, self).__init__()
        self.csv_file = csv_file
        self.stock_data = self._load_csv_data()
        # Stock data
        # self.stock_data = stock_data
        self.n_days = 5  # Number of days to consider for the state
        self.day = 0

        # Financial info
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.highest_net_worth = initial_balance
        self.shares_held = 0
        self.average_share_cost = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        # Define action and observation space
        self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([2, 1]), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(30,), dtype=np.float32)
    def _load_csv_data(self):
        # Load and prepare the stock data from the CSV file
        df = pd.read_csv(self.csv_file)
        return df[['Open', 'High', 'Low', 'Close', 'Volume']].values
    def step(self, action):
        # Check for and handle potential numerical issues
        if np.isnan(self.balance):
            self.balance = 0
            print("Warning: Balance is NaN, resetting to 0.")

        # Make sure self.shares_held is not zero to avoid division by zero
        if self.shares_held == 0:
            self.average_share_cost = 0
        # Calculate the market value of the held stocks
        current_price = self.stock_data[self.day][3]  # Using Close price
        if np.isnan(current_price):
            print(f"Warning: Current price is NaN on day {self.day}.")
            current_price = 0
        market_value = self.shares_held * current_price

        # Action: [0] -> Buy/Sell/Hold, [1] -> Amount/Percentage
        action_type = action[0]
        amount = action[1]

        # Check for and handle potential numerical issues
        if np.isnan(self.balance):
            self.balance = 0
            print("Warning: Balance is NaN, resetting to 0.")

        if action_type < 1:  # Buy
            total_possible = int(self.balance / current_price)
            shares_bought = int(total_possible * amount)
            prev_cost = self.average_share_cost * self.shares_held
            additional_cost = shares_bought * current_price

            self.balance -= additional_cost
            self.shares_held += shares_bought
            self.average_share_cost = (prev_cost + additional_cost) / self.shares_held if self.shares_held > 0 else 0

        elif action_type < 2:  # Sell
            shares_sold = int(self.shares_held * amount)
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * current_price

        # Update the net worth
        net_worth = self.balance + market_value
        if np.isnan(net_worth):
            print("Warning: Net worth is NaN.")
            net_worth = 0
        self.highest_net_worth = max(self.highest_net_worth, net_worth)

        # Update the day
        self.day += 1
        done = self.day >= len(self.stock_data) - self.n_days

        # Calculate the reward
        # This is a simple reward function; modify it based on your needs
        reward = net_worth - self.initial_balance
        if np.isnan(reward):
            print("Warning: Reward is NaN.")
            reward = 0
        time_factor = self.day / 2000
        reward *= time_factor

        # Get the next observation
        new_state = self._next_observation()
        truncated = False

        return new_state, reward, done, truncated, {}

    def reset(self, *, seed=None, options=None):
        self.balance = self.initial_balance
        self.highest_net_worth = self.initial_balance
        self.shares_held = 0
        self.average_share_cost = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.day = 0

        # Reset the seed (optional, if we need deterministic behavior)
        # self.seed(seed)

        # Return the initial state and an empty info dictionary
        return self._next_observation(), {}

    def render(self, mode='human'):
        print(f'Day: {self.day}, Balance: {self.balance}, Shares held: {self.shares_held}, Total sales: {self.total_sales_value}')

    def normalize(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def _next_observation(self):
        # Use the last n_days of stock data and scale each value to between 0 and 1
        stock_matrix = self.stock_data[self.day:self.day + self.n_days]
        normalized_stock_data = self.normalize(stock_matrix)

        # Append additional financial information
        financial_info = np.array([self.balance, self.highest_net_worth, self.shares_held, self.average_share_cost, self.total_sales_value])
        normalized_financial_info = self.normalize(financial_info)

        # Combine stock data and financial information
        observation = np.append(normalized_stock_data.flatten(), normalized_financial_info)
        return observation

# Example usage
# np.random.seed(0)  # For reproducibility
# stock_data = np.random.rand(100, 5) * 100  # Dummy stock data
# env = StockTradingEnv(stock_data)
# state = env.reset()
# done = False
# while not done:
#     action = env.action_space.sample()  # Random action for demonstration
#     state, reward, done, _ = env.step(action)
#     env.render()
