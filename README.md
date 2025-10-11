# Stock Market Data Analysis

## Overview

This project analyzes historical stock prices to find trends, calculate returns, and visualize performance over time.  
The program downloads stock data from **Yahoo Finance** using the `yfinance` library and processes it with **Pandas** and **Matplotlib**.

The dataset includes companies like **Apple (AAPL)**, **Microsoft (MSFT)**, and **Google (GOOGL)**.  
It covers daily prices, returns, and volatility from 2019 to the present.

The purpose of this program is to learn how to work with real data, clean it, analyse it, and create visual charts and summaries.

---

## Data Analysis Results

### Questions and Answers

1. **What is the average annual return and volatility of each stock over the last 5 years?**  
   - AAPL: Around 22% return, moderate volatility  
   - MSFT: Around 18% return, stable growth  
   - GOOGL: Around 15% return, higher volatility  

2. **How have the stock prices changed over time?**  
   - All three companies show strong long-term growth trends.  
   - Apple shows the best overall performance during the period.  

3. **How are the stocks correlated?**  
   - AAPL, MSFT, and GOOGL are strongly correlated, meaning they often move in the same direction.

---

## Development Environment

**Tools Used**
- Visual Studio Code (VS Code)
- Git (for version control)

**Programming Language & Libraries**
- **Python 3.10+**
---

## Useful Websites

* [Yahoo Finance](https://finance.yahoo.com/)  
* [yfinance Documentation](https://pypi.org/project/yfinance/)  
* [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html)  
* [Pandas Documentation](https://pandas.pydata.org/docs/)  

---

## Future Work

* Add more stocks automatically from a list or file  
* Include machine learning predictions for stock trends  
* Create an interactive dashboard (e.g., Streamlit or Dash)
Install all dependencies using the command below:

```bash
pip install pandas numpy matplotlib seaborn yfinance
