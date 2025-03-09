#!/usr/bin/env python3
import os
import csv
import requests
from datetime import datetime
from typing import Dict, List
from magentic import prompt
from dotenv import load_dotenv

# Load local .env if present
load_dotenv()

AV_API_KEY = os.getenv("AV_API_KEY")

###############################################################################
# 1) Data Retrieval with Time Stamps
###############################################################################
def get_daily_price(ticker: str) -> Dict[str, Dict[str, str]]:
    """
    Fetch daily time series from Alpha Vantage (entire available range).
    We'll locally filter by date if needed.
    
    Returns a dict:
      {
        "YYYY-MM-DD": {
            "1. open": ...,
            "2. high": ...,
            "3. low": ...,
            "4. close": ...,
            "5. volume": ...
        },
        ...
      }
    }
    """
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={AV_API_KEY}"
    resp = requests.get(url, timeout=30).json()
    return resp.get("Time Series (Daily)", {})

def filter_daily_prices_by_date(
    data: Dict[str, Dict[str, str]], 
    start_date: str, 
    end_date: str
) -> Dict[str, Dict[str, str]]:
    """
    Filter the daily price dictionary to only include data between start_date and end_date (inclusive).
    Dates must be in 'YYYY-MM-DD' format.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt   = datetime.strptime(end_date,   "%Y-%m-%d")
    filtered = {}
    for date_str, values in data.items():
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue
        if start_dt <= dt <= end_dt:
            filtered[date_str] = values
    return filtered

def get_news_sentiment(
    ticker: str, 
    time_from: str = None, 
    time_to: str = None, 
    limit: int = 5
) -> List[dict]:
    """
    Fetch sentiment analysis on financial news for `ticker` from Alpha Vantage,
    with optional date range via `time_from` and `time_to` in "YYYYMMDD" format.
    E.g. time_from="20230401", time_to="20230501".
    
    If time_from/time_to are None, it won't filter by date.
    """
    base_url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}"
    if time_from:
        base_url += f"&time_from={time_from}"
    if time_to:
        base_url += f"&time_to={time_to}"
    base_url += f"&apikey={AV_API_KEY}"

    resp = requests.get(base_url, timeout=30).json()
    feed = resp.get("feed", [])[:limit]
    fields = ["time_published", "title", "summary", "overall_sentiment_score", "overall_sentiment_label"]
    return [{fld: art.get(fld, "") for fld in fields} for art in feed]

def get_earnings_calendar(ticker: str) -> List[list]:
    """Fetch upcoming earnings data for a given ticker (12-month horizon)."""
    url = (
        f"https://www.alphavantage.co/query?function=EARNINGS_CALENDAR"
        f"&symbol={ticker}&horizon=12month&apikey={AV_API_KEY}"
    )
    resp = requests.get(url, timeout=30)
    decoded = resp.content.decode("utf-8")
    cr = csv.reader(decoded.splitlines(), delimiter=",")
    return list(cr)  # Each row is a list of strings

def get_company_overview(ticker: str) -> dict:
    """Fetch fundamental data (market cap, P/E, sector, etc.)."""
    url = (
        f"https://www.alphavantage.co/query?function=OVERVIEW"
        f"&symbol={ticker}&apikey={AV_API_KEY}"
    )
    return requests.get(url, timeout=30).json()

def get_sector_performance() -> dict:
    """Fetch market-wide sector performance data."""
    url = f"https://www.alphavantage.co/query?function=SECTOR&apikey={AV_API_KEY}"
    return requests.get(url, timeout=30).json()


###############################################################################
# 2) Format Data for the LLM
###############################################################################
def format_daily_prices(
    price_data: Dict[str, Dict[str, str]]
) -> str:
    """
    Sort by date (ascending) and create a textual summary:
    2023-08-01: Close=145.21, High=147.00, Low=143.50, Volume=28900000
    """
    lines = []
    for date_str in sorted(price_data.keys()):
        row = price_data[date_str]
        c = row.get("4. close", "N/A")
        h = row.get("2. high", "N/A")
        l = row.get("3. low",  "N/A")
        v = row.get("5. volume","N/A")
        lines.append(f"{date_str}: Close={c}, High={h}, Low={l}, Volume={v}")
    return "\n".join(lines)

def format_news_data(news_list: List[dict]) -> str:
    """
    Include timestamps from 'time_published' 
    plus short summary for each news item.
    """
    lines = []
    for item in news_list:
        tstamp = item["time_published"]
        title  = item["title"]
        sent   = item["overall_sentiment_label"]
        score  = item["overall_sentiment_score"]
        lines.append(f"{tstamp} => Title={title}, Sentiment={sent} ({score})")
    return "\n".join(lines)

def format_earnings_data(earnings: List[list]) -> str:
    """
    Each row is typically something like:
      [date, symbol, reportedEPS, consensusEPS, estimatedEPS, ...]
    We'll just join columns by comma.
    """
    lines = []
    for row in earnings:
        line = ", ".join(row)
        lines.append(line)
    return "\n".join(lines)

def format_company_overview(overview: dict) -> str:
    """
    Just list key-value pairs.
    """
    return "\n".join(f"{k}={v}" for k, v in overview.items())

def format_sector_perf(sector: dict) -> str:
    """
    The sector data can be big; show some key fields.
    """
    lines = []
    for k, v in sector.items():
        lines.append(f"{k}: {v}")
    return "\n".join(lines)


###############################################################################
# 3) LLM Prompt: Next-Day Price Forecast
###############################################################################
@prompt(
    """
    You are a stock forecasting assistant.
    The user has asked you to predict the next day's closing price (in float) for a given stock,
    based on data from {start_date} to {end_date}:

    Daily Prices:
    {daily_prices}

    News (time-stamped):
    {news_data}

    Earnings Calendar:
    {earnings_data}

    Company Overview:
    {overview_data}

    Sector Performance:
    {sector_data}

    Return only a float (e.g., 145.32). No disclaimers or extra text.
    """
)
def llm_predict_price(
    start_date: str,
    end_date: str,
    daily_prices: str,
    news_data: str,
    earnings_data: str,
    overview_data: str,
    sector_data: str
) -> str:
    """
    Expects a numeric string in response.
    We'll parse it into float in code.
    """


###############################################################################
# 4) Main Script
###############################################################################
def main():
    import sys
    if len(sys.argv) < 4:
        print("Usage: python price_predict_with_timestamp.py <TICKER> <START_DATE> <END_DATE>")
        print("Example: python price_predict_with_timestamp.py AAPL 2023-01-01 2023-01-05")
        sys.exit(1)

    ticker = sys.argv[1].upper()
    start_date = sys.argv[2]  # "YYYY-MM-DD"
    end_date   = sys.argv[3]  # "YYYY-MM-DD"

    print(f"[INFO] Predicting next-day close for {ticker}, from {start_date} to {end_date}.")

    # 1) Retrieve raw data
    raw_price_dict = get_daily_price(ticker)
    filtered_price_dict = filter_daily_prices_by_date(raw_price_dict, start_date, end_date)

    # For the news sentiment, alpha vantage expects YYYYMMDD format
    # We'll remove hyphens:
    time_from = start_date.replace("-", "")
    time_to   = end_date.replace("-", "")
    news_list = get_news_sentiment(ticker, time_from=time_from, time_to=time_to, limit=5)

    earnings_list = get_earnings_calendar(ticker)
    overview_dict = get_company_overview(ticker)
    sector_dict   = get_sector_performance()

    if not filtered_price_dict:
        print("[WARNING] No price data retrieved in that date range. The prompt may be incomplete.")

    # 2) Format each data source
    daily_str   = format_daily_prices(filtered_price_dict)
    news_str    = format_news_data(news_list)
    earn_str    = format_earnings_data(earnings_list)
    over_str    = format_company_overview(overview_dict)
    sect_str    = format_sector_perf(sector_dict)

    # 3) LLM Prompt
    raw_llm_resp = llm_predict_price(
        start_date=start_date,
        end_date=end_date,
        daily_prices=daily_str,
        news_data=news_str,
        earnings_data=earn_str,
        overview_data=over_str,
        sector_data=sect_str
    )
    print("[DEBUG] LLM raw response:", raw_llm_resp)

    # 4) Parse float
    try:
        pred_price = float(raw_llm_resp.strip())
        print(f"\n[RESULT] Predicted Next-Day Close for {ticker} = {pred_price:.2f}")
    except ValueError:
        print("[ERROR] LLM did not return a valid float.")


if __name__ == "__main__":
    main()
