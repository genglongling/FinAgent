# 🚀 FinAgent: Beyond Rational Frontiers, An Adaptive Boosting  Weighted Multi-LLM Agent for Financial Tasks and Stock Return Prediction

## 1. Background

### 1.1  Machine Learning
Traditional Machine Learning relies on tedious work for dataset selection and feature engineering.

### 1.2  LLM Limitations 

Large Language Models (LLMs) excel at pattern recognition but struggle with complex tasks that require:  

- 📏 **1) Math Induction, Machine Learning**  
- 🧠 **2) Deliberate reasoning**
- ⚠️ **3) Constraint management, Complex planning, scheduling, and optimization** 
- ⏳ **4) Temporal and Spacial awareness**  

---

## 2. Weighted Multi-Agent Collaborative Intelligence Framework (W-MACI)  

W-MACI is a general multi-agent system with:  

1) 🏗️ **7 Multi-Agent tools**  
   - Our newest version Fin-Agent support Magentic multi-agent tool.
   - Our previous MACI-framework could support multi-agent tools: LangGraph, AutoGen, Crewai, LangChain, and to be extended on LlamaIndex, Haystack.

2) 🤝 **20+ LLM Agents**  
   - **[OpenAI LLMs:](https://openai.com/)** including gpt-4, gpt-4o, gpt-4o-mini, etc.
   - **[Ollama:](https://ollama.com/)** including Llama 3.3, DeepSeek-R1, Phi-4, Mistral, Gemma 2, etc.
   - **[Anthropic:](https://www.anthropic.com/)** including Claud 3.7.
   - **[Mistral:](https://mistral.ai/)** 
   - **[LiteLLM:](https://docs.litellm.ai/)** 
   - **or any other OpenAI schema-compatible model** 
   - **FinAgent(ours)**
   - **Your Self-developed LLMs(ours)**

3) 📡 **Interactive Front-End App**  
   - User could interact with LLMs.
   - Adapts to unexpected changes in real-time.
  
4) 🤖 **Interactive Back-End App**  
   - Access to financial database such as [Alphavantage](https://www.alphavantage.co/documentation/), and to be extended on yahoo api.
   - Adapts to unexpected changes in real-time financial information. 

---

## 3. Bagging optimization algorithm (Bagging-MACI)  

Bagging-MACI is designed to overcome these ML and LLM limitations using our new bagging method:  

### 1) Agent Registration:  

- 🔍 **(1) Agent 1**: Company News Analysis (done)
- 🏗 **(2) Agent 2**: Company Structure and Relationship Analysis  (done)
- 📈 **(3) Agent 3**: Company Risk Analysis   (done)
- 👥 **(4) Agent 4**: With and Cross Sector Analysis  (new)
- 🚀 **(5) Agent 5**: Stock Prediction and Forecasting (new)
- 🏗️ **(6) Agent 6**: Plot Generation (new)
  
###  2) Bagging  
   - We run **Bagging** algorithm on the above selected, 6 agents.
   - Adapts to unexpected changes in real-time.  


---

## ⚙️ 4. Experiment Set-up  

We plan to evaluate our multi-agent temporal planning framework on **S&P 500 stocks (2018-2024)**, incorporating historical trading data, reports data, and other relevant financial data, and also **real-time financial data** from alphavantage. Our focus will be on three major market sectors.  

All experiments will utilize publicly available data from sources like:  

- 📈 **Yahoo Finance** (Stock Prices)  
- 📜 **SEC EDGAR** (Financial Reports)  

### ⚖️ 4.1 Baselines & Comparisons:  

1. 📊 **Single factors/Multifactors:**  
   - Single factors include: Stock price (close, open, high, volumn).
   - Multi factors include: Market shifts/trends, within-sector impact, news, macro economic factors

2. 📊 **Baseline Models:**  
   - Traditional machine learning methods (**Prophet, Logistic Regression, SVM, LSTM networks**). 
   - Potential fine-tuning with: boosting, ensembles, transformers, other methods

3. 🆚 **Our model: W-MACI and Bagging-MACI**  
   - Regression: Evaluating prediction accuracy (**MAE, MSE**) for stock price prediction.  
   - Classification: **Directional accuracy** for stock trend prediction.   

4. 🧩 **Ablation Studies:**  
   - Testing different combinations of agents and their impact on performance.  

5. 📊 **Robustness & Scalability:**  
   - Assessing performance across different market conditions and unseen stocks (out-of-sample validation).  

---

## 🎓 5. Contribution  

1. 💻 **GitHub Setup and Experiment for prophet, W-MACI, and Bagging-MACI** – *Longling Gloria Geng*  
2. 💻 **Code Testing and Experiment for XGBoost** – *Yunong Liu*  
3. 💻 **Code Testing and Experiment for LSTM, gpt-4o-mini** – *Wendy Yin*  

---
# 🚀 6. How to Run the Code

## 6.1 (Optional) Create and Activate a Virtual Environment
It is recommended to use a virtual environment to manage dependencies:

```sh
python3 -m venv env
source env/bin/activate  # On macOS/Linux
env\Scripts\activate     # On Windows
```

## 6.2 Install Dependencies
Ensure you have all necessary dependencies installed:

```sh
pip install -r requirements.txt
```

Or install manually:

```sh
pip install pandas numpy matplotlib prophet
```

## 6.3 Test on Machine Learning models
### 1) Download & Place the S&P 500 Stocks Data
The dataset is available on Kaggle:  
🔗 [S&P 500 Stocks Dataset](https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks)

Extract and place the CSV file inside the `sp500_stocks/` directory:

```sh
mkdir -p sp500_stocks
mv path/to/sp500_stocks.csv sp500_stocks/
```

### 2) Execute the Python Script
Run the stock prediction script:

```sh
cd models
python3 svr.py
python3 xgb.py
python3 main_prophet1.py
python3 LSTM.py
```

### 3) Wait for the Script to Finish
The script will:  
✅ Predict stock prices for **2 years into the future**  
✅ Load and preprocess the stock data  
✅ Train a **Prophet forecasting model**  
✅ Generate & save plots showing historical vs. forecasted values

### 4) View Generated Plots
Once the script completes, you’ll find the forecasted plots in the project folder:  
- `AAPL_forecast.png` → Forecast for **Apple**  
- `TSLA_forecast.png` → Forecast for **Tesla**  
- `META_forecast.png` → Forecast for **Meta**
- other plots etc.

### 5) (optional) Build your own ML models
```sh
cd models
python3 main.py
```

## 6.4 Test on single-LLM model
```sh
cd models
python3 LLM_rag_pipeline.py
```

This **README** provides an overview of the **CS229 Benchmarking Multi-Agent LLM with Machine Learning for Stock Prediction** project, highlighting its **motivations, project plan, methodologies, demo, and future directions.** 🚀  
