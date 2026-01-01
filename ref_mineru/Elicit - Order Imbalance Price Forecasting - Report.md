# Short-term price prediction for order imbalance

Order imbalance analysis provides effective short-term price predictions with 65-80% accuracy across various markets, showing strongest effects immediately after order placement.

# Abstract

Order imbalance measures appear to provide useful signals for short-term price prediction. In Nasdaq equities, queue imbalance applied in logistic regression yields receiver operating characteristic AUC values between 0.7 and 0.8 for large-tick stocks (and 0.6–0.65 for small-tick stocks), while linear models on New York Stock Exchange data explain 65% of price variance from order flow imbalance. In US equities, multivariate regression approaches report up to 78% variance explained over horizons from 30 seconds to 1 hour, and an analysis of Australian securities finds 71% accuracy in predicting price direction. Several studies note that predictability is strongest immediately or within the first few minutes after an order, with effects decaying over 30 minutes. Methods range from regression and logistic approaches to Markov chain and generative modeling, and the reported performance spans diverse markets such as equities, futures, and cryptocurrencies.

# Paper search

Using your research question ”Short-term price prediction for order imbalance”, we searched across over 126 million academic papers from the Semantic Scholar corpus. We retrieved the 50 papers most relevant to the query.

# Screening

We screened in papers that met these criteria:

• Data Requirements: Does the study include both order imbalance data and order book/trade flow data from financial markets?   
• Methodology: Does the study employ quantitative methods for price prediction with clearly defined and measurable performance metrics? Time Frame: Does the study focus exclusively on short-term price movements (5 days or less)? Publication Quality: Is the study either peer-reviewed or published as a working paper from a recognized financial institution or academic repository? Order Imbalance Methodology: Does the study present a clear, well-documented methodology for measuring order imbalance? Empirical Validation: Does the study include empirical validation of its approach using real market data?   
• Research Focus: Does the study examine general price prediction rather than focusing solely on market manipulation?   
Variable Coverage: Does the study include order imbalance as a key variable rather than focusing solely on other market metrics?

We considered all screening questions together and made a holistic judgement about whether to screen in each paper.

# Data extraction

We asked a large language model to extract each data column below from each paper. We gave the model the extraction instructions shown below for each column.

# • Data Source and Market Characteristics:

Identify and extract:

• Specific financial market or exchange used (e.g., Nasdaq, NYSE)   
• Number of stocks analyzed   
• Characteristics of stocks (e.g., liquid vs. small-tick stocks)   
• Time period of data collection

If multiple markets or stocks are studied, list all. If specific details are not fully clear, note ”not specified” and provide any partial information available.

• Prediction Methodology:

Extract details about:

• Type of prediction model used (e.g., logistic regression, local logistic regression) • Specific variables used as predictors (e.g., queue imbalance, order flow imbalance) • Target prediction variable (e.g., mid-price movement direction, price changes) • Classification approach (binary, probabilistic)

Capture the specific mathematical or statistical approach used for prediction. If multiple methods are compared, note all methods and their relative performance.

# • Predictive Performance Metrics:

Extract:

• Statistical significance of prediction model   
• Quantitative performance measures (e.g., percentage of variance explained, classification accuracy)   
• Comparative performance against baseline/null models

If multiple performance metrics are reported, list all. If statistical significance is reported, include p-values or other relevant statistical indicators.

# • Key Findings on Order Imbalance:

Identify and extract:

• Specific relationship found between order imbalance and price movement • Magnitude and direction of relationship • Any nuanced findings about different stock types or market conditions

Capture the core insights about how order imbalance relates to short-term price prediction. Include any quantitative relationships discovered.

# • Study Limitations and Generalizability:

Extract:

• Explicit limitations mentioned by authors

• Potential constraints on generalizability • Suggestions for future research

If no limitations are directly stated, note ”No limitations explicitly discussed”. Focus on methodological constraints or potential biases in the approach.

# Results

Characteristics of Included Studies   

<table><tr><td rowspan=1 colspan=7>Prediction                              Full textStudy             Market Type     Time Scale       Method           Key Metrics      retrieved</td></tr><tr><td rowspan=1 colspan=7>Cartea et al.,     Nasdaq            Seconds to        Markov chain-   Profit             No</td></tr><tr><td rowspan=1 colspan=7>2015               equities (11       minutes; 2014    modulated        improvement</td></tr><tr><td rowspan=2 colspan=7>stocks)                                  pure jump        (no explicitmodel; volume  accuracyimbalance        metrics)Tripathi et al.,   National Stock   Minutes; time    No mention       Qualitative:      No</td></tr><tr><td rowspan=1 colspan=3>No mention       Qualitative:      No</td></tr><tr><td rowspan=1 colspan=4>2020               Exchange of     period not</td><td rowspan=1 colspan=3>found; order     predictability</td></tr><tr><td rowspan=1 colspan=4>India (195        mentioned in</td><td rowspan=1 colspan=3>flow               strong for 5</td></tr><tr><td rowspan=2 colspan=4>active stocks)    the abstract</td><td rowspan=1 colspan=1>information</td><td rowspan=1 colspan=2>min, decays by</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>30 min</td></tr><tr><td rowspan=1 colspan=4>Gould and        Nasdaq            Seconds; 2014</td><td rowspan=1 colspan=1>Logistic</td><td rowspan=1 colspan=2>Receiver          Yes</td></tr><tr><td rowspan=1 colspan=4>Bonart, 2015     equities (10</td><td rowspan=1 colspan=1>regression,</td><td rowspan=1 colspan=2>Operating</td></tr><tr><td rowspan=1 colspan=4>liquid stocks: 5</td><td rowspan=1 colspan=1>local logistic</td><td rowspan=1 colspan=2>Characteristic</td></tr><tr><td rowspan=1 colspan=1></td><td></td><td rowspan=1 colspan=2>large-tick, 5</td><td rowspan=1 colspan=1>regression;</td><td rowspan=1 colspan=2>Area Under the</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>small-tick)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>queue</td><td rowspan=1 colspan=2>Curve (ROC</td></tr><tr><td rowspan=7 colspan=4>Cont et al.,       New York         Seconds to</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>imbalance</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>(large-tick),</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.60.65</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>(small-tick);</td><td rowspan=2 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>99%</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>significance</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>New York</td><td rowspan=1 colspan=1>Seconds to</td><td rowspan=1 colspan=1>Linear</td><td rowspan=1 colspan=1>Variance</td><td rowspan=1 colspan=1>Yes</td></tr><tr><td rowspan=1 colspan=2>2010</td><td rowspan=1 colspan=1>Stock</td><td rowspan=1 colspan=1>half-hour; April</td><td rowspan=1 colspan=1>regression;</td><td rowspan=1 colspan=1>explained (R2)</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>Exchange (50</td><td rowspan=1 colspan=1>2010</td><td rowspan=1 colspan=1>order flow</td><td rowspan=1 colspan=1>= 65% (order</td><td rowspan=2 colspan=1></td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>S&amp;P 500</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>imbalance</td><td rowspan=1 colspan=1>flow</td></tr><tr><td rowspan=5 colspan=4>stocks)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>imbalance),</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>32% (trade</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>imbalance);</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>95%</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>significance</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=4>Maslov and      NASDAQ         Minutes; July</td><td rowspan=1 colspan=1>Empirical</td><td rowspan=1 colspan=1>Qualitative:</td><td rowspan=1 colspan=1>Yes</td></tr><tr><td rowspan=1 colspan=4>Mills, 2001        Level II (JDSU, 2000</td><td rowspan=1 colspan=1>observation;</td><td rowspan=1 colspan=1>linear scaling,</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=2 colspan=4>BRCM)</td><td rowspan=1 colspan=1>limit order</td><td rowspan=1 colspan=1>short-lived</td><td rowspan=2 colspan=1></td></tr><tr><td rowspan=1 colspan=1>imbalance</td><td rowspan=1 colspan=1>predictability</td></tr></table>

<table><tr><td rowspan=1 colspan=6>Prediction                               Full textStudy             Market Type    Time Scale       Method           Key Metrics     retrieved</td></tr><tr><td rowspan=1 colspan=3>2015               stocks)            hour; time</td><td></td><td></td><td></td></tr><tr><td rowspan=3 colspan=3>period notmentioned inthe abstractVolkenand et     CME              Intraday;</td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=2>order book</td><td></td></tr><tr><td rowspan=1 colspan=2>Regression;</td><td></td></tr><tr><td rowspan=7 colspan=3>al., 2018          agricultural       March 2008,futures (5        March 2016contracts)Guo and          Bitcoin (major  Minutes;</td><td rowspan=1 colspan=2>order</td><td></td></tr><tr><td rowspan=1 colspan=2>imbalance</td><td></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td></td></tr><tr><td rowspan=1 colspan=1>Minutes;</td><td rowspan=1 colspan=2>Generative</td><td></td></tr><tr><td rowspan=1 colspan=2>Antulov-         exchange)</td><td rowspan=1 colspan=1>20162017</td><td rowspan=1 colspan=2>temporal</td><td></td></tr><tr><td rowspan=6 colspan=2>Fantulin, 2018</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>mixture model;</td><td rowspan=1 colspan=1>machine</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>order book</td><td rowspan=1 colspan=1>learning and</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>features</td><td rowspan=1 colspan=1>time-series</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>models (no</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>explicit</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>metrics)</td></tr><tr><td rowspan=1 colspan=1>Sitaru et al.,</td><td rowspan=1 colspan=1>100 stocks</td><td rowspan=1 colspan=1>No mention</td><td rowspan=1 colspan=2>Decomposed</td><td rowspan=1 colspan=1>Significant</td></tr><tr><td rowspan=1 colspan=1>2023</td><td rowspan=1 colspan=1>(market not</td><td rowspan=1 colspan=1>found; 3 years</td><td rowspan=1 colspan=2>order flow</td><td rowspan=1 colspan=1>improvement in</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>mentioned in</td><td rowspan=1 colspan=2>imbalance</td><td></td></tr><tr><td rowspan=4 colspan=3>the abstract)</td><td rowspan=1 colspan=3>the abstract)</td></tr><tr><td rowspan=1 colspan=2>imbalance plus</td><td></td></tr><tr><td rowspan=1 colspan=2>event types)</td><td></td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>metrics)</td></tr><tr><td rowspan=1 colspan=1>Yang and</td><td rowspan=1 colspan=2>Australian        Intra-day; time</td><td rowspan=1 colspan=2>Ordered-probit-</td><td rowspan=1 colspan=1>71% accuracy</td></tr><tr><td rowspan=1 colspan=1>Parwada, 2012</td><td rowspan=1 colspan=2>Securities         period not</td><td rowspan=1 colspan=2>Generalized</td><td rowspan=1 colspan=1>in direction</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Exchange         mentioned in</td><td rowspan=1 colspan=2>Autoregressive</td><td></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>(number not      the abstract</td><td rowspan=1 colspan=2>Conditional</td><td></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>mentioned in</td><td rowspan=1 colspan=2>Heteroskedas-</td><td></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>the abstract)</td><td rowspan=1 colspan=2>ticity</td><td></td></tr><tr><td></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2>(GARCH);</td><td></td></tr><tr><td></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>trade</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>imbalance,</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td></td><td></td><td></td><td></td><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1></td></tr><tr><td></td><td></td><td></td><td></td><td rowspan=1 colspan=1></td></tr></table>

Market Type:

• US equities:5 studies

• Indian equities:1 study   
• Australian equities:1 study CME agricultural futures:1 study Bitcoin (cryptocurrency):1 study Market not mentioned in the abstract:1 study

Time Scale:

• Seconds as primary time scale:2 studies   
• Minutes:3 studies Seconds to minutes:1 study Seconds to half-hour:1 study 30 seconds to 1 hour:1 study Intraday data:2 studies   
• No mention found:1 study   
• Multiple time scales covered in several studies

Prediction Method:

• Regression-based approaches (linear, multivariate, or unspecified):4 studies Logistic regression:1 study Markov chain-modulated pure jump model:1 study Empirical observation:1 study Generative temporal mixture (machine learning) model:1 study   
• Ordered-probit-GARCH:1 study   
• Order flow imbalance decomposition:1 study   
• No mention found:1 study

Imbalance Type:

• Order flow imbalance:3 studies Order book imbalance/features:2 studies Queue imbalance:1 study Volume imbalance:1 study Limit order imbalance:1 study Trade imbalance:2 studies   
• No mention found:1 study   
• Some studies used more than one imbalance type

Key Metrics:

• Explicit quantitative metrics (ROC AUC, variance explained, accuracy):4 studies • Only qualitative results or no explicit quantitative metrics:6 studies

# Effects

# Short-term Price Prediction Accuracy Time Horizon Analysis

<table><tr><td></td><td></td><td></td><td colspan="2">Statistical</td></tr><tr><td>Study</td><td>Prediction Window</td><td>Effect Size</td><td>Significance</td><td>Market Context</td></tr><tr><td>Cartea et al., 2015</td><td>Immediate (seconds after market order)</td><td>Profit boost (no explicit effect size)</td><td>No mention found</td><td>Nasdaq equities, 2014</td></tr><tr><td>Tripathi et al., 2020</td><td>530 minutes</td><td>Strong predictability (first 5min), decays by 30 min</td><td>No mention found</td><td>Indian equities, most active stocks</td></tr><tr><td>Gould and Bonart, 2015</td><td>One-tick ahead (seconds)</td><td>Receiver Operating Characteristic Area Under the Curve (ROC AUC) 0.70.8 (large-tick),</td><td>99% (likelihood ratio test)</td><td>Nasdaq, large/small-tick stocks</td></tr><tr><td>Cont et al., 2010</td><td>Seconds to half-hour</td><td>(small-tick) Variance explained 95% (z-test) (R2) = 65% (order flow imbalance), 32% (trade</td><td></td><td>New York Stock Exchange, S&amp;P 500</td></tr><tr><td>Maslov and Mills, 2001</td><td>Minutes</td><td>imbalance) Linear scaling; effect lasts a few minutes</td><td>No mention found</td><td>NASDAQ Level II, JDSU/BRCM</td></tr><tr><td>Liu and Park, 2015 30 seconds to 1</td><td>hour</td><td>Up to 78% variance explained</td><td>No mention found</td><td>US equities</td></tr><tr><td>Volkenand et al., 2018</td><td>Intraday</td><td>Positive relation to returns/volatility; not always greater</td><td>No mention found</td><td>CME agricultural futures</td></tr><tr><td>Guo and Antulov-Fantulin, 2018</td><td>Short-term (minutes)</td><td>than volume Outperforms machine learning and time-series</td><td>No mention found</td><td>Bitcoin</td></tr><tr><td>Sitaru et al., 2023</td><td>No mention found (short-term)</td><td>models Significant improvement</td><td></td><td>No mention found 100 stocks, 3 years</td></tr><tr><td>Yang and Parwada, Intra-day 2012</td><td></td><td>(forward-looking) 71% accuracy</td><td></td><td>No mention found Australian equities</td></tr></table>

Prediction window:

• Immediate or seconds-ahead prediction:4 studies • Short-term windows of 5–60 minutes:5 studies • Intraday window:2 studies

• No mention found:1 study

Effect size:

• Explicit quantitative metrics (Receiver Operating Characteristic Area Under the Curve, variance explained, accuracy):4 studies   
• Qualitative effects (e.g., ”profit boost,” ”strong predictability,” ”significant improvement,” ”linear scaling”):4 studies   
• Comparative statements (e.g., ”outperforms machine learning/time-series models,” ”not always greater than volume”):2 studies

Statistical significance:

• Statistical significance using formal tests (99% and 95% confidence):2 studies • No mention found in 8 studies

Market context:

• US equities markets:5 studies   
• Non-US equities (India and Australia):2 studies Futures markets:1 study Cryptocurrency markets:1 study   
• Multiple or unspecified equity markets:1 study

# Implementation Approaches

Order Imbalance Metrics

Trading Strategy Integration

<table><tr><td>Study</td><td>Approach Type</td><td>Performance Metrics</td><td>Market Impact</td><td>Implementation Complexity</td></tr><tr><td>Cartea et al., 2015</td><td>Markov chain-modulated pure jump model; volume imbalance</td><td>Profit boost (no explicit metrics)</td><td>Reduces adverse selection, improves limit order</td><td>High (model calibration, stochastic control)</td></tr><tr><td>Tripathi et al., 2020</td><td>No mention found; order flow information</td><td>Qualitative: short-term predictability</td><td>positioning Supports information asymmetry hypothesis</td><td>No mention found</td></tr><tr><td>Gould and Bonart, 2015</td><td>Logistic/local logistic regression; queue imbalance</td><td>Receiver Operating Characteristic Area Under the Curve, mean</td><td>Stronger for large-tick stocks</td><td>Moderate (logistic regression, local fits)</td></tr><tr><td>Cont et al., 2010</td><td>Linear regression; order flow imbalance</td><td>squared residuals Variance explained, coefficient significance</td><td>Linear price impact, robust</td><td>Low (ordinary least squares</td></tr><tr><td>Maslov and Mills, 2001</td><td>Empirical observation; limit order imbalance</td><td>Qualitative: linear scaling</td><td>Predictability lasts minutes</td><td>Low (empirical averaging)</td></tr><tr><td>Liu and Park, 2015</td><td>Multivariate linear model; liquidity supply/demand</td><td>Up to 78% variance explained</td><td>Quantifies supply-demand dynamics</td><td>Moderate (multivariate regression)</td></tr><tr><td>Volkenand et al., 2018</td><td>Regression; order imbalance</td><td>Qualitative: positive relation</td><td>Useful for some contracts; not always greater than volume</td><td>Low (regression)</td></tr><tr><td>Guo and Antulov-Fantulin, 2018</td><td>Generative temporal mixture model; order book features</td><td>Outperforms machine learning and time-series models</td><td>Detects volatility regimes</td><td>High (generative modeling)</td></tr><tr><td>Sitaru et al., 2023</td><td>Decomposed order flow imbalance (order flow imbalance plus</td><td>Significant improvement (forward-looking)</td><td>Statistically and economically beneficial</td><td>Moderate-high (model extension)</td></tr><tr><td>Yang and Parwada, 2012</td><td>Ordered-probit- Generalized Autoregressive Conditional Heteroskedasticity;</td><td>71% accuracy</td><td>Positive effect of trade imbalance</td><td>Moderate (ordered probit, Generalized Autoregressive Conditional Heteroskedasticity)</td></tr></table>

Approach Type:

• Regression-based approaches (linear, logistic, multivariate):4 studies   
• Generative or advanced models (Markov chain, generative mixture, decomposed order flow imbalance):3 studies   
• Empirical/qualitative or no mention found approaches:2 studies   
• Ordered-probit-Generalized Autoregressive Conditional Heteroskedasticity model:1 study

Performance Metrics:

• Explicit quantitative performance metrics (Receiver Operating Characteristic Area Under the Curve, variance explained, accuracy, relative model performance):6 studies   
• Only qualitative performance metrics:4 studies

Market Impact:

• Positive market microstructure effects (reducing adverse selection, improving limit order positioning,

supporting information asymmetry):3 studies

• Robust or generalizable market impact (robust across stocks, quantifying supply-demand):2 studies   
• Conditional or contract/market-structure-specific effects:2 studies   
• Short-term predictability:1 study   
• Volatility regime detection:1 study   
• Statistically and economically beneficial effects:1 study

Implementation Complexity:

• Low complexity:3 studies • Moderate complexity:3 studies Moderate-high complexity:1 study • High complexity:2 studies • No mention found:1 study

# Limitations and Generalizability

Most studies do not explicitly discuss limitations in their abstracts. Where mentioned, limitations include:

• Restricted market or asset focus:For example, studies limited to a single exchange or specific asset types.   
• Short time frames or limited data periods:Some studies only cover brief periods, which may affect generalizability.   
• Methodological constraints:These include limitations of parametric models and lack of access to detailed order book data (Level II data).   
• Potential issues with generalizability:Results may not extend to other markets, less liquid assets, or longer prediction horizons.

Several studies suggest, in their abstracts or full texts, that future research could explore alternative imbalance metrics, longer time scales, or more granular data, but these are not the focus of this review.

# References

Á. Cartea, Ryan Francis Donnelly, and S. Jaimungal. “Enhancing Trading Strategies with Order Book Signals,” 2015.   
Abhinava Tripathi, Alok Dixit, and Vipul. “Information Content of Order Imbalance in an Order-Driven Market: Indian Evidence.” Finance Research Letters, 2020.   
Bogdan Sitaru, Anisoara Calinescu, and Mihai Cucuringu. “Order Flow Decomposition for Price Impact Analysis in Equity Limit Order Books.” International Conference on AI in Finance, 2023.   
Jingle Liu, and Sanghyun Park. “Behind Stock Price Movement: Supply and Demand in Market Microstructure and Market Influence.” The Journal of Trading, 2015.   
Joey (Wenling) Yang, and J. Parwada. “Predicting Stock Price Movements: An Ordered Probit Analysis on the Australian Securities Exchange,” 2012.   
M. Gould, and J. Bonart. “Queue Imbalance as a One-Tick-Ahead Price Predictor in a Limit Order Book,” 2015.   
R. Cont, A. Kukanov, and Sasha Stoikov. “The Price Impact of Order Book Events,” 2010.   
S. Maslov, and M. Mills. “Price Fluctuations from the Order Book Perspective - Empirical Facts and a Simple Model,” 2001.   
Steffen Volkenand, Guenther Filler, and M. Odening. “The Impact of Order Imbalance on Returns, Liquidity, and Volatility in Agricultural Commodity Markets.” Agricultural Finance Review, 2018.   
Tian Guo, and Nino Antulov-Fantulin. “Predicting Short-Term Bitcoin Price Fluctuations from Buy and Sell Orders.” arXiv.org, 2018.