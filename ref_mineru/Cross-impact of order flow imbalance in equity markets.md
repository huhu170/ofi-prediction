# Cross-impact of order flow imbalance in equity markets

Rama Cont, Mihai Cucuringu & Chao Zhang

To cite this article: Rama Cont, Mihai Cucuringu & Chao Zhang (2023) Cross-impact of order flow imbalance in equity markets, Quantitative Finance, 23:10, 1373-1393, DOI: 10.1080/14697688.2023.2236159

To link to this article: https://doi.org/10.1080/14697688.2023.2236159

![](images/133eac0b5f4acc7e5fecc661601387f9143be7df61d94e4d190f8d5c99e50888.jpg)

© 2023 iStockphoto LP

# Cross-impact of order flow imbalance in equity markets

RAMA CONT $\textcircled{1}$ †‡, MIHAI CUCURINGU †§¶- and CHAO ZHANG $\textcircled{1}$ †§¶\*

†Mathematical Institute, University of Oxford, Oxford, UK ‡Oxford Suzhou Centre for Advanced Research, Suzhou, People’s Republic of China $\ S$ Department of Statistics, University of Oxford, Oxford, UK ¶Oxford-Man Institute of Quantitative Finance, University of Oxford, Oxford, UK -The Alan Turing Institute, London, UK

Received 24 September 2022; accepted 7 July 2023; published online 17 August 2023

We investigate the impact of order flow imbalance (OFI) on price movements in equity markets in a multi-asset setting. First, we propose a systematic approach for combining OFIs at the top levels of the limit order book into an integrated OFI variable which better explains price impact, compared to the best-level OFI. We show that once the information from multiple levels is integrated into OFI, multi-asset models with cross-impact do not provide additional explanatory power for contemporaneous impact compared to a sparse model without cross-impact terms. On the other hand, we show that lagged cross-asset OFIs do improve the forecasting of future returns. We also establish that this lagged cross-impact mainly manifests at short-term horizons and decays rapidly in time.

Keywords: Market impact; Cross-impact; Order flow imbalance; Return prediction

JEL Codes: C31, C53, G14

# 1. Introduction

Accurately estimating and forecasting the impact of trading behavior of market participants on the price movements of assets carries practical implications for both practitioners and academics, such as trading cost analysis and optimal execution of trades. The impact of trades on asset prices, known as price impact, has been the focus of many studies and modeling efforts (Lillo et al. 2003, Cont et al. 2014). In a multi-asset setting, several studied have focused on the concept of cross-impact, which attempts to describe the impact of trading a given asset on the price of other assets (see Pasquariello and Vega 2015, Benzaquen et al. 2017, Capponi and Cont 2020).

Several studies have investigated contemporaneous crossimpact of order flow on returns by examining their cross-correlation structure. For example, Hasbrouck and Seppi (2001) revealed that commonality in returns among Dow 30 stocks is mostly attributed to order flow commonality. Tomas et al. (2022) built a principled approach to choosing a cross-impact model for various markets. Capponi and Cont (2020) showed that the positive covariance between returns of a specific stock and order flow imbalances of other stocks does not necessarily constitute evidence of crossimpact. They further demonstrated that, as long as the common factor in order flow imbalances is taken into account, adding cross-impact terms only marginally improves model performance, and thus may be disregarded. Our study complements Capponi and Cont (2020) in several ways: unlike Capponi and Cont (2020) which focuses on in-sample performance, we also consider the forecasting power of cross-order flow using both single and multi-level OFIs. To the best of our knowledge, there have been no studies that examine the influence of order flows on price movements in a multi-asset setting, while also taking into account the deeper levels in the limit order book (LOB).†

A more challenging problem than explaining contemporaneous returns is to examine the impact of trade orders on prices over future horizons, which has received a lot less attention in the literature, despite its important economic implications. Some studies have examined the relationship between order imbalances and future daily returns. $\ddagger$ Chordia et al. (2002) revealed that daily stock market returns are strongly related to contemporaneous and lagged order imbalances. Chordia and Subrahmanyam (2004) further found that there exists a positive relation between lagged order imbalances and daily individual stock returns. The authors also showed that imbalance-based trading strategies, i.e. buy if the previous day’s imbalance is positive, and sell if the previous day’s imbalance is negative, are able to yield statistically significant profits. Pasquariello and Vega (2015) provided empirical evidence of cross-asset informational effects in NYSE and NASDAQ stocks between 1993 and 2004, and demonstrated that the daily order flow imbalance in one stock, or across one industry, has a significant and persistent impact on daily returns of other stocks or industries. Rosenbaum and Tomas (2021) provided a characterization of the class of cross-impact kernels for a market that employs Hawkes processes to model trades and applied their method to two instruments from E-Mini Futures.

Given the recent progress in high-frequency trading (HFT), it is increasingly crucial to obtain accurate estimations of the cross-impact on future intraday returns. Benzaquen et al. (2017) introduced a multivariate linear model (see Kyle 1985) to describe the structure of cross-impact and found that a significant fraction of the covariance of stock returns can be accounted for by this model. Wang et al. (2016a, 2018) empirically analyzed and discussed the impact of trading a specific stock on the average price change of the whole market or of individual sectors. Schneider and Lillo (2019) derived theoretical limits for the size and form of crossimpact and verified them on sovereign bonds data. However, when modeling cross-impact, these methods do not consider the possibility of high correlations between cross-asset order flows, which may result in overfitting issues. This is also evidenced by studies such as Benzaquen et al. (2017) and Tomas et al. (2022). Moreover, these studies mainly investigated the cross-impact coefficients for a fixed time period (i.e. in a static setting), ignoring the temporal dynamics of cross-impact.

In recent years, machine learning models including deep neural networks, have achieved substantial developments, leading to their applications in financial markets, especially for the task of modeling stock returns. For example,

Huck (2019) utilized state-of-the-art techniques, such as random forests, to construct a portfolio over a period of 22 years, and the results demonstrated the power of machine learning models to produce profitable trading signals. Krauss et al. (2017) applied a series of machine learning methods to forecast the probability of a stock outperforming the market index, and then constructed long-short portfolios from the predicted one-day-ahead trading signals. Gu et al. (2020) employed a set of machine learning methods to make onemonth-ahead return forecasts, and demonstrated the potential of machine learning approaches in empirical asset pricing, due to their ability to handle nonlinear interactions. Ait-Sahalia et al. (2022) investigated the predictability of high-frequency stock returns and durations using LASSO and tree methods via many relevant predictors derived from returns and order flows. Tashiro et al. (2019) and Kolm et al. (2023) applied deep neural networks with LOB-based features to predict high-frequency returns. Nonetheless, to the best of our knowledge, cross-asset order flow imbalances have not been considered as predictors for forecasting future high-frequency returns in the literature, which is one of the main directions we explore in the second half of this paper.

# 1.1. Main contributions

The present study makes two main contributions to the literature regarding the contemporaneous and predictive crossimpact of order flow imbalances on price returns.

First, we revisit the significance of contemporaneous crossimpact by considering various definitions of order flow imbalance (OFI). Instead of only looking at the best-level orders, we systematically examine the impact of multi-level order flows in a cross-asset setting. Our results show that, once information from multi-level order flow is incorporated in the definition of order flow imbalance, cross-impact terms do not provide additional explanatory power for contemporaneous impact, compared to a parsimonious model without crossimpact. To the best of our knowledge, this is the first study to comprehensively analyze the relations between contemporaneous individual returns and multi-level orders in both single-asset and multi-asset settings.

Furthermore, we consider the associated forecasting problem and investigate the predictive power of the cross-asset order flows on future price returns. Our results suggest that cross-impact terms do provide significant information content for intraday forecasting of future returns over a short horizon of up to several minutes, but their predictability decays quickly through time.

# 1.2. Outline

Section 2 describes our dataset and defines the variables of interest. Section 3 discusses modeling of contemporaneous cross-impact. In Section 4, we first discuss the out-of-sample forecasting performance of cross-impact models over oneminute-ahead horizon from two perspectives: $R ^ { 2 }$ values and economic gains, and then examine the predictability over longer horizons. Finally, we conclude the analysis in Section 5 and highlight potential future research directions.

# 2. Data and variables

# 2.1. Data

We use the Nasdaq ITCH data from LOBSTER to compute the independent and dependent variables. Our data includes the top 100 components of S&P 500 index, existing from 2017-01-01 to 2019-12-31.†

Cont et al. (2014) found that over short time intervals, price changes are mainly driven by the Order Flow Imbalance (henceforth denoted as OFI). Kolm et al. (2023) also demonstrated that forecasting deep learning models trained on OFIs significantly outperform most models trained directly on order books or returns. Therefore, we adopt the OFIs as features in our below analysis.

During the interval $( t - h , t ]$ , we enumerate the observations of all order book updates by $n$ . Given two consecutive order book states for a given stock $i$ at $n - 1$ and $n$ , we compute the bid order flows (OFm,bi,n ) and ask order flows $( \mathrm { O F } _ { i , n } ^ { m , a } )$ of stock $i$ at level $m$ at time $n$ as

$$
\begin{array} { r l r } & { } & { \mathrm { i f } \mathrm { } { P } _ { i , n } ^ { m , b } > { P } _ { i , n - 1 } ^ { m , b } , } \\ & { } & { \mathrm { O F } _ { i , n } ^ { m , b } : = \left\{ \begin{array} { l l } { q _ { i , n } ^ { m , b } , } & { \mathrm { i f } { P } _ { i , n } ^ { m , b } = P _ { i , n - 1 } ^ { m , b } , } \\ { q _ { i , n } ^ { m , b } - q _ { i , n - 1 } ^ { m , b } , } & { \mathrm { i f } { P } _ { i , n } ^ { m , b } = P _ { i , n - 1 } ^ { m , b } , } \\ { - q _ { i , n } ^ { m , b } , } & { \mathrm { i f } { P } _ { i , n } ^ { m , b } < P _ { i , n - 1 } ^ { m , b } , } \end{array} \right. } \\ & { } & { \mathrm { O F } _ { i , n } ^ { m , a } : = \left\{ \begin{array} { l l } { - q _ { i , n } ^ { m , a } , } & { \mathrm { i f } { P } _ { i , n } ^ { m , a } > { P } _ { i , n - 1 } ^ { m , a } , } \\ { q _ { i , n } ^ { m , a } - q _ { i , n - 1 } ^ { m , a } , } & { \mathrm { i f } { P } _ { i , n } ^ { m , a } = P _ { i , n - 1 } ^ { m , a } } \\ { q _ { i , n } ^ { m , a } , } & { \mathrm { i f } { P } _ { i , n } ^ { m , a } < P _ { i , n - 1 } ^ { m , a } , } \end{array} \right. } \end{array}
$$

i,n of shares) of stock m,a where, Pm,bi,n and $q _ { i , n } ^ { m , b }$ $i$ denote the bid price and size (in number at level $m$ , respectively. Similarly, $P _ { i , n } ^ { m , a }$ and $q _ { i , n } ^ { m , a }$ denote the ask price and ask size at level , respectively. Note that the variable $\mathrm { O F } _ { i , t } ^ { m , b }$ is positive when (i) the bid price increase; (ii) the bid price remains the same and the bid size increases. $\mathrm { { O F } } _ { i , t } ^ { m , b }$ is negative when (i) the bid price decreases; (ii) the bid price remains the same and the bid size decreases. One can perform an analogous analysis and interpretation for the ask order flows OFm,ai,t .

2.1.1. Best-level OFI. It calculates the accumulative OFIs at the best bid/ask side during a given time interval (see Cont et al. 2014, Kolm et al. 2023), and is defined $\mathrm { { a s } \ddag }$

$$
\mathrm { O F I } _ { i , t } ^ { 1 , h } : = \sum _ { n = N ( t - h ) + 1 } ^ { N ( t ) } \mathrm { O F } _ { i , n } ^ { 1 , b } - \mathrm { O F } _ { i , n } ^ { 1 , a } ,
$$

where $N ( t - h ) + 1$ and $N ( t )$ are the indexes of the first and the last order book event in the interval $( t - h , t ]$ .

2.1.2. Deeper-level OFI. A natural extension of the bestlevel OFI defined in equation (1) is deeper-level OFI (see Xu et al. 2018, Kolm et al. 2023). We define OFI at level $m$ $( m \ge 1 ) ,$ as follows

$$
\mathrm { O F I } _ { i , t } ^ { m , h } : = \sum _ { n = N ( t - h ) + 1 } ^ { N ( t ) } \mathrm { O F } _ { i , n } ^ { m , b } - \mathrm { O F } _ { i , n } ^ { m , a } .
$$

Due to the intraday pattern in limit order depth, we use the average size to scale OFIs at the corresponding levels (consistent with Ahn et al. 2001, Harris and Panchapagesan 2005), and consider

$$
\mathrm { o f } _ { i , t } ^ { m , h } = \frac { { \mathrm { O F I } } _ { i , t } ^ { m , h } } { Q _ { i , t } ^ { M , h } } ,
$$

where $\begin{array} { r } { Q _ { i , t } ^ { M , h } = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \frac { 1 } { 2 \Delta N ( t ) } \sum _ { n = N ( t - h ) + 1 } ^ { N ( t ) } [ q _ { i , n } ^ { m , b } + q _ { i , n } ^ { m , a } ] } \end{array}$ i s the average order book depth across the first $M$ levels and $\Delta N ( t ) = N ( t ) - N ( t - h )$ is the number of events during $( t - h , t ]$ . In this paper, we consider the top $M = 1 0$ levels e the multi-level OFI vector as. $\mathbf { o } \mathbf { \hat { n } } _ { i , t } ^ { ( h ) } =$ $( \mathrm { o f f } _ { i , t } ^ { 1 , h } , \dots , \mathrm { o f f } _ { i , t } ^ { 1 0 , h } ) ^ { T }$

2.1.3. Integrated OFI. Our following analysis in Section 2.2 will show that there exist strong correlations between multilevel OFIs, and that the first principal component can explain over $89 \%$ of the total variance among multi-level OFIs. In order to make use of the information embedded in multiple LOB levels and avoid overfitting, we propose an integrated version of OFIs via Principal Components Analysis (PCA) as shown in equation (4), which only preserves the first principal component. $\ S$ We further normalize the first principal component by dividing by its $l _ { 1 }$ norm so that the weights of multi-level OFIs in constructing integrated OFIs sum to 1, leading to

$$
\mathrm { o f f } _ { i , t } ^ { I , h } = \frac { { \pmb w } _ { 1 } ^ { T } \mathbf { o f } _ { i , t } ^ { ( h ) } } { \| { \pmb w } _ { 1 } \| _ { 1 } } ,
$$

where $w _ { 1 }$ is the first principal vector computed from historical data. To the best of our knowledge, this is the first work to aggregate multi-level OFIs into a single variable.

2.1.4. Logarithmic returns. Our dependent variable is the logarithmic asset return. Specifically, we define the returns

Table 1. Summary statistics of OFIs and returns.   

<table><tr><td></td><td>Mean (bp)</td><td>Std (bp)</td><td>Skewness</td><td>Kurtosis</td><td>10% (bp)</td><td>25% (bp)</td><td>50% (bp)</td><td>75% (bp)</td><td>90% (bp)</td></tr><tr><td>ofi1,(1m)</td><td>-0.01</td><td>6.26</td><td>-0.04</td><td>1.89</td><td>-7.97</td><td>-3.45</td><td>0.03</td><td>3.47</td><td>7.90</td></tr><tr><td>of ,(1m)</td><td>0.01</td><td>6.86</td><td>-0.04</td><td>1.04</td><td>- 8.86</td><td>- 3.88</td><td>0.02</td><td>3.95</td><td>8.85</td></tr><tr><td>of,(1m)</td><td>-0.01</td><td>7.05</td><td>-0.04</td><td>0.71</td><td> 9.26</td><td>-4.08</td><td>0.01</td><td>4.11</td><td>9.19</td></tr><tr><td>f,(1m)</td><td>- 0.02</td><td>7.22</td><td>-0.05</td><td>0.68</td><td>-9.50</td><td>− 4.21</td><td>0.01</td><td>4.24</td><td>9.40</td></tr><tr><td>of5,(1m)</td><td>-0.03</td><td>7.14</td><td>-0.05</td><td>0.79</td><td>-9.38</td><td>-4.14</td><td>0.01</td><td>4.15</td><td>9.25</td></tr><tr><td>of6,(1m)</td><td>-0.03</td><td>6.87</td><td>-0.04</td><td>0.96</td><td>-8.98</td><td>3.94</td><td>0.01</td><td>3.95</td><td>8.85</td></tr><tr><td>of7,(1m)</td><td>-0.03</td><td>6.39</td><td>-0.05</td><td>1.29</td><td>-8.31</td><td>- 3.59</td><td>0.01</td><td>3.59</td><td>8.16</td></tr><tr><td>of8,(1m)</td><td>-0.03</td><td>6.03</td><td>-0.05</td><td>1.59</td><td>-7.80</td><td>- 3.37</td><td>0.01</td><td>3.36</td><td>7.66</td></tr><tr><td>of9(1m)</td><td>- 0.05</td><td>5.71</td><td>-0.05</td><td>1.96</td><td>7.38</td><td>3.18</td><td>0.01</td><td>3.14</td><td>7.19</td></tr><tr><td>o100,(1m)</td><td>-0.05</td><td>5.38</td><td>-0.05</td><td>2.52</td><td>-6.92</td><td>-2.97</td><td>0.01</td><td>2.91</td><td>6.74</td></tr><tr><td>ofI,(1m)</td><td>0.01</td><td>6.53</td><td>-0.05</td><td>0.76</td><td> 8.52</td><td>-3.81</td><td>0.05</td><td>3.89</td><td>8.47</td></tr><tr><td>r(1m)</td><td>0.02</td><td>4.81</td><td>-0.04</td><td>1.85</td><td> 6.22</td><td>- 2.71</td><td>0.00</td><td>2.79</td><td>6.23</td></tr></table>

Note: These statistics are computed at the minute level across each stock and the full sample period. $1  { \mathrm { b p } } = 0 . 0 0 0 1 = 0 . 0 1 \%$ .

![](images/a85f336c9d7c8730ac6fefb2ff085490df71e5655ffc7c04d54d3da532137f62.jpg)  
Figure 1. Correlation matrix of multi-level OFIs. (a) Average. (b) AAPL. (c) JPM and (d) JNJ. Note: Plot (a) is averaged across each stock and each trading day, Plots (b)-(d): correlation matrix of Apple (AAPL), JPMorgan Chase (JPM), and Johnson & Johnson (JNJ) averaged across each trading day. The $x \cdot$ -axis and $y$ -axis represent different levels of OFIs.

over the interval $( t - h , t ]$ as follows:

$$
r _ { i , t } ^ { ( h ) } = \log \left( \frac { P _ { i , t } } { P _ { i , t - h } } \right) ,
$$

where $P _ { i , t }$ is the mid-price at time $t ,$ , i.e. $\begin{array} { r } { P _ { i , t } = \frac { P _ { i , t } ^ { 1 , b } + P _ { i , t } ^ { 1 , a } } { 2 } } \end{array}$

# 2.2. Summary statistics

Table 1 presents the summary statistics of multi-level OFIs, integrated OFIs, and returns for the top 100 components of S&P 500 index. These descriptive statistics (e.g. mean, std, etc) are computed at the minute level and aggregated across trading days and stocks.

Figure 1 reveals that even though the correlation structure of multi-level OFIs may vary across stocks, they all show strong relationships (above $7 5 \%$ ). It is worth pointing out that the best-level OFI exhibits the smallest correlation with any of the remaining nine levels, a pattern that persists across different stocks. Table 2 further reveals that the first principal component explains more than $89 \%$ of the total variance.

In figure 2, we show statistics pertaining to the weights attributed to the top 10 levels in the first principal component. Plot figure 2(a) shows the average weights, and the one standard deviation bars, across all stocks in the universe. Plot figure 2(a) reveals that the best-level OFI has the smallest weight in the first principal component, but the highest standard deviation, hinting that it fluctuates significantly across stocks. Plots (b–d) show various patterns for the first principal component of multi-level OFIs, for each quantile bucket of various stock characteristics, in particular, for volume, volatility and spread. For instance, in figure 2(b), the red curve shows the average weights in the first principal component for each of the 10 levels, where the average is taken over all the top $2 5 \%$ largest volume stocks. A striking pattern that emerges from this figure is that for high-volume (red line in figure 2(b)), and low-volatility stocks (blue line in figure 2(c)), OFIs deeper in the LOB receive more weights in the first component. However, for low-volume (blue line in figure 2(b)), and large-spread stocks (red line in figure 2(d)), the best-level OFIs account more than the deeper-level OFIs.

# 3. Contemporaneous cross-impact

In this section, we study the existence of contemporaneous cross-impact by comparing it with the price impact model studied in Cont et al. (2014).

# 3.1. Models

3.1.1. Price impact of best-level OFIs. We first pay attention to the price impact of best-level OFI $\mathrm { ( o f } _ { i , t } ^ { \mathbf { \bar { l } } , h } .$ ) on contemporaneous returns $( r _ { i , t } ^ { ( h ) } )$ that materialize over the same

Table 2. Average percentage and the standard deviation (in parentheses) of variance attributed to each principal component.   

<table><tr><td>Principal Component</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td></tr><tr><td>Explained Variance Ratio</td><td>89.06 (6.12)</td><td>4.99 (3.52)</td><td>2.28 (1.26)</td><td>1.28 (0.74)</td><td>0.80 (0.48)</td><td>0.54 (0.34)</td><td>0.39 (0.25)</td><td>0.29 (0.19)</td><td>0.21 (0.15)</td><td>0.15 (0.11)</td></tr></table>

Note: The table reports the ratio (in percentage points) between the variance of each principal component and the total variance averaged across each stock and trading day.

![](images/a2e361d866457076cb3be02d24f8c6a9d1aab2ea919e5354aa8aea64c87ea9b3.jpg)  
Figure 2. First principal component of multi-level OFIs, in quantile buckets for various stock characteristics. (a) Average. (b) Volume. (c) Volatility and (d) Spread. Note: The $x \cdot$ -axis indexes the top 10 levels of the OFIs. Volume: trading volume on the previous trading day. Volatility: volatility of 1-minute returns during the previous trading day. Spread: average bid-ask spread during the previous trading day. $[ 0 \% , 2 5 \% )$ , respectively $[ 7 5 \%$ , $100 \% ]$ , denote the subset of stocks with the lowest, respectively highest, $2 5 \%$ values for a given stock characteristic.

time bucket as the OFI, via the model

$$
\mathbf { P I } ^ { [ 1 ] } : ~ r _ { i , t } ^ { ( h ) } = \alpha _ { i } ^ { [ 1 ] } + \beta _ { i } ^ { [ 1 ] } \mathrm { o f } _ { i , t } ^ { 1 , h } + \epsilon _ { i , t } ^ { [ 1 ] } .
$$

Here, $\alpha _ { i } ^ { [ 1 ] }$ 1] and β [1]i are the intercept and slope coefficients, respectively. $\epsilon _ { i , t } ^ { [ 1 ] }$ is a noise term summarizing the influences of other factors, such as the OFIs at even deeper levels, and potentially the trading behaviors of other stocks. For the sake of simplicity, we refer to the above regression model as $\mathbf { P I } ^ { [ 1 ] }$ and use ordinary least squares (OLS) to estimate it.

3.1.2. Price impact of integrated OFIs. The second model specification takes into account the impact of multi-level OFIs by leveraging the integrated OFIs, which we set up as follows and use OLS for estimation.

$$
\mathbf { P I } ^ { I } : ~ r _ { i , t } ^ { ( h ) } = \alpha _ { i } ^ { I } + \beta _ { i } ^ { I } \mathrm { o f } _ { i , t } ^ { I , h } + \epsilon _ { i , t } ^ { I } .
$$

3.1.3. Cross-impact of best-level OFIs. Assuming there are $N$ stocks in the studied universe, we incorporate the multiasset best-level OFIs, $\mathrm { o f f } _ { j , t } ^ { 1 , h } ( j = 1 , \ldots , N )$ , as candidate features to help fit the returns of the ith stock $r _ { i , t } ^ { ( h ) }$ . For simplicity, we denote the impact from itself (stock $i$ ) as Self and that from other stocks as Cross, as shown below,

$$
\mathbf { C I } ^ { [ 1 ] } : \quad r _ { i , t } ^ { ( h ) } = \alpha _ { i } ^ { [ 1 ] } + \underbrace { \beta _ { i , i } ^ { [ 1 ] } \mathrm { o f } _ { i , t } ^ { 1 , h } } _ { S e l f } + \sum _ { j \neq i } \underbrace { \beta _ { i , j } ^ { [ 1 ] } \mathrm { o f } _ { j , t } ^ { 1 , h } } _ { C r o s s } + \eta _ { i , t } ^ { [ 1 ] } .
$$

Therefore, $\beta _ { i , j } ^ { [ 1 ] }$ represents the influence of the jth stock’s bestlevel OFIs on the returns of stock $i$ .

3.1.4. Cross-impact of integrated OFIs. Finally, we incorporate the cross-asset integrated OFIs to explore the impact of

multi-level OFIs from other assets, resulting in the following $\mathbf { C I } ^ { I }$ model,

$$
\mathbf { C } \mathbf { I } ^ { I } : \quad r _ { i , t } ^ { ( h ) } = \alpha _ { i } ^ { I } + \underbrace { \beta _ { i , i } ^ { I } \mathrm { o f } _ { i , t } ^ { I , h } } _ { S e l f } + \sum _ { j \neq i } \underbrace { \beta _ { i , j } ^ { I } \mathrm { o f } _ { j , t } ^ { I , h } } _ { C r o s s } + \eta _ { i , t } ^ { I } .
$$

3.1.5. Sparsity of cross-impact. As we are aware, OLS regression becomes ill-posed when there are fewer observations than parameters. Recall that we are now considering $N \approx 1 0 0$ independent variables in equations (8) and (9). Assuming the time interval is one minute and we are interested in estimating the intraday cross-impact models, e.g. relying on the $3 0 \mathrm { - m i n }$ estimation window and 1-min returns (as in Cont et al. 2014), then it seems inappropriate to estimate $\dot { \mathbf { C } } \mathbf { I } ^ { [ 1 ] }$ and $\mathbf { C I } ^ { I }$ for intraday scenarios using the OLS regression with more variables than observations. Moreover, the multicollinearity issue of features contradicts the necessary condition for a well-posed OLS. As displayed in figure 3, a significant portion of the cross-asset correlations based on the best-level OFIs cannot be ignored. For example, approximately $10 \%$ of correlations are larger than 0.30. Last, Capponi and Cont (2020) found that a certain number of cross-impact coefficients from their OLS regressions are not statistically significant at the $1 \%$ significance level.

With all the above considerations in mind, we assume that there is a small number of assets having a significant impact on a specific stock i, as opposed to the entire universe, in $\bar { \mathbf { C I } } ^ { [ 1 ] }$ and $\mathbf { C I } ^ { I }$ . To this end, we apply the Least Absolute Shrinkage and Selection Operator $( \mathrm { L A S S O } ) \dagger$ to solve equations (8)

![](images/15d2ae80a346bfbde48b25be1ef6ccc9487ecf86b784fe856647e1c9da0140e8.jpg)  
Figure 3. Distribution of correlations based on the best-level OFIs. Note: The orange vertical line represents the average correlation.

and (9). The sparsity of cross-impact terms also facilitates the explanation of coefficients. Note that even though the sparsity of the cross-impact terms is not theoretically guaranteed, our empirical evidence confirms this modeling assumption.

# 3.2. Empirical results

For a more representative and fair comparison with previous studies, we apply a similar procedure described in Cont et al. (2014) to our experiments. We exclude the first and last 30 minutes of the trading day due to the increased volatility near the opening and closing sessions, in line with Hasbrouck and Saar (2002), Chordia et al. (2002) and Chordia and Subrahmanyam (2004), Cont et al. (2014), Capponi and Cont (2020). In particular, we use each non-overlapping 30- minute estimation window during the intraday time interval $1 0 { : } 0 0 \mathrm { a m } { - } 3 { : } 3 0 \mathrm { p m }$ to estimate the regressions, namely equations (6)–(9). Within each window, returns and OFIs are computed for every minute.

3.2.1. In-sample performance. We first measure the model performance via in-sample adjusted- $R ^ { 2 }$ , denoted as the insample $R ^ { 2 }$ or IS $R ^ { 2 }$ . From table 3, we first observe that $\mathbf { P I } ^ { [ 1 ] }$ can explain $7 1 . 1 6 \%$ of the in-sample variation of a stock’s contemporaneous returns, consistent with the findings of Cont et al. (2014). Meanwhile, $\mathbf { P } \mathbf { I } ^ { I }$ displays higher and more consistent explanation power, with an average adjusted $R ^ { 2 }$ value of $8 7 . 1 4 \%$ and a standard deviation of $9 . 1 6 \%$ , indicating the effectiveness of our integrated OFIs. $^ \dagger$

Table 3 also shows that the in-sample $R ^ { 2 }$ values increase as cross-asset OFIs are included as additional features, which is not surprising given that $\mathbf { P I } ^ { [ 1 ] }$ (respectively, $\mathbf { P } \mathbf { I } ^ { I }$ ) is a nested model of $\mathbf { C I } ^ { [ 1 ] }$ (respectively, $\mathbf { C } \mathbf { I } ^ { I }$ ). However, the increments of the in-sample $R ^ { 2 }$ are smaller when using integrated OFIs $( 8 7 . 8 5 \% - 8 7 . 1 4 \% = 0 . 7 1 \%$ ), compared to the counterpart using best-level OFIs $( 7 3 . 8 7 \% - 7 1 . 1 6 \% = 2 . 7 1 \% )$ ). This indicates that cross-asset multi-level OFIs may not provide additional information on the variance in returns compared to the price impact model with integrated OFIs.

Table 3. In-sample performance for contemporaneous returns.   

<table><tr><td></td><td colspan="2">Best-level OFIs</td><td colspan="2">Integrated OFIs</td></tr><tr><td></td><td>PI[1]</td><td>CI[1]</td><td>PII</td><td>CII</td></tr><tr><td>IS R²</td><td>71.16</td><td>73.87</td><td>87.14</td><td>87.85</td></tr><tr><td></td><td>(13.80)</td><td>(12.23)</td><td>(9.16)</td><td>(8.58)</td></tr></table>

Note: The table reports the mean values and standard deviations (in parentheses) of in-sample $R ^ { 2 }$ (in percentage points) of various models when modeling contemporaneous returns. The models include $\mathbf { P I } ^ { [ 1 ] }$ (equation (6)), $\mathbf { C } \mathbf { \hat { I } } ^ { [ 1 ] }$ (equation (8)), $\mathbf { P } \mathbf { \Phi } _ { } ^ { I }$ (equation (7)), and $\mathbf { C I } ^ { I }$ (equation (9)). These statistics are averaged across each stock and each regression window.

Next, we take a closer look at the cross-impact coefficients based on either the best-level or integrated OFIs, i.e. $\beta _ { i , j } ^ { [ 1 ] }$ and $\beta _ { i , j } ^ { I }$ $( i , j = 1 , \dots , N )$ . Table 4 reveals the frequency of selfimpact and cross-impact variables selected by LASSO, i.e. the frequency of $\beta _ { i , j } ^ { \left[ 1 \right] } \neq 0$ (respectively, $\beta _ { i , j } ^ { I } \neq 0$ ). We observe that self-impact variables are consistently chosen in both $\mathbf { C I } ^ { [ 1 ] }$ and $\mathbf { C } \mathbf { I } ^ { I }$ , as found in Cont et al. (2014). However, another interesting observation is that the frequency of a cross-asset integrated OFI variable selected by $\mathbf { C I } ^ { I }$ is around $1 / 2$ of its counterpart in $\mathbf { C I } ^ { [ 1 ] }$ . When we turn to the size of the average regression coefficients as shown in table 4, we obtain reasonably consistent results. The self-impact is much higher than the cross-impact in both the $\mathbf { C I } ^ { [ 1 ] }$ and $\mathbf { C } \mathbf { I } ^ { I }$ models, while the cross-impact coefficients in $\mathbf { C I } ^ { I }$ are about $1 / 3$ in scale of their counterparts in $\mathbf { C I } ^ { [ 1 ] }$ . This difference in scale may suggest that the cross-impact terms are less important in the $\mathbf { C I } ^ { I }$ model, however, it is worth noting that even small cross-term coefficients can have a non-negligeable effect when aggregated at the portfolio level.

![](images/9acada298ca8f8cf3a3f3b50418805daeb39f417f30ac421791cc7c9ddac65c6.jpg)  
Figure 4. Barplot of singular values for the coefficient matrix in contemporaneous cross-impact models. Note: We perform Singular Value Decomposition (SVD) on the coefficient matrix to obtain the singular values. Singular values are in descending order and the coefficients are averaged over each regression window between 2017–2019. The $x$ -axis represents the singular value rank, and the $y$ -axis represents the singular values.

Table 4. Summary statistics of coefficients in the cross-impact models $\mathbf { C I } ^ { [ 1 ] }$ and $\mathbf { C I } ^ { I }$ .   

<table><tr><td></td><td colspan="2">Frequency (%)</td><td colspan="2">Magnitude</td></tr><tr><td></td><td>CI[1]</td><td>CI</td><td>CI[1]</td><td>CI</td></tr><tr><td>Self</td><td>99.85</td><td>99.96</td><td>1.02</td><td>1.24</td></tr><tr><td></td><td>(0.34)</td><td>(0.18)</td><td>(0.31)</td><td>(0.34)</td></tr><tr><td>Cross</td><td>17.34</td><td>8.29</td><td>4.5e-3</td><td>1.6e-3</td></tr><tr><td></td><td>(2.78)</td><td>(2.56)</td><td>(1.3e-³)</td><td>(0.7e-³)</td></tr></table>

Note: The table is calculated over each stock and each regression window. The first two columns describe the frequency of Self and Cross variables chosen by the corresponding model with a standard deviation (in parentheses); The last two columns describe the magnitude of Self and Cross coefficients in the corresponding model with a standard deviation (in parentheses).

structure, in accordance with previous studies (e.g. Benzaquen et al. 2017). This behavior could be fueled by index arbitrage strategies, where traders may, for example, trade an entire basket of stocks coming from the same sector against an index.

Figure 5(b) presents the network of cross-impact coefficients based on integrated OFIs, i.e. $[ \beta _ { i , j } ^ { I } ] _ { j \neq i }$ . Compared with figure 5(a), the connections in figure 5(b) are much weaker, implying that the cross-impact from stocks can be potentially explained by a stock’s own multi-level OFIs, to a large extent. Note that there is only one connection from GOOGL to GOOG, as pointed out at the top of figure $5 ( \mathrm { b } )$ . This stems from the fact that both stock ticker symbols pertain to Alphabet (Google). Our study also reveals that OFIs of GOOGL have more influence on the returns of GOOG, not the other way around. The main reason might be that GOOGL shares have voting rights, while GOOG shares do not.

In addition, cross-impact being large/small is a statement about a matrix, more related to its singular values and relative magnitudes, rather than the individual value of the coefficients. Figure 4 shows a comparison of the top 20 singular values of the coefficient matrices given by the best-level and integrated OFIs.† The relatively large singular values of the best-level OFI matrix are a consequence of the higher edge density, and thus average degree, of the network. Note that both networks exhibit a large top singular value of the adjacency matrix (akin to the usual market mode in Laloux et al. 2000), and the integrated OFI network has a faster decay of the spectrum, thus revealing its low-rank structure.

We visualize a network for each coefficient matrix, which only preserves the edges larger than a given threshold (following Kenett et al. 2010, Curme et al. 2015), as shown in figure 5. We color stocks according to the GICS sector division, and sort them by their market capitalization within each sector. $\ddagger$ As one can see from figure 5(a), the cross-impact coefficient matrix $[ \beta _ { i , j } ^ { [ 1 ] } ] _ { j \neq i }$ displays a sectorial

In figures 5(c,d), we set lower threshold values (75th, respectively, 25th percentile of coefficients) in order to promote more edges in the networks based on integrated OFIs. Interestingly, we observe only four connections in figure 5(c). Except from bidirectional links between GOOGL and GOOG, there exists a one-way link from Cigna (CI) to Anthem (ANTM), and another one-way link from Duke Energy (DUK) to NextEra Energy (NEE). Anthem announced to acquire Cigna in 2015. After a prolonged breakup, this merger finally failed in 2020. Therefore, it is unsurprising that the OFIs of Cigna can affect the price movements of Anthem. Conversely, Anthem’s OFIs also have an impact on the price movements of Cigna, but to a lesser extent. Further research should be undertaken to investigate this phenomenon. In terms of the second pair, Duke Energy rebuffed NextEra’s acquisition interest in 2020. Note that 2020 is not in our sample period. This finding hints that certain market participants may have noticed the special relationship between Duke Energy and NextEra Energy before this mega-merger was proposed.

3.2.2. Out-of-sample performance. Although the in-sample estimation yields interesting findings, practitioners are eventually concerned about the out-of-sample estimation. Therefore, we propose to perform the following out-of-sample tests.

![](images/11acb5e075d08cb25c4705880ca1c571653a22855f09007ece46a806f673ee8b.jpg)  
Figure 5. Illustrations of the coefficient networks constructed from contemporaneous cross-impact models. (a) Threshold $= 9 5 \mathrm { t h }$ percentile, based on best-level OFIs. (b) Threshold $= 9 5 \mathrm { t h }$ percentile, based on integrated OFIs. (c) Threshold $= 7 5 \mathrm { t h }$ percentile, based on integrated OFIs and (d) Threshold $= 2 5 \mathrm { t h }$ percentile, based on integrated OFIs.

Note: To render the networks more interpretable and for ease of visualization, we only plot the top $5 \%$ largest (a-b), or top $2 5 \%$ largest (c), or top $7 5 \%$ largest (d), in magnitude coefficients. The coefficients are averaged over each regression window between 2017–2019. Nodes are colored by the GICS structure and sorted by market capitalization. Green links represent positive values while black links represent negative values. The width of edges is proportional to the absolute values of their respective coefficients.

We use the above-fitted models to estimate returns on the following 30-minute data and compute the corresponding $R ^ { 2 }$ denoted as out-of-sample $R ^ { 2 }$ or $ { \mathbf { O S } } R ^ { 2 } .$ $^ \dagger$

Table 5 reports the average values and their standard deviations of out-of-sample $R ^ { 2 }$ of $\mathbf { P I } ^ { [ 1 ] }$ , $\mathbf { C I } ^ { [ 1 ] }$ , $\mathbf { P } \mathbf { I } ^ { I }$ , and $\mathbf { C I } ^ { I }$ . We first focus on the models using best-level OFIs. It appears $\mathbf { C I } ^ { [ 1 ] }$ has a slight advantage compared with $\mathbf { P I } ^ { [ 1 ] }$ for out-of-sample tests with an improvement of $1 . 3 9 \%$ $( = 6 6 . 0 3 \% - 6 4 . 6 4 \% )$ . However, when involving multi-level or integrated OFIs, the performance of $\mathbf { C } \mathbf { I } ^ { I }$ is slightly worse than $\mathbf { P } \mathbf { I } ^ { I }$ , indicating that the cross-impact model with integrated OFIs cannot provide extra explanatory power to the price impact model with integrated OFIs. Overall, we observe that the models using integrated OFIs unveil significant and consistent improvements over those using only best-level OFIs.

Table 5. Out-of-sample performance for contemporaneous returns.   

<table><tr><td rowspan="2"></td><td colspan="2">Best-level OFIs</td><td colspan="2">Integrated OFIs</td></tr><tr><td>P[1]</td><td>CI[1]</td><td>PI</td><td>CI</td></tr><tr><td>OS R²</td><td>64.64 (21.82)</td><td>66.03 (19.51)</td><td>83.83 (16.90)</td><td>83.62 (14.53)</td></tr></table>

Note: The table reports the mean values and standard deviations (in parentheses) of out-of-sample $R ^ { 2 }$ (in percentage points) of various models when modeling contemporaneous returns. The models include $\mathbf { P I } ^ { [ 1 ] }$ (equation (6)), $\mathbf { C } \hat { \mathbf { I } } ^ { [ 1 ] }$ (equation (8)), $\mathbf { P } \mathbf { I } ^ { I }$ (equation (7)), and $\mathbf { C I } ^ { I }$ (equation (9)). These statistics are averaged across each stock and each regression window.

In general, we observe strong evidence implying $\mathbf { C I } ^ { [ 1 ] }$ provides a better out-of-sample estimate than $\mathbf { P } \bar { [ 1 ] }$ , while for $\mathbf { P } \mathbf { I } ^ { I }$ and $\mathbf { C I } ^ { I }$ , the evidence is opposite. However, it is important to note that these conclusions are based on a point estimate and do not necessarily indicate statistical significance. Therefore, we perform the following hypothesis test for each stock on the out-of-sample data to assess statistical significance,

$$
\begin{array} { r l } & { \mathcal { H } _ { 0 } : \mathbb { E } \left[ R _ { \mathrm { O S } } ^ { 2 } \left( { \bf C I } ^ { [ 1 ] } \right) - R _ { \mathrm { O S } } ^ { 2 } \left( { \bf P I } ^ { [ 1 ] } \right) \right] \leq 0 \mathrm { v s } . } \\ & { \mathcal { H } _ { 1 } : \mathbb { E } \left[ R _ { \mathrm { O S } } ^ { 2 } \left( { \bf C I } ^ { [ 1 ] } \right) - R _ { \mathrm { O S } } ^ { 2 } \left( { \bf P I } ^ { [ 1 ] } \right) \right] > 0 . } \end{array}
$$

We employ the approach from Giacomini and White (2006) and Chinco et al. (2019) to assess statistical significance through a Wald-type test (see Ward and Ahlquist 2018). Theorem 1 in Giacomini and White (2006) implies that we can use a standard $t$ -test to evaluate the statistical significance of changes in $R ^ { 2 }$ . A $p$ -value less than a given significance level $\alpha$ rejects the null hypothesis in favor of the alternative at the $1 - \alpha$ confidence level, implying $\mathbf { C I } ^ { [ 1 ] }$ has significantly better estimation than $\mathbf { P I } ^ { [ 1 ] }$ . We also implement this test for the comparison between $\mathbf { P } \mathbf { I } ^ { I }$ and $\mathbf { C I } ^ { I }$ .

Figure 6 illustrates the main results from the above hypothesis tests. When using only the best-level OFIs, the crossimpact model is superior to the price impact model for $9 1 . 0 \%$ $( 9 4 . 4 \% )$ of stocks, at the $1 \%$ $( 5 \% )$ confidence level. However, when examining the models using integrated OFIs, we reject the null hypothesis (i.e. in favor of the cross-impact model) only for $2 8 . 1 \%$ $( 3 3 . 7 \% )$ of stocks at the $1 \%$ $( 5 \% )$ confidence level. As expected, cross-impact terms can significantly improve the explanatory power of the price impact model for GOOG and GOOGL.

Dynamics of limit order book may depend on the tickto-price ratio, or alternatively, the fraction of time that the bid-ask spread is equal to one tick for a given stock (Curato and Lillo 2015). We examine whether this dependence also extends to cross-asset OFIs. $^ \dagger$ Our findings, presented in table 6, suggest that cross-asset OFIs can better explain the price dynamics of stocks with a larger tick-to-price ratio.

Table 6. Out-of-sample $R ^ { 2 }$ of various contemporaneous models sorted by tick-to-price ratio.   

<table><tr><td></td><td>[0%,25%)</td><td>[25%,50%)</td><td>[50%,75%)</td><td>[75%,100%]</td></tr><tr><td>P[1]</td><td>44.38</td><td>62.51</td><td>77.55</td><td>70.70</td></tr><tr><td>CI[1]</td><td>53.01</td><td>66.32</td><td>72.50</td><td>78.34</td></tr><tr><td>PI</td><td>68.14</td><td>84.58</td><td>88.14</td><td>89.86</td></tr><tr><td>CI</td><td>72.01</td><td>84.80</td><td>88.51</td><td>91.01</td></tr></table>

Note: $[ 0 \% , 2 5 \% )$ , respectively $[ 7 5 \%$ , $100 \% ]$ , denote the subset of stocks with the lowest, respectively highest, $2 5 \%$ values according to the tick-to-price ratio.

# 3.3. Discussion about contemporaneous cross-impact

3.3.1. Impact on stocks. In summary, our previous results mainly show that when considering only the best-level OFI of a single stock, the addition of the best-level OFI from other stocks slightly increases the explanatory power. On the other hand, when the information from multiple levels is integrated into the OFI, the improvement is negligible. In the meantime, it is unsurprising that taking into account more levels in the LOB $( \mathbf { P } \mathbf { I } ^ { I } )$ could better explain price changes, compared to only considering best-level orders $( \mathbf { P I } ^ { [ 1 ] } )$ .

After observing these results, several natural questions may arise: How can the above facts be reconciled $\mathrm { ? \ddag }$ How do the cross-asset best-level OFIs interact with the multi-level OFIs, when modeling contemporaneous returns?

To address these questions, we consider the following scenario, also depicted in figure 7. For simplicity, we denote the order from trading strategy $A$ on stock $i$ (respectively, $j )$ as $A _ { i }$ (respectively, $A _ { j }$ ). Analogously, we define orders from strategy $B$ and $s$ . Let us next consider the orders of stock i. There are three orders from different portfolios, given by $A _ { i }$ , $B _ { i }$ and $S _ { i } . A _ { i }$ is at the third bid level of stock $i$ and linked to an order at the best ask level of stock $j$ , i.e. $A _ { j }$ . Also, $B _ { i }$ is at the best ask level of stock $i$ and linked to an order at the best bid level of stock $j .$ , i.e. $B _ { j }$ . Finally, $S _ { i }$ is an individual bid order at the best level of stock $i$ .

We observe that the best-level limit orders from stock $j$ may be linked to price movements of stock $i$ through paths $B _ { j } $ $B _ { i }  \mathrm { o f } _ { i } ^ { 1 }  r _ { i }$ and $A _ { j }  A _ { i }  \mathrm { o f } _ { i } ^ { 3 }  r _ { i }$ . Thus the price impact model for stock $i$ which only utilizes its own best-level orders will ignore the information of $A _ { i }$ , while the crossimpact model can partially collect it along the path $A _ { j } \to A _ { i }$ . This might illustrate why the best-level OFIs of multiple assets can provide slightly additional explanatory power to the price impact model using only the best-level OFIs.

![](images/f97e264c854064d50b2053216c548c3c3330e5eea4849c00c2c218c70f2d2af6.jpg)  
Figure 6. Mean differences of out-of-sample $R ^ { 2 }$ between CI and PI models. (a) $R _ { \mathrm { O S } } ^ { 2 } ( { \bf C I } ^ { [ 1 ] } ) - R _ { \mathrm { O S } } ^ { 2 } ( { \bf P I } ^ { [ 1 ] } )$ and (b) $R _ { \mathrm { O S } } ^ { 2 } ( { \bf C I } ^ { I } ) - R _ { \mathrm { O S } } ^ { 2 } ( { \bf P I } ^ { I } )$ . Note: A positive (negative) number indicates superiority for the CI (PI) model. The $y$ -axis represents the average difference of $\mathrm { O S } R ^ { 2 }$ between CI and PI, while the $x \cdot$ -axis lists the stock symbols. Stars indicate the $p$ -values, with orange, green, and blue representing significance at the $1 \%$ , $5 \%$ , and $10 \%$ levels, respectively.

Nonetheless, if we can integrate multi-level OFIs in an efficient way (in our example, aggregate order imbalances caused by orders $A _ { i }$ , $B _ { i }$ and $S _ { i }$ ), then there is no need to consider OFIs from other stocks for modeling price dynamics. For example, information hidden in the path $A _ { j }  \stackrel { \cdot } { A } _ { i }  \mathrm { o f f } _ { i } ^ { 3 }  r _ { i }$ can be leveraged as long as $A _ { i }$ is well absorbed into new integrated OFIs. In this sense, for stock $i$ , cross-asset best-level OFIs (including $A _ { j }$ ) are surrogates of its own OFIs at different levels (here $A _ { i }$ ), to a certain extent. The likelihood of this relationship is attributed to massive portfolio trades that submit or cancel limit orders across a variety of assets at different levels.† We put forward this mechanism which potentially explains why the cross-impact model with integrated OFIs cannot provide additional explanatory power compared to the price impact model with integrated OFIs. $^ { \ddagger }$

3.3.2. Impact on portfolios. A related question is about the aggregation of cross-impact at the portfolio level. Let us consider the OFI of portfolio $p$ as $\begin{array} { r } { \mathrm { o f f } _ { p , t } ^ { \hat { 1 } , h } : = \sum _ { i = 1 } ^ { N } w _ { i } \mathrm { o f f } _ { i , t } ^ { 1 , h } } \end{array}$ , where $w _ { i }$ is the weight of asset $i$ in a portfolio. Then the price impact for portfolio $p$ is

$$
r _ { p , t } ^ { ( h ) } = a _ { p } ^ { [ 1 ] } + \beta _ { p } ^ { [ 1 ] } \mathrm { o f } _ { p , t } ^ { 1 , h } + e _ { p , t } ^ { [ 1 ] }
$$

![](images/cb7b033c76a9c89428bf946aaa351c77bfaf7b8e5c505b85a4d3bc3c92cf1802.jpg)  
Figure 7. Illustration of the cross-impact model. Note: The orders at different levels of each stock may come from single-asset and multi-asset trading strategies. The returns of stock $i$ are potentially influenced by orders of stock $j$ through the connections $B _ { j }  B _ { i }  \mathrm { o f f } _ { i } ^ { 1 }  r _ { i }$ and $A _ { j }  \bar { A _ { i } }  \mathrm { o f f } _ { i } ^ { 3 }  r _ { i }$ . Information along the path $A _ { j }  A _ { i }  \mathrm { o f f } _ { i } ^ { 3 }  r _ { i }$ can be collected by the price impact model with integrated OFIs but not by the price impact model with only best-level OFIs.

$$
= a _ { p } ^ { [ 1 ] } + \beta _ { p } ^ { [ 1 ] } \sum _ { i = 1 } ^ { N } w _ { i } \mathrm { o f } _ { i , t } ^ { 1 , h } + e _ { p , t } ^ { [ 1 ] } ,
$$

where $a _ { p } ^ { [ 1 ] }$ is the intercept and $e _ { p , t } ^ { [ 1 ] }$ is the noise term.

On the other hand, the cross-impact for portfolio $p$ is

$$
\begin{array} { l } { { \displaystyle r _ { p , t } ^ { ( h ) } = \sum _ { i = 1 } ^ { N } w _ { i } r _ { p , t } ^ { ( h ) } } } \\ { ~ = \sum _ { i = 1 } ^ { N } w _ { i } \left( \alpha _ { i } ^ { [ 1 ] } + \beta _ { i } ^ { [ 1 ] } \mathsf { o f } _ { i , t } ^ { 1 , h } + \epsilon _ { i , t } ^ { [ 1 ] } \right) }  \\ { { \displaystyle ~ = \alpha _ { p } ^ { [ 1 ] } + \sum _ { i = 1 } ^ { N } \beta _ { i } ^ { [ 1 ] } w _ { i } \mathsf { o f } _ { i , t } ^ { 1 , h } + \epsilon _ { p , t } ^ { [ 1 ] } } , }  \end{array}
$$

where $\begin{array} { r } { \alpha _ { p } ^ { [ 1 ] } = \sum _ { i = 1 } ^ { N } w _ { i } \alpha _ { i } ^ { [ 1 ] } } \end{array}$ is the intercept.

Comparing (10) and (11) shows that cross-impact at portfolio level depends on the angles of $\vec { \beta } = ( \beta _ { 1 } ^ { [ 1 ] } , \cdot \cdot \cdot , \beta _ { N } ^ { [ 1 ] } )$ and $\vec { w } = ( w _ { 1 } , \ldots , w _ { N } )$ . On one hand, if individual assets exhibit a universal pattern of price dynamics, i.e. $\beta _ { i } ^ { [ 1 ] } \approx \beta _ { j } ^ { [ 1 ] } , \forall i \neq j ,$ we may conclude that the portfolio return is driven by its own order flows, to a large extent. On the other hand, if the portfolio places most of the weight on a specific stock or a set of stocks with similar patterns of price dynamics, then the portfolio returns are also driven by its own order flows, rather than by cross-impact. In other scenarios, it is necessary to consider the cross-impact. One can perform an analogous analysis for models with integrated OFIs.

In fact, the mechanism depicted in figure 7 aligns with this analysis. For example, orders placed on stock $i$ represented by $A _ { i }$ and $S _ { i }$ have the potential to affect the price movements of that stock, which subsequently affect the returns of portfolio $B$ , even if $A _ { i }$ and $S _ { i }$ have no direct association with $B$ . Additionally, $A _ { j }$ may have a different influence on the performance of $B$ because of the (potentially) different price dynamics of stock $j$ . Therefore, it may be necessary to take into account cross-asset OFIs when developing models for portfolio returns.

To examine the potential of cross-asset OFIs in explaining portfolio returns, we choose two widely-used portfolio construction methods: the equal-weighted portfolio (EW), and the eigenportfolio $^ \dagger$ using the 1st principal component (Eig1). Table 7 summarizes the out-of-sample $R ^ { 2 }$ of various models on these two portfolios. As table 7 shows, there is a significant difference between $\mathbf { P I } ^ { [ 1 ] }$ and $\mathbf { C I } ^ { [ 1 ] }$ , $\mathbf { P } \mathbf { I } ^ { I }$ and $\mathbf { C I } ^ { I }$ , indicating that cross-impact is a crucial factor when modeling portfolio returns. Again, the models using the integrated OFIs outperform their counterparts using the best-level OFIs.

# 4. Forecasting future returns

In the previous section, the definitions of price impact and cross-impact are based on contemporaneous OFIs and returns, meaning that both quantities pertain to the same bucket of time. In this section, we extend the above studies to future returns, and probe into the forward-looking price impact and cross-impact models.

Table 7. Out-of-sample performance on portfolio returns.   

<table><tr><td></td><td colspan="2">Best-level OFIs</td><td colspan="2">Integrated OFIs</td></tr><tr><td></td><td>PI[]</td><td>CI[1]</td><td>PII</td><td>CI</td></tr><tr><td>EW</td><td>79.29</td><td>81.03</td><td>85.26</td><td>87.97</td></tr><tr><td></td><td>(6.24)</td><td>(6.16)</td><td>(3.54)</td><td>(2.78)</td></tr><tr><td>Eig1</td><td>80.73</td><td>81.95</td><td>84.69</td><td>87.70</td></tr><tr><td></td><td>(6.75)</td><td>(6.36)</td><td>(3.70)</td><td>(2.84)</td></tr></table>

Note: EW: equal-weighted portfolio. Eig1: eigenportfolio using the 1st principal component of multi-asset returns.

# 4.1. Predictive models

We first propose the following forward-looking price impact and cross-impact models, denoted as $\mathbf { F P I } ^ { [ 1 ] }$ , $\mathbf { F P \bar { F } } ^ { I }$ , FCI[1], and $\mathbf { F C I } ^ { I }$ , respectively. FPI[1] $( \mathbf { F P I } ^ { I } )$ uses the lagged best-level (integrated) OFIs of stock $i$ to predict its own future return $r _ { i , t + f } ^ { ( f ) }$ during $( t , t + f ]$ , while $\mathbf { F C I } ^ { [ 1 ] } ( \mathbf { F C I } ^ { I } )$ involves the lagged multi-asset best-level (integrated) OFIs. We employ OLS to fit the forward-looking price impact models and LASSO to fit the cross-impact models.

$$
\begin{array} { l } { { \displaystyle { \bf F } { \bf P } { \bf \Pi } ^ { [ 1 ] } : \quad r _ { i , i + f } ^ { ( f ) } = \alpha _ { i } ^ { [ 1 ] } + \sum _ { k \in { \cal L } } \beta _ { i } ^ { [ 1 ] , k } \mathrm { o f } _ { i , t } ^ { [ \delta h ] } + \epsilon _ { i , i + f } ^ { [ 1 ] } , } } \\ { { \displaystyle { \bf F } { \bf C } { \bf I } ^ { [ 1 ] } : \quad r _ { i , i + f } ^ { ( f ) } = \alpha _ { i } ^ { [ 1 ] } + \sum _ { j = 1 } ^ { N } \sum _ { k \in { \cal L } } \beta _ { i j } ^ { [ 1 ] , k } \mathrm { o f } _ { i , t } ^ { [ \delta h ] } + \eta _ { i , i + f } ^ { [ 1 ] } , } } \\ { { \displaystyle { \bf F } { \bf P } { \bf I } ^ { I } : \quad r _ { i , i + f } ^ { ( f ) } = \alpha _ { i } ^ { I } + \sum _ { k \in { \cal L } } \beta _ { i } ^ { I , k } \mathrm { o f } _ { i , t } ^ { I , ( k ) } + \epsilon _ { i , i + f } ^ { I } , } } \\ { { \displaystyle { \bf F } { \bf C } { \bf I } ^ { I } : \quad r _ { i , i + f } ^ { ( f ) } = \alpha _ { i } ^ { I } + \sum _ { j = 1 } ^ { N } \sum _ { k \in { \cal L } } \beta _ { i , i } ^ { I , k } \mathrm { o f } _ { i , t } ^ { I , ( k ) } + \eta _ { i + f } ^ { I } , } } \end{array}
$$

where $f$ is the forecasting horizon of future returns and $L =$ {1, 2, 3, 5, 10, 20, 30} represents the set of lags.

Furthermore, we compare OFI-based models with returnbased models studied in previous works, where lagged returns are involved as predictors. AR (equation (16)) is an autoregressive (AR) model using various returns over different time horizons, inspired by Corsi (2009), Ait-Sahalia et al. (2022). CAR (Chinco et al. 2019) uses the entire cross-section lagged returns as candidate predictors, as detailed in equation (17). We employ OLS to fit the ARs and LASSO to fit the CARs.

$$
\begin{array} { r l } { \mathbf { A } \mathbf { R } : } & { r _ { i , t + f } ^ { ( f ) } = \alpha _ { i } + \displaystyle \sum _ { k \in L } \beta _ { i } ^ { r , k } r _ { i , t } ^ { ( k h ) } + \epsilon _ { i , t + f } } \\ { \mathbf { C } \mathbf { A } \mathbf { R } : } & { r _ { i , t + f } ^ { ( f ) } = \alpha _ { i } + \displaystyle \sum _ { j = 1 } ^ { N } \sum _ { k \in L } \beta _ { i , j } ^ { r , k } r _ { i , t } ^ { ( k h ) } + \eta _ { i , t + f } } \end{array}
$$

# 4.2. Empirical results

In this experiment, observations associated with returns and OFIs are computed minutely, i.e. $h = 1$ minute.† Following Chinco et al. (2019), we use data from the previous 30 minutes to estimate the model parameters and apply the fitted model to forecast future $f$ -minute returns. We then move one minute forward and repeat this procedure to compute the rolling $f$ -minute-ahead return forecasts. For all models, we initially focus on the 1-minute forecasting horizon. In Section 4.3, we consider return forecasts over longer horizons, including $f \in \{ 2 , 3 , 5 , 1 0 , 2 0 , 3 0 \}$ minutes, to assess the strength and duration of price impact and cross-impact.

Table 8. Out-of-sample performance for one-minute-ahead returns.   
Note: The table reports the mean values and standard deviations (in parentheses) of out-of-sample $R ^ { 2 }$ of various models when modeling one-minute-ahead returns. The predictive models include FPI[1] (equation (12)), $\mathbf { F C I } ^ { [ 1 ] }$ (equation (13)), $\mathbf { F P I } ^ { I }$ (equation (14)), $\mathbf { F C I } ^ { I }$ (equation (15)), AR (equation (16)) and CAR (equation (17)). These statistics are averaged across each stock and each regression window.   

<table><tr><td></td><td colspan="2">Best-level OFIs</td><td colspan="2">Integrated OFIs</td><td colspan="2">Returns</td></tr><tr><td></td><td>FPI[1]</td><td>FCI[]</td><td>FPI!</td><td>FCI!</td><td>AR</td><td>CAR</td></tr><tr><td>OS R²</td><td>0.37 (0.10)</td><td>-0.10 (0.05)</td><td>-0.36 (0.08)</td><td>-0.10 (0.05)</td><td>0.36 (0.11)</td><td>-0.10 (0.05)</td></tr></table>

Following the analysis of Cartea et al. (2018) and Chinco et al. (2019), we demonstrate the effectiveness of the forwardlooking price impact and cross-impact models from two perspectives: (1) statistical performance, and (2) economic gain.

4.2.1. Statistical performance. Table 8 summarizes the outof-sample $R ^ { 2 }$ values of the aforementioned predictive models when predicting the subsequent 1-minute returns, i.e. $f = 1$ . It appears the cross-impact models $\mathbf { F C I } ^ { [ 1 ] }$ (respectively, $\mathbf { F C I } ^ { I }$ , CAR) achieve higher out-of-sample $R ^ { 2 }$ statistics compared to the price impact models $\mathbf { F P I } ^ { [ 1 ] }$ (respectively, $\mathbf { F P I } ^ { I }$ , AR). We also implement the same hypothesis test described in Section 3 to investigate the statistical significance (unreported) of these results. We observe that the cross-impact models exhibit significantly superior performance than the price impact models across all stocks, at the $1 \%$ confidence level.

Most of the empirical literature in return prediction focuses its evaluations on out-of-sample $R ^ { 2 }$ . However, we remark that negative $R ^ { 2 }$ values do not imply that the forecasts are economically meaningless (see more discussions in Choi et al. 2022, Kelly et al. 2022). $\ddagger$ To emphasize this point, we will incorporate these return forecasts into a forecastbased trading strategy, and showcase their profitability in the following subsection.

![](images/e6a092de8831e9f9decf7fb89a0ae0a58281073fc9c33fa3c27c3bbcd473d047.jpg)  
Figure 8. Network structure of the coefficient matrix constructed from forward-looking cross-impact models. (a) Based on best-level OFIs. (b) Based on integrated OFIs and (c) Based on returns.   
Note: The coefficients are averaged over 2017–2019. To render the networks more interpretable and for ease of visualization, we only plot the top $5 \%$ largest in magnitude coefficients. Nodes are colored by the GICS structure and sorted by market capitalization. Green links represent positive values while black links represent negative values. The width of edges is proportional to the absolute values of their respective coefficients.

Table 9. Group degree centrality for each GICS sector.   

<table><tr><td></td><td colspan="3">Group In-degree Centrality</td><td colspan="3">Group Out-degree Centrality</td></tr><tr><td></td><td>Best-level OFIs</td><td>Integrated OOFIs</td><td>Returns</td><td>Best-level OFIs</td><td>Integrated OOFIs</td><td>Returns</td></tr><tr><td>Information Technology</td><td>0.12</td><td>0.36</td><td>0.26</td><td>0.46</td><td>0.62</td><td>0.59</td></tr><tr><td>Communication Services</td><td>0.06</td><td>0.24</td><td>0.20</td><td>0.85</td><td>0.74</td><td>0.60</td></tr><tr><td>Consumer Discretionary</td><td>0.09</td><td>0.20</td><td>0.15</td><td>0.86</td><td>0.51</td><td>0.17</td></tr><tr><td>Consumer Staples</td><td>0.03</td><td>0.15</td><td>0.09</td><td>0.00</td><td>0.11</td><td>0.01</td></tr><tr><td>Health Care</td><td>0.10</td><td>0.37</td><td>0.19</td><td>0.12</td><td>0.22</td><td>0.59</td></tr><tr><td>Financials</td><td>0.12</td><td>0.21</td><td>0.17</td><td>0.03</td><td>0.41</td><td>0.08</td></tr><tr><td>Industrials</td><td>0.10</td><td>0.19</td><td>0.20</td><td>0.00</td><td>0.39</td><td>0.27</td></tr><tr><td>Utilities</td><td>0.00</td><td>0.07</td><td>0.04</td><td>0.00</td><td>0.06</td><td>0.00</td></tr><tr><td>Energy</td><td>0.06</td><td>0.07</td><td>0.04</td><td>0.00</td><td>0.14</td><td>0.00</td></tr><tr><td>Real Estate</td><td>0.00</td><td>0.05</td><td>0.02</td><td>0.00</td><td>0.00</td><td>0.01</td></tr></table>

Note: According to the threshold networks as shown in figure 8, we compute the fraction of stocks outside of a specific sector connected to stocks in this specific sector. The color of each sector in this table corresponds to the color in figure 8.

Considering the different magnitudes of the OFIs and returns, we first normalize the coefficient matrix of each model by dividing by the average of the absolute coefficients. Figure A2 (deferred to Appendix 5) shows the average coefficient matrices of $\mathbf { F C I } ^ { [ 1 ] }$ , $\mathbf { \widetilde { F } C I } ^ { I }$ , and CAR. For example, as revealed in figure A2(a) $( \mathbf { F C I } ^ { [ 1 ] } )$ , for a specific stock, the main influence comes from its own OFI, i.e. the absolute values of diagonal elements are significantly larger than the off-diagonal ones. We observe that cross-impact is often negative, consistent with Pasquariello and Vega (2015). Except for the self-impact, most stocks are also influenced by stocks in Communication Services, Consumer Discretionary and Information Technology.

To better illustrate the interactions between different stocks, we construct a network for each normalized coefficient matrix and only preserve the cross-asset edges (i.e. off-diagonal elements) larger than the 95th percentile of coefficients. Figure 8 illustrates some of the main characteristics of the coefficient networks for $\mathbf { F C I } ^ { [ 1 ] }$ , $\mathbf { F C I } ^ { I }$ , and CAR. For example, we again observe that there are more edges from Communication Services, Consumer Discretionary and Information Technology, indicating they may contain more predictive power for others.

To gain a better understanding of the structural properties of the resulting network, we aggregate node centrality measures (see Everett and Borgatti 1999) at the sector level, and also perform a spectral analysis of the adjacency matrix. From table 9, we observe that the out-degree centrality of Communication Services, Consumer Discretionary and Information Technology is significantly larger than that of others, consistent with previous findings. Figure 8(c) also shows that the network based on returns contains more inner-sector connections than the other two counterparts, thus implying a sectorial structure. Table 10 presents the top five most significant stocks in terms of out-degree centrality in each network, which exhibit more impact on the prices of other stocks.

Figure 9 shows a barplot with the average value for the top 20 largest singular values of the network adjacency matrix, for best-level OFIs, integrated OFIs, and returns, where the average is performed over all constructed networks. For ease of visualization and comparison, we first normalize the adjacency matrix before computing the top singular values, which exhibit a fast decay. In addition to the significantly large top singular value revealing that the networks have a strong rank-1 structure, the next 6-8 singular values are likely to correspond to the more prominent industry sectors.

Table 10. Top 5 stocks according to node out-degree centrality in threshold networks.   

<table><tr><td>Best-level OFIs</td><td>Integrated OFIs</td><td>Returns</td></tr><tr><td>AMZN</td><td>NFLX</td><td>NVDA</td></tr><tr><td>GOOG</td><td>AMZN</td><td>NFLX</td></tr><tr><td>GOOGL</td><td>NVDA</td><td>ISRG</td></tr><tr><td>NVDA</td><td>GS</td><td>AVGO</td></tr><tr><td>NFLX</td><td>FB</td><td>GE</td></tr></table>

Note: The out-degree centrality for a node is the fraction of nodes its outgoing edges are connected to.

4.2.2. Economic gains. On the basis of return forecasts, we employ a portfolio construction method, proposed by Chinco et al. (2019), to evaluate the economic gains of the aforementioned predictive models.

Forecast-implied portfolio. For a specific forecasting model $F$ , the motivations of portfolio construction can be summarized as follows.

• It only executes an order when the one-minuteahead return forecast exceeds the bid-ask spread. • It buys/sells more shares of the ith stock when the absolute value of one-minute-ahead return forecast for ith stock is higher. It buys/sells more shares of the ith stock when the one-minute-ahead return forecasts for the ith stock tend to be less volatile throughout the trading day.

This strategy allocates a fraction $w _ { i , t }$ of its capital to the ith stock

$$
w _ { i , t } \stackrel { \mathrm { d e f } } { = } \frac { 1 _ { \left\{ \left| f _ { i , t } ^ { F } \right| > s p r d _ { i , t } \right\} } \cdot f _ { i , t } ^ { F } / \sigma _ { i , t } ^ { F } } { \sum _ { n = 1 } ^ { N } 1 _ { \left\{ \left| f _ { n , t } ^ { F } \right| > s p r d _ { n , t } \right\} } \cdot \left| f _ { n , t } ^ { F } \right| / \sigma _ { n , t } ^ { F } } ,
$$

where $f _ { i , t } ^ { F }$ represents the one-minute-ahead return forecast according to model $F$ for minute $( t + 1 ) , s p r d _ { i , t }$ represents the relative bid-ask spread at time $t$ , $\sigma _ { i , t } ^ { F }$ represents the standard deviation of the model’s one-minute-ahead return forecasts for the ith stock during the previous 30 minutes of trading, i.e. the standard deviation of in-sample fits. The denominator is the total investment so that the strategy is self-financed. If there are no stocks with forecasts that exceed the spread in a given minute, then we set $w _ { i , t } = 0 , \forall i$ .

Table 11. Economic performance of the forecast-implied trading strategy.   
Note: The table reports the mean values and standard deviations (in parentheses) of annualized PnLs of forecast-implied trading strategy of various models for forecasting one-minute-ahead returns. The predictive models include $\mathbf { \nabla } \mathbf { F P I } ^ { [ 1 ] }$ (equation (12)), FCI[1] (equation (13)), $\mathbf { F P I } ^ { I }$ (equation (14)), $\mathbf { F C } \mathbf { \dot { I } } ^ { \dot { I } }$ (equation (15)), AR (equation (16)) and CAR (equation (17)). These statistics are averaged over 2017-2019.   

<table><tr><td rowspan="2"></td><td colspan="2">Best-level OFIs</td><td colspan="2">Integrated OFIs</td><td colspan="2">Returns</td></tr><tr><td>FPI[1]</td><td>FCI[1]</td><td>FPI!</td><td>FCI</td><td>AR</td><td>CAR</td></tr><tr><td>PnL</td><td>0.21 (0.12)</td><td>0.43 (0.17)</td><td>0.23 (0.13)</td><td>0.39 (0.19)</td><td>0.23 (0.13)</td><td>0.40 (0.18)</td></tr></table>

Finally, we compute the profit and loss $( P n L )$ of the resulting portfolios on each trading day by summing the strategy’s minutely returns as in Chinco et al. (2019).

Table 11 compares the performance (annualized PnL) of the forecast-implied strategies, based on forecast returns from various predictive models. It is worth noting that in the following analysis, the strategy ignores trading costs, as this is not the focus of our paper. Table 11 shows that portfolios based on forecasts of the forward-looking cross-impact model outperform those based on forecasts of the forward-looking price impact model.

# 4.3. Longer forecasting horizons

One-minute-ahead return forecasts are not the only time horizon of interest to practitioners and academics. Additionally, we evaluate the performance of the above models and examine the forecasting ability of cross-impact terms over longer prediction horizons.

Figure 10 illustrates the model predictability from the perspective of raw annualized PnL across multiple horizons. $^ \dagger$ Due to the similar performance of $\mathbf { F P I } ^ { [ 1 ] }$ and $\mathbf { F P I } ^ { I }$ (respectively, $\mathbf { F C I } ^ { [ 1 ] }$ and $\mathbf { F C I } ^ { \bar { I } }$ ) over longer horizons, we only show the curves of $\mathbf { F P I } ^ { [ 1 ] }$ , $\mathbf { F C I } ^ { I }$ , AR, CAR, and a benchmark (S&P100 ETF). It appears that superior forecasting ability arises from cross-asset terms at short horizons. However, the PnL of cross-asset models declines more quickly over longer horizons. A further study with more focus on the reasons for the predictability of cross-asset OFIs over multiple horizons is therefore suggested. Finally, the models in which each stock only relies on its own returns/OFIs marginally outperform their counterparts which use the entire cross-sectional predictors.

![](images/a576d4e17cb67d187571245f0191331ca497389ad3e4760bc2c4f7aab55536eb.jpg)  
Figure 9. Barplot of normalized singular values for the average coefficient matrix in forward-looking cross-impact models. Note: We perform Singular Value Decomposition (SVD) on the coefficient matrix and obtain the singular values. The $x$ -axis represents the singular value rank, and the $y$ -axis shows the normalized singular values. The coefficients are averaged over 2017–2019.

![](images/02fa6e023f1f93162d8a18ddf46dc9ba89208454d3752ecdccd78173cefad686.jpg)  
Figure 10. Annualized $\mathrm { P n L }$ as a function of the forecasting horizon. Note: The $x$ -axis represents the prediction horizon (in minutes), while the $y$ -axis represents the annualized $\mathrm { P n L }$ . The grey horizontal line is the performance of the S&P100 ETF index.

# 4.4. Discussion about predictive cross-impact

Tables 8 and 11 reveal that, in contrast to the price impact model, multi-asset OFIs can provide considerably more additional explanatory power for future returns compared to contemporaneous returns. A possible explanation for this asymmetric phenomenon is that there exists a time lag between when the OFIs of a given stock are formed (a so-called flow formation period) and the actual time when traders notice this change of flow and incorporate it into their trading model (see Buccheri et al. 2021). $\ddagger$ For example, assume a trader submitted an unexpectedly large amount of buy limit orders of Apple (AAPL) at $1 0 { : } 0 0 \mathrm { a m }$ , at either the first level or potentially deeper in the book. Other traders may notice this anomaly and adjust their portfolios (including Apple) at a later time, for example, 10:01 am. In this case, the OFIs of Apple may indicate future price changes of other stocks.

Consistent with our explanation, Hou (2007) argued that the gradual diffusion of industry information is a leading cause of the lead-lag effect in stock returns. Cohen and Frazzini (2008) found that certain stock prices do not promptly incorporate news pertaining to economically related firms, due to the presence of investors subject to attention constraints. Further research should be undertaken to investigate the origins of the success of multi-asset OFIs in predicting future returns.

It is also interesting to note that forward-looking models using integrated OFIs cannot significantly outperform models using the best-level OFIs. This phenomenon might stem from the fact that the integrated OFIs do not explicitly take into account the level information (distance of a given level to the best bid/ask) of multi-level OFIs, and are agnostic to different sizes resting at different levels on the bid and ask sides of the book. Previous studies (such as Hasbrouck and Saar 2002, Cao et al. 2009, Cenesizoglu et al. 2022) demonstrated that traders might strategically choose to place their orders in different levels of the book depending on various factors, therefore limit orders at different price levels may contain different information content with respect to predicting future returns. A further study with more focus on the impact of multi-level OFIs over different time horizons is suggested.

# 5. Conclusion

We have systematically examined the impact of OFIs from multiple perspectives. The main contributions can be summarized as follows.

First, we verify the effects of multi-level and cross-asset OFIs on contemporaneous price dynamics. We introduce a new procedure to examine the cross-impact on contemporaneous returns. Under the sparsity assumption of cross-impact coefficients, we use LASSO to describe such a structure and compare the performances with the price impact model which only utilizes a stock’s own OFIs. We implement models with the best-level OFIs and integrated OFIs, respectively. The results first demonstrate that our integrated OFIs provide higher explanatory power for price movements than the widely-used best-level OFIs. More interestingly, in comparison with the price impact model using best-level OFIs, the cross-impact model exhibits additional explanatory power. However, the cross-impact model with integrated OFIs cannot provide extra explanatory power to the price impact model with integrated OFIs, indicating the effectiveness of our integrated OFIs.

In addition, we apply the price impact and cross-impact models to the challenging task of predicting future returns. The results reveal that involving cross-asset OFIs can increase out-of-sample $R ^ { 2 }$ . We subsequently demonstrate that this increase in out-of-sample $R ^ { 2 }$ leads to additional economic profits, when incorporated in common trading strategies, thus providing evidence of cross-impact over short future horizons. We also find that predictability of cross-impact terms vanishes quickly over longer horizons.

Future research directions. There are a number of interesting avenues to explore in future research. One such direction pertains to the assessment of whether cross-asset multi-level OFIs can improve the forecast of future returns (in the present work, we only considered the best-level OFI and integrated OFI due to limited computing power). Another interesting direction pertains to performing a similar analysis as in the present paper, but for the last 15–30 minutes of the trading day, where a significant fraction of the total daily trading volume occurs. For example, for the first few months of 2020 in the US equity market, about $23 \%$ of trading volume in the 3000 largest stocks by market value has taken place after $3 { : } 3 0 \mathrm { p m }$ , compared with about $4 \%$ from 12: pm to $1 \mathrm { p m }$ (Banerji 2020). It would be an interesting study to explore the interplay between the OFI dynamics and this surge of trading activity at the end of U.S. market hours.

# Acknowledgments

We thank Robert Engle, Álvaro Cartea, Slavi Marinov and seminar participants at the 11th Bachelier World Congress 2022; the 14th Annual SoFiE Conference; University of Oxford for helpful comments. We also thank the Oxford Suzhou Centre for Advanced Research for providing the computational facilities. Earlier versions of this paper circulated under the titles ‘Price Impact of Order Flow Imbalance: Multi-level, Cross-asset and Forecasting’ and ‘Cross-Impact of Order Flow Imbalance: Contemporaneous and Predictive’. First draft: December 2021.

# Disclosure statement

No potential conflict of interest was reported by the author(s).

# Funding

Chao Zhang has been supported by the Clarendon Fund Scholarship and G-Research PhD prize in Maths and Data Science.

# ORCID

Rama Cont $\textcircled{1}$ http://orcid.org/0000-0003-1164-6053 Mihai Cucuringu $\textcircled{1}$ http://orcid.org/0000-0002-8464-2152 Chao Zhang $\textcircled{1}$ http://orcid.org/0000-0002-0013-336X

# References

Ahn, H.-J., Bae, K.-H. and Chan, K., Limit orders, depth, and volatility: Evidence from the stock exchange of Hong Kong. J. Finance, 2001, 56(2), 767–788.   
Ait-Sahalia, Y., Fan, J., Xue, L. and Zhou, Y., How and when are high-frequency stock returns predictable? Available at SSRN 4095405, 2022.   
Avellaneda, M. and Lee, J.-H., Statistical arbitrage in the US equities market. Quant. Finance, 2010, 10(7), 761–782.   
Banerji, G., The 30 minutes that can make or break the trading day. 2020. https://www.wsj.com/articles/the-30-minutes-that-can-mak e-or-break-the-trading-day-11583886131?reflink=desktopwebsha re_permalink.   
Benzaquen, M., Mastromatteo, I., Eisler, Z. and Bouchaud, J.-P., Dissecting cross-impact on stock markets: An empirical analysis. J. Stat. Mech. Theory Exp., 2017, 2017(2), 023406.   
Buccheri, G., Corsi, F. and Peluso, S., High-frequency lead-lag effects and cross-asset linkages: A multi-asset lagged adjustment model. J. Bus. Econ. Stat., 2021, 39(3), 605–621.   
Cao, C., Hansch, O. and Wang, X., The information content of an open limit-order book. J. Futures Markets, 2009, 29(1), 16–41.   
Capponi, F. and Cont, R., Multi-asset market impact and order flow commonality. SSRN. 2020. https://papers.ssrn.com/sol3/papers.cf m?abstract_id=3706390.   
Cartea, Á., Donnelly, R. and Jaimungal, S., Enhancing trading strategies with order book signals. Appl. Math. Finance, 2018, 25(1), 1–35.   
Cenesizoglu, T., Dionne, G. and Zhou, X., Asymmetric effects of the limit order book on price dynamics. J. Empir. Finance, 2022, 65, 77–98.   
Chakrabarty, B., Hendershott, T., Nawn, S. and Pascual, R., Order exposure in high frequency markets. 2022. Available at SSRN 3074049.   
Chinco, A., Clark-Joseph, A.D. and Ye, M., Sparse signals in the cross-section of returns. J. Finance, 2019, 74(1), 449–492.   
Choi, D., Jiang, W. and Zhang, C., Alpha go everywhere: Machine learning and international stock returns. 2022. Available at SSRN 3489679.   
Chordia, T. and Subrahmanyam, A., Order imbalance and individual stock returns: Theory and evidence. J. Financ. Econ., 2004, 72(3), 485–518.   
Chordia, T., Roll, R. and Subrahmanyam, A., Order imbalance, liquidity, and market returns. J. Financ. Econ., 2002, 65(1), 111–130.   
Cohen, L. and Frazzini, A., Economic links and predictable returns. J. Finance, 2008, 63(4), 1977–2011.   
Cont, R., Kukanov, A. and Stoikov, S., The price impact of order book events. J. Financ. Econom., 2014, 12(1), 47–88.   
Corsi, F., A simple approximate long-memory model of realized volatility. J. Financ. Econom., 2009, 7(2), 174–196.   
Curato, G. and Lillo, F., How tick size affects the high frequency scaling of stock return distributions. In Financial Econometrics and Empirical Market Microstructure, pp. 55–76, 2015 (Springer).   
Curme, C., Tumminello, M., Mantegna, R.N., Stanley, H.E. and Kenett, D.Y., Emergence of statistically validated financial intraday lead-lag relationships. Quant. Finance, 2015, 15(8), 1375– 1386.   
Epps, T.W., Comovements in stock prices in the very short run. J. Am. Stat. Assoc., 1979, 74(366a), 291–298.   
Everett, M.G. and Borgatti, S.P., The centrality of groups and classes. J. Math. Sociol., 1999, 23(3), 181–201.   
Giacomini, R. and White, H., Tests of conditional predictive ability. Econometrica, 2006, 74(6), 1545–1578.   
Gu, S., Kelly, B. and Xiu, D., Empirical asset pricing via machine learning. Rev. Financ. Stud., 2020, 33(5), 2223–2273.   
Harris, L.E. and Panchapagesan, V., The information content of the limit order book: Evidence from NYSE specialist trading decisions. J. Financ. Mark., 2005, 8(1), 25–67.   
Hasbrouck, J. and Saar, G., Limit orders and volatility in a hybrid market: The Island ECN. Stern School of Business Dept. of Finance Working Paper FIN-01-025, 2002.   
Hasbrouck, J. and Seppi, D.J., Common factors in prices, order flows, and liquidity. J. Financ. Econ., 2001, 59(3), 383–411.   
Hastie, T., Tibshirani, R. and Friedman, J., The Elements of Statistical Learning: Data Mining, Inference, and Prediction, 2009 (Springer Science & Business Media: New York).   
Hautsch, N. and Huang, R., The market impact of a limit order. J. Econ. Dyn. Control, 2012, 36(4), 501–522.   
Hou, K., Industry information diffusion and the lead-lag effect in stock returns. Rev. Financ. Stud., 2007, 20(4), 1113–1138.   
Huck, N., Large data sets and machine learning: Applications to statistical arbitrage. Eur. J. Oper. Res., 2019, 278(1), 330–342.   
Kelly, B.T., Malamud, S. and Zhou, K., The virtue of complexity in return prediction. J. Finance, 2022, forthcoming.   
Kenett, D.Y., Tumminello, M., Madi, A., Gur-Gershgoren, G., Mantegna, R.N. and Ben-Jacob, E., Dominating clasp of the financial sector revealed by partial correlation analysis of the stock market. PLoS. ONE., 2010, 5(12), e15032.   
Kolm, P.N., Turiel, J. and Westray, N., Deep order flow imbalance: Extracting alpha at multiple horizons from the limit order book. Math. Finance, 2023, forthcoming.   
Krauss, C., Do, X.A. and Huck, N., Deep neural networks, gradientboosted trees, random forests: Statistical arbitrage on the S&P 500. Eur. J. Oper. Res., 2017, 259(2), 689–702.   
Kyle, A.S., Continuous auctions and insider trading. Econometrica, 1985, 53(6), 1315–1335.   
Laloux, L., Cizeau, P., Potters, M. and Bouchaud, J.-P., Random matrix theory and financial correlations. Int. J. Theor. Appl. Finance, 2000, 3(03), 391–397.   
Lillo, F., Farmer, J.D. and Mantegna, R.N., Master curve for priceimpact function. Nature, 2003, 421(6919), 129–130.   
Menzly, L. and Ozbas, O., Market segmentation and crosspredictability of returns. J. Finance, 2010, 65(4), 1555–1580.   
Pasquariello, P. and Vega, C., Strategic cross-trading in the US stock market. Rev. Financ., 2015, 19(1), 229–282.   
Renò, R., A closer look at the epps effect. Int. J. Theor. Appl. Finance, 2003, 6(01), 87–102.   
Rosenbaum, M. and Tomas, M., A characterisation of cross-impact kernels. arXiv preprint arXiv:2107.08684. 2021.   
Schneider, M. and Lillo, F., Cross-impact and no-dynamic-arbitrage. Quant. Finance, 2019, 19(1), 137–154.   
Sirignano, J.A., Deep learning for limit order books. Quant. Finance, 2019, 19(4), 549–570.   
Tashiro, D., Matsushima, H., Izumi, K. and Sakaji, H., Encoding of high-frequency order information and prediction of shortterm stock price by deep learning. Quant. Finance, 2019, 19(9), 1499–1506.   
Tomas, M., Mastromatteo, I. and Benzaquen, M., How to build a cross-impact model from first principles: Theoretical requirements and empirical results. Quant. Finance, 2022, 22(6), 1017–1036.   
Tóth, B. and Kertész, J., The epps effect revisited. Quant. Finance, 2009, 9(7), 793–802.   
Wang, S., Schäfer, R. and Guhr, T., Average cross-responses in correlated financial markets. Eur. Phys. J. B, 2016a, 89(9), 207.   
Wang, S., Schäfer, R. and Guhr, T., Cross-response in correlated financial markets: Individual stocks. Eur. Phys. J. B, 2016b, 89(4), 105.   
Wang, S., Neusüß, S. and Guhr, T., Statistical properties of market collective responses. Eur. Phys. J. B, 2018, 91, 1–11.   
Ward, M.D. and Ahlquist, J.S., Maximum Likelihood for Social Science: Strategies for Analysis, 2018 (Cambridge University Press: Cambridge).   
Xu, K., Gould, M.D. and Howison, S.D., Multi-level order-flow imbalance in a limit order book. Mark. Microstruct. Liq., 2018, 4(3–4), 1950011.   
Zhang, L., Estimating covariation: Epps effect, microstructure noise. J. Econom., 2011, 160(1), 33–47.

# Appendices

# Appendix 1. Aggregation of multi-level OFIs

Table 2 presents evidence of the effectiveness of PCA in selecting weights for combining multi-level OFIs. However, figure 2 shows that the weights derived from PCA are not extremely different. This prompts us to consider a simpler method, namely the simple average (SA), to achieve similar performance. Table A1 reveals that the explained variance of the SA increases as more levels are included. Nonetheless, the SA across 10 levels is inferior to the first principal component (PC) in terms of EVR, i.e. $8 5 . 0 7 \%$ of SA vs $8 9 . 0 6 \%$ of PC in table 2. Additionally, table A2 provides further evidence that PC consistently performs better than SA across different subsets grouped by stock-specific characteristics.

We then perform the price impact and cross-impact analysis with simple average multi-level OFIs, denoted as $\mathbf { P } \mathbf { I } ^ { S A }$ and $\mathbf { C I } ^ { S A }$ , respectively. As shown in table A3, $\mathbf { P I } ^ { S A }$ has a similar performance with $\mathbf { C I } ^ { S A }$ , consistent with our main analysis. This again confirms that as long as multi-level orders are taken into account, adding crossimpact terms cannot significantly improve model performance. On the other hand, $\mathbf { P } \mathbf { \Phi } _ { } ^ { I }$ is slightly better than $\mathbf { P I } ^ { S A }$ . A future research direction might be to devise various weighting schemes that average the OFI information across the multiple levels, where the weights could be given, for example, by an inverse function of the distance of each price level to the mid-price or applying tensor-SVD/PCA on this data.

# Appendix 2. Contemporaneous price impact of multi-level OFIs

To explicitly identify the impact of deeper-level OFIs, we also consider an extended version of $\mathbf { \bar { P } } _ { \mathbf { I } } ^ { [ 1 ] }$ by incorporating multi-level OFIs as features in the model

$$
\mathbf { P I } ^ { [ m ] } : ~ r _ { i , t } ^ { ( h ) } = \alpha _ { i } ^ { [ m ] } + \sum _ { k = 1 } ^ { m } \beta _ { i } ^ { [ m ] , k } \mathrm { o f f } _ { i , t } ^ { k , ( h ) } + \epsilon _ { i , t } ^ { [ m ] } .
$$

Recall that $\mathrm { o f f } _ { i , t } ^ { k , ( h ) }$ is the OFI at level $k$ . We refer to this model as $\mathbf { P I } ^ { [ m ] }$ , and use OLS to estimate it.

The top panel of table A4 shows that the in-sample $R ^ { 2 }$ values increase as more multi-level OFIs are included as features, which is not surprising given that $\mathbf { P I } ^ { [ m ] }$ is a nested model of $\mathbf { P I } ^ { [ m + 1 ] }$ . However the increments of the in-sample $R ^ { 2 }$ are descending, indicating that much deeper LOB data might be unable to provide additional information. This argument is confirmed by the models’ performance on out-of-sample data, as shown at the bottom panel of table A4. Out-of-sample $R ^ { 2 }$ reaches a peak at $\mathbf { P I } ^ { [ 8 ] }$ .

Impact comparison between multi-level OFIs. An interesting question is whether the OFIs at different price levels contribute evenly in terms of price impact. Based on Figure 1(a), we conclude that multi-level OFIs have different contributions to price movements. Generally, OFIs at the second-best level manifest greater influence than OFIs at the best level in model $\mathbf { P I } ^ { [ 1 0 ] }$ , which is perhaps counter-intuitive, at first sight.

![](images/f1e541fa4320ae87261e7212e62a211aa8b2af35965c796c3b5d1fafc18fd111.jpg)  
Figure A1. Coefficients of the model $\mathbf { P I } ^ { [ 1 0 ] }$ . (a) Average. (b) Volume. (c) Volatility and (d) Spread. Note: Plot (a) reports average coefficients and one standard deviation (error bars); Plots (b)-(d) show coefficients sorted by stock characteristics. Volume: trading volume on the previous trading day. Volatility: volatility of one-minute returns during the previous trading day. Spread: average bid-ask spread during the previous trading day. $[ 0 \% , 2 5 \% )$ , respectively $[ 7 5 \%$ , $100 \% ]$ , denote the subset of stocks with the lowest, respectively highest, $2 5 \%$ values for a given stock characteristic. The $x$ -axis represents different levels of OFIs and the $y .$ -axis represents the coefficients.

![](images/3f0f353e1742974891ade618cc455b9be13d98bb503e97c87a752f465949b354.jpg)  
Figure A2. Average coefficient matrices constructed from forward-looking cross-impact models. (a) Coefficient matrix of $\mathbf { F C I } ^ { [ 1 ] }$

![](images/f99b6bcff2338b2453682d618a47fa0aed935501e574f352799fb3b6dbcb30fb.jpg)  
Figure A2. Continued. (b) Coefficient matrix of FCII

We further investigate how the coefficients vary across stocks with different characteristics, such as volume, volatility, and bidask spread. Figure A1(b–d) reveals that for stocks with high-volume and small-spread, order flow posted deeper in the LOB has more influence on price movements. The results regarding spread are in line with Xu et al. (2018), where it is observed that for large-spread stocks (AMZN, TSLA, and NFLX), the coefficients of $\mathbf { o } \hbar ^ { m }$ (OFIs at the mth level) tend to get smaller as the LOB level $m$ increases, while for small-spread stocks (ORCL, CSCO, and MU), the coefficients of ofim may become larger as $m$ increases.

Cont et al. (2014) concluded that the effect of ofi $^ m ( m \geq 2 )$ on price changes is only second-order or null. There are two likely causes for the differences between their findings and ours. First, the data used in Cont et al. (2014) includes 50 stocks (randomly picked from S&P500 constituents) for a single month in 2010, while we use the top 100 large-cap stocks for 36 months during 2017-2019. Second, Cont et al. (2014) considered the average of the coefficients across 50 stocks. In our work, we first group 100 stocks by firm characteristics, and then study the average coefficients of each subset. Therefore, our results are based on a more granular analysis, across a significantly longer period of time.

# Appendix 3. Comparison with Capponi and Cont (2020)

One closely related work is Capponi and Cont (2020) (CC hereafter), where the authors propose a two-step procedure to justify the significance of cross-impact terms and render a different conclusion about cross-impact.

InOFIs $( \mathrm { o f i } _ { i , t } ^ { 1 , ( h ) } )$ step, the authors use OLS to deco into the common factor of OFIs $( \bar { F } _ { \mathrm { o f f } , t } ^ { ( h ) } )$ each stock’s, that is the first principal component of the multi-asset order flow imbalances, and obtain the idiosyncratic components $( \tau _ { i , t } ^ { ( h ) } )$ of the OFIs, for each

$$
\begin{array} { r } { \mathrm { o f l } _ { i , t } ^ { 1 , ( h ) } = \mu _ { i } + \gamma _ { i } F _ { \mathrm { o f f } , t } ^ { ( h ) } + \tau _ { i , t } ^ { ( h ) } . } \end{array}
$$

In the second step, they regress returns $( r _ { i , t } ^ { ( h ) } )$ of stock $i$ against (i) the common factor of OFIs (F(h)ofi,t ), (ii) the idiosyncratic components of its own OFIs $( \tau _ { i , t } ^ { ( h ) } )$ , and (iii) the idiosyncratic components of the OFIs of other stocks $( \tau _ { j , t } ^ { ( h ) } , j \neq i )$ . Finally, we arrive at the cross-impact model proposed by Capponi and Cont (2020) in equation (A3), denoted as CICC.

![](images/7d21c3fa1385b8ece0a20dccca65dffdbb4bf221a361f05a6d78873a4fe86469.jpg)  
Figure A2. Continued. (c) Coefficient matrix of CAR.

$$
\mathbf { C I } ^ { C C } : \quad r _ { i , t } ^ { ( h ) } = \alpha _ { i } ^ { C C } + \beta _ { i 0 } ^ { C C } F _ { \mathrm { o f } , t } ^ { ( h ) } + \sum _ { j = 1 } ^ { N } \beta _ { i j } ^ { C C } \tau _ { j , t } ^ { ( h ) } + \eta _ { i , t } ^ { C C } .
$$

$\mathbf { C I } ^ { C C }$ is compared with a parsimonious model $\mathbf { P I } ^ { C C }$ (equation (A4)), in which only the common order flow factor and a stock’s own idiosyncratic OFI are utilized.

that introducing the common factor leads to quite small changes in the model’s explanatory power of price dynamics in the in-sample and out-of-sample tests. Moreover, the models employing integrated OFIs continually outperform others.

Capponi and Cont (2020) claim that the main determinants of impact are from idiosyncratic order flow imbalance as well as a market order flow factor common across stocks; we conclude that as long as the multi-level OFIs are included, additional cross-impact terms are not necessary. The results also reveal that the sparse price impact model with integrated (or multi-level) OFIs can explain the price dynamics better than the models proposed by Capponi and Cont (2020).

$$
\mathbf { P I } ^ { C C } : \quad r _ { i , t } ^ { ( h ) } = \alpha _ { i } ^ { C C } + \beta _ { i 0 } ^ { C C } F _ { \mathrm { o f } , t } ^ { ( h ) } + \beta _ { i i } ^ { C C } \tau _ { i , t } ^ { ( h ) } + \epsilon _ { i , t } ^ { C C } .
$$

We estimate the $\mathbf { P I } ^ { C C }$ and $\mathbf { C I } ^ { C C }$ models on historical data, under the same setting as in Section 3.2. Given that there are more features than observations, we employ LASSO in the second step to testify the intraday cross-impact of the idiosyncratic OFIs.

Similarly, we present the both in-sample and out-of-sample $R ^ { 2 }$ values of $\mathbf { P I } ^ { C C }$ and $\mathbf { C I } ^ { C C }$ in table A5. We observe small improvements $( 1 . 3 7 \%$ in in-sample tests, $0 . 5 8 \%$ in out-of-sample tests) from $\mathbf { P I } ^ { C C }$ to $\mathbf { C I } ^ { C C }$ . From considering tables 3, 5, and A5, we also observe

# Appendix 4. High-frequency updates of contemporaneous models

In this experiment, we use a 30-minute window to estimate contemporaneous models. We then apply the estimated coefficients to fit data in the next one minute, and repeat this procedure every minute. The results summarized in table A6 reveal similar conclusions as in Section 3, illustrating the robustness of our findings.

Table A1. Average percentage and the standard deviation (in parentheses) of variance attributed to SA across multiple levels.   

<table><tr><td>Average across</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td></tr><tr><td>Explained Variance Ratio</td><td>13.82 (9.04)</td><td>22.81 (10.09)</td><td>32.53 (10.35)</td><td>42.73 (10.17)</td><td>52.58 (9.58)</td><td>61.45 (8.69)</td><td>68.82 (7.71)</td><td>75.10 (7.10)</td><td>80.49 (7.23)</td><td>85.07 (8.14)</td></tr></table>

Notes: The table reports the ratio (in percentage points) between the variance of SA across multiple levels and the total variance averaged across each stock and trading day.

Table A2. Explained variance ratio by different integration ways of multi-level OFIs, sorted by stock characteristics.   

<table><tr><td colspan="3">Volume</td><td>Volatility</td><td>Spread</td></tr><tr><td>[0%,25%)</td><td>SA</td><td>83.02</td><td>85.09</td><td>81.84</td></tr><tr><td rowspan="3">[25%,50%)</td><td>PC</td><td>85.79</td><td>89.64</td><td>89.90</td></tr><tr><td>SA</td><td>87.33</td><td>84.75</td><td>87.97</td></tr><tr><td>PC</td><td>89.93</td><td>89.12</td><td>91.17</td></tr><tr><td>[50%,75%)</td><td>SA</td><td>86.93</td><td>85.23</td><td>87.57</td></tr><tr><td rowspan="3">[75%,100%]</td><td>PC</td><td>90.76</td><td>89.03</td><td>89.81</td></tr><tr><td>SA</td><td>83.55</td><td>85.23</td><td>83.03</td></tr><tr><td>PC</td><td>90.23</td><td>88.48</td><td>85.53</td></tr></table>

Notes: SA: taking the simple average of OFIs across 10 levels. PC: taking the first principal component of OFIs across 10 levels. Volume: trading volume on the previous trading day. Volatility: volatility of one-minute returns during the previous trading day. Spread: average bid-ask spread during the previous trading day. $[ 0 \% , 2 5 \% )$ , respectively $[ 7 5 \%$ , $100 \% ]$ , denote the subset of stocks with the lowest, respectively highest, $2 5 \%$ values for a given stock characteristic.

Table A3. Out-of-sample performance of models based on different aggregations of multi-level OFIs.   

<table><tr><td></td><td>PISA</td><td>CISA</td><td>PII</td><td>CI</td></tr><tr><td>OS R²</td><td>82.34</td><td>82.51</td><td>83.83</td><td>83.62</td></tr><tr><td></td><td>(18.02)</td><td>(14.27)</td><td>(16.90)</td><td>(14.53)</td></tr></table>

Notes: $\mathbf { P I } ^ { S A }$ (resp. $\mathbf { C } \mathbf { I } ^ { S A }$ ): price impact (resp. crossimpact) model using the simple average of OFIs across 10 levels.

# Appendix 5. Additional results of Section 4

One interesting question to consider when examining the predictability of cross-impact is whether the lead-lag effect is linked to the frequency of LOB updates. As shown in table A7, the results indicate that assets with a higher frequency of book updates tend to lead ‘slower’ assets, which aligns with the findings reported by Kolm et al. (2023).

Table A5. Performance of CC’s models.   

<table><tr><td></td><td>PICC</td><td>CICC</td></tr><tr><td>IS R2</td><td>72.58</td><td>73.95</td></tr><tr><td></td><td>(13.22)</td><td>(12.56)</td></tr><tr><td>OS R²</td><td>64.78</td><td>65.36</td></tr><tr><td></td><td>(19.95)</td><td>(18.68)</td></tr></table>

Note: The table reports the mean values and standard deviations (in parentheses) of both in-sample and outof-sample $R ^ { 2 }$ (in percentage points) of $\mathbf { P I } ^ { \hat { C C } }$ and $\mathbf { C } \mathbf { I } ^ { C C }$ when modeling contemporaneous returns. These statistics are averaged across each stock and each regression window.

Table A6. Performance of various models under one-minute update frequency.   

<table><tr><td rowspan="2"></td><td colspan="2">Best-level OFIs</td><td colspan="2">Integrated OFIs</td></tr><tr><td>PI[1]</td><td>CI[]</td><td>PII</td><td>CII</td></tr><tr><td>IS R²</td><td>70.80</td><td>73.55</td><td>86.10</td><td>86.84</td></tr><tr><td rowspan="3">OS R²</td><td>(13.10)</td><td>(12.73)</td><td>(9.64)</td><td>(8.79)</td></tr><tr><td>59.67</td><td>61.46</td><td>78.88</td><td>78.91</td></tr><tr><td>(23.15)</td><td>(18.96)</td><td>(16.78)</td><td>(15.02)</td></tr></table>

Note: The table reports the mean values and standard deviations (in parentheses) of both in-sample and out-of-sample $R ^ { 2 }$ (in percentage points) of various models when modeling contemporaneous returns in one-minute update frequency. The models include $\mathbf { P I } ^ { [ 1 ] }$ (equation (6)), $\mathbf { C I ^ { [ 1 ] } }$ (equation (8)), $\mathbf { P } \mathbf { \overset { . } { F } }$ (equation (7)), and $\mathbf { C I } ^ { I }$ (equation (9)). These statistics are averaged across each stock and each regression window.

Table A7. Out-of-sample $R ^ { 2 }$ of various predictive models sorted by book updates.   

<table><tr><td></td><td>[0%,25%)</td><td>[25%,50%)</td><td>[50%,75%)</td><td>[75%,100%]</td></tr><tr><td>FPI[1]</td><td>-0.38</td><td>-0.37</td><td>-0.36</td><td>-0.34</td></tr><tr><td>FCI[1]</td><td>-0.12</td><td>-0.11</td><td>-0.10</td><td>-0.09</td></tr><tr><td>FPI</td><td>-0.37</td><td>-0.35</td><td>-0.35</td><td>- 0.33</td></tr><tr><td>FCI</td><td>-0.12</td><td>-0.11</td><td>-0.10</td><td>-0.09</td></tr></table>

Notes: $[ 0 \% , 2 5 \% )$ , respectively $[ 7 5 \%$ , $100 \% ]$ , denote the subset of stocks with the lowest, respectively highest, $2 5 \%$ values according to the frequencies of book updates.

Table A4. Performance of price impact models with multi-level OFIs.   

<table><tr><td></td><td>PI[1]</td><td>P[2]</td><td>P[3]</td><td>PI[4]</td><td>PI[5]</td><td>PI[6]</td><td>P[7]</td><td>PI[8]</td><td>P[9]</td><td>P[10]</td></tr><tr><td>IS R²</td><td>71.16 (13.80)</td><td>81.61 (11.80)</td><td>85.07 (10.76)</td><td>86.69 (10.30)</td><td>87.66 (10.05)</td><td>88.30 (9.86)</td><td>88.74 (9.71)</td><td>89.04 (9.57)</td><td>89.24 (9.45)</td><td>89.38 (9.34)</td></tr><tr><td>OS R²</td><td>64.64 (21.82)</td><td>75.81 (19.83)</td><td>79.47 (18.87)</td><td>81.13 (18.61)</td><td>82.05 (18.58)</td><td>82.65 (18.65)</td><td>83.01 (18.78)</td><td>83.16 (18.93)</td><td>83.15 (19.49)</td><td>83.11 (20.93)</td></tr></table>

Note: The table reports the mean values and standard deviations (in parentheses) of both in-sample and out-of-sample $R ^ { 2 }$ (in percentage points) of $\mathbf { P I } ^ { [ m ] }$ $( m = 1 , \ldots , 1 0 )$ when modeling contemporaneous returns. These statistics are averaged across each stock and each regression window.