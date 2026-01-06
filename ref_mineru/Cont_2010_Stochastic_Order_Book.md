# A Stochastic Model for Order Book Dynamics

**Authors:** Rama Cont, Sasha Stoikov, Rishi Talreja

**Affiliation:** IEOR Dept, Columbia University, New York

**Contact:** rama.cont@columbia.edu, sashastoikov@gmail.com, rt2146@columbia.edu

**Journal:** Operations Research, Vol. 58, No. 3, May–June 2010, pp. 549–563

**DOI:** https://doi.org/10.1287/opre.1090.0780

**SSRN:** https://ssrn.com/abstract=1273160

---

## Abstract

We propose a stochastic model for the continuous-time dynamics of a limit order book. The model strikes a balance between two desirable features: it captures key empirical properties of order book dynamics and its analytical tractability allows for fast computation of various quantities of interest without resorting to simulation. We describe a simple parameter estimation procedure based on high-frequency observations of the order book and illustrate the results on data from the Tokyo stock exchange. Using Laplace transform methods, we are able to efficiently compute probabilities of various events, conditional on the state of the order book: an increase in the mid-price, execution of an order at the bid before the ask quote moves, and execution of both a buy and a sell order at the best quotes before the price moves. Comparison with high-frequency data shows that our model can capture accurately the short term dynamics of the limit order book.

---

## Keywords

Limit order book, financial engineering, Laplace transform inversion, queueing systems, simulation.

---

## Contents

1. Introduction
2. A continuous-time model for a stylized limit order book
   - 2.1 Limit order books
   - 2.2 Dynamics of the order book
3. Parameter estimation
   - 3.1 Description of the data set
   - 3.2 Estimation procedure
4. Laplace transform methods for computing conditional probabilities
   - 4.1 Laplace transforms and first-passage times of birth-death processes
   - 4.2 Direction of price moves
   - 4.3 Executing an order before the mid-price moves
   - 4.4 Making the spread
5. Numerical Results
   - 5.1 Long term behavior
   - 5.2 Conditional distributions
6. Conclusion

---

## 1 Introduction (Key Excerpts)

"The evolution of prices in financial markets results from the interaction of buy and sell orders through a rather complex dynamic process."

"The dynamics of a limit order book resembles in many aspects that of a queuing system. Limit orders wait in a queue to be executed against market orders (or canceled). Drawing inspiration from this analogy, we model a limit order book as a continuous-time Markov process that tracks the number of limit orders at each price level in the book."

"The model strikes a balance between three desirable features: it can be easily calibrated to high-frequency data, reproduces various empirical features of order books and is analytically tractable. In particular, we show that our model is simple enough to allow the use of Laplace transform techniques from the queueing literature to compute various conditional probabilities."

"These include the probability of the mid-price increasing in the next move, the probability of executing an order at the bid before the ask quote moves and the probability of executing both a buy and a sell order at the best quotes before the price moves, given the state of the order book."

---

## Model Description (Section 2)

"The model proposed here is admittedly simpler in structure than some others existing in the literature: it does not incorporate strategic interaction of traders as in game theoretic approaches Parlour (1998), Foucault et al. (2005) and Rosu (forthcoming), nor does it account for 'long memory' features of the order flow as pointed out by Bouchaud et al. (2002) and Bouchaud et al. (2008). However, contrarily to these models, it leads to an analytically tractable framework where parameters can be easily estimated from empirical data and various quantities of interest may be computed efficiently."

---

## 6 Conclusion (Key Excerpts)

"We have proposed a stylized stochastic model describing the dynamics of a limit order book, where the occurrence of different types of events -market orders, limit orders and cancellations- are described in terms of independent Poisson processes."

"The formulation of the model, which can be viewed as a queuing system, is entirely based on observable quantities and its parameters can be easily estimated from observations of the events in the order book. The model is simple enough to allow analytical computation of various conditional probabilities of order book events via Laplace transform methods, yet rich enough to capture adequately the short-term behavior of the order book."

"One by-product of this study is to show how far a stochastic model can go in reproducing the dynamic properties of a limit order book without resorting to detailed behavioral assumptions about market participants or introducing unobservable parameters describing agent preferences, as in the market microstructure literature."

---

## Numerical Results (Tables 3-5)

### Table 4: Probability of executing a bid order before a change in mid-price

| b\a | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| 1 | .497 | .641 | .709 | .749 | .776 |
| 2 | .302 | .449 | .535 | .591 | .631 |
| 3 | .206 | .336 | .422 | .483 | .528 |
| 4 | .152 | .263 | .344 | .404 | .452 |
| 5 | .118 | .213 | .287 | .346 | .393 |

(Laplace transform method results; simulation results show 95% confidence intervals matching these values)

### Table 5: Probability of making the spread

| b\a | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| 1 | .266 | .308 | .309 | .300 | .288 |
| 2 | .308 | .386 | .406 | .406 | .400 |
| 3 | .309 | .406 | .441 | .452 | .452 |
| 4 | .300 | .406 | .452 | .471 | .479 |
| 5 | .288 | .400 | .452 | .479 | .491 |

(Laplace transform method results)

---

## References (Selected)

- Bouchaud, J. P., D. Farmer, F. Lillo. 2008. How markets slowly digest changes in supply and demand. Handbook of Financial Markets.
- Bouchaud, Jean-Philippe, Marc Mézard, Marc Potters. 2002. Statistical properties of stock order books: empirical results and models. Quantitative Finance 2 251–256.
- Foucault, T., O. Kadan, E. Kandel. 2005. Limit order book as a market for liquidity. Review of Financial Studies 18(4) 1171–1217.
- Parlour, Ch. A. 1998. Price dynamics in limit order markets. Review of Financial Studies 11(4) 789–816.
- Smith, E., J. D. Farmer, L. Gillemot, S. Krishnamurthy. 2003. Statistical theory of the continuous double auction. Quantitative Finance 3(6) 481–514.

---

## 其他文献中的引用

引用格式（Operations Research标准）：
> R. Cont, S. Stoikov, and R. Talreja, "A stochastic model for order book dynamics," Operations Research, vol. 58, no. 3, pp. 549–563, 2010.

该论文在以下ref_mineru文献中被引用：
- ssrn-1748022.md: "[21] R. CONT, S. STOIKOV, AND R. TALREJA, A stochastic model for order book dynamics, Operations Research, 58 (2010), pp. 549–563."
- 1907.06230v2.md: "R. Cont, S. Stoikov, and R. Talreja. A stochastic model for order book dynamics. Operations Research, 58(3):549–563, 2010."
- 1810.09965v1.md: "[5] R. Cont, S. Stoikov, R. Talreja, A stochastic model for order book dynamics, Operations research 58 (3) (2010) 549–563."
- 1410.1900v2.md: "[18] Cont R., Stoikov S., Talreja R. A stochastic model for order book dynamics // Operations Research, 2010. Vol. 58. No. 3. P. 549–563."
- 2402.17359v2.md: "Cont, Rama, Sasha Stoikov, and Rishi Talreja (2010). 'A Stochastic Model for Order Book Dynamics'. In: Operations Research 58.3, pp. 549–563."

---

**PDF来源:** Columbia University (http://www.columbia.edu/~ww2040/orderbook.pdf)

**本地文件:** Cont_2010_Stochastic_Order_Book.pdf
