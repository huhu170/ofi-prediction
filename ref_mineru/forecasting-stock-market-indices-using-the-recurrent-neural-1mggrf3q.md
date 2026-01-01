# Article Forecasting Stock Market Indices Using the Recurrent Neural Network Based Hybrid Models: CNN-LSTM, GRU-CNN, and Ensemble Models

Hyunsun Song $\ast ( 0 )$ and Hyunjun Choi $\textcircled{1}$

Citation: Song, H.; Choi, H.

Forecasting Stock Market Indices Using the Recurrent Neural Network Based Hybrid Models: CNN-LSTM, GRU-CNN, and Ensemble Models. Appl. Sci. 2023, 13, 4644. https:// doi.org/10.3390/app13074644

Academic Editors: Amerigo Capria and Seung-Hoon Yoo

Received: 19 December 2022   
Revised: 27 March 2023   
Accepted: 4 April 2023   
Published: 6 April 2023

Department of Nano & Semiconductor Engineering, Tech University of Korea, Siheung-si 15073, Republic of Korea \* Correspondence: hyunsun0113@tukorea.ac.kr; Tel.: $+ 8 2$ -31-8041-0380

Abstract: Various deep learning techniques have recently been developed in many fields due to the rapid advancement of technology and computing power. These techniques have been widely applied in finance for stock market prediction, portfolio optimization, risk management, and trading strategies. Forecasting stock indices with noisy data is a complex and challenging task, but it plays an important role in the appropriate timing of buying or selling stocks, which is one of the most popular and valuable areas in finance. In this work, we propose novel hybrid models for forecasting the onetime-step and multi-time-step close prices of DAX, DOW, and S&P500 indices by utilizing recurrent neural network (RNN)–based models; convolutional neural network-long short-term memory (CNN-LSTM), gated recurrent unit (GRU)-CNN, and ensemble models. We propose the averaging of the high and low prices of stock market indices as a novel feature. The experimental results confirmed that our models outperformed the traditional machine-learning models in $4 8 . 1 \%$ and $4 0 . 7 \%$ of the cases in terms of the mean squared error (MSE) and mean absolute error (MAE), respectively, in the case of one-time-step forecasting and $8 1 . 5 \%$ of the cases in terms of the MSE and MAE in the case of multi-time-step forecasting.

Keywords: deep learning; convolutional neural networks; recurrent neural networks; long short-term memory; gated recurrent unit; ensemble model; feature engineering

# 1. Introduction

Forecasting stock market indices is one of the most critical yet challenging areas in finance, as a key task in investment management. The stock market indices are used to formulate and implement economic policy, and they are also used to inform decisions about the timing and size of various investments, such as stocks and real estate for investors.

In finance, stock market forecasting is one of the most challenging tasks due to the inherently volatile, noisy, dynamic, nonlinear, complex, non-parametric, non-stationary, and chaotic nature of stock markets, making any prediction model subject to large errors [1,2]. Additionally, price fluctuations are influenced not only by historical stock trading data, but also by nonlinear factors, such as political factors, investor behavior, and unexpected events [3–6].

To overcome these difficulties, numerous studies have been conducted over the past decades to predict various types of financial time-series data.

Linear models, such as the autoregressive and moving average (ARMA) and autoregressive integrated moving average (ARIMA) models have achieved high predictive accuracy in predicting stock market trends. However, traditional statistical models assume that financial time series are linear, which is not the case in real-world scenarios. Meanwhile, as many machine learning techniques capture nonlinear relationships from the data [7], they might be very useful for decision-making with respect to financial market investments [8].

A variety of deep learning models has been shown to significantly improve upon previous machine learning models in tasks, such as speech recognition, image captioning, question answering, natural language processing, autonomous self-driving cars, sports, arts, and regression tasks [9–11]. Deep-learning-based models have also been widely used in financial areas, such as forecasting stock price and index, portfolio optimization, risk management, financial information processing, and trade execution strategies.

In particular, RNNs, LSTMs, and GRUs have been designed to deal with time-series data and have been shown to perform better than traditional time-series models when a series of previous events is essential to predict future events. Thus, they have been actively applied to tasks, such as stock market index prediction and language translation [12,13].

RNNs have many advantages when processing short sequences. However, when the distance between the relevant information and the point using the information increases, the learning ability of the network is significantly reduced. The reason for this problem is that the back-propagation algorithm has difficulty in long-term dependency learning. In the process of updating the weights, the gradient disappears as values smaller than one are continuously multiplied, which is called the vanishing gradient problem. To solve the long-term dependency problem of RNNs, the LSTM and GRU models have been proposed, which represent the transformation algorithms of RNNs. In the LSTM model, a structure called the cell state is added to resolve long-term dependencies. Additionally, three additional input, forget, and output gates are added where the data are computed, which partially solves the problem of long-term dependencies by storing each state value in a memory space cell. The GRU is similar to LSTM; however, it has only update and reset gates and no output gate, thus being a simpler model with fewer parameters than LSTM. Recently, the LSTM model has shown great success in various domains, including speech recognition and machine translation, outperforming vanilla RNNs and conventional machine learning algorithms.

In this study, we propose hybrid models based on a variation of RNN models, such as LSTMs and GRUs, to improve stock market index prediction performance. The proposed models are divided into three types: a CNN-LSTM model that stacks a one-dimensional CNN and LSTM, a GRU-CNN model that stacks GRU and a one-dimensional CNN, and an ensemble model that takes the average value of each output result by placing RNN, LSTM, and GRU in parallel. The experiments were conducted on various daily stock market indices (i.e., Deutscher Aktienindex (DAX), Dow Jones Industrial Average (DOW), and Standard and Poor’s 500 (S&P500)) for the periods from 1 January 2017 through 31 December 2019 and from 1 January 2019 through 31 December 2021 for three years before and after the COVID-19 pandemic, respectively. Additionally, we considered a long period of time from 1 January 2000 through 31 December 2019 for the DOW and S&P500 and from 24 October 2014 through 31 December 2019 for DAX because the DAX data were only available from 24 October 2014 in the pandas DataReader module.

We considered the look-back periods of 5, 21, and 42 days and look-ahead periods of one day for one-time-step and five days for multi-time-step prediction. To verify the robustness of our results, we compared our models with conventional deep-learning models such as RNN, LSTM, GRU, and WaveNet.

The main contributions of this study include the following:

Novel RNN-based hybrid models are proposed to forecast one-time-step and multitime-step closing prices of the DAX, DOW, and S&P500 indices by utilizing neural network structures: CNN-LSTM, GRU-CNN, and ensemble models. − The novel feature, which is the average of the high and low prices of stock market indices, is used as an input feature. Comparisons between the proposed and traditional benchmark models with various look-back periods and features are presented. The experimental results indicate that the proposed models outperform the benchmark models in $4 8 . 1 \%$ and $4 0 . 7 \%$ of the cases in terms of the mean squared error (MSE) and mean absolute error (MAE), respectively, in the case of one-time-step forecasting and $8 1 . 5 \%$ of the cases in terms of the MSE and MAE in the case of multi-time-step forecasting.

Further, compared with previous studies that involved using open, high, and low prices, and trading volume of stock market indices as features, in this study, we evaluate the performance of our models by adding a novel feature to reduce the influence of the highest and lowest prices. The results confirm that the newly proposed feature contributes to improving the performance of the models in forecasting stock market indices.   
In particular, the ensemble model provides significant results for one-time-step forecasting.

The remainder of this paper is organized as follows. Section 2 presents an overview of deep learning models and reviews the relevant existing literature on stock market forecasting. Section 3 describes the proposed models designed using RNN-based hybrid architectures and provides the implementation details of the experiment, including the data and experimental setting. In Section 4, we present the experimental results, where we evaluate the proposed models on three stock market indices, compare them with benchmark models, and analyze the effect of the novel feature. Section 5 discusses the implications and advantages of the proposed models. Finally, Section 6 summarizes the conclusions of the study.

# 2. Background and Related Work

# 2.1. Deep-Learning Background

In this subsection, we review the artificial neural network (ANN), multilayer perceptron (MLP), CNN, RNN, LSTM, and GRU.

# 2.1.1. ANN

ANNs, also known as feedforward neural networks, are computing systems inspired by the biological human brain and consist of input, hidden, and output layers with connected neurons, wherein connections between neurons do not form a cycle. An ANN is capable of learning nonlinear functions and processing information in parallel [14]. Each neuron computes the weighted sum of all of its inputs, and a nonlinear activation function is applied to this sum to produce the output result of each neuron. The weights are adjusted to minimize a metric of the difference between the actual and predicted values of the data using the back-propagation algorithm [15].

# 2.1.2. MLP

The perceptron was proposed by [16] in 1943, representing an algorithm for the supervised learning of binary classifiers. As a linear classifier, a single-layer perceptron is the simplest feedforward neural network. Minsky and Papert [17] showed that a singlelayer perceptron is incapable of learning the exclusive or (XOR) problem, whereas an MLP is capable of solving the XOR problem.

An MLP is a fully connected class of ANN. Attempts to solve linearly inseparable problems, such as the XOR problem, have led to different variations in the number of layers and neurons as well as nonlinear activation functions, such as a logistic sigmoid function or a hyperbolic tangent function [18].

# 2.1.3. CNN

The CNN was proposed to automatically learn spatial hierarchies of features in tasks, such as image recognition and speech recognition [19], by exploiting the spatial relationships among the pixels in an image. In [20], a CNN is composed of convolutional layers, pooling layers, and fully connected layers, and is trained with the adaptive moment estimation (Adam) optimizer on mini batches [21]. The convolutional layers extract the useful features, while the pooling layers reduce the dimensions of the feature maps. The rectified linear unit (ReLU) is applied as a nonlinear activation function [22], and a dropout layer is used as a regularization method in which the output of each hidden neuron is set to zero with a given probability [23].

# 2.1.4. RNN

The assumption of a traditional neural network is that all the inputs are independent of each other, which makes them ineffective when dealing with sequential data and varied sizes of inputs and outputs [24]. The RNN is an extension of the conventional feedforward neural network and is well suited to sequential data, such as time series, gene sequences, and weather data.

An RNN has memory loops and handles a variable length of input sequence by having a recurrent hidden state [25]. It is known to have a shortcoming of a significant decrease in learning ability as the gradient gradually decreases during back-propagation when the distance between the relevant information and the point is long, which is called the vanishing gradient problem [26]. Errors from later time steps are difficult to propagate back to previous time steps, which results in difficulty in training deep RNNs to preserve information over multiple time steps because the gradients tend to either vanish or explode as they cycle through feedback loops [27]. To address this problem, Hochreiter and Schmidhuber [26] proposed the LSTM, which is capable of solving the vanishing gradient problem using memory cells.

# 2.1.5. LSTM

The LSTM was proposed by [26] as a variant of the vanilla RNN to overcome the vanishing or exploding gradient problem by adding the cell state to the hidden state of an RNN. The LSTM is composed of a cell state and three gates: input, output, and forget gates. The following equations describe the LSTM architecture.

The forget gate $\mathbf { f } _ { t }$ determines which information is input to forget or keep from the previous cell state $\mathbf { C } _ { t - 1 }$ and is computed as

$$
\mathbf { f } _ { t } = \sigma ( \mathbf { W _ { f } } \cdot \left[ \mathbf { h } _ { t - 1 } , \mathbf { x } _ { t } \right] + \mathbf { b _ { f } } ) ,
$$

where $\mathbf { x } _ { t }$ is the input vector at time $t$ the function $\sigma$ is a logistic sigmoid function.

The input gate $\mathbf { i } _ { t }$ determines which information is updated to the cell state $\mathbf { C } _ { t }$ and is computed by

$$
\mathbf { i } _ { t } = \sigma ( \mathbf { W _ { i } } \cdot [ \mathbf { h } _ { t - 1 } , \mathbf { x } _ { t } ] + \mathbf { b _ { i } } ) .
$$

The candidate value $\widetilde { \mathbf { C } } _ { t }$ that can be added to the state is created by a tanh activation function and is computed by

$$
\widetilde { \mathbf { C } } _ { \mathbf { t } } = \operatorname { t a n h } ( \mathbf { W } _ { \mathbf { C } } \cdot [ \mathbf { h } _ { t - 1 } , \mathbf { x } _ { t } ] + \mathbf { b } _ { \mathbf { C } } ) .
$$

The cell state $\mathbf { C } _ { t }$ can store information over long periods of time by updating the internal state and is computed by

$$
\mathbf { C } _ { t } = \mathbf { f } _ { t } \odot \mathbf { C } _ { t - 1 } + \mathbf { i } _ { t } \odot { \widetilde { \mathbf { C } } } _ { t } ,
$$

where the operator $\odot$ represents the element-wise Hadamard product.

The output gate $\mathbf { o } _ { t }$ determines what information from the cell state to be used as an output by taking the logistic sigmoid activation function, and is computed by

$$
\mathbf { 0 } _ { t } = \sigma ( \mathbf { W _ { 0 } } \cdot [ \mathbf { h } _ { t - 1 } , \mathbf { x } _ { t } ] + \mathbf { b _ { 0 } } ) ,
$$

and the output $\mathbf { h } _ { t }$ is computed as

$$
\mathbf { h } _ { t } = \mathbf { o } _ { t } \odot \operatorname { t a n h } ( \mathbf { C } _ { t } ) ,
$$

where $\mathbf { W } _ { * }$ and $\mathbf { b } _ { * }$ represent weight matrices and bias vectors, respectively.

Following [26], the gates decide which information to be forgotten or to be remembered; therefore, the LSTM is suitable for managing long-term dependencies and forecasting time series with different numbers of time steps. Further, it can generalize and handle noise, distributed representations, and continuous values well.

# 2.1.6. GRU

The GRU, proposed in [28], is a simpler variation of LSTM and has fewer parameters than LSTM. The LSTM has update, input, forget, and output gates and maintains the internal memory state, whereas the GRU has only update and reset gates. It combines the forget and input gates of LSTM into a single update gate and has fewer tensor operations, resulting in faster training than LSTM.

The GRU merges the cell and hidden states. It performs well in sequence learning tasks and overcomes the problems of vanishing or exploding gradients in vanilla RNNs when learning long-term dependencies [29]. It also tends to perform better than LSTM on fewer training data, whereas LSTM is more efficient in remembering longer sequences [30–32]. The following equations describe how memory cells at each hidden layer are updated at each time step [33]. The reset gate $\mathbf { r } _ { t }$ controls the influence of $\mathbf { h } _ { t - 1 }$ and is computed as

$$
\mathbf { r } _ { t } = \sigma ( \mathbf { W } _ { \mathbf { r } } \cdot [ \mathbf { h } _ { t - 1 } , \mathbf { x } _ { t } ] ) ,
$$

where $\mathbf { x } _ { t }$ and $\mathbf { h } _ { t - 1 }$ are the input and the previous hidden state, respectively.

The update gate $\mathbf { z } _ { t }$ specifies whether to ignore the current information $\mathbf { x } _ { t }$ and is computed as

$$
\mathbf { z } _ { t } = \sigma \big ( \mathbf { W } _ { \mathbf { z } } \cdot [ \mathbf { h } _ { t - 1 } , \mathbf { x } _ { t } ] \big ) .
$$

The computation of candidate activation $\mathbf { h } _ { t }$ is similar to that of the traditional recurrent unit, that is,

$$
\mathbf { h } _ { t } = \mathbf { z } _ { t } \odot \mathbf { h } _ { t - 1 } + \left( 1 - \mathbf { z } _ { t } \right) \odot \widetilde { \mathbf { h } } _ { t } ,
$$

where

$$
\widetilde { \mathbf { h } } _ { t } = \operatorname { t a n h } ( \mathbf { W } _ { \mathbf { h } } \cdot \left[ \mathbf { r } _ { t } \odot \mathbf { h } _ { t - 1 } , \mathbf { x } _ { t } \right] ) ,
$$

$\mathbf { W _ { r } } , \mathbf { W _ { z } }$ and ${ \bf W _ { h } }$ are weight matrices which are learned.

# 2.2. Related Work

A stock market index is an important indicator of changes in the share prices of different companies, thus informing investment decisions. It is also more advantageous to invest in an index fund than to invest in individual stocks because it keeps costs low and removes the need to constantly manage reports from many companies. However, stock market index forecasting is extremely challenging because of the multiple factors affecting the stock market, such as politics, global economic conditions, unexpected events, and the financial performance of companies listed on the stock market.

Recently, deep-learning models have been extensively applied to numerous areas in finance, such as the forecasting future prices of stocks, prediction of stock price movements, portfolio management, risk assessment, and trading strategies [34–39]. Using deep learningbased models, such as CNNs, RNNs, LSTMs, and GRUs, studies have shown that such models outperform classical methods for time series forecasting tasks because of their ability to handle nonlinearity [19,25,26,33].

CNN models have been used in different time series forecasting applications. Chen et al. [40] and Sezer and Ozbayoglu [41] transformed time-series data into two-dimensional image data and used them as inputs for a CNN to classify the movement of the data. Meanwhile, Gross et al. [42] interpreted multivariate time series as space-time pictures.

RNN-based models have been used to predict time-series data. Fischer and Krauss [43] showed that LSTM outperformed memory-free classification methods, such as random forests, deep ANNs, and logistic regression classifiers, in prediction tasks. Dutta et al. [44] proposed the GRU model with recurrent dropout to predict the daily cryptocurrency prices.

Other deep learning models have been applied for time series forecasting. Heaton et al. [45] stacked autoencoders to predict and classify stock prices and their movements. Abrishami et al. [7] used a variational autoencoder to remove noise from the data and stacked LSTM to predict the close price of stocks. Wang et al. [2] used wavelet transform to forecast time-series data.

Moreover, various architectures combining deep learning-based models have been proposed in the literature. Ilyas et al. [46] combined technical and content features via learning time series and textual data, Livieris et al. [47] introduced the CNN-LSTM model to predict gold prices and movements, while Daradkeh [6] integrated a CNN and a bidirectional LSTM to predict stock trends. Zhang et al. [4] combined attention and LSTM models for financial time series prediction. Livieris and Pintelas [48] proposed ensemble learning strategies with advanced deep learning models for forecasting cryptocurrency prices and movements. Bao et al. [24] combined wavelet transforms, stacked autoencoders, and LSTM to forecast the closing stock prices for the next day by eliminating noise from the data and generating deep high-level features. Meanwhile, Zhang et al. [49] proposed a novel architecture of a generative adversarial network (GAN) with an MLP as the discriminator and an LSTM as the generator for forecasting the closing price of stocks.

Further, Leung et al. [50] proposed a two-timescale duplex neurodynamic approach for solving the portfolio optimization problem, and several studies have applied an LSTM to construct a portfolio [36,51–55].

This study proposes three models by combining CNN and RNN-based models for predicting the stock market index. Additionally, in contrast to existing studies, which employed open, high, and low prices, trading volume, and change in stock market indices, we introduce a novel input feature: the average of high and low prices. Furthermore, the three proposed models are evaluated on three daily stock market indices with two different optimizers and four different features. Finally, we compare the performance of the proposed models with conventional benchmark models with respect to forecasting the closing prices of the stock market indices.

# 3. Materials and Methods

# 3.1. Proposed Models

Following Livieris and Pintelas [48], by combining prediction models, a bias is added, which in turn reduces the variance, resulting in a better performance than that of single models. Therefore, we propose three RNN-based hybrid models that predict the stock market indices for one-time-step and multi-time-step at a time.

# 3.1.1. Proposed CNN-LSTM Model

CNNs can effectively learn the internal representations of time-series data [47]. The one-dimensional convolutional layer filters out the noise, extracts spatial features, and reduces the number of parameters. The causal convolution ensures that the output at time t derives only inputs from time $t - 1$ . RNNs are considered the best sequential deep-learning models for forecasting time-series data. To this end, we combine a one-dimensional CNN and an LSTM in a new model: CNN-LSTM. The CNN-LSTM model consists of (1) a onedimensional convolutional layer, (2) an LSTM layer, (3) a batch-normalization layer, (4) a dropout layer, and (5) a dense layer.

To determine the best-performing parameters, we examined different variants of the model: the number of hidden layers (1 and 2), the number of neurons (64 and 128), the batch size (32 and 64), and the dropout rate (0.2 and 0.5).

The best-performing CNN-LSTM model comprised a one-dimensional convolutional layer with 32 filters of size 3 with a stride of 1, causal padding, and the ReLU activation function; an LSTM layer with 128 units and tanh activation function; a batch-normalization layer; a dropout layer with a rate of 0.2; and a dense layer with a prediction window size of units and the ReLU activation function. Figure 1 illustrates the architecture of the proposed CNN-LSTM model, while Table 1 summarizes the configuration.

![](images/cc3200e28858530d745a10c79fdd0050b57ec1345b376180f4a9ca9451addfdc.jpg)  
Figure 1. (a) Architecture of the CNN-LSTM model. (b) Architecture of the GRU-CNN model. (c) Architecture of the ensemble model.

Table 1. Model configuration of the proposed models.   

<table><tr><td>Model</td><td>Description</td></tr><tr><td>CNN-LSTM</td><td>One-dimensional convolutional layer with 32 filters of size 3 with a stride of 1 LSTM layer with 128 units and tanh activation function Batch-normalization layer Dropout layer with a rate of 0.2</td></tr><tr><td>GRU-CNN</td><td>Dense layer with a prediction window size of units GRU layer with 128 units and the tanh activation One-dimensional convolutional layer with 32 filters of size 3 with a stride of 1 One-dimensional global max-pooling layer Batch-normalization layer Dense layer with 10 units and the ReLU activation Dropout layer with a rate of 0.2</td></tr><tr><td>Ensemble</td><td>Dense layer with a prediction window size of units RNN layer with 128 units and the tanh activation function LSTM layer with 128 units and the tanh activation function GRU layer with 128 units and the tanh activation function Average of all the hidden states from RNN, LSTM, and GRU Dropout layer with a rate of 0.2 Dense layer with 32 units and the ReLU activation function Dense layer with a prediction window size of units</td></tr></table>

# 3.1.2. Proposed GRU-CNN Model

The GRU is simpler than LSTM, has the ability to train sequential patterns, and takes less time to train the model with improved network performance. To utilize both GRU and one-dimensional CNN, we propose a stacked architecture where a GRU and a onedimensional CNN are combined, namely the GRU-CNN model. The parameters used for the GRU-CNN model were similar to those of the CNN-LSTM model, as described in Section 3.1.1. The difference between the CNN-LSTM and GRU-CNN models is in the order of stacking the RNN and CNN layers.

The GRU-CNN model consists of a GRU layer with 128 units and the tanh activation function; a one-dimensional convolutional layer with 32 filters of size 3 with a stride of 1, causal padding, and the ReLU activation function; a one-dimensional global max-pooling layer; a batch-normalization layer; a dense layer with 10 units and the ReLU activation function; a dropout layer with a rate of 0.2; and a dense layer with a prediction window size of units and the ReLU activation function. In the GRU-CNN model, the GRU layer returns a sequence, and the one-dimensional global max-pooling layer takes only important features and reduces the dimension of the feature map. The architecture of the proposed GRU-CNN model is illustrated in Figure 1, while the configuration is listed in Table 1.

# 3.1.3. Proposed Ensemble Model

While evaluating the performance of the benchmark models, various RNN models, such as RNN, LSTM, and GRU, exhibited high predictive performance on different types of datasets. There are three types of widely employed ensemble learning strategies: ensemble averaging, bagging, and stacking. Based on the results of the benchmarks, the CNN-LSTM, and the GRU-CNN as implemented above, we propose an average ensemble of three RNNbased models to achieve averaged high performance for various datasets. The proposed ensemble model can utilize the representations of the RNN, LSTM, and GRU models. The parameters used for the ensemble model were similar to those of the CNN-LSTM and GRU-CNN models, as described in Section 3.1.1.

The ensemble model consists of an RNN layer with 128 units and the tanh activation function; an LSTM layer with 128 units and the tanh activation function; a GRU layer with 128 units and the tanh activation function; followed by taking the average of all the hidden states from RNN, LSTM, and GRU; a dropout layer with a rate of 0.2; a dense layer with 32 units and the ReLU activation function; and a dense layer with a prediction window size of units and the ReLU activation function. Figure 1 illustrates the details of each layer of the proposed ensemble model, while Table 1 presents the configuration.

# 3.2. Implementation Details

In this subsection, we present an extensive empirical analysis of the proposed models on three datasets. First, we describe the datasets and the experimental setting used to demonstrate the validity of our financial time-series prediction models. Next, we evaluate the performance of our models on several datasets and compare them with those of conventional deep learning models.

# 3.2.1. Dataset

We evaluated the performance of the proposed models on daily stock market indices to verify the robustness of our models. We considered three stock market indices from major stock markets listed below.

(1) DAX: Deutscher Aktienindex, which is a stock market index consisting of the 40 (expanded from 30 in 2021) major German blue-chip companies trading on the Frankfurt stock exchange.   
(2) DOW: Dow Jones Industrial Average, which is a stock market index of 30 prominent companies in the United States.   
(3) S&P500: Standard and Poor’s 500, which is a stock market index of 500 large companies in the United States.

The DOW is the most influential and widely used stock market index in the literature. We considered three types of periods for all three indices: the period from 1 January 2000 through 31 December 2019 for DOW and S&P500 and from 24 October 2014 through 31 December 2019 for DAX as long periods; from 1 January 2017 through 31 December 2019 and from 1 January 2019 through 31 December 2021 as short periods before and after the COVID-19 pandemic, respectively.

The historical prices of each stock market index were obtained using the Finance-DataReader open-source library available in the pandas DataReader module of the Python programming language [56]. The raw data included six features: open, high, low, and close prices, trading volume, and change. The incomplete data were removed.

Before feeding the raw data into our models, we pre-processed the data. We normalized the raw data using Scikit-learn’s MinMaxScaler tool, as follows:

$$
x = \frac { x - x _ { m a x } } { x _ { m a x } - x _ { m i n } } ,
$$

where $x$ is the input feature of the stock market index and $x _ { m a x }$ and $x _ { m i n }$ are the maximum and minimum values of each input feature, respectively. Granger [57] suggested holding approximately $2 0 \%$ of the data for out-of-sample testing. Following this suggestion, the first $8 0 \%$ of the data were used as the training set for in-sample training, while the remaining $2 0 \%$ were used as the test set, to ensure that our models were evaluated on unseen outof-sample data. The first $9 0 \%$ of the training set was used to train the network and to iteratively adjust its parameters such that the loss function was minimized. The trained network predicted the remaining $1 0 \%$ for validation, and the validation loss was computed after each epoch.

# 3.2.2. Generation of the Inputs and Outputs Using the Sliding Window Technique

This subsection describes the generation of the inputs and outputs. The daily open, high, and low prices, trading volume, and change were commonly used as input features in other studies. However, in the current study, we introduce a novel feature named medium, which is the average of high and low prices, to reduce the influence of the unusually extreme highest and lowest prices and to ensure generalizability.

For each stock market index, the partial features of daily open, high, low, and medium prices, trading volume, and change (OHLMVC) were used as the input to train the model, and the daily close prices were used as the output to predict one-time-step and multi-timestep ahead.

For the input and output generation, the normalized data were segmented using the sliding window technique, by which a fixed window size of time-series data was chosen as the input and a fixed number of the following observations was chosen as the output. This process was repeated for the entire dataset by sliding the window in intervals of one time step to obtain the next input and output. We trained the proposed models to look at $m$ consecutive past data of features. The input at time $t$ was denoted by

$$
\begin{array} { r } { \mathbf { X } _ { t } = \left( \mathbf { x } _ { t } ^ { O } , \mathbf { x } _ { t } ^ { H } , \mathbf { x } _ { t } ^ { L } , \mathbf { x } _ { t } ^ { M } , \mathbf { x } _ { t } ^ { V } , \mathbf { x } _ { t } ^ { C h } \right) \in \mathbb { R } ^ { m \times 6 } , } \end{array}
$$

where for each $k \in \{ O , H , L , M , V , C h \} ,$

$$
\mathbf { x } _ { t } ^ { k } = \left( x _ { t - m + 1 } ^ { k } , \cdot \cdot \cdot , x _ { t - 1 } ^ { k } , x _ { t } ^ { k } \right) ^ { T } \in \mathbb { R } ^ { m } ,
$$

$\mathbf { x } ^ { O } , \mathbf { x } ^ { H } , \mathbf { x } ^ { L } , \mathbf { x } ^ { M } , \mathbf { x } ^ { V } ,$ and $\mathbf { x } ^ { C h }$ are the daily open, high, low, and medium prices, trading volume, and change from time $t - m + 1$ to time $t ,$ , respectively.

The input $\mathbf { \boldsymbol { x } } _ { t }$ was fed sequentially into the proposed models to predict the following $n$ daily close prices of stock market indices, with the output denoted by

$$
\mathbf { y } _ { t + 1 } ^ { C } = \left( y _ { t + 1 } , y _ { t + 2 } , . . . y _ { t + n } \right) ^ { T } \in \mathbb { R } ^ { n } .
$$

The look-back periods of 5, 21, and 42 days were considered as one week, one month, and two months, respectively; while the look-ahead periods of one and five days were considered to predict the future one-time-step or multi-time-step ahead. Figure 2 illustrates the sliding window technique.

![](images/dc0c671fe0b343e0e61e48ba1bb93a757adae1ee1816a1875f887d4b5ddf9bad.jpg)  
Figure 2. Sliding window technique.

# 3.2.3. Software and Hardware

The proposed models were implemented, trained, and analyzed in Python 3.7.6 [58] with the Keras library 2.4.3 [59] as a high-level neural network API using TensorFlow 2.3.1 as back-end [60], relying on NumPy 1.19.2 [61], Pandas 0.25.3 [56], and Scikit-learn 1.0.2 [62]. The code used for producing the figures and analysis is available on GitHub at https://github.com/hyunsunsong/Project.

All experiments were performed using a workstation equipped with an Intel Xeon Silver 4208 CPU at 2.10 GHz x8, Nvidia GPU TITAN, and 12 GB RAM on each board.

# 3.2.4. Experimental Setting

The proposed models were trained with the Huber loss function, which combines the characteristics of MSE and MAE and is less susceptible to outliers in the data than the MSE loss function [63]. It behaves quadratically for small residuals and linearly for large residuals [64]. The parameters of the network were learned to minimize the average of the Huber loss function over the entire training dataset.

The network weights and biases were initialized with the Glorot–Xavier uniform method and zeros, respectively. Glorot and Bengio [65] proposed the Glorot–Xavier uniform method to adopt a properly scaled uniform distribution for initialization.

The successful applications of neural networks require regularization [66]. Introduced by [23], the dropout regularization technique randomly drops a fraction of the units with a specified probability, along with connections during training, while all units are presented during testing. We applied the dropout values of 0.2 and 0.5 to reduce overfitting and have observed that higher dropout value result in a decline in performance. Therefore, we settled on the relatively low dropout value of 0.2 as studied in [67].

The batch size and maximum number of epochs were set to 32 and 50, respectively, and an early stopping patience of 10 was applied [68]. That is, once the validation loss no longer decreased for the patience period, the training was stopped, and the weights of the model with the lowest validation loss were restored using ModelCheckpoint callback in the Keras library [59].

The optimization algorithms used for training were the Adam and root mean square propagation (RMSProp) [69], which are adaptive learning rate methods. The RMSProp is usually a viable choice for RNNs [59]. We compared the performance of the proposed models using two different optimizers.

We applied learning rates of 0.001 and 0.0005 and found that a learning rate of 0.0005 resulted in a better performance. Therefore, the learning rate was set to 0.0005.

The ReLU activation function proposed in [22] was used for the dense layers, and the data shuffling technique was not used during training.

# 3.2.5. Predictive Performance Metrics

In this study, we adopted the MSE and MAE as evaluation metrics to compare the performance of the proposed models with that of conventional benchmark models for forecasting time-series data, which are calculated as follows:

$$
\mathrm { M S E } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } ( y _ { t } - \hat { y _ { t } } ) ^ { 2 } ,
$$

$$
\mathrm { M A E } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \lvert y _ { t } - \hat { y _ { t } } \rvert ,
$$

where $T$ is the number of prediction time horizons; $y _ { t }$ and $\hat { y _ { t } }$ are the true and predicted values, respectively, during one-time-step prediction. During multi-time-step prediction, we only used the value of the last time step; thus, $y _ { t }$ and $\hat { y _ { t } }$ represent the true and predicted values of the last time step, respectively.

# 4. Experimental Results

In this section, we present the experimental results of the proposed models using historical time-series data for three stock market indices: DAX, DOW, and S&P500. We first describe the details of the benchmark models used for comparison. Second, we compare the results for the proposed models and conventional benchmarks with respect to one-time-step and multi-time-step predictions on three datasets over three different periods. Third, we present the results of the impact of different features and optimizers on the performance of the proposed models.

# 4.1. Benchmark Models

For benchmark comparison, we deploy several conventional deep learning models, such as RNN, LSTM, and GRU, to examine whether the proposed models outperform the benchmarks. In addition, we utilize WaveNet, which combines causal filters with dilated convolutions, so that the model learns long-range temporal dependencies in time-series data [70]. The benchmark models and corresponding architectures are listed below.

1. RNN: Two RNN layers with 128 units and a dense layer with a look-ahead period of units;   
2. LSTM: An LSTM layer with 128 units and a dense layer with a look-ahead period of units;   
3. GRU: A GRU layer with 128 units and a dense layer with a look-ahead period of units;   
4. WaveNet: A simpler architecture of an audio generative model based on Pixel-CNN [71], as described in [70].

Table 2 lists the training setting for the benchmark models. All benchmark models were trained with 50 epochs, an early stopping patience of 10, a learning rate of 0.0005, a batch size of 32, the MSE loss function, the Adam optimizer, and the ReLU activation function.

Table 2. Hyperparameter setting for the benchmark models.   

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Number of epochs</td><td>50</td></tr><tr><td>Early stopping patience</td><td>10</td></tr><tr><td>Learning rate</td><td>0.0005</td></tr><tr><td>Batch size</td><td>32</td></tr><tr><td>Loss function</td><td>MSE</td></tr><tr><td>Optimizer</td><td>Adam</td></tr><tr><td>Activation function</td><td>ReLU</td></tr></table>

# 4.2. One-Time-Step Prediction Comparisons between Proposed and Benchmark Models

In this subsection, we provide the experimental results of the proposed models to predict the one-time-step ahead of the three stock market indices. We evaluated the performance of the proposed models with various look-back periods of 5, 21, and 42 days as one week, one month, and two months, respectively, for different periods. The proposed and benchmark models were implemented as described in previous sections. The Adam optimizer and OHLV features were used for all methods in Table 3.

Table 3. Comparison of one-time-step prediction between proposed and benchmark models.   

<table><tr><td rowspan="2">Look-Back Period</td><td rowspan="2">Metric</td><td rowspan="2">Model</td><td colspan="3">2000-2019 1</td><td colspan="3">20172019</td><td colspan="3">20192021</td></tr><tr><td>DAX</td><td>DOW</td><td>S&amp;P500</td><td>DAX</td><td>DOW</td><td>S&amp;P500</td><td>DAX</td><td>DOW</td><td>S&amp;P500</td></tr><tr><td rowspan="9">5 days</td><td rowspan="9">MSE</td><td>RNN LSTM</td><td>0.1505 0.1505</td><td>0.5739 0.5739</td><td>0.5618 0.5618</td><td>0.1146 0.1146</td><td>0.6942 0.6942</td><td>0.6174 0.6174</td><td>0.7626 0.7626</td><td>0.8576 0.8576</td><td>0.7843 0.7843</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GRU</td><td>0.1505</td><td>0.5739</td><td>0.0004</td><td>0.0012</td><td>0.6942</td><td>0.6174</td><td>0.7626</td><td>0.8576</td><td>0.0049</td></tr><tr><td>WaveNet</td><td>0.4040</td><td>0.0940</td><td>0.0886</td><td>0.4929</td><td>0.0411</td><td>0.0672</td><td>0.0177</td><td>0.0075</td><td>0.0189</td></tr><tr><td>CNN-LSTM</td><td>0.0079</td><td>0.0004</td><td>0.0040</td><td>0.0539</td><td>0.0154</td><td>0.1333</td><td>0.0868</td><td>0.0045</td><td>0.0032</td></tr><tr><td>GRU-CNN</td><td>0.0011</td><td>0.0132</td><td>0.0042</td><td>0.0014</td><td>0.0069</td><td>0.0135</td><td>0.0111</td><td>0.0103</td><td>0.7843</td></tr><tr><td>Ensemble</td><td>0.0017</td><td>0.0075</td><td>0.0059</td><td>0.0012</td><td>0.0023</td><td>0.0011</td><td>0.0009</td><td>0.0029</td><td>0.0003</td></tr><tr><td>RNN</td><td>0.3756</td><td>0.7410</td><td>0.7376</td><td>0.3207</td><td>0.8282</td><td>0.7779</td><td>0.8717</td><td>0.9253</td><td>0.8834</td></tr><tr><td>LSTM</td><td>0.3756</td><td>0.7410</td><td>0.7376</td><td>0.3207</td><td>0.8282</td><td>0.7779</td><td>0.8717</td><td>0.9253</td><td>0.8834</td></tr><tr><td rowspan="6">MAE</td><td>GRU</td><td>0.3756</td><td>0.7410</td><td>0.0185</td><td>0.0273</td><td>0.8282</td><td>0.7779</td><td>0.8717</td><td>0.9253</td><td>0.0674</td></tr><tr><td>WaveNet</td><td>0.6284</td><td>0.2645</td><td>0.2680</td><td>0.6942</td><td>0.1841</td><td>0.2384</td><td>0.1224</td><td>0.0781</td><td>0.1235</td></tr><tr><td>CNN-LSTM</td><td>0.0827</td><td>0.0162</td><td>0.0600</td><td>0.2120</td><td>0.1188</td><td>0.3520</td><td>0.2909</td><td>0.0650</td><td>0.0544</td></tr><tr><td>GRU-CNN</td><td>0.0255</td><td>0.1031</td><td>0.0593</td><td>0.0279</td><td>0.0779</td><td>0.1113</td><td>0.1022</td><td>0.0991</td><td>0.8834</td></tr><tr><td>Ensemble</td><td>0.0336</td><td>0.0740</td><td>0.0651</td><td>0.0262</td><td>0.0418</td><td>0.0279</td><td>0.0244</td><td>0.0468</td><td>0.0143</td></tr><tr><td>RNN</td><td>0.1509</td><td>0.5058</td><td>0.5341</td><td>0.0982</td><td>0.6552</td><td>0.6482</td><td>0.7252</td><td>0.8119</td><td>0.8070</td></tr><tr><td rowspan="9">MSE 21 days</td><td rowspan="9"></td><td>LSTM</td><td>0.1658</td><td>0.5734</td><td>0.5602</td><td>0.1066</td><td>0.7420</td><td>0.6781</td><td>0.7555</td><td>0.8855</td><td>0.8273</td></tr><tr><td>GRU</td><td>0.1658</td><td>0.5734</td><td>0.0002</td><td>0.0008</td><td>0.7420</td><td>0.6781</td><td>0.7555</td><td>0.8855</td><td>0.8273</td></tr><tr><td>WaveNet</td><td>0.3877</td><td>0.0903</td><td>0.0856</td><td>0.4931</td><td>0.0371</td><td>0.0605</td><td>0.0196</td><td>0.0065</td><td>0.0154</td></tr><tr><td>CNN-LSTM</td><td>0.0078</td><td>0.0159</td><td>0.0033</td><td>0.0632</td><td>0.0148</td><td>0.1074</td><td>0.1507</td><td>0.1950</td><td>0.2035</td></tr><tr><td>GRU-CNN</td><td>0.0014</td><td>0.0356</td><td>0.0149</td><td>0.1066</td><td>0.0170</td><td>0.0193</td><td>0.0090</td><td>0.0033</td><td>0.0018</td></tr><tr><td>Ensemble</td><td>0.0008</td><td>0.0007</td><td>0.0014</td><td>0.0009</td><td>0.0023</td><td>0.0011</td><td>0.0008</td><td>0.0011</td><td>0.0004</td></tr><tr><td></td><td>0.3769</td><td></td><td></td><td>0.2927</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>RNN</td><td>0.3965</td><td>0.6934 0.7418</td><td>0.7188 0.7375</td><td>0.3074</td><td>0.8025 0.8559</td><td>0.7941 0.8145</td><td>0.8497 0.8680</td><td>0.8987 0.9401</td><td>0.8959</td></tr><tr><td>LSTM GRU</td><td>0.3965</td><td>0.7418</td><td>0.0136</td><td>0.0222</td><td>0.8559</td><td>0.8145</td><td>0.8680</td><td>0.9401</td><td>0.9074 0.9074</td></tr><tr><td>WaveNet</td><td>0.6166</td><td>0.2596</td><td>0.2639</td><td>0.6936</td><td>0.1736</td><td>0.2251</td><td>0.1320</td><td>0.0722</td><td>0.1120</td></tr><tr><td rowspan="5">MAE</td><td></td><td>0.0742</td><td>0.1178</td><td>0.0473</td><td>0.2296</td><td>0.1177</td><td>0.3223</td><td>0.3865</td><td>0.4398</td><td>0.4479</td></tr><tr><td>CNN-LSTM</td><td>0.0290</td><td>0.1741</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GRU-CNN</td><td>0.0211</td><td>0.0220</td><td>0.1135</td><td>0.3074</td><td>0.1224</td><td>0.1317</td><td>0.0888</td><td>0.0542</td><td>0.0384</td></tr><tr><td>Ensemble</td><td></td><td></td><td>0.0312</td><td>0.0240</td><td>0.0426</td><td>0.0280</td><td>0.0201</td><td>0.0281</td><td>0.0160</td></tr><tr><td>RNN LSTM</td><td>0.1619</td><td>0.4885</td><td>0.4736</td><td>0.1147 0.1228</td><td>0.5921</td><td>0.5475</td><td>0.5888</td><td>0.7551</td><td>0.6797</td></tr><tr><td rowspan="9">42 days</td><td></td><td>0.1683</td><td>0.5904</td><td>0.5766</td><td></td><td>0.7307</td><td>0.6683</td><td>0.7352</td><td>0.8806</td><td>0.8292</td></tr><tr><td>GRU</td><td>0.1683</td><td>0.5904</td><td>0.0013</td><td>0.0009</td><td>0.7307</td><td>0.6683</td><td>0.7352</td><td>0.8806</td><td>0.8292</td></tr><tr><td>WaveNet</td><td>0.3732</td><td>0.0856</td><td>0.0816</td><td>0.5011</td><td>0.0378</td><td>0.0595</td><td>0.0211</td><td>0.0058</td><td>0.0125</td></tr><tr><td>CNN-LSTM</td><td>0.0025</td><td>0.0342</td><td>0.0210</td><td>0.0496</td><td>0.0783</td><td>0.0392</td><td>0.0833</td><td>0.0759</td><td>0.0004</td></tr><tr><td>GRU-CNN</td><td>0.0035</td><td>0.0459</td><td>0.0220</td><td>0.0075</td><td>0.0128</td><td>0.0128</td><td>0.0019</td><td>0.0712</td><td>0.0270</td></tr><tr><td>Ensemble RNN</td><td>0.0007 0.3940</td><td>0.0012</td><td>0.0007</td><td>0.0015</td><td>0.0013</td><td>0.0039</td><td>0.0009</td><td>0.0004</td><td>0.0007</td></tr><tr></table>

1 The period from 24 October 2014 through 31 December 2019 for the DAX.

Table 3 compares our models and the benchmark models for the different look-back periods for one-time-step prediction, where the best performance results are marked in bold for each stock market index, period, and metric.

According to the results in Table 3, increasing the look-back period slightly enhances the performance across all operating conditions by keeping all other hyperparameters constant. Moreover, a very long sequence length, such as the look-back period of 42, increases the performance. From 1 January 2000 through 31 December 2019, the proposed models improved the benchmarks in $7 7 . 8 \%$ and $7 7 . 8 \%$ of cases in terms of MSE and MAE, respectively. Additionally, our models outperformed the benchmarks in $6 6 . 7 \%$ and $7 7 . 8 \%$ of cases in terms of MSE and MAE, respectively, for the period from 1 January 2017 through 31 December 2019, before the COVID-19 pandemic, and in $1 0 0 \%$ of cases for the period from 1 January 2019 through 31 December 2021, after the COVID-19 pandemic, in terms of both MSE and MAE. Some results were the same for different benchmarks because the training algorithm might find the local optima. Further, an overall comparison between the ensemble model and other models in Table 3 indicates that the ensemble model significantly outperformed the other models.

We evaluated the performance of the proposed models with four different features (i.e., MV, MVC, OHLV, and OHLMVC) in addition to a novel feature, medium, defined as the average of high and low prices. In comparison, OHLVs have been commonly used as features in other studies.

In addition, we evaluated the performance of our models using two different optimizers, Adam and RMSProp, by keeping all other hyperparameters constant. The average MSE and MAE over the three periods for the impact of different features and optimizers of the proposed models are shown in Tables 4 and 5, where the best performance results are marked in bold for each stock market index, a look-back period, and optimizer.

Table 4. Comparison of different optimizers and features in terms of average MSE over the three periods for one-time-step prediction.   

<table><tr><td rowspan="2">Look-Back Period</td><td rowspan="2">Optimizer</td><td rowspan="2">Feature</td><td colspan="3">CNN-LSTM</td><td colspan="3">GRU-CNN</td><td colspan="3">Ensemble</td></tr><tr><td>DAX</td><td>DOW</td><td>S&amp;P500</td><td>DAX</td><td>DOW</td><td>S&amp;P500</td><td>DAX</td><td>DOW</td><td>S&amp;P500</td></tr><tr><td rowspan="8">5 days</td><td rowspan="4">Adam</td><td>MV</td><td>0.0406</td><td>0.0228</td><td>0.0154</td><td>0.0025</td><td>0.0070</td><td>0.2717</td><td>0.0014</td><td>0.0039</td><td>0.0026</td></tr><tr><td>MVC</td><td>0.0534</td><td>0.0366</td><td>0.0097</td><td>0.0026</td><td>0.0054</td><td>0.0053</td><td>0.0014</td><td>0.0039</td><td>0.0026</td></tr><tr><td>OHLV OHLMVC</td><td>0.0495</td><td>0.0068</td><td>0.0468</td><td>0.0045</td><td>0.0101</td><td>0.2674</td><td>0.0013</td><td>0.0042</td><td>0.0025</td></tr><tr><td></td><td>0.0496</td><td>0.0025</td><td>0.0030</td><td>0.0070</td><td>0.0096</td><td>0.2668</td><td>0.0013</td><td>0.0042</td><td>0.0025</td></tr><tr><td rowspan="4">RMSProp</td><td>MV</td><td>0.0231</td><td>0.0239</td><td>0.0189</td><td>0.0023</td><td>0.0088</td><td>0.0063</td><td>0.0013</td><td>0.0031</td><td>0.0014</td></tr><tr><td>MVC</td><td>0.0972</td><td>0.0796</td><td>0.0270</td><td>0.0035</td><td>0.0197</td><td>0.0171</td><td>0.0013</td><td>0.0031</td><td>0.0014</td></tr><tr><td>OHLV</td><td>0.0175</td><td>0.0133</td><td>0.0206</td><td>0.0019</td><td>0.0144</td><td>0.2223</td><td>0.0013</td><td>0.0031</td><td>0.0014</td></tr><tr><td>OHLMVC</td><td>0.0330</td><td>0.0152</td><td>0.0396</td><td>0.0043</td><td>0.2513</td><td>0.0107</td><td>0.0013</td><td>0.0031</td><td>0.0014</td></tr><tr><td rowspan="8">21 days</td><td rowspan="4">Adam</td><td>MV</td><td>0.0174</td><td>0.0454</td><td>0.0217</td><td>0.0070</td><td>0.0133</td><td>0.0180</td><td>0.0010</td><td>0.0029</td><td>0.0011</td></tr><tr><td>MVC</td><td>0.1042</td><td>0.0489</td><td>0.1143</td><td>0.0030</td><td>0.0141</td><td>0.0115</td><td>0.0021</td><td>0.0040</td><td>0.0080</td></tr><tr><td>OHLV</td><td>0.0739</td><td>0.0752</td><td>0.1047</td><td>0.0390</td><td>0.0186</td><td>0.0120</td><td>0.0008</td><td>0.0013</td><td>0.0010</td></tr><tr><td>OHLMVC</td><td>0.0406</td><td>0.0228</td><td>0.0154</td><td>0.0054</td><td>0.0253</td><td>0.0138</td><td>0.0021</td><td>0.0040</td><td>0.0080</td></tr><tr><td rowspan="4">RMSProp</td><td>MV</td><td>0.0295</td><td>0.1721</td><td>0.2440</td><td>0.0163</td><td>0.0183</td><td>0.0242</td><td>0.0010</td><td>0.0042</td><td>0.0060</td></tr><tr><td>MVC</td><td>0.0318</td><td>0.0608</td><td>0.1752</td><td>0.0395</td><td>0.0306</td><td>0.0178</td><td>0.0017</td><td>0.0185</td><td>0.0038</td></tr><tr><td>OHLV</td><td>0.0477</td><td>0.1299</td><td>0.3043</td><td>0.0157</td><td>0.0229</td><td>0.0137</td><td>0.0008</td><td>0.0014</td><td>0.0015</td></tr><tr><td>OHLMVC</td><td>0.0406</td><td>0.0228</td><td>0.0154</td><td>0.0284</td><td>0.0244</td><td>0.0253</td><td>0.0021</td><td>0.0040</td><td>0.0080</td></tr><tr><td rowspan="8">42 days</td><td rowspan="4">Adam</td><td>MV</td><td>0.0443</td><td>0.0790</td><td>0.0105</td><td>0.0163</td><td>0.0169</td><td>0.2459</td><td>0.0014</td><td>0.0028</td><td>0.0022</td></tr><tr><td>MVC</td><td>0.0566</td><td>0.0585</td><td>0.0426</td><td>0.0227</td><td>0.0271</td><td>0.2339</td><td>0.0010</td><td>0.2455</td><td>0.0129</td></tr><tr><td>OHLV</td><td>0.0451</td><td>0.0628</td><td>0.0202</td><td>0.0043</td><td>0.0433</td><td>0.0206</td><td>0.0010</td><td>0.0010</td><td>0.0018</td></tr><tr><td>OHLMVC</td><td>0.0488</td><td>0.1413</td><td>0.0102</td><td>0.0088</td><td>0.0126</td><td>0.0124</td><td>0.0019</td><td>0.0026</td><td>0.0012</td></tr><tr><td rowspan="4">RMSProp</td><td>MV</td><td>0.1339</td><td>0.0792</td><td>0.1816</td><td>0.0084</td><td>0.0503</td><td>0.0223</td><td>0.0017</td><td>0.0036</td><td>0.0035</td></tr><tr><td>MVC</td><td>0.0659</td><td>0.2908</td><td>0.1823</td><td>0.0191</td><td>0.0609</td><td>0.3112</td><td>0.0012</td><td>0.0057</td><td>0.0049</td></tr><tr><td>OHLV</td><td>0.3795</td><td>0.3173</td><td>0.0588</td><td>0.0079</td><td>0.0394</td><td>0.0341</td><td>0.0008</td><td>0.0021</td><td>0.2784</td></tr><tr><td>OHLMVC</td><td>0.0496</td><td>0.0607</td><td>0.6410</td><td>0.0100</td><td>0.0256</td><td>0.0229</td><td>0.0010</td><td>0.0021</td><td>0.0017</td></tr></table>

Regarding the one-time-step prediction, Table 4 shows that the CNN-LSTM, GRU-CNN, and ensemble models with the novel medium feature outperformed the other models in $8 3 . 3 \%$ , $3 3 . 3 \%$ , and $0 \%$ of cases with the DAX dataset; $8 3 . 3 \%$ , $1 0 0 \%$ , and $1 6 . 7 \%$ of cases with the DOW dataset; and $8 3 . 3 \%$ , $8 3 . 3 \%$ , and $3 3 . 3 \%$ of cases with the ${ \sf S } \& { \sf P } { \sf 5 0 0 }$ dataset, respectively, in terms of the average MSE over the three periods.

Table 5 shows that the CNN-LSTM, GRU-CNN, and ensemble models incorporating the medium feature outperformed the other models in $8 3 . 3 \%$ , $3 3 . 3 \%$ , and $1 6 . 7 \%$ of cases with the DAX dataset; $8 3 . 3 \%$ , $1 0 0 \%$ , and $1 6 . 7 \%$ of cases with the DOW dataset; and $6 6 . 7 \%$ $6 6 . 7 \%$ and $3 3 . 3 \%$ of cases with the S&P500 dataset, respectively, in terms of the average MAE over the three periods.

An overall comparison between the models incorporating the medium feature and the models without the medium feature shows that adding the medium feature improves the performances of all models.

Table 5. Comparison of different optimizers and features in terms of average MAE over the three periods for one-time-step prediction.   

<table><tr><td rowspan="2">Look-Back Period</td><td rowspan="2">Optimizer</td><td rowspan="2">Feature</td><td colspan="3">CNN-LSTM</td><td colspan="3">GRU-CNN</td><td colspan="3">Ensemble</td></tr><tr><td>DAX</td><td>DOW</td><td>S&amp;P500</td><td>DAX</td><td>DOW</td><td>S&amp;P500</td><td>DAX</td><td>DOW</td><td>S&amp;P500</td></tr><tr><td rowspan="6">5 days</td><td rowspan="3">Adam</td><td>MV MVC</td><td>0.1635 0.2047</td><td>0.1285 0.1433</td><td>0.1161 0.0810</td><td>0.0405 0.0413</td><td>0.0778 0.0640</td><td>0.3662</td><td>0.0293 0.0259</td><td>0.0460 0.0578</td><td>0.0391 0.0423</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>0.0584</td><td></td><td></td><td></td></tr><tr><td>OHLV</td><td>0.1952 0.1806</td><td>0.0667</td><td>0.1554</td><td>0.0519 0.0639</td><td>0.0934</td><td>0.3513</td><td>0.0280</td><td>0.0542</td><td>0.0358</td></tr><tr><td></td><td>OHLMVC</td><td>0.0421</td><td>0.0466</td><td></td><td>0.0860</td><td>0.3503</td><td>0.0227</td><td>0.0373</td><td>0.2706</td></tr><tr><td rowspan="3">RMSProp</td><td>MV</td><td>0.1344</td><td>0.1391</td><td>0.1186</td><td>0.0375</td><td>0.0765</td><td>0.0690</td><td>0.0288</td><td>0.0426</td><td>0.0311</td></tr><tr><td>MVC</td><td>0.2370</td><td>0.2125</td><td>0.1433</td><td>0.0472</td><td>0.0972</td><td>0.1134</td><td>0.0295</td><td>0.0556</td><td>0.0582</td></tr><tr><td>OHLV OHLMVC</td><td>0.1057 0.1438</td><td>0.0813 0.1012</td><td>0.0999 0.1770</td><td>0.0353 0.0578</td><td>0.0926 0.3581</td><td>0.3573 0.0872</td><td>0.0230 0.0284</td><td>0.0328 0.0666</td><td>0.0279 0.0545</td></tr><tr><td rowspan="6">21 days</td><td rowspan="3">Adam</td><td>MV</td><td>0.0810</td><td>0.2076</td><td>0.1332</td><td>0.0679</td><td>0.1048</td><td>0.1066</td><td>0.0248</td><td>0.0439</td><td>0.0268</td></tr><tr><td>MVC</td><td>0.2429</td><td>0.1804</td><td>0.2890</td><td>0.0455</td><td>0.1116</td><td>0.0979</td><td>0.0340</td><td>0.0496</td><td>0.0603</td></tr><tr><td>OHLV</td><td>0.2301</td><td>0.2251</td><td>0.2725</td><td>0.1417</td><td>0.1169</td><td>0.0945</td><td>0.0217</td><td>0.0309</td><td></td></tr><tr><td>OHLMVC</td><td>0.1635</td><td>0.1285</td><td>0.1161</td><td></td><td>0.0616</td><td></td><td></td><td>0.0340</td><td>0.0496</td><td>0.0251 0.0603</td></tr><tr><td rowspan="3">RMSProp</td><td></td><td></td><td></td><td></td><td></td><td>0.1519</td><td>0.1057</td><td></td><td></td><td></td></tr><tr><td>MV</td><td>0.1454 0.1376</td><td>0.3760</td><td>0.3348</td><td>0.1017</td><td>0.1177</td><td>0.1387</td><td>0.0236</td><td>0.0526</td><td>0.0645</td></tr><tr><td>MVC OHLV</td><td>0.1783</td><td>0.2006 0.3155</td><td>0.3381 0.4366</td><td>0.1432 0.0902</td><td>0.1485 0.1353</td><td>0.1213 0.0930</td><td>0.0342 0.0216</td><td>0.0940 0.0291</td><td>0.0542 0.0328</td></tr><tr><td rowspan="6">42 days</td><td rowspan="3">Adam</td><td>OHLMVC</td><td>0.1635</td><td>0.1285</td><td>0.1161</td><td>0.1149</td><td>0.1475</td><td>0.1484</td><td>0.0340</td><td>0.0496</td><td>0.0603</td></tr><tr><td>MV</td><td>0.1664</td><td>0.2570</td><td>0.0850</td><td>0.1085</td><td>0.1189</td><td>0.3836</td><td>0.0284</td><td>0.0421</td><td>0.0334</td></tr><tr><td>MVC</td><td>0.2052</td><td>0.1808</td><td>0.1750</td><td>0.1227</td><td>0.1499</td><td>0.3378</td><td>0.0245</td><td>0.3110</td><td>0.0719</td></tr><tr><td>OHLV OHLMVC</td><td>0.1730</td><td>0.2454</td><td>0.1164</td><td></td><td>0.0523</td><td>0.1892</td><td>0.1361</td><td>0.0241</td><td>0.0263</td><td>0.0330</td></tr><tr><td rowspan="3">RMSProp</td><td></td><td>0.1970</td><td>0.3401</td><td>0.0741</td><td>0.0816</td><td>0.1013</td><td>0.1014</td><td>0.0336</td><td>0.0409</td><td>0.0275</td></tr><tr><td>MV</td><td>0.3173</td><td>0.2390</td><td>0.2914</td><td>0.0753</td><td>0.2114</td><td>0.1191</td><td>0.0330</td><td>0.0510</td><td>0.0512</td></tr><tr><td>MVC OHLV</td><td>0.2235 0.4822</td><td>0.5177 0.5251</td><td>0.4117 0.2263</td><td>0.1108 0.0692</td><td>0.2255 0.1869</td><td>0.4436 0.1710</td><td>0.0259 0.0211</td><td>0.0607 0.0371</td><td>0.0615 0.3333</td></tr></table>

In addition, the proposed models were trained for 1500 epochs with the RMSProp optimizer and MV features to achieve higher performance than that of the model trained as described in Section 3.2.4. Figures 3–5 compare the actual and predicted close prices of the DAX, DOW, and S&P500 indices, respectively, for the different look-back periods. In Figures 3–5, the left, middle, and right plots correspond to the look-back periods of 5, 21, and 42 days, respectively. Further, the look-back period and stock market index evidently affect the model performance.

![](images/5be146d60a8391dabd5fc71c767ae9b7c7bb4ccf360b8b3571e861e311cf6d38.jpg)  
Figure 3. Comparison of true and predicted close prices of the DAX index between different look-back periods for one-time-step prediction over the period from 24 October 2014 through 31 December 2019.

![](images/b05c0ec0934266f7a73043cf22306234ddc6cd9c18a5f900d19712cefd88a685.jpg)  
Figure 4. Comparison of true and predicted close prices of the DOW index between different look-back periods for one-time-step prediction over the period from 1 January 2000 through 31 December 2019.

![](images/a8f9d8a9fc634b3fa7b612431f050d4f344f88a0fba6614f0f2c9cf1ee2909db.jpg)  
Figure 5. Comparison of true and predicted close prices of the S&P500 index between different look-back periods for one-time-step prediction over the period from 1 January 2000 through 31 December 2019.

Moreover, the proposed models were trained for 1500 epochs with the Adam optimizer and a look-back period of 5 days. The comparisons of true and predicted close prices of the DAX, DOW, and S&P500 indices between different input features for one-time-step prediction are provided in Figures 6–8, respectively.

![](images/ca1d2fe94f8e5ff509d83d3a9c8422f47a528e2b550550c3a3c011f5523290a8.jpg)  
Figure 6. Comparison of true and predicted close prices of the DAX index between different input features for one-time-step prediction over the period from 24 October 2014 through 31 December 2019.

![](images/03649901815d37dd50c55212d488a9728b95c5f67c9f46e848c07853b7c8cc58.jpg)  
Figure 7. Comparison of true and predicted close prices of the DOW index between different input features for one-time-step prediction over the period from 1 January 2000 through 31 December 2019.

![](images/b628ed7d435ccdf870f000acb1e3ab53d993d79671b9b53e85e95a65221c4c78.jpg)  
Figure 8. Comparison of true and predicted close prices of the ${ \mathrm { S } } \& { \mathrm { P } } { \mathrm { 5 0 0 } }$ index between different input features for one-time-step prediction over the period from 1 January 2000 through 31 December 2019.

# 4.3. Multi-Time-Step Prediction Comparisons between Proposed and Benchmark Models

In this subsection, we evaluated the performance of the proposed models with various look-back periods and provided the experimental results to predict multi-time-step ahead for the three stock market indices. The look-back periods of 5, 21, and 42 days and the look-ahead period of five days were adopted for each period. The proposed and benchmark models were implemented as described in previous sections. The Adam optimizer and OHLV features were used for all methods in Table 6.

Table 6 compares the proposed and benchmark models in terms of different look-back periods for five-time-step prediction, where the best performance results are marked in bold for each stock market index, period, and metric.

From the table, the proposed models outperformed the benchmarks in $6 6 . 7 \%$ and $6 6 . 7 \%$ of cases for the period from 1 January 2000 through 31 December 2019; in $2 2 . 2 \%$ and $1 1 . 1 \%$ of cases for the period from 1 January 2017 through 31 December 2019, before the COVID-19 pandemic; and in $5 5 . 6 \%$ and $5 5 . 6 \%$ of cases for the period from 1 January 2019 through 31 December 2021, after the COVID-19 pandemic in terms of MSE and MAE, respectively.

For long-term predictions, the MSE and MAE were not as good as for short-term predictions. Specifically, the results showed that the errors grew with the increase in prediction steps, highlighting that long-term predictions are more challenging than shortterm ones. Nonetheless, the ensemble model still outperformed conventional benchmark models in long-term predictions.

Table 6. Comparison of five-time-step prediction between proposed and benchmark models.   

<table><tr><td rowspan="2">Look-Back Period</td><td rowspan="2">Metric</td><td rowspan="2">Model</td><td colspan="3">2000-2019 1</td><td colspan="3">20172019</td><td colspan="3">20192021</td></tr><tr><td>DAX</td><td>DOW</td><td>S&amp;P500</td><td>DAX</td><td>DOW</td><td>S&amp;P500</td><td>DAX</td><td>DOW</td><td>S&amp;P500</td></tr><tr><td rowspan="9">5 days</td><td rowspan="9">MSE</td><td>LSTM</td><td>0.0561 0.1153</td><td>0.3050 0.4377</td><td>0.2887 0.4247</td><td>0.0421 0.0901</td><td>0.3856 0.5388</td><td>0.3378 0.4804</td><td>0.4136 0.5686</td><td>0.4975 0.6558</td><td>0.4463 0.5995</td></tr><tr><td>GRU</td><td>0.0034</td><td>0.0012</td><td>0.0023</td><td>0.0045</td><td>0.0031</td><td>0.0031</td><td>0.0011</td><td>0.0030</td><td>0.0015</td></tr><tr><td>WaveNet</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>0.0383</td><td>0.3106</td><td>0.2995</td><td>0.0218</td><td>0.3857</td><td>0.3240</td><td>0.4604</td><td>0.5215</td><td>0.4606</td></tr><tr><td>CNN-LSTM</td><td>0.0070</td><td>0.0529</td><td>0.0865</td><td>0.0361</td><td>0.0070</td><td>0.0699</td><td>0.0737</td><td>0.1186</td><td>0.1112</td></tr><tr><td>GRU-CNN</td><td>0.0023</td><td>0.0153</td><td>0.0049</td><td>0.0042</td><td>0.0299</td><td>0.0359</td><td>0.0153</td><td>0.0147</td><td>0.0261</td></tr><tr><td>Ensemble</td><td>0.0041</td><td>0.0037</td><td>0.0037</td><td>0.0036</td><td>0.0070</td><td>0.0053</td><td>0.0013</td><td>0.0021</td><td>0.0009</td></tr><tr><td>RNN</td><td>0.2228</td><td>0.5325</td><td>0.5223</td><td>0.1854</td><td>0.6141</td><td>0.5702</td><td>0.6407</td><td>0.7033</td><td>0.6650</td></tr><tr><td>LSTM</td><td>0.3285</td><td>0.6469</td><td>0.6411</td><td>0.2837</td><td>0.7294</td><td>0.6856</td><td>0.7527</td><td>0.8090</td><td>0.7723</td></tr><tr><td rowspan="7">MAE</td><td>GRU</td><td>0.0459</td><td>0.0285</td><td>0.0407</td><td>0.0561</td><td>0.0388</td><td>0.0479</td><td>0.0348</td><td>0.0309</td><td>0.0224</td></tr><tr><td>WaveNet</td><td>0.1732</td><td>0.5355</td><td>0.5319</td><td>0.1251</td><td>0.6155</td><td>0.5606</td><td>0.6766</td><td>0.7212</td><td>0.6762</td></tr><tr><td>CNN-LSTM</td><td>0.0695</td><td>0.2155</td><td>0.2896</td><td>0.1582</td><td>0.0687</td><td>0.2522</td><td>0.2685</td><td>0.3423</td><td>0.3303</td></tr><tr><td>GRU-CNN</td><td>0.0374</td><td>0.1141</td><td>0.0631</td><td>0.0526</td><td>0.1649</td><td>0.1791</td><td>0.1195</td><td>0.1173</td><td>0.1585</td></tr><tr><td>Ensemble</td><td>0.0524</td><td>0.0522</td><td>0.0498</td><td>0.0460</td><td>0.0743</td><td>0.0645</td><td>0.0289</td><td>0.0394</td><td>0.0230</td></tr><tr><td>RNN</td><td>0.1692</td><td>0.5765</td><td>0.5631</td><td>0.1083</td><td>0.6957</td><td>0.6176</td><td>0.7504</td><td>0.8649</td><td>0.7967</td></tr><tr><td rowspan="8">21 days</td><td rowspan="8">MSE</td><td>LSTM</td><td>0.1127</td><td>0.3921</td><td>0.3816</td><td>0.0734</td><td>0.4769</td><td>0.4226</td><td>0.5105</td><td>0.5959</td><td>0.5468</td></tr><tr><td>GRU</td><td>0.0023</td><td>0.0022</td><td>0.0015</td><td>0.0026</td><td>0.0036</td><td>0.0042</td><td>0.0016</td><td>0.0034</td><td>0.0141</td></tr><tr><td>WaveNet</td><td>0.0409</td><td>0.3148</td><td>0.3031</td><td>0.0227</td><td>0.3970</td><td>0.3374</td><td>0.4482</td><td>0.5305</td><td>0.4756</td></tr><tr><td>CNN-LSTM</td><td>0.0199</td><td>0.1772</td><td>0.0423</td><td>0.0432</td><td>0.0444</td><td>0.0973</td><td>0.0472</td><td>0.1140</td><td>0.0839</td></tr><tr><td>GRU-CNN</td><td>0.0030</td><td>0.0278</td><td>0.0087</td><td>0.0092</td><td>0.0173</td><td>0.0217</td><td>0.0029</td><td>0.0071</td><td>0.0361</td></tr><tr><td>Ensemble</td><td>0.0022</td><td>0.0017</td><td>0.0019</td><td>0.0048</td><td>0.0050</td><td>0.0094</td><td>0.0015</td><td>0.0012</td><td>0.0009</td></tr><tr><td>RNN</td><td>0.4009</td><td>0.7440</td><td>0.7396</td><td>0.3095</td><td>0.8297</td><td>0.7794</td><td></td><td></td><td></td></tr><tr><td>LSTM</td><td>0.3263</td><td>0.6129</td><td>0.6083</td><td>0.2530</td><td>0.6863</td><td>0.6437</td><td>0.8650 0.7132</td><td>0.9293 0.7713</td><td>0.8911 0.7380</td></tr><tr><td rowspan="6">MAE</td><td>GRU</td><td>0.0364</td><td>0.0403</td><td>0.0330</td><td>0.0438</td><td>0.0520</td><td>0.0540</td><td>0.0350</td><td>0.0510</td><td>0.1154</td></tr><tr><td>WaveNet</td><td>0.1832</td><td>0.5405</td><td>0.5361</td><td>0.1264</td><td>0.6247</td><td>0.5727</td><td>0.6679</td><td>0.7275</td><td>0.6877</td></tr><tr><td>CNN-LSTM</td><td>0.1318</td><td>0.4034</td><td>0.1973</td><td>0.1794</td><td>0.2017</td><td>0.3030</td><td>0.2148</td><td>0.3363</td><td>0.2865</td></tr><tr><td>GRU-CNN</td><td>0.0429</td><td>0.1471</td><td>0.0832</td><td>0.0842</td><td>0.1213</td><td>0.1379</td><td>0.0476</td><td></td><td></td></tr><tr><td>Ensemble</td><td>0.0363</td><td>0.0348</td><td>0.0364</td><td>0.0545</td><td>0.0614</td><td>0.0834</td><td></td><td>0.0787</td><td>0.1876</td></tr><tr><td>RNN</td><td></td><td></td><td></td><td>0.1045</td><td></td><td></td><td>0.0309</td><td>0.0288</td><td>0.0252</td></tr><tr><td rowspan="9">42 days</td><td></td><td>0.1562</td><td>0.5930 0.4016</td><td>0.5795 0.3910</td><td></td><td>0.7432</td><td>0.6845 0.4690</td><td>0.7360</td><td>0.8848 0.6070</td><td>0.8377 0.5733</td></tr><tr><td>LSTM</td><td>0.1144</td><td></td><td></td><td>0.0849</td><td>0.5090</td><td></td><td>0.4994</td><td></td><td></td></tr><tr><td>GRU</td><td>0.0030</td><td>0.0038</td><td>0.0024</td><td>0.0040</td><td>0.0023</td><td>0.0046</td><td>0.0013</td><td>0.0011</td><td>0.0017</td></tr><tr><td>WaveNet</td><td>0.0445</td><td>0.3203</td><td>0.3080</td><td>0.0242</td><td>0.3983</td><td>0.3445</td><td>0.4413</td><td>0.5374</td><td>0.4911</td></tr><tr><td>CNN-LSTM</td><td>0.0079</td><td>0.0388</td><td>0.0654</td><td>0.0294</td><td>0.0296</td><td>0.0064</td><td>0.0709</td><td>0.2007</td><td>0.0598</td></tr><tr><td>GRU-CNN</td><td>0.0047</td><td>0.0305</td><td>0.0060</td><td>0.0099</td><td>0.0318</td><td>0.0274</td><td>0.0357</td><td>0.0860</td><td>0.0462</td></tr><tr><td>Ensemble</td><td>0.0023</td><td>0.0021</td><td>0.0011</td><td>0.0037</td><td>0.0269</td><td>0.0164</td><td>0.0022</td><td>0.0012</td><td>0.0011</td></tr><tr></table>

1 The period from 24 October 2014 through 31 December 2019 for the DAX.

We evaluated the performance of the proposed models with four different features and two different optimizers.

The average MSE and MAE for the use of different features and optimizers of the proposed models over the three periods are shown in Tables 7 and 8, where the best performance results are marked in bold for each stock market index, a look-back period, and optimizer.

For multi-time-step prediction, in terms of the average MSE over the three periods, Table 7 confirms that the CNN-LSTM, GRU-CNN, and ensemble models with the introduced medium feature outperformed the other models in $6 6 . 7 \%$ , $8 3 . 3 \%$ , and $8 3 . 3 \%$ of cases with the DAX dataset; $6 6 . 7 \%$ , $6 6 . 7 \%$ , and $5 0 \%$ of cases with the DOW dataset; and $6 6 . 7 \%$ , $5 0 \%$ , and $8 3 . 3 \%$ of cases with the S&P500 dataset.

Further, for multi-time-step prediction, in terms of the average MAE over the three periods, Table 8 shows that the CNN-LSTM, GRU-CNN, and ensemble models with the novel medium feature outperformed the other models in $8 3 . 3 \%$ , $1 0 0 \%$ , and $8 3 . 3 \%$ of cases with the DAX dataset; $6 6 . 7 \%$ , $1 0 0 \%$ , and $6 6 . 7 \%$ of cases with the DOW dataset; and $6 6 . 7 \%$ $5 0 \%$ , and $8 3 . 3 \%$ of cases with the S&P500 dataset.

Table 7. Comparison of different optimizers and features in terms of average MSE over the three periods for five-time-step prediction.   

<table><tr><td rowspan="2">Look-Back Period</td><td rowspan="2">Optimizer</td><td rowspan="2">Feature</td><td colspan="3">CNN-LSTM</td><td colspan="3">GRU-CNN</td><td colspan="3">Ensemble</td></tr><tr><td>DAX</td><td>DOW</td><td>S&amp;P500</td><td>DAX</td><td>DOW</td><td>S&amp;P500</td><td>DAX</td><td>DOW</td><td>S&amp;P500</td></tr><tr><td rowspan="6">5 days</td><td rowspan="3">Adam</td><td>MV</td><td>0.0293</td><td>0.0573</td><td>0.0636</td><td>0.0085</td><td>0.0099</td><td>0.0160</td><td>0.0532</td><td>0.2397</td><td>0.0048</td></tr><tr><td>MVC</td><td>0.0149</td><td>0.0531</td><td>0.1266</td><td>0.0038</td><td>0.0229</td><td>0.0128</td><td>0.0028</td><td>0.0084</td><td>0.0120</td></tr><tr><td>OHLV</td><td>0.0389</td><td>0.0595</td><td>0.0892</td><td>0.0073</td><td>0.0200</td><td>0.0223</td><td>0.0030</td><td>0.0043</td><td>0.0033</td></tr><tr><td rowspan="3"></td><td>OHLMVC</td><td>0.0523</td><td>0.0945</td><td>0.0803</td><td>0.0058</td><td>0.0227</td><td>0.0155</td><td>0.0024</td><td>0.0051</td><td>0.0029</td></tr><tr><td>MV</td><td>0.0255</td><td>0.0848</td><td>0.0340</td><td>0.0036</td><td>0.0146</td><td>0.0263</td><td>0.0033</td><td>0.0074</td><td>0.0052</td></tr><tr><td>MVC</td><td>0.0184 0.0131</td><td>0.0333</td><td>0.0926</td><td>0.0034</td><td>0.0186</td><td>0.0269</td><td>0.0031</td><td>0.0141</td><td>0.0065</td></tr><tr><td rowspan="6"></td><td rowspan="3"></td><td>OHLV OHLMVC</td><td>0.0167</td><td>0.0372 0.0720</td><td>0.0507 0.0803</td><td>0.0038 0.0031</td><td>0.0022 0.0253</td><td>0.0201 0.0322</td><td>0.0033 0.0027</td><td>0.0037 0.0044</td><td>0.0068 0.0050</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MV MVC</td><td>0.0443 0.1774</td><td>0.0541 0.1768</td><td>0.1509 0.2014</td><td>0.0395 0.0066</td><td>0.0273</td><td>0.0106</td><td>0.0030</td><td>0.0052 0.0077</td><td>0.1904</td></tr><tr><td rowspan="3">Adam</td><td>OHLV</td><td>0.0368</td><td>0.1119</td><td>0.0745</td><td>0.0050</td><td>0.0107 0.0174</td><td>0.0217 0.0222</td><td>0.0036 0.0028</td><td>0.0026</td><td>0.0040 0.0041</td></tr><tr><td>OHLMVC</td><td>0.0990</td><td>0.1957</td><td>0.1828</td><td>0.0044</td><td>0.0182</td><td>0.0105</td><td>0.0027</td><td>0.0026</td><td>0.0025</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="5"></td><td rowspan="3">RMSProp</td><td>MV MVC</td><td>0.2070 0.0811</td><td>0.1048 0.0669</td><td>0.0475 0.0804</td><td>0.0243 0.0125</td><td>0.0205</td><td>0.0517</td><td>0.0026 0.0042</td><td>0.0102 0.0154</td><td>0.0059 0.0124</td></tr><tr><td>OHLV</td><td>0.1223</td><td></td><td>0.1723</td><td>0.0147</td><td>0.0154</td><td>0.0519</td><td></td><td>0.2354</td><td></td></tr><tr><td>OHLMVC</td><td>0.0640</td><td>0.0316 0.1376</td><td>0.1346</td><td>0.0264</td><td>0.0152 0.0189</td><td>0.0543 0.0496</td><td>0.0044 0.0376</td><td>0.0035</td><td>0.0062 0.0089</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="5">42 days</td><td rowspan="3">Adam</td><td>MV</td><td>0.0327 0.0640</td><td>0.2019 0.0928</td><td>0.1374</td><td>0.0532</td><td>0.0477</td><td>0.0562</td><td>0.0031</td><td>0.0065 0.2501</td><td>0.0046 0.0075</td></tr><tr><td>MVC</td><td>0.0360</td><td>0.0897</td><td>0.0850 0.0439</td><td>0.0115 0.0168</td><td>0.0443 0.0494</td><td>0.0320</td><td>0.0062</td><td>0.0101</td><td>0.0062</td></tr><tr><td>OHLV OHLMVC</td><td>0.0756</td><td>0.1798</td><td>0.1960</td><td>0.0079</td><td>0.0355</td><td>0.0265 0.0455</td><td>0.0027 0.0028</td><td>0.0045</td><td>0.0093</td></tr><tr><td rowspan="3"></td><td>MV</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>RMSProp MVC</td><td>0.0506 0.1086</td><td>0.2216 0.4098</td><td>0.1771 0.1259</td><td>0.0111 0.0084</td><td>0.0486 0.0418</td><td>0.0392</td><td>0.0036</td><td>0.0156</td><td>0.0093 0.0125</td></tr><tr><td>OHLV OHLMVC</td><td>0.1241</td><td>0.3247</td><td>0.1819</td><td>0.0052</td><td>0.0471</td><td>0.0648 0.0301</td><td>0.0033 0.2476</td><td>0.3016 0.0065</td><td>0.0042</td></tr></table>

In addition, the proposed models were trained for 1500 epochs with the Adam optimizer and OHLV features to achieve higher performance than that of the model as described in Section 3.2.4. Figures 9–11 compare the actual and predicted close prices of the DAX, DOW, and S&P500 indices, respectively, with respect to the different look-back periods. In Figures 9–11, the left, middle, and right plots correspond to the look-back periods of 5, 21, and 42 days, respectively. The look-back period and stock market index also evidently affect the model performance.

Table 8. Comparison of different optimizers and features in terms of average MAE over the three periods for five-time-step prediction.   

<table><tr><td rowspan="2">Look-Back Period</td><td rowspan="2">Optimizer</td><td rowspan="2">Feature</td><td colspan="3">CNN-LSTM</td><td colspan="3">GRU-CNN</td><td colspan="3">Ensemble</td></tr><tr><td>DAX</td><td>DOW</td><td>S&amp;P500</td><td>DAX</td><td>DOW</td><td>S&amp;P500</td><td>DAX</td><td>DOW</td><td>S&amp;P500</td></tr><tr><td rowspan="6">5 days</td><td rowspan="3">Adam</td><td>MV</td><td>0.1349</td><td>0.1894</td><td>0.2386</td><td>0.0786</td><td>0.0920</td><td>0.1012</td><td>0.1561</td><td>0.3217</td><td>0.0551</td></tr><tr><td>MVC</td><td>0.0963</td><td>0.2239</td><td>0.3409</td><td>0.0528</td><td>0.1421</td><td>0.1017</td><td>0.0409</td><td>0.0781</td><td>0.0832</td></tr><tr><td>OHLV</td><td>0.1654</td><td>0.2088</td><td>0.2907</td><td>0.0698</td><td>0.1321</td><td>0.1336</td><td>0.0424</td><td>0.0553</td><td>0.0457</td></tr><tr><td rowspan="3"></td><td>OHLMVC</td><td>0.1963</td><td>0.2775</td><td>0.2648</td><td>0.0631</td><td>0.1414</td><td>0.1085</td><td>0.0376</td><td>0.0572</td><td>0.0427</td></tr><tr><td>MV</td><td>0.1277</td><td>0.2466</td><td>0.1705</td><td>0.0476</td><td>0.1098</td><td>0.1520</td><td>0.0448</td><td>0.0698</td><td>0.0569</td></tr><tr><td>MVC</td><td>0.1008</td><td>0.1616</td><td>0.2727</td><td>0.0472</td><td>0.1208</td><td>0.1544</td><td>0.0429</td><td>0.1027</td><td>0.0606</td></tr><tr><td rowspan="8"></td><td rowspan="4"></td><td>OHLV OHLMVC</td><td>0.0868 0.1114</td><td>0.1747 0.2308</td><td>0.1881 0.2130</td><td>0.0525 0.0434</td><td>0.1397 0.1452</td><td>0.1304 0.1567</td><td>0.0447 0.0400</td><td>0.0493 0.0531</td><td>0.0620 0.0560</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MV MVC</td><td>0.1697</td><td>0.1928</td><td>0.3751</td><td>0.0828</td><td>0.1555</td><td>0.0897</td><td>0.0414</td><td>0.0603</td><td>0.2791</td></tr><tr><td>OHLV</td><td>0.2673 0.1753</td><td>0.4012 0.3138</td><td>0.4062 0.2623</td><td>0.0699</td><td>0.0929</td><td>0.1394</td><td>0.0457</td><td>0.0770</td><td>0.0484</td></tr><tr><td rowspan="3"></td><td></td><td>0.2692</td><td>0.4271</td><td>0.4016</td><td>0.0582 0.0546</td><td>0.1157 0.1249</td><td>0.1363</td><td>0.0406</td><td>0.0417</td><td>0.0483</td></tr><tr><td>OHLMVC</td><td></td><td></td><td></td><td></td><td></td><td>0.0928</td><td>0.0396</td><td>0.0412</td><td>0.0350</td></tr><tr><td>MV</td><td>0.3389</td><td>0.2637</td><td>0.1844</td><td>0.1218</td><td>0.1333</td><td>0.1993</td><td>0.0424</td><td>0.0859</td><td>0.0633</td></tr><tr><td rowspan="5"></td><td rowspan="3">RMSProp</td><td>MVC</td><td>0.2507</td><td>0.2386</td><td>0.2320</td><td>0.0980</td><td>0.0982</td><td>0.2102</td><td>0.0535</td><td>0.1073</td><td>0.0969 0.0671</td></tr><tr><td>OHLV OHLMVC</td><td>0.2850 0.1878</td><td>0.1612 0.3017</td><td>0.3277</td><td>0.0990</td><td>0.1096</td><td>0.2077</td><td>0.0547</td><td>0.3192</td><td></td></tr><tr><td></td><td></td><td></td><td>0.3532</td><td>0.1293</td><td>0.1242</td><td>0.2014</td><td>0.1280</td><td>0.0476</td><td>0.0810</td></tr><tr><td rowspan="3">Adam</td><td>MV</td><td>0.1471</td><td>0.4368</td><td>0.3548</td><td>0.1859</td><td>0.2023</td><td>0.1943</td><td>0.0415</td><td>0.0648</td><td>0.0476</td></tr><tr><td>MVC OHLV</td><td>0.1884 0.1539</td><td>0.2731 0.2643</td><td>0.2670 0.1849</td><td>0.0921 0.1104</td><td>0.1885 0.2039</td><td>0.1613</td><td>0.0616</td><td>0.3178</td><td>0.0592</td></tr><tr><td>OHLMVC</td><td>0.2555</td><td>0.3332</td><td>0.3816</td><td>0.0785</td><td>0.1707</td><td>0.1443</td><td>0.0394</td><td>0.0725</td><td>0.0557</td></tr><tr><td rowspan="6">42 days</td><td rowspan="3">RMSProp</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.1935</td><td>0.0406</td><td>0.0543</td><td>0.0753</td></tr><tr><td>MV</td><td>0.2048</td><td>0.4303</td><td>0.3616</td><td>0.0904</td><td>0.2019</td><td>0.1691</td><td>0.0473</td><td>0.1037</td><td>0.0786</td></tr><tr><td>MVC</td><td>0.2602</td><td>0.5322</td><td>0.2999</td><td>0.0744</td><td>0.1889</td><td>0.2186</td><td>0.0445</td><td>0.3711</td><td>0.0873</td></tr><tr><td>OHLV</td><td>0.2904</td><td>0.5606</td><td>0.3259</td><td></td><td>0.0628 0.2044</td><td></td><td>0.1510</td><td>0.3172</td><td>0.0683</td><td>0.0493</td></tr><tr><td></td><td>OHLMVC</td><td>0.2310</td><td>0.3181</td><td>0.2358</td><td>0.0622</td><td>0.1973</td><td>0.2204</td><td>0.0417</td><td>0.0531</td><td>0.0657</td></tr></table>

![](images/0f94a849374bab996f2d6d9af16c10334278050a454d2aeb98be9f7bd843b135.jpg)  
Figure 9. Comparison of true and predicted close prices of the DAX index between different look-back periods for five-time-step prediction over the period from 24 October 2014 through 31 December 2019.

![](images/4deda6bebb1658b2d8885a191ff8e204f98e3713568545c96da5ac6882082471.jpg)  
Figure 10. Comparison of true and predicted close prices of the DOW index between different look-back periods for five-time-step prediction over the period from 1 January 2000 through 31 December 2019.

![](images/b65699e195e14a1e2c3aeb66ab3372599bb0530c95ce461af73029847a754c26.jpg)  
Figure 11. Comparison of true and predicted close prices of the S&P500 index between different look-back periods for five-time-step prediction over the period from 1 January 2000 through 31 December 2019.

Moreover, the proposed models were trained for 1500 epochs with the Adam optimizer and a look-back period of 5 days. The comparisons of true and predicted close prices of the DAX, DOW, and S&P500 indices between different input features for five-time-step prediction are provided in Figures 12–14, respectively.

![](images/aac0f6cc1a38830ae9a33f5622bebcdfd8aa5424c21990958501168b9696a37b.jpg)  
Figure 12. Comparison of true and predicted close prices of the DAX index between different input features for five-time-step prediction over the period from 24 October 2014 through 31 December 2019.

![](images/22d011ca5c3b9788c7a47d7c1e8dcf5d95389a4a34ddddb824b7e6ae7ac1d6b8.jpg)  
Figure 13. Comparison of true and predicted close prices of the DOW index between different input features for five-time-step prediction over the period from 1 January 2000 through 31 December 2019.

![](images/b35caec7d96cf2cffe2e3dd0d5955b88e98a6043504941e5c8831134cb3ab9be.jpg)  
Figure 14. Comparison of true and predicted close prices of the S&P500 index between different input features for five-time-step prediction over the period from 1 January 2000 through 31 December 2019.

# 5. Discussion

Various deep-learning techniques have been applied extensively in the field of finance for stock market prediction, portfolio optimization, risk management, and trading strategies. Although forecasting stock market indices with noisy data is a complex and challenging process, it significantly affects the appropriate timing of buying or selling investment assets for investors as they reduce the risk, which is one of the most valuable areas in finance.

Combining multiple deep-learning models results in a better performance [48]. We proposed to integrate RNNs, namely, CNN-LSTM, GRU-CNN, and ensemble models. The proposed models were evaluated to forecast the one-time-step and multi-time-step closing prices of stock market indices using various stock market indices, look-back periods, optimizers, features, and the learning rate.

The experimental results revealed that the proposed models that combine variants of RNNs outperformed the traditional machine learning models, such as RNN, LSTM, GRU, and WaveNet in most cases. In particular, the ensemble model produced significant results for one-time-step forecasting. Moreover, compared with the performance of previous studies that used open, high, and low prices and trading volume of stock market indices as features, that of our models improved by incorporating the proposed novel feature, which is the average of the high and low prices. Furthermore, our models with MV features provided favorable results in numerous cases. Notably, reducing the number of features could be interpreted as circumventing the overfitting.

The performance of the proposed and benchmark models with the Adam optimizer and OHLV features over three periods were evaluated to predict one-time-step and fivetime-step using look-back periods of 5, 21, and 42 days as provided in Tables 3 and 6, respectively. The comparisons of the average MSE and MAE over three periods for different look-back and look-ahead periods are provided in Figures 15 and 16, respectively. An overall comparison between the ensemble model and other models in Figures 15 and 16 indicates that the ensemble model significantly outperformed the other models.

![](images/d2d4d1e449bf00912d15e1be5dd0278e50715900ce1e26a057d41ac012366f70.jpg)  
Figure 15. Comparison of the average MSE over three periods for different look-back and lookahead periods using RNN, LSTM, GRU, WaveNet, CNN-LSTM, GRU-CNN, and ensemble with OHLV features.

In addition, the performance of the proposed and benchmark models over three periods were evaluated to compare the impact of four different input features (i.e., MV, MVC, OHLV, and OHLMVC) for one-time-step and five-time-step predictions with three look-back periods and two optimizers as described in Section 3.2. The comparisons of the average MSE and MAE of the proposed and benchmark models over all periods, optimizers, look-back, and look-ahead periods are provided in Figures 17 and 18, respectively. The proposed models outperform the benchmark models and the performance of our models improves by incorporating the proposed medium feature.

![](images/6ab8940602c91b111e783a6c0da4d8c56dee44e8c9188b080faf944542a54526.jpg)  
Figure 16. Comparison of the average MAE over three periods for different look-back and look-ahead periods using RNN, LSTM, GRU, WaveNet, CNN-LSTM, GRU-CNN, and ensemble with OHLV features.

![](images/8d89e07545d0ddaae1a87f13d3519cae97970d3259e033f776a5f46f0bebd0c9.jpg)  
Figure 17. Comparison of the average MSE of all models using MV, MVC, OHLV, and OHLMVC.

![](images/b3a7d400dcac4e1016fea68d95178084ebcf44510614f4278a78bbc2fd425a53.jpg)  
Figure 18. Comparison of the average MAE of all models using MV, MVC, OHLV, and OHLMVC.

During the course of this study, the Russia–Ukraine crisis escalated on 24 February 2022. Additional experiments were conducted to examine the impact of this crisis on each stock market index for the period from 1 January 2021 through 15 February 2023.

We evaluated the performance of the proposed and benchmark models to predict one-time-step and five-time-step ahead with various look-back periods of 5, 21, and 42 days as one week, one month, and two months, respectively. The architectures of the proposed and benchmark models have been described in Sections 3.1 and 4.1, respectively.

The proposed and benchmark models were implemented with 50 epochs, an early stopping patience of 10, a batch size of 32, a learning rate of 0.0005, the Adam optimizer, the ReLU activation function, and OHLV features. The network weights and biases were initialized with the Glorot-Xavier uniform method and zeros, respectively. The proposed and benchmark models were trained with the Huber loss function and MSE loss function, respectively.

Table 9 compares our models with the benchmark models for the different look-back periods for one-time-step and five-time-step predictions, where the best performance results are marked in bold for each stock market index, period, and metric. Table 9 indicates that the proposed models improved the benchmarks in several cases and that the ensemble model significantly outperformed the other models.

Table 9. Comparison of one-time-step and five-time-step predictions between proposed and benchmark models for the period from 1 January 2021 through 15 February 2023.   

<table><tr><td rowspan="2">Look-Back Period</td><td rowspan="2">Metric</td><td rowspan="2">Model</td><td colspan="3">One-Time-Step Prediction</td><td colspan="3">Five-Time-Step Prediction</td></tr><tr><td>DAX</td><td>DOW</td><td>S&amp;P500</td><td>DAX</td><td>DOW</td><td>S&amp;P500</td></tr><tr><td rowspan="20">5 days</td><td rowspan="8">MSE</td><td>RNN</td><td>0.1738</td><td>0.3059</td><td>0.1043</td><td>0.0250</td><td>0.0219</td><td>0.1122</td></tr><tr><td>LSTM</td><td>0.1735</td><td>0.3060</td><td>0.1043</td><td>0.0058</td><td>0.0076</td><td>0.0064</td></tr><tr><td>GRU</td><td>0.1735</td><td>0.3060</td><td>0.1043</td><td>0.0081</td><td>0.0187</td><td>0.0058</td></tr><tr><td>WaveNet</td><td>0.4870</td><td>0.3120</td><td>0.5712</td><td>0.0477</td><td>0.1230</td><td>0.0177</td></tr><tr><td>CNN-LSTM</td><td>0.1435</td><td>0.0560</td><td>0.0988</td><td>0.0831</td><td>0.0447</td><td>0.0912</td></tr><tr><td>GRU-CNN</td><td>0.0370</td><td>0.3060</td><td>0.0053</td><td>0.0101</td><td>0.0166</td><td>0.0090</td></tr><tr><td>Ensemble</td><td>0.0017</td><td>0.0051</td><td>0.0027</td><td>0.0126</td><td>0.0122</td><td>0.0057</td></tr><tr><td>RNN</td><td>0.3738</td><td>0.5163</td><td>0.2934</td><td>0.1468</td><td>0.1266</td><td>0.3053</td></tr><tr><td rowspan="5">MAE</td><td>LSTM</td><td>0.3726</td><td>0.5165</td><td>0.2934</td><td>0.0593</td><td>0.0640</td><td>0.0585</td></tr><tr><td>GRU</td><td>0.3727</td><td>0.5164</td><td>0.2934</td><td>0.0802</td><td>0.1251</td><td>0.0629</td></tr><tr><td>WaveNet</td><td>0.6741</td><td>0.5185</td><td>0.7456</td><td>0.1925</td><td>0.3177</td><td>0.1110</td></tr><tr><td>CNN-LSTM</td><td>0.3440</td><td>0.1587</td><td>0.2864</td><td>0.2345</td><td>0.1551</td><td>0.2666</td></tr><tr><td>GRU-CNN</td><td>0.1733</td><td>0.5165</td><td>0.0557</td><td>0.0901</td><td>0.1139</td><td>0.0739</td></tr><tr><td>Ensemble</td><td></td><td>0.0345</td><td>0.0628</td><td>0.0420</td><td>0.1025</td><td>0.0968</td><td>0.0605</td></tr><tr><td rowspan="11">21 days</td><td rowspan="8">MSE</td><td>RNN</td><td>0.1787</td><td>0.3315</td><td>0.0995</td><td>0.1917</td><td>0.3485</td><td>0.1081</td></tr><tr><td>LSTM</td><td>0.1785</td><td>0.3315</td><td>0.0995</td><td>0.0133</td><td>0.0070</td><td>0.0060</td></tr><tr><td>GRU</td><td>0.1785</td><td>0.3315</td><td>0.0995</td><td>0.0126</td><td>0.0155</td><td>0.0056</td></tr><tr><td>WaveNet</td><td>0.4126</td><td>0.2232</td><td>0.5259</td><td>0.0526</td><td>0.1446</td><td>0.0177</td></tr><tr><td>CNN-LSTM</td><td>0.0805</td><td>0.0225</td><td>0.0680</td><td>0.0399</td><td>0.0127</td><td>0.0595</td></tr><tr><td>GRU-CNN</td><td>0.0159</td><td>0.0307</td><td>0.0058</td><td>0.0245</td><td>0.0127</td><td>0.0079</td></tr><tr><td>Ensemble</td><td>0.0018</td><td>0.0128</td><td>0.0026</td><td>0.0115</td><td>0.0124</td><td>0.0059</td></tr><tr><td>RNN</td><td>0.3947</td><td>0.5588</td><td>0.2950</td><td>0.4151</td><td>0.5807</td><td>0.3127</td></tr><tr><td rowspan="7">MAE</td><td>LSTM</td><td>0.3940</td><td>0.5588</td><td>0.2950</td><td>0.1019</td><td>0.0645</td><td>0.0625</td></tr><tr><td>GRU</td><td>0.3940</td><td>0.5587</td><td>0.2950</td><td>0.1032</td><td>0.1121</td><td>0.0634</td></tr><tr><td>WaveNet</td><td>0.6244</td><td>0.4501</td><td>0.7168</td><td>0.2015</td><td>0.3605</td><td>0.1094</td></tr><tr><td>CNN-LSTM</td><td>0.2490</td><td>0.1076</td><td>0.2367</td><td>0.1460</td><td>0.0808</td><td>0.2231</td></tr><tr><td>GRU-CNN</td><td>0.1133</td><td>0.1626</td><td>0.0634</td><td>0.1429</td><td>0.0994</td><td>0.0678</td></tr><tr><td>Ensemble</td><td>0.0358</td><td>0.1027</td><td>0.0402</td><td>0.0966</td><td>0.0980</td><td>0.0650</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="15">42 days MAE</td><td rowspan="5">MSE</td><td>RNN LSTM</td><td>0.2434 0.2434</td><td>0.4011 0.4011</td><td>0.1446</td><td>0.0481 0.0071</td><td>0.0071</td><td>0.1150</td></tr><tr><td></td><td></td><td></td><td>0.1446</td><td></td><td>0.0040</td><td>0.0066</td></tr><tr><td>GRU</td><td>0.2434</td><td>0.4011</td><td>0.1446</td><td>0.0079</td><td>0.0122</td><td>0.0044</td></tr><tr><td>WaveNet</td><td>0.3106</td><td>0.1505</td><td>0.4595</td><td>0.0685</td><td>0.1746</td><td>0.0215</td></tr><tr><td>CNN-LSTM</td><td>0.0435</td><td>0.0027</td><td>0.0433</td><td>0.0115</td><td>0.0033</td><td>0.0597</td></tr><tr><td>GRU-CNN Ensemble</td><td></td><td>0.0425</td><td>0.4011</td><td>0.0092</td><td>0.0351 0.0092</td><td>0.0261</td><td>0.0075 0.0083</td></tr><tr><td rowspan="8"></td><td>RNN</td><td>0.0028</td><td>0.0057</td><td>0.0024</td><td></td><td>0.0079</td><td></td></tr><tr><td>LSTM</td><td>0.4857 0.4857</td><td>0.6314 0.6314</td><td>0.3694</td><td>0.2009 0.0668</td><td>0.0693 0.0517</td><td>0.3288 0.0643</td></tr><tr><td>GRU</td><td>0.4857</td><td>0.6314</td><td>0.3694 0.3694</td><td>0.0768</td><td>0.0985</td><td>0.0548</td></tr><tr><td>WaveNet</td><td>0.5511</td><td>0.3843</td><td></td><td>0.2488</td><td>0.4143</td><td>0.1246</td></tr><tr><td></td><td></td><td></td><td>0.6730</td><td></td><td></td><td></td></tr><tr><td>CNN-LSTM</td><td>0.1961</td><td>0.0456</td><td>0.1878</td><td>0.0847</td><td>0.0472</td><td>0.2295</td></tr><tr><td>GRU-CNN</td><td>0.2030</td><td>0.6314</td><td>0.0878</td><td>0.1805</td><td>0.1358</td><td>0.0711</td></tr><tr><td>Ensemble</td><td>0.0476</td><td>0.0693</td><td>0.0414</td><td>0.0846</td><td>0.0771</td><td>0.0778</td></tr></table>

Further, compared with other forecasting methods in other fields, the proposed framework herein can be applied to forecasting time-series data, such as energy consumption, oil price, gas concentration, air quality, and river flow. Moreover, the performance of forecasting can be improved by combining different types of RNN-based models and constructing a portfolio using predicted stock market prices in future studies.

# 6. Conclusions

In this paper, we proposed three RNN-based hybrid models, namely CNN-LSTM, GRU-CNN, and ensemble models, to make one-time-step and multi-time-step predictions of the closing price of three stock market indices in different financial markets. We evaluated and compared the performance of the proposed models with conventional benchmarks (i.e., RNN, LSTM, GRU, and WaveNet) over three different periods: a long period of more than 15 years and two short periods of three years before and after the COVID-19 pandemic. The proposed models significantly outperformed the benchmark models by achieving high predictive performance for various sizes of look-back and look-ahead periods in terms of MSE and MAE. Moreover, we found that the proposed ensemble model was comparable to the GRU, which performed well among benchmarks and outperformed the benchmarks in many cases.

Additionally, we introduced a novel feature, medium, which is the average of high and low prices, and evaluated the performance of the proposed models with four different features and two different optimizers. The results indicated that incorporating the novel feature improved model performance. Overall, our experiments verified that the proposed models outperformed the benchmark models in many cases and that incorporating the medium feature improved their performance.

Author Contributions: Conceptualization, H.S. and H.C.; methodology, H.S. and H.C.; software, H.S. and H.C.; validation, H.S. and H.C.; formal analysis, H.S. and H.C.; writing—original draft preparation, H.S. and H.C.; writing—review and editing, H.S.; supervision, H.S.; project administration, H.S.; funding acquisition, H.S. All authors have read and agreed to the published version of the manuscript.

Funding: This work was supported by the Basic Science Research Program through the National Research Foundation of Korea (NRF) grant funded by the Ministry of Science and ICT (MSIT, Korea) (No. NRF-2020R1G1A1A01006808).

Institutional Review Board Statement: Not applicable.

Informed Consent Statement: Not applicable.

Data Availability Statement: Data used in this study was obtained using the FinanceDataReader open-source library.

Conflicts of Interest: The authors declare no conflict of interest.

# Abbreviations

The following abbreviations are used in this manuscript:

Adam Adaptive Moment Estimation   
ANN Artificial Neural Network   
ARIMA Autoregressive Integrated Moving Average   
ARMA Autoregressive and Moving Average   
CNN Convolutional Neural Network   
DAX Deutscher Aktienindex   
DOW Dow Jones Industrial Average   
GAN Generative Adversarial Network   
GRU Gated Recurrent Unit   
LSTM Long Short Term Memory   
MAE Mean Absolute Error   
MLP Multilayer Perceptron

MSE Mean Squared Error ReLU Rectified Linear Unit RMSProp Root Mean Square Propagation RNN Recurrent Neural Network S&P500 Standard and Poor’s 500

# References

1. Tan, T.; Quek, C.; Ng, G. Brain-inspired genetic complementary learning for stock market prediction. In Proceedings of the IEEE Congress on Evolutionary Computation, Edinburgh, UK, 2–5 September 2005; Volume 3, pp. 2653–2660. [CrossRef]   
2. Wang, J.Z.; Wang, J.J.; Zhang, Z.G.; Guo, S.P. Forecasting stock indices with back propagation neural network. Expert Syst. Appl. 2011, 38, 14346–14355. [CrossRef]   
3. Fama, E.F. The behavior of stock market prices. J. Bus. 1965, 38, 34–105. [CrossRef]   
4. Zhang, X.; Liang, X.; Zhiyuli, A.; Zhang, S.; Xu, R.; Wu, B. AT-LSTM: An Attention-based LSTM Model for Financial Time Series Prediction. IOP Conf. Ser. Mater. Sci. Eng. 2019, 569, 052037. [CrossRef]   
5. Shields, R.; Zein, S.A.E.; Brunet, N.V. An Analysis on the NASDAQ’s Potential for Sustainable Investment Practices during the Financial Shock from COVID-19. Sustainability 2021, 13, 3748. [CrossRef]   
6. Daradkeh, M.K. A Hybrid Data Analytics Framework with Sentiment Convergence and Multi-Feature Fusion for Stock Trend Prediction. Electronics 2022, 11, 250. [CrossRef]   
7. Abrishami, S.; Turek, M.; Choudhury, A.R.; Kumar, P. Enhancing Profit by Predicting Stock Prices using Deep Neural Networks. In Proceedings of the IEEE 31st International Conference on Tools with Artificial Intelligence (ICTAI), Portland, OR, USA, 4–6 November 2019; pp. 1551–1556.   
8. Aggarwal, S.; Aggarwal, S. Deep Investment in Financial Markets using Deep Learning Models. Int. J. Comput. Appl. 2017, 162, 40–43. [CrossRef]   
9. Graves, A.; Rahman Mohamed, A.; Hinton, G. Speech Recognition With Deep Recurrent Neural Networks. In Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing, Vancouver, BC, Canada, 26–31 May 2013; pp. 6645–6649.   
10. Xu, K.; Ba, J.; Kiros, R.; Cho, K.; Courville, A.; Salakhudinov, R.; Zemel, R.; Bengio, Y. Show, Attend and Tell: Neural Image Caption Generation with Visual Attention. In Proceedings of the 32nd International Conference on Machine Learning, Lille, France, 7–9 July 2015; Volume 37, pp. 2048–2057.   
11. Zhu, Y.; Groth, O.; Bernstein, M.S.; Li, F. Visual7W: Grounded Question Answering in Images. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV, USA, 26 June–1 July 2016; pp. 4995–5004.   
12. Ren, B. The use of machine translation algorithm based on residual and LSTM neural network in translation teaching. PLoS ONE 2020, 15, e0240663. [CrossRef] [PubMed]   
13. Bhandari, H.N.; Rimal, B.; Pokhrel, N.R.; Rimal, R.; Dahal, K.R.; Khatri, R.K. Predicting stock market index using LSTM. Mach. Learn. Appl. 2022, 9, 100320. [CrossRef]   
14. Walczak, S.; Cerpa, N. Artificial Neural Networks. In Encyclopedia of Physical Science and Technology, 3rd ed.; Academic Press: New York, NY, USA, 2003; pp. 631–645. [CrossRef]   
15. Rumelhart, D.E.; Hinton, G.E.; Williams, R.J. Learning representations by back-propagating errors. Nature 1986, 323, 533–536. [CrossRef]   
16. Mcculloch, W.; Pitts, W. A Logical Calculus of Ideas Immanent in Nervous Activity. Bull. Math. Biophys. 1943, 5, 115–133. [CrossRef]   
17. Minsky, M.; Papert, S. Perceptrons: An Introduction to Computational Geometry; MIT Press: Cambridge, MA, USA, 1969.   
18. Popescu, M.C.; Balas, V.E.; Perescu-Popescu, L.; Mastorakis, N. Multilayer Perceptron and Neural Networks. WSEAS Trans. Circuits Syst. 2009, 8, 579–588.   
19. Lecun, Y.; Bengio, Y. Convolutional Networks for Images, Speech, and Time-Series; MIT Press: Cambridge, MA, USA, 1997.   
20. Krizhevsky, A.; Sutskever, I.; Hinton, G.E. ImageNet Classification with Deep Convolutional Neural Networks. In Proceedings of the 25th International Conference on Neural Information Processing Systems, NIPS’12, Lake Tahoe, NV, USA, 3–6 December 2012; Curran Associates Inc.: Red Hook, NY, USA, 2012; Volume 1, pp. 1097–1105.   
21. Kingma, D.P.; Ba, J.L. Adam: A Method for Stochastic Optimization. In Proceedings of the 3rd International Conference for Learning Representations (ICLR), San Diego, CA, USA, 7–9 May 2015.   
22. Nair, V.; Hinton, G.E. Rectified Linear Units Improve Restricted Boltzmann Machines. In Proceedings of the 27th International Conference on Machine Learning (ICML), Haifa, Israel, 21–24 June 2010; pp. 807–814.   
23. Srivastava, N.; Hinton, G.; Krizhevsky, A.; Sutskever, I.; Salakhutdinov, R. Dropout: A Simple Way to Prevent Neural Networks from Overfitting. J. Mach. Learn. Res. 2014, 15, 1929–1958.   
24. Bao, W.; Yue, J.; Rao, Y. A deep learning framework for financial time series using stacked autoencoders and long-short term memory. PLoS ONE 2017, 12, e0180944. [CrossRef]   
25. Rumelhart, D.E.; McClelland, J.L. Learning Internal Representations by Error Propagation. In Parallel Distributed Processing: Explorations in the Microstructure of Cognition: Foundations; MIT Press: Cambridge, MA, USA, 1987; pp. 318–362.   
26. Hochreiter, S.; Schmidhuber, J. Long Short-Term Memory. Neural Comput. 1997, 9, 1735–1780. [CrossRef] [PubMed] 1994, 5, 157–166. [CrossRef]   
28. Cho, K.; van Merriënboer, B.; Bahdanau, D.; Bengio, Y. On the Properties of Neural Machine Translation: Encoder–Decoder Approaches. In Proceedings of the SSST-8, Eighth Workshop on Syntax, Semantics and Structure in Statistical Translation, Doha, Qatar, 25 October 2014; Association for Computational Linguistics: Cedarville, OH, USA, 2014; pp. 103–111.   
29. Shen, G.; Tan, Q.; Zhang, H.; Zeng, P.; Xu, J. Deep Learning with Gated Recurrent Unit Networks for Financial Sequence Predictions. Procedia Comput. Sci. 2018, 131, 895–903.   
30. Chung, J.; Gulcehre, C.; Cho, K.; Bengio, Y. Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling. In Proceedings of the NIPS 2014 Workshop on Deep Learning, Montreal, QC, Canada, 13 December 2014.   
31. Kaiser, L.; Sutskever, I. Neural GPUs Learn Algorithms. In Proceedings of the 4th International Conference on Learning Representations, ICLR, San Juan, PR, USA, 2–4 May 2016.   
32. Yin, W.; Kann, K.; Yu, M.; Schütze, H. Comparative Study of CNN and RNN for Natural Language Processing. arXiv 2017, arXiv:1702.01923.   
33. Cho, K.; van Merriënboer, B.; Gulcehre, C.; Bahdanau, D.; Bougares, F.; Schwenk, H.; Bengio, Y. Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), Doha, Qatar, 25–29 October 2014; Association for Computational Linguistics: Doha, Qatar, 2014; pp. 1724–1734.   
34. Nguyen, T.T.; Yoon, S. A Novel Approach to Short-Term Stock Price Movement Prediction using Transfer Learning. Appl. Sci. 2019, 9, 4745. . [CrossRef]   
35. Kamal, I.M.; Bae, H.; Sunghyun, S.; Yun, H. DERN: Deep Ensemble Learning Model for Short- and Long-Term Prediction of Baltic Dry Index. Appl. Sci. 2020, 10, 1504. [CrossRef]   
36. Ta, V.D.; Liu, C.M.; Tadesse, D.A. Portfolio Optimization-Based Stock Prediction Using Long-Short Term Memory Network in Quantitative Trading. Appl. Sci. 2020, 10, 437. [CrossRef]   
37. Rouf, N.; Malik, M.B.; Arif, T.; Sharma, S.; Singh, S.; Aich, S.; Kim, H.C. Stock Market Prediction Using Machine Learning Techniques: A Decade Survey on Methodologies, Recent Developments, and Future Directions. Electronics 2021, 10, 2717. [CrossRef]   
38. Aldhyani, T.H.H.; Alzahrani, A. Framework for Predicting and Modeling Stock Market Prices Based on Deep Learning Algorithms. Electronics 2022, 11, 3149. [CrossRef]   
39. Lin, Y.L.; Lai, C.J.; Pai, P.F. Using Deep Learning Techniques in Forecasting Stock Markets by Hybrid Data with Multilingual Sentiment Analysis. Electronics 2022, 11, 3513. [CrossRef]   
40. Chen, J.F.; Chen, W.L.; Huang, C.P.; Huang, S.H.; Chen, A.P. Financial Time-Series Data Analysis Using Deep Convolutional Neural Networks. In Proceedings of the 7th International Conference on Cloud Computing and Big Data (CCBD), Macau, China, 16–18 November 2016; pp. 87–92. [CrossRef]   
41. Sezer, O.B.; Ozbayoglu, A.M. Algorithmic financial trading with deep convolutional neural networks: Time series to image conversion approach. Appl. Soft Comput. 2018, 70, 525–538. [CrossRef]   
42. Gross, W.; Lange, S.; Bödecker, J.; Blum, M. Predicting Time Series with Space-Time Convolutional and Recurrent Neural Networks. In Proceedings of the European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, Bruges, Belgium, 26–28 April 2017; pp. 26–28.   
43. Fischer, T.; Krauss, C. Deep learning with long short-term memory networks for financial market predictions. Eur. J. Oper. Res. 2018, 270, 654–669. [CrossRef]   
44. Dutta, A.; Kumar, S.; Basu, M. A Gated Recurrent Unit Approach to Bitcoin Price Prediction. J. Risk Financ. Manag. 2020, 13, 23. [CrossRef]   
45. Heaton, J.; Polson, N.; Witte, J. Deep Learning for Finance: Deep Portfolios. Appl. Stoch. Model. Bus. Ind. 2016, 33, 3–12. [CrossRef]   
46. Ilyas, Q.M.; Iqbal, K.; Ijaz, S.; Mehmood, A.; Bhatia, S. A Hybrid Model to Predict Stock Closing Price Using Novel Features and a Fully Modified Hodrick–Prescott Filter. Electronics 2022, 11, 3588. [CrossRef]   
47. Livieris, I.E.; Pintelas, E.; Pintelas, P. A CNN-LSTM model for gold price time-series forecasting. Neural Comput. Appl. 2019, 32, 17351–17360. [CrossRef]   
48. Livieris, I.E.; Pintelas, E.; Stavroyiannis, S.; Pintelas, P. Ensemble Deep Learning Models for Forecasting Cryptocurrency Time-Series. Algorithms 2020, 13, 121. [CrossRef]   
49. Zhang, K.; Zhong, G.; Dong, J.; Wang, S.; Wang, Y. Stock Market Prediction Based on Generative Adversarial Network. Procedia Comput. Sci. 2018, 147, 400–406. [CrossRef]   
50. Leung, M.F.; Wang, J.; Che, H. Cardinality-constrained portfolio selection via two-timescale duplex neurodynamic optimization. Neural Netw. 2022, 153, 399–410. [CrossRef] [PubMed]   
51. Troiano, L.; Villa, E.M.; Loia, V. Replicating a Trading Strategy by Means of LSTM for Financial Industry Applications. IEEE Trans. Ind. Inform. 2018, 14, 3226–3234. [CrossRef]   
52. Chalvatzis, C.; Hristu-Varsakelis, D. High-performance stock index trading: Making effective use of a deep LSTM neural network. arXiv 2019, arXiv:1902.03125.   
53. Park, S.; Song, H.; Lee, S. Linear programing models for portfolio optimization using a benchmark. Eur. J. Financ. 2019, 25, 435–457. [CrossRef]   
54. Lee, S.I.; Yoo, S.J. Threshold-based portfolio: The role of the threshold and its applications. J. Supercomput. 2020, 76, 8040–8057. [CrossRef]   
55. Sen, J.; Dutta, A.; Mehtab, S. Stock Portfolio Optimization Using a Deep Learning LSTM Model. In Proceedings of the IEEE Mysore Sub Section International Conference, Hassan, India, 24–25 October 2021; pp. 263–271.   
56. McKinney, W. Data Structures for Statistical Computing in Python. In Proceedings of the 9th Python in Science Conference, Austin, TX, USA, 28 June– 3 July 2010; pp. 56–61.   
57. Granger, C.W.J. Strategies for Modelling Nonlinear Time-Series Relationships. Econ. Rec. 1993, 69, 233–238. [CrossRef]   
58. Python Core Team. Python: A Dynamic, Open Source Programming Language. Python Software Foundation. 2019. Available online: https://www.python.org (accessed on 18 December 2022).   
59. Keras. 2015. Available online: https://keras.io (accessed on 18 December 2022).   
60. Abadi, M.; Agarwal, A.; Barham, P.; Brevdo, E.; Chen, Z.; Citro, C.; Corrado, G.S.; Davis, A.; Dean, J.; Devin, M.; et al. TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems. arXiv 2015, arXiv:1603.04467.   
61. van der Walt, S.; Colbert, S.C.; Varoquaux, G. The NumPy Array: A Structure for Efficient Numerical Computation. Comput. Sci. Eng. 2011, 13, 22–30. [CrossRef]   
62. Pedregosa, F.; Varoquaux, G.; Gramfort, A.; Michel, V.; Thirion, B.; Grisel, O.; Blondel, M.; Prettenhofer, P.; Weiss, R.; Dubourg, V.; et al. Scikit-Learn: Machine Learning in Python. J. Mach. Learn. Res. 2011, 12, 2825–2830.   
63. Huber, P.J. Robust Estimation of a Location Parameter. Ann. Math. Stat. 1964, 35, 73–101. [CrossRef]   
64. Ku, J.; Mozifian, M.; Lee, J.; Harakeh, A.; Waslander, S.L. Joint 3D Proposal Generation and Object Detection from View Aggregation. In Proceedings of the 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), Madrid, Spain, 1–5 October 2018; pp. 1–8. [CrossRef]   
65. Glorot, X.; Bengio, Y. Understanding the difficulty of training deep feedforward neural networks. In Proceedings of the 13th International Conference on Artificial Intelligence and Statistics, Sardinia, Italy, 13–15 May 2010; Volume 9, pp. 249–256.   
66. Zaremba, W.; Sutskever, I.; Vinyals, O. Recurrent Neural Network Regularization. arXiv 2014, arXiv:1409.2329.   
67. Gal, Y.; Ghahramani, Z. A Theoretically Grounded Application of Dropout in Recurrent Neural Networks. In Advances in Neural Information Processing Systems; Lee, D., Sugiyama, M., Luxburg, U., Guyon, I., Garnett, R., Eds.; Curran Associates, Inc.: Red Hook, NY, USA, 2016; Volume 29.   
68. Yao, Y.; Rosasco, L.; Caponnetto, A. On Early Stopping in Gradient Descent Learning. Constr. Approx. 2007, 26, 289–315. [CrossRef]   
69. Tieleman, T.; Hinton, G. Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude. COURSERA Neural Netw. Mach. Learn. 2012, 4, 26–31.   
70. van den Oord, A.; Dieleman, S.; Zen, H.; Simonyan, K.; Vinyals, O.; Graves, A.; Kalchbrenner, N.; Senior, A.W.; Kavukcuoglu, K. WaveNet: A Generative Model for Raw Audio. In Proceedings of the 9th ISCA Speech Synthesis Workshop, Sunnyvale, CA, USA, 13–15 September 2016; p. 125.   
71. van den Oord, A.; Kalchbrenner, N.; Espeholt, L.; Koray K.; Vinyals, O.; Graves, A. Conditional Image Generation with PixelCNN Decoders. In Advances in Neural Information Processing Systems; Curran Associates, Inc.: Red Hook, NY, USA, 2016; Volume 29.