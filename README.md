# Driver2Vec
Reproduction of Driver2Vec paper. 
Link to blog: https://achillebailly.github.io/Driver2Vec/

Authors: Danish Khan, Achille Bailly and Mingjia He 

Original paper[1]: Yang, J., Zhao, R., Zhu, M., Hallac, D., Sodnik, J., & Leskovec, J. (2021). Driver2vec: Driver identification from automotive data. arXiv preprint arXiv:2102.05234.

## Introduction
The neural network architecture Driver2Vec is discussed and used to detect drivers from automotive data in this blogpost. Yang et al. published a paper in 2021 that explained and evaluated Driver2Vec, which outperformed other architectures at that time. Driver2Vec (is the first architecture that) blends temporal convolution with triplet loss using time series data [1]. With this embedding, it is possible to classify different driving styles. The purpose of this reproducibility project is to recreate the Driver2Vec architecture and reproduce Table 5 from the original paper. The purpose of this blog post is to give a full explanation of this architecture as well as to develop it from the ground up.

Researchers employ sensors found in modern vehicles to determine distinct driving patterns. In this manner, the efficacy is not dependent on invasive data, such as facial recognition or fingerprints. A system like this may detect who is driving the car and alter its vehicle settings accordingly. Furthermore, a system that recognizes driver types with high accuracy may be used to identify unfamiliar driving patterns, lowering the chance of theft.

## Method

### Temporal Convolutional Network (TCN)
Temporal Convolutional Networks (TCN) combines the architecture of convolutional networks and recurrent networks. 
The principle of TCN consists of two aspects: 
1) The output of TCN has the same length as the input. 
2) TCN uses causal convolutions, where an output at a specific time step is only depend on the input from this time step and earlier in the previous layer.

To ensure the first principle, zero padding is applied. As shown in Figure 1, the zero padding is applied on the left side of the input tensor and ensure causal convolution. In this case, the kernel size is 3 and the input length is 4. With a padding size of 2, the output length is equal to the input length. 

<div align=center><img width="380" height="220" alt="zero padding" src="https://user-images.githubusercontent.com/101323945/161212963-e3fcf12a-edd9-4c15-9f1a-f37c42b28ab2.png"/></div>

<p align="center">Figuer 1 zero padding[2]</p>

One of the problems of casual convolution is that the history size it can cover is linear in the depth of network. Simple casual convolution could be challenging when dealing with sequential tasks that require a long history coverage, as very deep network would have many parameters, which may expand training time and lead to overfitting. Thus, dilated convolution is used to increase the receptive field size while having a small number of layers. Dilation is the name for the interval length between the elements in a layer used to compute one element of the next layer. The convolution with a dilation of one is a simple regular convolution. In TCN, dilation exponentially increases as progress through the layers. As shown in Figure 2, as the network moves deeper, the elements in the next layer cover larger range of elements in the previous layer.

<div align=center><img width="350" height="240" alt="zero padding" src="https://user-images.githubusercontent.com/101323945/161215806-812c7e4f-661e-49a6-b189-e8ad72517d3c.png"/></div>

<p align="center">Figure 2 An example of dilated causal convolution[3]</p>

TCN employs generic residual module in place of a convolutional layer. The structure of residual connection is shown in Figure 3, in each residual block, TCN has two layers including dilated causal convolution, weight normalization, rectified linear unit (ReLU) and dropout. 

<div align=center><img width="580" height="280" alt="zero padding" src="https://user-images.githubusercontent.com/101323945/161216087-b0570b3b-dcf5-4b4b-87ef-6c2ea2abfc77.png"/></div>

<p align="center">Figure 3 The residual module in TCN[3]</p>


### Haar Wavelet Transform 

Driver2vec applied Haar wavelet transformation to generates two vectors in the frequency domain. Wavelet Transform decomposes a time series function into a set of wavelets. A Wavelet is an oscillation use to decompose the signal, which has two characteristics, scale and location. Large scale can capture low frequency information and conversely, small scale is designed for high frequency information. Location defines the time and space of the wavelet. 

The essence of Wavelet Transform is to how much of a wavelet is in a signal for a particular scale and location. The process of Wavelet Transform consists of four steps: 

1) the wavelet moves across the entire signal with various location;
2) the coefficients of trend and fluctuation for at each time step is calculated use scalar product (in following equations);
3) increase the wavelet scale and repeat the process.



<div align=center>
  
  ![](https://latex.codecogs.com/svg.image?a_{m}=f\bullet&space;W_{m})
  
  ![](https://latex.codecogs.com/svg.image?d_{m}=f\bullet&space;V_{m})
  
</div>

Most specifically, the Haar transform decomposes a discrete signal into two sub-signals of half its length, one is a running average or trend and the other is a running difference or fluctuation. As shown in the following equations, the first trend subsignal is computed from the average of two values and fluctuation, the second trend subsignal, is computed by taking a running difference, as shown in Equation 2. This structure enable transform to detect small fluctuations feature in signals. Figure 3 shows how Haar transform derives sub-signals for the signal f=(4, 6, 10, 12, 8, 6, 5, 5)

<div align=center>
  
  ![](https://latex.codecogs.com/svg.image?a_{m}=\frac{f_{2m-1}&plus;f_{2m&plus;1}}{\sqrt{2}})
  
  ![](https://latex.codecogs.com/svg.image?a_{m}=\frac{f_{2m-1}-f_{2m&plus;1}}{\sqrt{2}})
  
</div>

<div align=center><img width="550" height="260" alt="zero padding" src="https://user-images.githubusercontent.com/101323945/161373770-d9e80326-a68f-4522-9e99-5868b88a912d.png"/></div>

<p align="center">Figure 4 An exampel for Haar transform[4]</p>

### Full architecture

The two vectors that the wavelet transform outputs are then fed through a Fully Connected (FC) layer to map them to a 15 dimensional vector. Both of them are concatenated with the last output of the TCN and fed through a final FC layer with Batch Normalization and a sigmoid activation function to get our final embedding.

### Triplet Margin Loss

Once the output of the wavelet transform and the TCN are combined throught a Fully Connected layer to form the embedding, we need a way to train the network. With no ground truth to compare the output to, the *triplet margin loss* is used. At its core, this criterion pulls together the embeddings that are supposed to be close and pushes away the ones that are not. Mathematically, it is defined as follows:

<div align=center>

  ![](https://latex.codecogs.com/svg.image?\bold{L}(x_{r},x_{p},x_{n})=max(0,D_{rp}^{2}&plus;D_{rp}^{2}&plus;margin))
  
</div>

Where ![](https://latex.codecogs.com/svg.image?x_{r,p,n}) are the embeddings for the anchor, positive and negative samples respectively, ![](https://latex.codecogs.com/svg.image?D_{rp}) (resp. ![](https://latex.codecogs.com/svg.image?D_{rn})) is the distance (usually euclidian) between the anchor and the positive embdeggings (resp. negative) and the margin is a positive number.

With the available dataset being so limited, choosing the positive and negative samples for each anchor at random is probably enough. In most cases however, the most efficient way of choosing them is to pick the worst ones for each anchor (see [4]), i.e. chossing the positive sample that is the farthest away and the negative one that is the closest. Again, for more detail on how to actually do that efficiently, go to the website referenced in [4] for a very detailed explanation.

### Gradient Boosting Decision Trees (LightGBM)
Before introducing Light GBM, we first illustrate what is boosting and how it can work. The goal of boosting is improving the prediction power converting weak learners into strong learners. The basic logit is to build a model on the training dataset, and then build the next model to rectify the errors present in the previous one. In this procedure, weights of observations are updates according to the rule that wrongly classified observations would have increasing weights. So, only those misclassified observations get selected in the next model and the procedure iterate until the errors are minimized. 

Gradient Boosting trains many models in an additive and sequential manner, using gradient decent to minimize the loss function One of the most popular types of gradient boosting is boosted decision trees. There are two different strategies to compute the trees: level-wise and leaf-wise, as shown in the following figure. The level-wise strategy grows the tree level by level. In this strategy, each node splits the data prioritizing the nodes closer to the tree root. The leaf-wise strategy grows the tree by splitting the data at the nodes with the highest loss change. 


<div align=center><img width="520" height="160" alt="" src="https://user-images.githubusercontent.com/101323945/163355753-4eda483e-61ea-4634-aacf-e4f736e58a45.png"/></div>
<p align="center">Figure 5 The level-wise strategy[5]</p>

<div align=center><img width="520" height="160" alt="" src="https://user-images.githubusercontent.com/101323945/163355769-b5cb412b-249e-43ef-9729-180b50527689.png"/></div>
<p align="center">Figure 6 The leaf-wise strategy[5]</p>


However, conventional gradient decision tree could be inefficient when dealing with large scale data set. That is why Light GBM is proposed, which is a gradient boosting decision tree with Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB). 
Light GBM is based on tree-based learning algorithms growing tree vertically (leaf-wise).  It is designed to be  distributed and efficient with the following advantages[6]:

•	Faster training speed and higher efficiency.

•	Lower memory usage.

•	Better accuracy.

•	Support of parallel, distributed, and GPU learning.

•	Capable of handling large-scale data.


GOSS is design for the sampling process with the aim to reduce computation cost and not lose much training accuracy. The instances with large gradients would be better kept considering those bearing more information gain, and the instances with small gradients will be randomly drop. EFB tries to effectively reduce the number of features in a nearly lossless manner. A feature scanning algorithm is designed to build feature histograms from the feature bundles. The algorithms used in presented in the following figures (refer to the work of Ke. G et al.[7]).



<div align=center><img width="440" height="250" alt="" src="https://user-images.githubusercontent.com/101323945/163355909-37c69ea4-0575-4709-94b9-7012e2874b38.png"/>
</div>
<div align=center><img width="440" height="250" alt="" src="https://user-images.githubusercontent.com/101323945/163355924-f46f4075-726f-4311-bf30-49ab9dd5f620.png"/>
 </div>
<p align="center">Figure 7 Algorithms for Light GBM[7]</p>

  
Light GBM has been widely used due to its ability to handle the large size of data and takes lower memory to run. But it should be noted that there are shortcomings: Light GBM is sensitive to overfitting and can easily overfit small data. 


## References
[1]: Yang, J., Zhao, R., Zhu, M., Hallac, D., Sodnik, J., & Leskovec, J. (2021). Driver2vec: Driver identification from automotive data. arXiv preprint arXiv:2102.05234.

[2]: Francesco, L. (2021). Temporal Convolutional Networks and Forecasting. https://unit8.com/resources/temporal-convolutional-networks-and-forecasting/

[3]: Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv preprint arXiv:1803.01271.

[4]: Good explanation and implementation (in Tensorflow) of the Triplet Loss: https://omoindrot.github.io/triplet-loss

[5] What is LightGBM https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc

[6] LightGBM’s documentation, Microsoft Corporation https://lightgbm.readthedocs.io/en/latest/

[7] Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T. Y. (2017). Lightgbm: A highly efficient gradient boosting decision tree. Advances in neural information processing systems, 30.


