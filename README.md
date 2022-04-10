# Driver2Vec
Reproduction of Driver2Vec paper. 
Link to blog: https://achillebailly.github.io/Driver2Vec/

Authors: Danish Khan, Achille Bailly and Mingjia He 

[//]: # (Original paper[1]: Yang, J., Zhao, R., Zhu, M., Hallac, D., Sodnik, J., & Leskovec, J. (2021). Driver2vec: Driver identification from automotive data. arXiv preprint arXiv:2102.05234.)

## Introduction
The neural network architecture Driver2Vec is discussed and used to detect drivers from automotive data in this blogplot. Yang et al. published a paper in 2021 that explained and evaluated Driver2Vec, which outperformed other architectures at that time. Driver2Vec (is the first architecture that) blends temporal convolution with triplet loss using time series data [1]. With this embedding, it is possible to classify different driving styles. The purpose of this blog post is to give a full explanation of this architecture as well as to develop it from the ground up.

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

### Gradient Boosting Decision Trees (LightGBM)

## Data
## Results
## Reference
[1] Yang, J., Zhao, R., Zhu, M., Hallac, D., Sodnik, J., & Leskovec, J. (2021). Driver2vec: Driver identification from automotive data. arXiv preprint arXiv:2102.05234.

[2] Francesco, L. (2021). Temporal Convolutional Networks and Forecasting. https://unit8.com/resources/temporal-convolutional-networks-and-forecasting/

[3] Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv preprint arXiv:1803.01271.

[4] Haar Wavelets http://dsp-book.narod.ru/PWSA/8276_01.pdf
