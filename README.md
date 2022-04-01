# Driver2Vec
Reproduction of Driver2Vec paper. 

Authors: [your name] and Mingjia He 

Original paper[1]: Yang, J., Zhao, R., Zhu, M., Hallac, D., Sodnik, J., & Leskovec, J. (2021). Driver2vec: Driver identification from automotive data. arXiv preprint arXiv:2102.05234.

## Introduction


## Method

### Temporal Convolutional Network (TCN)
Temporal Convolutional Networks (TCN) combines the architecture of convolutional networks and recurrent networks. 
The principle of TCN consists of two aspects: 
1) The output of TCN has the same length as the input. 
2) TCN uses causal convolutions, where an output at a specific time step is only depend on the input from this time step and earlier in the previous layer.

To ensure the first principle, zero padding is applied. As shown in Figure 1, the zero padding is applied on the left side of the input tensor and ensure causal convolution. In this case, the kernel size is 3 and the input length is 4. With a padding size of 2, the output length could be equal to the input length. 

<div align=center><img width="380" height="220" alt="zero padding" src="https://user-images.githubusercontent.com/101323945/161212963-e3fcf12a-edd9-4c15-9f1a-f37c42b28ab2.png"/></div>

<p align="center">Figuer 1 zero padding[2]</p>

One of the problems of the casual convolution is that the history size it can cover is linear in the depth of network. Simple casual convolution could be challenging when dealing with sequential tasks that require a long history coverage, as very deep network would have many parameters, which may expand training time and lead to overfitting. Thus, dilated convolution is used to increase the receptive field size while having a small number of layers. Dilation means the interval length between the elements in a layer used to compute one element of the next layer. The convolution with a dilation of one is a simple regular convolution, In TCN, dilation exponentially increases as the lay moves deeper. As shown in Figure 2, as the network moves deeper, the elements in the next layer cover larger range of elements in the former layer.

<div align=center><img width="350" height="240" alt="zero padding" src="https://user-images.githubusercontent.com/101323945/161215806-812c7e4f-661e-49a6-b189-e8ad72517d3c.png"/></div>
<p align="center">Figure 2 An example of dilated causal convolution[1]</p>

TCN employs generic residual module in place of a convolutional layer. The structure of residual connect is shown in Figure 3, in each residual block, TCN has two layers including dilated causal convolution, weight normalization, rectified linear unit (ReLU) and dropout. 

<div align=center><img width="580" height="280" alt="zero padding" src="https://user-images.githubusercontent.com/101323945/161216087-b0570b3b-dcf5-4b4b-87ef-6c2ea2abfc77.png"/></div>
<p align="center">Figure 3 The residual module in TCN[1]</p>


### Haar Wavelet Transform 

### Gradient Boosting Decision Trees (LightGBM)

## Data
## Results
## Reference
[1] Yang, J., Zhao, R., Zhu, M., Hallac, D., Sodnik, J., & Leskovec, J. (2021). Driver2vec: Driver identification from automotive data. arXiv preprint arXiv:2102.05234.

[2] Francesco, L. (2021). Temporal Convolutional Networks and Forecasting. https://unit8.com/resources/temporal-convolutional-networks-and-forecasting/
