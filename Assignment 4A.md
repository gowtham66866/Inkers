We can have any number of layers in a nueral network.
Max Pooling is a 2X2 convolution which  reduces image by half by consdering the maximum value in the image.
1X1 convolution does reduce the image size but we can reduce the no of kernels.
3X3 conolution reduces the image size by 4 and is better than 5X5 and 7X7 convolutions as no of parameters used is less.
The receptive field at the final prediction layer or output layer to be equal to the size of the image.
The softmax function is used in the final layer of a neural network-based classifier.It is not probability but probability like.
The learning rate is a hyperparameter which determines to what extent newly acquired information replaces old information. 
We need a set of edges and gradients to be detected to be able to represent the whole image. Through experiments, we have learned that we should use around 32 or 64 kernels in the first layer, increasing the number of kernels slowly. We can assume we add 32 kernels in the first layer, 64 in second, 128 in thrid and so on.Kernel convolves on the image to obtain the feature map.
Batch normalization normalizes the output of a previous activation layer by subtracting the batch mean and dividing by the batch standard deviation to increase stability of the netwrok.
In image normalization is a process that the range of pixel intensity values. Applications have photographs with poor contrast due to glare, for example. Normalization is also called contrast stretching or histogram stretching.
Position of max pooling should not be close to input image.There should be difference of 2 layers atleast.
Transition layer is the layer where maxpooling is applied.
Position of transition layer is atleast 2 layers away from the input image.
Number of epochs is the no of times training set runs to train the model.We need to increase the number of epochs when we need to increase th accuracy of the model.
Dropout is used to increase accuracy of the test set to reduce over fitting. and reduces the difference between training and test set.
When the difference between training set and test set is large, then we use drop out to reduce overfitting.
The final global receptive field (at the final prediction layer or output layer) to be equal to the size of the image. Max pooling needs to be atleast 2 layers away from the final prediction layer.
Batch normalisation can be next to the final prediction layer.
We go ahead with a larger kernel when we need better accuracy.
We know our kernel is not going well when the accuracy does does not pick in the first couple of epochs in the model.
Batch size is the number of convolutions in batch.The batch size helps improve accuracy.
We need to add validation checks if check if our model is training correctly or not.
While training deep neural networks, it is quite useful to reduce learning rate as the training progresses. This can be done by using pre-defined learning rate schedules or adaptive learning rate methods. 
Adam variant employs dynamic bounds on learning rates to achieve a gradual and smooth transition to SGD. SGD is a variant of gradient descent. Instead of performing computations on the whole dataset that redundant and inefficient, SGD only computes on a small subset or random selection of data examples. SGD produces the same performance as regular gradient descent when the learning rate is low.
