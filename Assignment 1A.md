
What are Channels and Kernels (according to EVA)?
Kernel is filter,feature extractor and 3*3 metrics and pea in Pulav as per the example given in EVA.
Every channel has its own usage.Channels are expected to contain similar information.All vertical lines and colors.

Why should we only (well mostly) use 3x3 Kernels?
It is heavily optimised.Generally go for odd nos.
Even nos are not used as we cannot do a triangle using 4X4.
Using 3X3 twice is better than using 5X5 once as both convolve on 1X1.
Former has only 18 pixels and latter has 25 pixels.Fomer runs super fast on any other kernel.
3X3 is state of art and is haeavily optimised.

How many times do we need to perform 3x3 convolution operation to reach 1x1 from 199x199 (show calculations)
Answer is 100
199X199
197X197
195X195
193X193
191X191
189X189
187X187
185X185
183X183
181X181
179X179
177X177
175X175
173X173
171X171
169X169
167X167
165X165
163X163
161X161
159X159
157X157
155X155
153X153
151X151
149X149
147X147
145X145
143X143
141X141
139X139
137X137
135X135
133X133
131X131
129X129
127X127
125X125
123X123
121X121
119X119
117X117
115X115
113X113
111X111
109X109
107X107
105X105
103X103
101X101
99X99
97X97
95X95
93X93
91X91
89X89
87X87
85X85
83X83
81X81
79X79
77X77
75X75
73X73
71X71
69X69
67X67
65X65
63X63
61X61
59X59
57X57
55X55
53X53
51X51
49X49
47X47
45X45
43X43
41X41
39X39
37X37
35X35
33X33
31X31
29X29
27X27
25X25
23X23
21X21
19X19
17X17
15X15
13X13
11X11
9X9
7X7
5X5
3X3
1X1

How are kernels initialized? 
Kernels are updated via backpropagation when training although we always have a choice of not updating some/all kernels.
Also , it is possible to set some of these kernels with predetermined 'patterns' to accelerate the learning. This is equivalent to saying that we are using a weight initialization method for CNN.


What happens during the training of a DNN?
Neural network consist of different connected layers of nodes, each layer has many coefficients and bias, initialized randomly for example.Training a neural network to do something is basically trying to reach the optimal coefficients and bias that would match an input to the desired output.
If we want to design a neural network that can recognize if a picture is a picture of a dog or a cat, then given an input image, the neural network will try to predict whether the picture is a dog or a cat. With the result of the prediction, the neural net will compute the error between the result and the expected result. This error will be back propagated through the layers and will evolve the parameters of the layers in order to reduce the error for the next predictions.
We always need an initial data set to train your neural network. Neural networks is all about learning from data.






