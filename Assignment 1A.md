
What are Channels and Kernels (according to EVA)?
Kernel is filter,feature extractor and 3*3 metrics and pea in Pulav as per the example given in EVA.
Every channel has its own usage.Channels are expected to contain similar information.All vertical lines and colors.

Why should we only (well mostly) use 3x3 Kernels?
It is heavily optimised.Generally go for odd nos.
Even nos are not used as we cannot do a triangle using 4X4.
Using 3X3 twice is better than using 5X5 once as both convolve on 1X1.
Former has only 18 pixels and latter has 25 pixels.Fomer runs super fast on any other kernel.
#X3 is state of art and is haeavily optimised.


