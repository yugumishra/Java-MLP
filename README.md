# Java-MLP
A Java based MLP implementation built with standard java libraries capable of the MNIST identification task (~92% accuracy)

The code features a model system in which the structure of an MLP can be very easily generated. It also features custom Vector and Matrix classes in which all of the linear algebra is done. Backpropagation is also done manually, and the model learns through Stochastic Gradient Descent. Since this is a basic implementation of an MLP, it does not have advanced features like Batch Normalization or more advanced optimizers like Adam or RMSProp. This makes the achieved performance that much more amazing, since ~92% accuracy is not far off from human classification levels. 

The entire code is within ~1k lines of code, and does the i/o for MNIST itself. It can also serialize the model for later training. The model is also tested on out of dataset images and can apply its learned patterns to those as well (it can identify a 7 that I wrote myself with 98% confidence). 

Sample Output:

Train (0) or read (1)?

0

Number of training epochs?

15

Size of training batch?

32

Epoch 1, ANN's MSE error: 0.003384316054478322. Time Taken (s): 4.842

Epoch 2, ANN's MSE error: 0.002207744208299484. Time Taken (s): 4.803

Epoch 3, ANN's MSE error: 0.002108677300802786. Time Taken (s): 4.668

Epoch 4, ANN's MSE error: 0.002066717824497569. Time Taken (s): 4.882

Epoch 5, ANN's MSE error: 0.0020507508695820635. Time Taken (s): 4.837

Epoch 6, ANN's MSE error: 0.0020340403332855667. Time Taken (s): 4.887

Epoch 7, ANN's MSE error: 0.0020272876782044358. Time Taken (s): 4.863

Epoch 8, ANN's MSE error: 0.002025573951536998. Time Taken (s): 4.874

Epoch 9, ANN's MSE error: 0.002017311516928116. Time Taken (s): 4.95

Epoch 10, ANN's MSE error: 0.002016633001138814. Time Taken (s): 4.99

Epoch 11, ANN's MSE error: 0.0020151465169063517. Time Taken (s): 4.949

Epoch 12, ANN's MSE error: 0.002014477589170922. Time Taken (s): 4.907

Epoch 13, ANN's MSE error: 0.00201351760077232. Time Taken (s): 4.907

Epoch 14, ANN's MSE error: 0.002012708490507474. Time Taken (s): 4.933

Epoch 15, ANN's MSE error: 0.0020113223235459356. Time Taken (s): 5.051


Testing on 10000 accuracy (%): 91.79


Prediction for digit (7) outside of dataset: 7



