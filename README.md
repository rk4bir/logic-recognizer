# LOGIC RECOGNIZER
> Goal of this project is to recognize logics behind the Logic-gates 
> of electronics, using DeepLearning Method. But you are not constrained
> to that. You can modify some pieces of code to make it useful to recognize 
> any logic :-)
> The recognizer was tested against **AND, NAND, NOR, EXNOR, EXNAND**
> logic-gates and the output is quite promising. It gives 100% accuracy
---

## LOGIC-GATE TRUTH TABLES
![image](/img/truth_table.gif)

### Schematic representation of 2-3-2 Network (Net)

![image](/img/2-3-2net.png)

### 2-3-2 Net with weights, biases and activations.

![image](/img/2-3-2detailnet.jpg)

> **NB:** You can modify the network structure as like you wish. e.g. 2-3-3-2.
---

### Sigmoid function
> Reduce any real number value in between 0 to 1.0
![image](/img/sigmoid.png)


---
# Introduction
We want to train our recognizer/classifer with some data, showing the output and and we want 
the network to test with any input data. We don't want to specify the logic manually instead we expect the network to understand/learn the logics itself by analyzing the input/label data. Although it's not neccessary to implement DeepLearning method to this simple if/else
job.

# STRUCTURE
* **Dataset:** 
	* Generates random data and label.
	* 2 input data i.e. x0 and x1
	* Split data and label to test and train portion

* **Network:**
	* Initialize random weights and biases
	* Forward propagation: activations calculation
	* Loss/Cost calculation: measurement of network's behaviour
	* Delta values (for weights & biases) calculation with **Gradient descent** method
	* Backward propagation: modification of connection strengths & biases by updating them
	* Testing the network's performance

> Output layer got 2 neurons, 1st neuron's activation means output is 0 and 2nd's means output is 1.
---

# STEPS
* Calculate layer by layer activations
* Calculate loss/cost function
* Calculate cost/loss derivative
* Calculate sigmoid derivative/derivative of activation
* Calculate delta and nabla values for weights and biases
* Define learning rate
* Update weights and biases

## Activation calculation
```python
a = self.sigmoid(np.dot(w,a) + b)
```
## Loss/Cost calculation
```python
# a[-1] is the activations of output layer and
# n is no. of training data
self.loss = np.sum(np.power(np.subtract(self.a[-1], label), 2)) / 2*self.dataset.batch_size
```
### Cost derivative
```python
cost_derivative = np.subtract(self.a[-1], y)
```
### Derivative of activation
```python
def sigmoid_derivative(a):
	return a * (1-a)
```

### Delta and nabla calculations
**For output layer**
```python
delta = cost_derivative * self.sigmoid_derivative(self.a[-1])
self.nabla_w[-1] = delta.dot(self.a[-2].T)
self.nabla_b[-1] = delta
```
**For hidden layers:** codes for the previous layer of output
```python
# delta in the right hand side is for output layer
delta = np.dot(self.weights[-1].T, delta) * self.sigmoid_derivative(self.a[-2])
self.nabla_b[-i] = delta
self.nabla_w[-i] = np.dot(delta, self.a[-3].T)
```

### Update weights and biases
```python
weight = weight - (learning_rate/self.dataset.batch_size)*nabla_w
bias = bias - (learning_rate/self.dataset.batch_size)*nabla_b
```


### Execute the codes

**Install dependencies**
```bash
pip3 install -r requirements.txt
```
**Codes were written for Python v3.6.6**

> Primarily I've set the network as [2-3-2] i.e. 1 hidden layer having 3 neurons and output
> layer having 2 neurons. You can change it in `n = Network([2,3,2]size=10000)`. 
> Off course you don't want to change first and last layer's neuron numbers. For 
> example You can have 2 hidden layer then you'd provide something like
> `n = Network([2,3,3,2]size=10000)`




**Run a logic gate**
```bash
python3 andGate.py
```