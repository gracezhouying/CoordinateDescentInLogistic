# CoordinateDescentInLogistic
Minimizing Logistic Loss Function with Coordinate Descent 

## Goal: 
Consider a standard unconstrained optimization problem: min L(w). 
Here, I would use coordinate descent method to minimize this loss function.

## Data:
There are three levels of y, so it would be a multilevel logistic regression. There are 42 variables.

## Which coordinate to choose in each iteration?
Given w, I will calculate the gradient of each w_j. 
Compare the absolute values of 42 gradients and find the index of the largest value. 
Choose this coordinate and update its corresponding w_index. 
I use the absolute gradient value as choosing standard since the larger the value is, the faster  will reaching optimal.

## How to set the new value to the coordinate?
The new value would be 
### w^{hat}_{l} = argmin L(w^{hat}_{0}, w^{hat}_{1},...,w^{hat}_{l-1},w,w^{hat}_{l+1},w^{hat}_{p})
I will use function scipy.optimize.fmin 𝑗 to calculate it
In scipy.optimize.fmin, Loss function have to be convex, thus it is possible to find minimum value and the corresponding w. 
Loss function should also be differentiable, since coordinate select is depend on gradient calculation.
