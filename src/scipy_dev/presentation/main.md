---
paginate: true
marp: true
author: Janos Gabler and Tim Mensinger
description: "Scipy 2022: Estimagic Tutorial"
theme: custom
---

<!-- ===================================================================================
# TITLE PAGE
==================================================================================== -->

<!-- paginate: false -->

## Practical Numerical Optimization with Estimagic, JaxOpt, Scipy

Scipy Conference 2022

<br/>

#### Janos Gabler & Tim Mensinger

University of Bonn

---

<!-- ===================================================================================
# First hour
==================================================================================== -->


<!-- paginate: true -->

### About Us

- link to webpage

---

### Index

1. What is numerical optimization
2. Introduction to `scipy.optimize`
3. Introduction to `estimagic`
4. Choosing algorithms
5. Advanced estimagic
6. Jax and Jaxopt

---

<!-- _class: lead -->
# What is numerical optimization

---

### Example problem

<!-- _class: split -->

<div class=leftcol>

- Parameters $x_1$, $x_2$
- Criterion $f(x_1, x_2) = x_1^2 + x_2^2$
- Want: $x_1^*, x_2^* = argmin_{x_1, x_2} f(x_1, x_2)$
- Possible extensions:
    - Constraints
    - Bounds

</div>



<div class=rightcol>

![sphere](../graphs/sphere.png)

</div>



---

### Brute force vs. smarter algorithm

<!-- Paper 1 -->
- contour plot of function with gridpoints
- contour plot of function with lbfgsb history

---

### Complexity of brute force

<!-- Paper 2 -->
- 1d grid of points
- 2d grid of points
- 3d grid of points
- lineplot of exponential growth of number of gridpoints with 10 points per dim

---

### What's (not) in this talk


- Covered
    - Nonlinear optimization of continuous parameters
    - Linear and nonlinear constraints
    - Global optimization
- Not covered
    - Linear programming
    - Mixed integer programming



---

<!-- _class: lead -->
# Introduction to scipy.optimize

---

### Solve example problem with scipy.optimize

```python
>>> import numpy as np
>>> from scipy.optimize import minimize

>>> def sphere(x):
>>>     return np.sum(x ** 2)

>>> x0 = np.ones(2)
>>> res = minimize(f, x0)
>>> res.fun
0.0
>>> res.x
array([0.0, 0.0])
```
---

### Features of scipy.optimize

- `minimize` as unified interface to 14 local optimizers
    - some support bounds
    - some support (non)linear constraints
- Parameters are 1d numpy arrays
- Numerical derivatives are calculated if necessary
- Maximization is done by minimizing $- f(x)$

---


<!-- _class: lead -->
# Practice Session 1: First optimization with scipy.optimize (15 min)

---

### Shortcomings of scipy.optimize

- Very limited number of algorithms
- If optimization crashes, all information is lost
- No parallelization
- Maximization via minimize is error-prone and cumbersome
- No diagnostics tools (e.g. visualization of histories)
- No feedback before optimization ends
- Parameters are 1d numpy arrays
- No built-in multistart, benchmarking, scaling, reparametrization or logging
- **In short**: scipy.optimize is a low level library

---

<!-- _class: lead -->
# Introduction to estimagic

---

### You can use it like scipy


---

### Params can be anything


---

### OptimizeResult



---

### Criterion plot

---

### Params plot


---

### Algorithms from scipy, nlopt, TAO, pygmo, ...


---

### Constraints via reparametrizations


---

### Closed-form or parallel numerical derivatives


---

### There is maximize


---

### Built in multistart framework


---

### Least squares optimizers


---

### Logging and Dashboard


---

### Harmonized `algo_options`

---

### The estimagic Team


---
<!-- ===================================================================================
# Second hour
==================================================================================== -->
<!-- _class: lead -->
# Break (5 min)

---


<!-- _class: lead -->
# Practice Session 2: Convert previous example to estimagic (15 min)

---

<!-- _class: lead -->
# Choosing algorithms

---

### Relevant problem properties


---

### Classes of algorithms


---

### `scipy_lbfgsb`


---

### `fides`


---

### `nlopt_bobyqa`


---

### `nlopt_neldermead`

---

### `nag_dfols`


---

### `scipy_ls_lm`

- Or other scipy ls. Need to benchmark


---

### `ipopt`


---
<!-- _class: lead -->
# Practice Session 3: Play with `algorithm` and `algo_options` (15 min)

---

### What is benchmarking

- highlight that we have a large number of problems
- cite paper about best practices
- ...

---

### Running benchmarks in estimagic


---
### Profile plots

- two column slide with normalized and absolute profile plot


---


### Convergence plots

---


### Built in benchmark suites and customization of problems

- Example
- Estimagic
- More-Wild
- ...
- How to add noise
- How to add bad scaling

---


<!-- _class: lead -->
# Practice Session 4: Benchmarking optimizers (10 min)

---


<!-- _class: lead -->
# Break (10 min)

---

<!-- ===================================================================================
# Third hour
==================================================================================== -->

### Terminology of constraints in estimagic

- bounds: handled by most algorithms
- estimagic constraints: handled via reparametrization and bounds
- nonlinear constraints: handled by some algorithms

---

### What is reparametrization

- simple example with increasing constraint

---

### Example problem in two flavors

- take the one from estimagic docs
- dict version
- df version

---

### Fixing parameters

- two columns, dict and df version


---

### Linear constraints


---

### What else can be done with reparametrization

- list constraint types
- link to docs


---


### Nonlinear constraints


---


<!-- _class: lead -->
# Practice Session 5: Constrained optimization (10 min)


---


### What is global optimization

- needs bounds to be well defined!

---

### Genetic algorithms

---

### Bayesian optimization

---

### Multistart optimization

---

### How to choose



---


### Example 1: Non-smooth, 5 Parameters



---

### Example 2: Smooth, 15 Parameters

- alpine on left, criterion plot on right

---

### Global optimization survival guide

- Do n



---

<!-- _class: lead -->

### Numerical instability during optimization


---

### Error handling in estimagic

---

### Scaling of optimization problems


---
### Scaling in estimagic



---
<!-- _class: lead -->
# Practice Session 6: Scaling of optimization problems (10 min)


---

<!-- ===================================================================================
# Last hour
==================================================================================== -->

### Other features and documentation of estimagic

---

### Numerical dervatives vs. automatic differentiation


---
### What is JAX


---

### Calculating derivatives with JAX


---

<!-- _class: lead -->
# Practice Session 7: Using JAX derivatives in estimagic (10 min)


---

### What is JAXopt and when to use it



---

### Simple optimization in JAXopt


---

### Vmap in JAX

---

### Vectorized optimization in JAXopt


---

### Differentiate an optimizer in JAXopt


---

<!-- _class: lead -->
# Practice Session 8: Vectorized optimization in JAXopt (15 min)

---

### Summary


---


<!-- ===================================================================================
# snippets
==================================================================================== -->
<!-- _class: lead -->
# Snippets

---
### A two-column slide
<!-- _class: split -->

<div class=leftcol>

bla

</div>

<div class=rightcol>

#### Title right column

```python
a = 1
```

</div>

---

<!-- ===================================================================================
# REFERENCES
==================================================================================== -->

### References

- JAX Opt
- scipy.optimize
