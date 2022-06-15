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

## Practical Numerical Optimization with Scipy, Estimagic and JAXopt

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
<!-- _class: split -->
<div class=leftcol>

<img src="../graphs/janos.jpg" alt="janos" width="200"/>

- Website: [janosg.com](https://janosg.com)
- GitHub: [janosg](https://github.com/janosg/)
- Submitted PhD thesis
- Original author of estimagic
- Looking for interesting jobs soon

</div>
<div class=rightcol>

<img src="../graphs/tim.jpeg" alt="tim" width="200"/>

- Website: [tmensinger.com](https://tmensinger.com)
- GitHub: [timmens](https://github.com/timmens)
- Core contributor of estimagic
- PhD student in Econ, University of Bonn


</div>

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
- Optimum at $(0, 0)$ with function value $0$

</div>

<div class=rightcol>

<img src="../../../bld/figures/sphere.png" alt="sphere" width="500" class="center"/>


</div>

---

### Brute force vs. smarter algorithm

<!-- Paper 1 -->
<!-- _class: split -->
<div class=leftcol>

<img src="../../../bld/figures/grid_search.png" alt="brute-force" width="400" class="center"/>

</div>
<div class=rightcol>

<img src="../../../bld/figures/gradient_descent.png" alt="smart" width="400" class="center"/>

</div>


---

### Complexity of brute force
<!-- _class: split -->
<style scoped>
section.split {
    grid-template-columns: 400px 700px;
}
</style>


<div class=leftcol>

<img src="../../../bld/figures/curse_of_dimensionality_v.png" alt="dimensionality" width="180" class="center"/>

</div>
<div class=rightcol>

<style scoped>
table {
  font-size: 30px;
}
</style>

| Number of <br /> Dimensions | Runtime (1 ms per evaluation, <br /> 100 points per dimension) |
| ----------------------------| ---------------------------------------------------------------|
| 1                           | 100 ms                                                         |
| 2                           | 10 s                                                           |
| 3                           | 16 min                                                         |
| 4                           | 27 hours                                                       |
| 5                           | 16 weeks                                                       |
| 6                           | 30 years                                                       |

</div>


---


### What's (not) in this talk


- Covered
    - Nonlinear optimization of continuous parameters
    - Linear and nonlinear constraints
    - Global optimization
    - Diagnostics and strategies for difficult probles
- Not covered
    - Linear programming
    - Mixed integer programming
    - Stochastic gradient descent


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
- No built-in multistart, benchmarking, scaling, reparametrization or logging
- Parameters are 1d numpy arrays

---


### Examples from real projects I

```python
def parse_parameters(x):
    """Parse the parameter vector into quantities we need."""
    num_types = int(len(x[54:]) / 6) + 1
    params = {
        'delta': x[0:1],
        'level': x[1:2],
        'coeffs_common': x[2:4],
        'coeffs_a': x[4:19],
        'coeffs_b': x[19:34],
        'coeffs_edu': x[34:41],
        'coeffs_home': x[41:44],
        'type_shares': x[44:44 + (num_types - 1) * 2],
        'type_shifts': x[44 + (num_types - 1) * 2:]
    }
    return params
```

---

### Examples from real projects II

```python
>>> scipy.optimize.minimize(func, x0)
---------------------------------------------------------------------------
LinAlgError                               Traceback (most recent call last)
<ipython-input-17-7459e5b4d8d4> in <module>
----> 1 scipy.optimize.minimize(func, x0)

     95
     96 def _raise_linalgerror_singular(err, flag):
---> 97     raise LinAlgError("Singular matrix")
     98

LinAlgError: Singular matrix
```
- After 5 hours and with no additional information

---

<!-- _class: lead -->
# Introduction to estimagic

---

### What is estimagic?

- Library for numerical optimization and nonlinear estimation
- Wraps many other optimizer libraries:
    - Scipy, Nlopt, TAO, Pygmo, ...
- Harmonized interface
- A lot of additional functionality

---

### You can use it like scipy
<!-- _class: split -->
<style scoped>
section.split {
    grid-template-columns: 550px 550px;
}
</style>

<div class=leftcol>

```python
>>> import estimagic as em

>>> def sphere(x):
>>>    return np.sum(x ** 2)

>>> res = em.minimize(
>>>     criterion=sphere,
>>>     params=np.arange(5),
>>>     algorithm="scipy_lbfgsb",
>>> )

>>> res.params
array([ 0., -0., -0., -0., -0.])
```
</div>
<div class=rightcol>

- There is also `maximize`
- Supports all scipy algorithms
    - `"scipy_neldermead"`
    - `"scipy_powell`
    - `"scipy_bfgs`
    - `"scipy_truncated_newton"`
    - ...

</div>

---

### Params can be anything

<!-- _class: split -->
<style scoped>
section.split {
    grid-template-columns: 650px 450px;
}
</style>

<div class=leftcol>

```python
>>> def dict_sphere(x):
>>>     out = (x["a"] ** 2 + x["b"] ** 2 + (x["c"] ** 2).sum()
>>>     return out


>>> res = minimize(
>>>     criterion=dict_sphere,
>>>     params={"a": 0, "b": 1, "c": pd.Series([2, 3, 4])},
>>>     algorithm="scipy_powell",
>>> )

>>> res.params
{'a': 0.,
 'b': 0.,
 'c': 0    0.
      1    0.
      2    0.
 dtype: float64}

```
</div>
<div class=rightcol>

- `params` can be (nested) dicts, lists, tuples or namedtuples containing numbers, arrays, Series and DataFrames.
- Special case: DataFrame with columns `"value"`, `"lower_bound"` and `"upper_bound"`
</div>



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

- Smoothness: Differentiable? With kinks? With discontinuities? Stochastic?
- Convexity: Are there local optima?
- Size: 2 parameters? 10? 100? 1000? More?
- Constraints: Bounds? Linear constraints? Nonlinear constraints?
- Special structure: Nonlinear least-squares, Log-likelihood function
- Goal: Do you need a global solution? How precise?

---

### `scipy_lbfgsb`

- Limited memory BFGS with support for bounds
- BFGS is a method to approximate hessians from multiple gradients
- Criterion must be differentiable
- Very fast and scales to a few thousand parameters
- Beats other BFGS implementations in many benchmarks
- Low overhead, i.e. works well for fast criterion functions

---

### `fides`

- Derivative based trust-region algorithm with support for bounds
- Developed by Fabian Fröhlich as a Python package
- Many advanced options to customize the optimization!
- Criterion must be differentiable
- Good solution if scipy_lbfgsb picks too extreme parameters that cause numerical overflow

---

### `nlopt_bobyqa`, `nag_pybobyqa`

- Derivative free trust region algorithm with support for bounds
- `nlopt` version has less overhead
- `nag` version has advanced options to deal with noise
- `nag` version is very sensitive to bad scaling of parameters
- Good choice for non-differentiable but not too noisy functions
- Slower than derivative based methods but faster than neldermead

---

### `scipy_neldermead`, `nlopt_neldermead`

- Very popular direct search method
- `nlopt` version supports bounds
- `nlopt` version requires much fewer criterion evaluations in most benchmarks
- The Nelder-Mead algorithm is never the best choice but also rarely the worst
- Immune to bad scaling of parameters

---

### `scipy_ls_lm`, `scipy_ls_trf`

- Derivative based optimizers for least squares problems
- Criterion needs the structure: $F(x) = \sum_i f_i(x)^2$
- In estimagic, criterion function must return a dictionary:

```python
def sphere_ls(x):
    # x are the least squares residuals in the sphere function
    return {"root_contributions": x, "value": x @ x}
```
- `scipy_ls_lm` is better for small problems without bounds
- `scipy_ls_trf` is better for problems with many parameters

---

### `nag_dfols`, `pounders`

- Derivative free trust region methods for nonlinear least-squares problems
- Both beat bobyqa for least-squares problems!
- `nag_dfols` is fastest and usually requires fewest criterion evaluations
- `nag_dfols` has advanced options to deal with noise
- `pounders` can do criterion evaluations in parallel
- Sensitive to bad scaling of parameters

---

### `ipopt`

- Interior point optimizer for problems with nonlinear constraints
- Probably the best open source optimizer for large constrained problems
- We wrap it via `cyipopt`
- Difficult to install on windows

---
<!-- _class: lead -->
# Practice Session 3: Play with `algorithm` and `algo_options` (15 min)

---

### What is benchmarking

- Compare multiple algorithms on functions with known optimum
- Benchmark functions should be similar to the problem you actually want to solve
    - similar number of parameters
    - similar w.r.t. differentiability or noise
- Benchmark functions should be fast!
- There are standardized benchmark sets and visualizations

---

### Running benchmarks in estimagic

<!-- _class: split -->
<style scoped>
section.split {
    grid-template-columns: 550px 550px;
}
</style>


<div class=leftcol>

```python
problems = em.get_benchmark_problems("estimagic")
optimizers = [
    "scipy_lbfgsb",
    "nag_dfols",
    "nlopt_bobyqa",
    "scipy_neldermead",
]
results = em.run_benchmark(
    problems=problems,
    optimize_options=optimizers,
    n_cores=4,
    max_criterion_evaluations=1000,
)
```

</div>
<div class=rightcol>

- Multiple benchmark sets
    - more_wild
    - estimagic
    - example
- Add noise or scaling problems
- Can pass additional options to govern minimization
- Benchmarks run in parallel

</div>

---
### Profile plots

<img src="../graphs/benchmark.png" alt="profile_plot" width="900"/>

---

### Convergence plots

<img src="../graphs/convergence_plot.png" alt="convergence_plot" width="900"/>

---


### Advanced options

<!-- _class: split -->
<style scoped>
section.split {
    grid-template-columns: 550px 550px;
}
</style>


<div class=leftcol>

```python
problems = em.get_benchmark_problems(
    name="example",
    additive_noise=True,
    additive_noise_options={
        "distribution": "normal",
        "std": 0.2,
    },
    scaling=True,
    scaling_options={
        "min_scale": 0.1,
        "max_scale": 1000,
    }
)
```

</div>
<div class=rightcol>

- Add additive noise
- Add bad scaling
- This would be a very difficult problem set

</div>

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

### Reparametrization example

- Assume we want to minimize $f(x_1, x_2) = \sqrt{x_2 - x_1} + x_2^2$
- Only defined if $x_1 \leq x_2$. Thus, this constraint should never be violated
- This is not a simple bound but a linear constraint!
- Let's solve this with reparametrization:
    - Define $\tilde{x}_2 = x_2 - x_1$ and $\tilde{f}(x_1, \tilde{x}_2) = \sqrt{\tilde{x}_2} + (x_1 + \tilde{x}_2)^2$
    - Calculate $argmin_{x_1 \in R, \tilde{x}_2 \in R^+}\tilde{f}(x_1, \tilde{x}_2)$
    - Translate the solution back into $x_1$ and $x_2$
- Easy to get confused and make mistakes when implementing this by hand
- Estimagic does this for you for many types of constraints

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
# REFERENCES
==================================================================================== -->

### References

- JAX Opt
- scipy.optimize
