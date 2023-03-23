---
# try also 'default' to start simple
theme: default
# random image from a curated Unsplash collection by Anthony
# like them? see https://unsplash.com/collections/94734566/slidev
#background: https://source.unsplash.com/collection/94734566/1920x1080
# apply any windi css classes to the current slide
class: "text-left"
# https://sli.dev/custom/highlighters.html
highlighter: shiki
# show line numbers in code blocks
lineNumbers: false
# some information about the slides, markdown enabled
info: |
  ## Slidev Starter Template
  Presentation slides for developers.

  Learn more at [Sli.dev](https://sli.dev)
# persist drawings in exports and build
drawings:
  persist: false
# page transition
transition: slide-left
# use UnoCSS
# css: unocss
---

## Installation


We assume you have done the following: Installed [miniconda](https://docs.conda.io/en/latest/miniconda.html) and 

```console
git clone https://github.com/OpenSourceEconomics/scipy-estimagic.git
cd scipy-estimagic
conda env create -f environment.yml
conda activate scipy-estimagic
```

- If you haven't done so, please do so until the first practice session

- Details: https://github.com/OpenSourceEconomics/scipy-estimagic


---
layout: cover
---

# Practical Numerical Optimization with Scipy, Estimagic and JAXopt

Scipy Conference 2022

<br/>

## Janos Gabler & Tim Mensinger

University of Bonn


---

## About Us
<br/>

<div class="grid grid-cols-2 gap-4">
<div>

<img src="/janos.jpg" alt="janos" width="200"/>

- Website: [janosg.com](https://janosg.com)
- GitHub: [janosg](https://github.com/janosg/)
- Started estimagic in 2019
- Just submitted PhD thesis, looking for jobs soon
</div>
<div>

<img src="/tim.jpeg" alt="tim" width="200"/>

- Website: [tmensinger.com](https://tmensinger.com)
- GitHub: [timmens](https://github.com/timmens)
- estimagic core contributor
- PhD student in Econ, University of Bonn

</div>
</div>





---
layout: section
---
## Sections
<br/>

1. Introduction to `scipy.optimize`

2. Introduction to `estimagic`

3. Choosing algorithms

4. Advanced topics

5. Jax and Jaxopt

---
layout: section
---


## Structure of each topic

<br/>

1. Summary of exercise you will solve

2. Some theory

3. Syntax in very simplified example

4. You solve a more difficult example in a notebook

5. Discuss one possible solution


---
layout: cover
---
# Introduction to scipy.optimize

---
layout: section
---
## Preview of practice session
<br/>


- Translate a criterion function from math to code
  
- Use `scipy.optimize` to minimize the criterion function

---
layout: section
---

## Example problem
<br/>


<div class="grid grid-cols-2 gap-4">
<div>

- **Criterion** $f(a, b) = a^2 + b^2$
- Parameters $a$, $b$
- Want: $a^*, b^* = \text{argmin} f(a, b)$
- Possible extensions:
    - Constraints
    - Bounds
- Optimum at $a^*=0$, $b^*=0$, $f(a^*,b^*) = 0$

</div>

<div>

<img src="/figures/sphere.png" alt="sphere" width="450" class="center"/>

</div>
</div>

---
layout: section
---


## Criterion plot
<br/>

<div class="grid grid-cols-2 gap-4">
<div>

```python
em.criterion_plot(res)
```
<img src="/figures/criterion_plot.png" alt="criterion" width="500" style="display: block;"/>


</div>
<div>

- First argument can be:
    - `OptimizeResult`
    - path to log file
    - list or dict thereof
- Dictionary keys are used for legend

</div>
</div>

---
layout: section
---

## Params plot

<br/>

<div class="grid grid-cols-[35%,65%] gap-4">


<div>


```python
# reminder: params looks like this
params = {
    "a": 0,
    "b": 1,
    "c": pd.Series([2, 3, 4])
}

em.params_plot(
    res,
    max_evaluations=300,
)
```

</div>
<div >

<img src="/figures/params_plot.png" alt="params_plot" width="500"/>

- Similar options as `criterion_plot`

</div>
</div>


---
layout: section
---
## Convergence plots
<br/>

<div class="grid grid-cols-[30%,70%] gap-4">
<div>

```python
subset = [
    "chebyquad_10",
    "chnrsbne",
    "penalty_1",
    "bdqrtic_8",
]
em.convergence_plot(
    problems,
    results,
    problem_subset=subset,
)
```
</div>
<div>

<img src="/convergence_plot.png" alt="convergence_plot" width="750"/>

</div>
</div>

---
layout: section
---
## Logging and Dashboard

<br/>
<div class="grid grid-cols-[50%,50%] gap-4">
<div>
```console
res = em.minimize(
    criterion=sphere,
    params=np.arange(5),
    algorithm="scipy_lbfgsb",
    logging="my_log.db",
 )

```
</div>

<div>
```console
from estimagic import OptimizeLogReader

reader = OptimizeLogReader("my_log.db")
 reader.read_history().keys()
dict_keys(['params', 'criterion', 'runtime'])

reader.read_iteration(1)["params"]
array([0., 0.817, 1.635, 2.452, 3.27 ])
```
</div>

</div>


- Persistent log in sqlite database
- No data loss ever
- Can be read during optimization
- Provides data for dashboard
- No SQL knowledge needed



---
