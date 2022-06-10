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

<!-- _footer: IZA and IFS, University of Bonn -->
<!-- paginate: false -->
## Practical Numerical Optimization with Estimagic, JaxOpt, Scipy

Scipy Conference 2022

<br/>

#### Janos Gabler & Tim Mensinger

---

<!-- ===================================================================================
# INDEX
==================================================================================== -->

<!-- paginate: true -->
## Index

1. Introduction
2. Installation
3. Why optimization is difficult

---

<!-- ===================================================================================
# INTRODUCTION
==================================================================================== -->

<!-- paginate: false -->
<!-- _class: lead -->
# Introduction

---

### Problem

- Parameter $x$
- Criterion function $f(x)$
- Constraints $C(x)$
- Bounds $a \leq x \leq b$

- minimize/maximize $f$ wrt constraints


---

### Running example

```python
def sphere(x):
    return np.sum(x ** 2)

def sphere_gradient(x):
    return 2 * x
```

---

<!-- ===================================================================================
# INSTALLATION
==================================================================================== -->

<!-- paginate: false -->
<!-- _class: lead -->
# Installation

---
<!-- paginate: true -->
### Installation

```console
$ conda config --add channels conda-forge
$ conda install -c conda-forge estimagic
```
or
```console
$ pip install estimagic
```

---

<!-- ===================================================================================
# WHY OPTIMIZATION IS DIFFICULT
==================================================================================== -->

<!-- paginate: false -->
<!-- _class: lead -->
# Why optimization is difficult

---
<!-- paginate: true -->

### Why grid search is infeasible

- Want precision up to 2 decimal places

- How many function evaluations do we need?

---
### A two-column slide
<!-- _class: split -->

<div class=leftcol>

#### Title left column
- listed item
- listed item
- listed item

</div>

<div class=rightcol>

#### Title right column

```python
def f(x):
    return x ** 2

g = converter.wrap(f, kwargs, key="value")
```

</div>

---

<!-- ===================================================================================
# REFERENCES
==================================================================================== -->

### References

- JAX Opt
- scipy.optimize
