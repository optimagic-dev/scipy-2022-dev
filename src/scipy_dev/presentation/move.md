

# move to closed form dereivative
---

<!-- _class: split -->

### Closed-form or parallel numerical derivatives
<div class=leftcol>

```python
>>> def sphere_gradient(params):
...     return 2 * params

>>> em.minimize(
...     criterion=sphere,
...     params=np.arange(5),
...     algorithm="scipy_lbfgsb",
...     derivative=sphere_gradient,
... )

>>> em.minimize(
...     criterion=sphere,
...     params=np.arange(5),
...     algorithm="scipy_lbfgsb",
...     numdiff_options={"n_cores": 6},
... )
```

</div>

<div class=rightcol>

- You can provide derivatives
- Otherwise, estimagic calculates them numerically
- Parallelization on (up to) as many cores as parameters

</div>

---

### Brute force vs. smarter algorithm
<!-- _class: split -->
<div class=leftcol>

<img src="../../../bld/figures/grid_search.png" alt="brute-force" width="400" class="center"/>

</div>
<div class=rightcol>

<img src="../../../bld/figures/gradient_descent.png" alt="smart" width="400" class="center"/>

</div>




---




### Advanced options

<!-- _class: split -->
<style scoped>
section.split {
    grid-template-columns: 450px 650px;
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

- Choices:
    - estimagic: small and large problems
    - more_wild: small least squares problems
    - example: subset of more_wild
- Add noise and scaling issues

</div>

---
