---
<!-- _class: split -->

### Use constraints with any optimizer

<style scoped>
section.split {
    grid-template-columns: 550px 550px;
}
</style>
<div class=leftcol>

```python
>>> res = em.minimize(
...     criterion=sphere,
...     params=np.array([0.1, 0.5, 0.4, 4, 5]),
...     algorithm="scipy_lbfgsb",
...     constraints=[{
...         "loc": [0, 1, 2],
...         "type": "probability"
...     }],
... )

>>> res.params
array([0.33334, 0.33333, 0.33333, -0., 0.])
```
</div>
<div class=rightcol>

- lbfgsb is unconstrained
- estimagic transforms constrained problems into unconstrained ones
- Supported constraints:
    - linear
    - probability
    - covariance
    - ...

</div>



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

<!-- _class: split -->

### Multistart framework
<div class=leftcol>


```python
>>> res = em.minimize(
...     criterion=sphere,
...     params=np.arange(5),
...     algorithm="scipy_neldermead",
...     soft_lower_bounds=np.full(5, -5),
...     soft_upper_bounds=np.full(5, 15),
...     multistart=True,
...     multistart_options={
...         "convergence.max_discoveries": 5,
...         "n_samples": 1000
...     },
... )
>>> res.params
array([0., 0., 0., 0.,  0.])
```


</div>
<div class=rightcol>

- Turn local optimizers global
- Inspired by [tiktak algorithm](https://github.com/serdarozkan/TikTak#tiktak)
- **Exploration** on random sample
- Local optimizations from best points
- Use any optimizer

</div>






---

<!-- _class: split -->

<style scoped>
section.split {
    grid-template-columns: 470px 630px;
}
</style>

### Exploit structure of $F$

<div class=leftcol>

```python
>>> def general_sphere(params):
...     contribs = params ** 2
...     out = {
...         "root_contributions": params,
...         "contributions": contribs,
...         "value": contribs.sum(),
...     }
...     return out

>>> res = em.minimize(
...     criterion=general_sphere,
...     params=np.arange(5),
...     algorithm="pounders",
... )
>>> res.params
array([0., 0., 0., 0., 0.])
```

</div>

<div class=rightcol>

- Common structures
    - $F(x) = \sum_if_i(x)^2$ (least squares)
    - $F(x) = \sum_if_i(x)$ (e.g. log-likelihood)
- Huge speed-ups
- Increased robustness

</div>

