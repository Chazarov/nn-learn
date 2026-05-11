# SOM (SELF-ORGANAIZED-MAP) loop


**Iteratively** for each input vector $x(t)$:

1. Find the **winner** $c$ — neuron with minimum Euclidean distance to $x(t)$
2. **Update weights** of all neurons using neighborhood function

## Weight Update Formula
$w_i(t+1) = w_i(t) + \alpha(t) \cdot h_{c,i}(t) \cdot (x(t) - w_i(t))$

## Weight Update Block

**Variables:**
- $w_i(t)$ — weights of $i$-th neuron at step $t$
- $x(t)$ — current input vector
- $c$ — winner index
- $\alpha(t)$ — learning rate $$, decreases over time [progler](https://progler.ru/blog/vychislit-exp-x-1-dlya-vseh-elementov-v-dannom-massive-numpy)
- $h_{c,i}(t)$ — neighborhood function: $h_{c,i}(t) = \exp\left(-\frac{d^2(c,i)}{2\sigma^2(t)}\right)$
- $d(c,i)$ — topological distance between neurons $c$ and $i$
- $\sigma(t)$ — neighborhood radius, shrinks over time