# neughbour function 
Converts the distance between neurons into the weight of influence in the learning process


## Гауссовская функция соседства

**Формула:**
$$h(d_i, \sigma) = e^{-\frac{d_i^2}{2\sigma^2}}$$

**Где:**
- $h(d_i, \sigma)$ — коэффициент влияния нейрона  
- $d_i$ — топологическое расстояние от победителя до нейрона $i$
- $\sigma$ — ширина окрестности (радиус принадлежности)

## Функция "мексиканской шляпы"

**Формула:**
$$
h(d_i, \sigma) = 
\begin{cases} 
\left(1 - \frac{d_i^2}{\sigma^2}\right) \cdot e^{-\frac{d_i^2}{2\sigma^2}}, & d_i < \sigma \\
0, & d_i \geq \sigma 
\end{cases}
$$

**Где:**
- $h(d_i, \sigma)$ — коэффициент влияния (может быть отрицательным)  
- $d_i$ — топологическое расстояние от победителя до нейрона $i$
- $\sigma$ — ширина окрестности (радиус возбуждения/торможения)