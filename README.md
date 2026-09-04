# Germibeta

Cálculo de la distribución beta discreta generalizada (DGBD) usando una regresión no lineal 
para cálculo de los exponentes.

$$
f(r) = A \frac {(N - 1 +r)^b}{r^a}  
$$

Donde r es la abundancia de cada rango, $f(r)$ es el valor ajustado al modelo, $(a,b)$ son los 
exponentes ajustados, N es la totalidad de rangos y $A$ es una constante de normalización.

El cálculo de exponentes se realiza por varios métodos, uno es una regresión multilineal y 
otro es una regresión no lineal (Marquadt-Levenberg)

---
## Métodos principales

La invocación principal desde python puede hacerse de la siguiente forma:


```python
import fbc as BC
bc = BC.Germibeta()
bc.carga_archivo('./data/fbc_brown.csv')
bc.ajuste()
params = gg.params
graf_datos(bc.f['vals'].array, params, {'titulo' : 'Brown noise', 'eje_x' : 'Rango', 'eje_y' : r'$\log(f)$'},'fbc_output.png')
```



---
## Estructura de directorio

```
.
│
├── doc/              manuales y ayuda 
├── data/             directorio de datos
├── requirements.txt  compatibilidades de python
└── README.md         este archivo
```
