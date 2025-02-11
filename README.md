# JAX vs TensorFlow vs PyTorch: Comparación de Frameworks de Deep Learning

## **Introducción**
JAX es una librería de Google diseñada para cálculos numéricos y aprendizaje profundo, combinando la facilidad de uso de NumPy con capacidades avanzadas como compilación Just-In-Time (JIT) y diferenciación automática. Es una alternativa emergente a TensorFlow y PyTorch, utilizada en proyectos de gran envergadura como DeepMind.

Este repositorio contiene una comparativa entre **JAX, TensorFlow y PyTorch**, destacando sus diferencias en **rendimiento, facilidad de uso y ecosistema**. También incluye un ejemplo práctico entrenando un modelo de clasificación en el dataset de **Wine**.

## **Puntos Tratados**
- **¿Qué es JAX?** Sus características principales y ventajas.
- **Comparación con TensorFlow y PyTorch** en términos de rendimiento, facilidad de uso y ecosistema.
- **Ecosistema de JAX**, explorando librerías como Flax, Optax y Stax.
- **Ejemplo práctico:** Implementación de un modelo de clasificación con JAX.

---

## **Comparación de Frameworks**
| Característica  | JAX | TensorFlow | PyTorch |
|---------------|-----|------------|--------|
| Lenguaje base | NumPy-like | Computación en grafos + Keras | Estilo dinámico |
| Diferenciación automática | Sí (grad) | Sí (AutoDiff) | Sí (Autograd) |
| JIT Compilation | Sí (XLA) | Parcial (XLA) | No |
| Soporte para GPUs/TPUs | Sí | Sí | Sí |
| Facilidad de uso | Intermedio | Avanzado | Fácil |

---

## **Ejemplo Práctico: Clasificación de Vinos con JAX**
Este repositorio contiene un **Jupyter Notebook** con un modelo de clasificación basado en redes neuronales para el dataset de **Wine**, utilizando JAX y Stax.

### **Requisitos**
Asegúrate de tener instaladas las siguientes dependencias:
```bash
pip install jax tensorflow torch numpy pandas matplotlib scikit-learn seaborn
```

### **Ejecución del Notebook**
Para probar el código, simplemente ejecuta el notebook:
```bash
jupyter notebook jax_comparison_ecosystem.ipynb
```

---

## **Referencias y Bibliografía**
- [JAX - GitHub Oficial](https://github.com/google/jax)
- [Optax - Optimizadores en JAX](https://github.com/deepmind/optax)
- [Flax - Framework de Deep Learning](https://github.com/google/flax)
- [Stax - Construcción de Modelos en JAX](https://www.kaggle.com/code/aakashnain/building-models-in-jax-part1-stax)

## **Autor**
Proyecto desarrollado por Yeray Mata Rodriguez
