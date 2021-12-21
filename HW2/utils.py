import numpy as np

def derivative(f, a, step_size) -> float:
    return f(a + step_size) - f(a) 