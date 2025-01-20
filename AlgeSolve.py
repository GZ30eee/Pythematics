import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from sympy.plotting import plot

# Helper Functions
def random_expression():
    x = sp.symbols('x')
    return sp.Add(*[sp.Rational(np.random.randint(-10, 10), np.random.randint(1, 10)) * x**i for i in range(np.random.randint(1, 5))])

def random_matrix(size):
    return sp.Matrix(np.random.randint(-10, 10, (size, size)))

def random_equations():
    n = np.random.randint(2, 4)
    x = sp.symbols(f'x0:{n}')
    A = np.random.randint(-10, 10, (n, n))
    b = np.random.randint(-10, 10, n)
    equations = [sp.Eq(sum(A[i, j] * x[j] for j in range(n)), b[i]) for i in range(n)]
    return equations

# Streamlit App
def main():
    st.title("Mathematical Operations App")

    operations = [
        "Simplify Expression",
        "Differentiate Expression",
        "Integrate Expression",
        "Expand Expression",
        "Factor Expression",
        "Partial Fraction Decompose",
        "Solve System of Equations",
        "Find Limit",
        "Find Limit at Infinity",
        "Find Roots",
        "Evaluate Expression",
        "Plot Function",
        "Inverse of Matrix"
    ]

    operation = st.selectbox("Select an operation:", operations)

    use_random = st.checkbox("Use Random Data")

    if operation in ["Simplify Expression", "Differentiate Expression", "Integrate Expression", 
                     "Expand Expression", "Factor Expression", "Partial Fraction Decompose", 
                     "Find Limit", "Find Limit at Infinity", "Find Roots", "Evaluate Expression", "Plot Function"]:

        x = sp.symbols('x')

        if use_random:
            expression = random_expression()
            st.write(f"Random Expression: {expression}")
        else:
            expression_input = st.text_input("Enter an expression (use 'x' as the variable):", "x**2 + 2*x + 1")
            expression = sp.sympify(expression_input)

        if operation == "Simplify Expression":
            result = sp.simplify(expression)
        elif operation == "Differentiate Expression":
            result = sp.diff(expression, x)
        elif operation == "Integrate Expression":
            result = sp.integrate(expression, x)
        elif operation == "Expand Expression":
            result = sp.expand(expression)
        elif operation == "Factor Expression":
            result = sp.factor(expression)
        elif operation == "Partial Fraction Decompose":
            result = sp.apart(expression)
        elif operation == "Find Limit":
            point = st.number_input("Enter the point at which to find the limit:", value=0.0)
            result = sp.limit(expression, x, point)
        elif operation == "Find Limit at Infinity":
            result = sp.limit(expression, x, sp.oo)
        elif operation == "Find Roots":
            result = sp.solveset(expression, x)
        elif operation == "Evaluate Expression":
            value = st.number_input("Enter the value of 'x':", value=0.0)
            result = expression.evalf(subs={x: value})
        elif operation == "Plot Function":
            plot(expression, show=False, xlabel="x", ylabel="y").save("plot.png")
            st.image("plot.png")
            return

        st.success(f"Result: {result}")

    elif operation == "Solve System of Equations":
        if use_random:
            equations = random_equations()
            st.write("Random System of Equations:")
            for eq in equations:
                st.latex(sp.latex(eq))
        else:
            num_eqs = st.number_input("Enter the number of equations:", min_value=2, value=2)
            variables = sp.symbols(f'x0:{num_eqs}')
            equations = []
            for i in range(num_eqs):
                eq_input = st.text_input(f"Enter equation {i + 1} (e.g., x0 + x1 = 2):")
                if eq_input:
                    equations.append(sp.sympify(eq_input))

        if st.button("Solve System"):
            try:
                result = sp.solve(equations)
                st.success(f"Solution: {result}")
            except Exception as e:
                st.error(f"Error: {e}")

    elif operation == "Inverse of Matrix":
        if use_random:
            size = np.random.randint(2, 5)
            matrix = random_matrix(size)
            st.write("Random Matrix:")
            st.latex(sp.latex(matrix))
        else:
            size = st.number_input("Enter the size of the matrix (n x n):", min_value=2, value=2)
            matrix = []
            for i in range(size):
                row = st.text_input(f"Enter row {i + 1} (space-separated):")
                if row:
                    matrix.append(list(map(float, row.split())))
            matrix = sp.Matrix(matrix)

        if st.button("Find Inverse"):
            try:
                result = matrix.inv()
                st.success("Inverse Matrix:")
                st.latex(sp.latex(result))
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
