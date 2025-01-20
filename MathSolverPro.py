import streamlit as st
import numpy as np

class NumericalMethods:
    # Newton-Raphson Method
    @staticmethod
    def newton_raphson(func, deriv, initial_guess, tolerance=1e-6, max_iterations=100):
        x = initial_guess
        for i in range(max_iterations):
            f_val = func(x)
            deriv_val = deriv(x)
            if abs(deriv_val) < 1e-10:
                raise ValueError("Derivative too small; method may not converge.")
            
            x_new = x - f_val / deriv_val
            if abs(x_new - x) < tolerance:
                return x_new
            x = x_new
        raise ValueError("Newton-Raphson did not converge within the maximum iterations.")
    
    # Simpson's Rule
    @staticmethod
    def simpsons_rule(func, a, b, n=100):
        if n % 2 == 1:
            n += 1  # Ensure even number of intervals
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        
        try:
            y = func(x)  # Ensure func(x) can handle NumPy array x
        except Exception as e:
            raise ValueError(f"Function evaluation failed: {e}")
        
        integral = h / 3 * (y[0] + 4 * sum(y[1:-1:2]) + 2 * sum(y[2:-2:2]) + y[-1])
        return integral

    # Gaussian Elimination
    @staticmethod
    def gaussian_elimination(A, b):
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        
        # Ensure b is a column vector
        if b.ndim == 1:
            b = b[:, np.newaxis]
        
        n = len(b)
        for i in range(n):
            # Partial Pivoting
            max_row = i + np.argmax(abs(A[i:, i]))
            A[[i, max_row]] = A[[max_row, i]]
            b[[i, max_row]] = b[[max_row, i]]

            # Make the diagonal element 1 and eliminate below
            for j in range(i + 1, n):
                factor = A[j, i] / A[i, i]
                A[j, i:] -= factor * A[i, i:]
                b[j] -= factor * b[i]

        # Back substitution
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
        return x

# Streamlit App
def main():
    st.title("Numerical Methods Solver")

    methods = NumericalMethods()
    
    method_choice = st.selectbox("Select the method you want to use:", 
                                   ["Newton-Raphson", "Simpson's Rule", "Gaussian Elimination"])

    if method_choice == "Newton-Raphson":
        st.subheader("Newton-Raphson Method")
        
        random_input = st.checkbox("Use Random Input")
        
        if random_input:
            func_input = "x**3 - x - 2"
            deriv_input = "3*x**2 - 1"
            initial_guess = np.random.uniform(-10, 10)  # Random initial guess between -10 and 10
            st.write(f"Function: {func_input}")
            st.write(f"Derivative: {deriv_input}")
            st.write(f"Initial Guess: {initial_guess}")
        else:
            func_input = st.text_input("Enter the function for which you want to find the root (e.g., x**3 - x - 2):")
            deriv_input = st.text_input("Enter the derivative of the function (e.g., 3*x**2 - 1):")
            initial_guess = st.number_input("Enter the initial guess:", value=0.0)

        if st.button("Find Root"):
            try:
                func = eval("lambda x: " + func_input)
                deriv = eval("lambda x: " + deriv_input)
                
                root = methods.newton_raphson(func, deriv, initial_guess)
                st.success(f"Root: {root}")
            except Exception as e:
                st.error(f"Error: {e}")

    if method_choice == "Simpson's Rule":
        st.subheader("Simpson's Rule Method")
        
        random_input = st.checkbox("Use Random Input")

        if random_input:
            func_input = "np.sin"  # Example function for integration
            a = np.random.uniform(0, np.pi / 2)  # Random start value between 0 and π/2
            b = np.random.uniform(a + 0.01, np.pi / 2)  # Random end value greater than a and less than π/2
            n = np.random.randint(2, 100) * 2  # Random even number of intervals between 2 and 100
            st.write(f"Function: {func_input}")
            st.write(f"Start Value (a): {a}")
            st.write(f"End Value (b): {b}")
            st.write(f"Number of Intervals (n): {n}")
        else:
            func_input = st.text_input("Enter the function you want to integrate (e.g., np.cos):")
            a = st.number_input("Enter the start value of the integration:", value=0.0)
            b = st.number_input("Enter the end value of the integration:", value=1.0)
            n = st.number_input("Enter the number of intervals (even number):", min_value=2, step=2)

        if st.button("Calculate Integral"):
            try:
                # Validate input function
                func_input = func_input.strip()
                if not func_input.startswith("np."):
                    raise ValueError("Please ensure your function uses NumPy, e.g., np.sin(x) or np.exp(x).")

                # Define and test the function
                func = eval(f"lambda x: {func_input}(x)")
                
                # Test the function with NumPy array
                x_test = np.linspace(a, b, 5)  # Test values
                y_test = func(x_test)  # Test the output
                
                if not isinstance(y_test, np.ndarray):
                    raise ValueError("Function must return a NumPy array when passed a NumPy array.")

                # Calculate the integral
                integral = methods.simpsons_rule(func, a, b, int(n))
                st.success(f"Integral value: {integral}")

            except SyntaxError:
                st.error("Syntax Error: Ensure your function is valid.")
            except NameError as e:
                st.error(f"Name Error: {e}. Use valid NumPy functions like np.sin, np.exp, etc.")
            except ValueError as e:
                st.error(f"Value Error: {e}")
            except Exception as e:
                st.error(f"Unexpected Error: {e}")

    elif method_choice == "Gaussian Elimination":
        st.subheader("Gaussian Elimination Method")
        
        random_input = st.checkbox("Use Random Input")

        if random_input:
            n = np.random.randint(2, 5)  # Random number of variables between 2 and 5
            A = np.random.rand(n, n).tolist()  # Random coefficients matrix A
            b = np.random.rand(n).tolist()  # Random constants vector b

            st.write(f"Number of Variables: {n}")
            for i in range(n):
                st.write(f"Row {i+1} Coefficients: {A[i]}")
            
            st.write(f"B Vector Constants: {b}")

            if st.button("Solve Random System"):
                try:
                    solution = methods.gaussian_elimination(A, b)
                    st.success(f"Solution: {solution}")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            n = st.number_input("Enter the number of variables in the system of equations:", min_value=1)

            A = []
            for i in range(n):
                row_input = st.text_input(f"Enter coefficients of row {i+1} (space-separated):", "")
                if row_input:
                    row = list(map(float, row_input.split()))
                    A.append(row)

            b_input = st.text_input("Enter constants of the equation (b vector):", "")

            if len(A) == n and b_input and len(b := list(map(float, b_input.split()))) == n:
                if st.button("Solve System"):
                    try:
                        solution = methods.gaussian_elimination(A, b)
                        st.success(f"Solution: {solution}")
                    except Exception as e:
                        st.error(f"Error: {e}")
if __name__ == "__main__":
    main()
