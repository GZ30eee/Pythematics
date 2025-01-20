import streamlit as st
from scipy.optimize import linprog, minimize
from pulp import LpProblem, LpVariable, LpMaximize, PULP_CBC_CMD
import numpy as np

# --- Helper Functions ---
def generate_random_lp():
    num_vars = np.random.randint(2, 6)
    num_constraints = np.random.randint(2, 6)

    c = np.random.randint(-10, 10, num_vars)
    A = np.random.randint(-10, 10, (num_constraints, num_vars))
    b = np.random.randint(1, 20, num_constraints)

    return c, A, b

def generate_random_nlp():
    num_vars = np.random.randint(2, 4)
    x0 = np.random.uniform(-10, 10, num_vars)
    return x0

def generate_random_qp():
    num_vars = np.random.randint(2, 4)
    Q = np.random.rand(num_vars, num_vars)
    Q = Q @ Q.T  # Make sure Q is positive semi-definite
    c = np.random.uniform(-10, 10, num_vars)
    return Q, c

def solve_linear_program(c, A, b):
    result = linprog(c, A_ub=A, b_ub=b, method='highs')
    return result

def solve_integer_program(c, A, b):
    prob = LpProblem("Integer_Programming", LpMaximize)
    vars = [LpVariable(f"x{i}", lowBound=0, cat='Integer') for i in range(len(c))]

    prob += sum(c[i] * vars[i] for i in range(len(c)))
    for row, rhs in zip(A, b):
        prob += sum(row[i] * vars[i] for i in range(len(c))) <= rhs

    prob.solve(PULP_CBC_CMD(msg=0))
    result = {
        "Objective": prob.objective.value(),
        "Variables": [v.varValue for v in vars],
        "Status": prob.status
    }
    return result

def solve_non_linear_program(func, x0, bounds, constraints):
    result = minimize(func, x0, bounds=bounds, constraints=constraints, method='SLSQP')
    return result

def quadratic_objective(x):
    return x[0]**2 + x[1]**2 + 2*x[0]*x[1] - 4*x[0] - 6*x[1]

# --- Streamlit UI ---
st.title("Optimization Problem Solver")
st.sidebar.title("Problem Settings")

# Problem Selection
problem_type = st.sidebar.selectbox("Select Problem Type", [
    "Linear Programming (LP)",
    "Non-Linear Programming (NLP)",
    "Quadratic Programming (QP)"
])

# Input Type
input_type = st.sidebar.radio("Input Type", ["Random Input", "User Input"])

# Linear Programming
if problem_type == "Linear Programming (LP)":
    if input_type == "Random Input":
        c, A, b = generate_random_lp()
        st.write("Randomly Generated LP Problem:")
        st.write("Objective Coefficients (c):", c)
        st.write("Constraint Matrix (A):", A)
        st.write("Constraint Bounds (b):", b)

        if st.button("Solve LP"):
            result = solve_linear_program(c, A, b)
            st.write("Results:")
            st.write("Objective Value:", result.fun)
            st.write("Variable Values:", result.x)
            st.write("Status:", result.message)
    
    else:
        num_vars = st.number_input("Number of Variables", min_value=1, step=1)
        num_constraints = st.number_input("Number of Constraints", min_value=1, step=1)

        c_input = st.text_area("Enter Objective Coefficients (comma-separated)")
        A_input = st.text_area("Enter Constraint Matrix (comma-separated rows)")
        b_input = st.text_area("Enter Constraint Bounds (comma-separated)")

        if st.button("Solve LP"):
            try:
                c = np.array([float(i) for i in c_input.split(",")])
                A = np.array([list(map(float, row.split(","))) for row in A_input.splitlines()])
                b = np.array([float(i) for i in b_input.split(",")])
                result = solve_linear_program(c, A, b)

                st.write("Results:")
                st.write("Objective Value:", result.fun)
                st.write("Variable Values:", result.x)
                st.write("Status:", result.message)
            except Exception as e:
                st.error(f"Error: {e}")

# Integer Programming
# elif problem_type == "Integer Programming (IP)":
#     if input_type == "Random Input":
#         c, A, b = generate_random_lp()
#         st.write("Randomly Generated IP Problem:")
#         st.write("Objective Coefficients (c):", c)
#         st.write("Constraint Matrix (A):", A)
#         st.write("Constraint Bounds (b):", b)

#         if st.button("Solve IP"):
#             result = solve_integer_program(c, A, b)
#             st.write("Results:")
#             st.write("Objective Value:", result["Objective"])
#             st.write("Variable Values:", result["Variables"])
#             st.write("Status:", result["Status"])
    
#     else:
#         num_vars_ip = st.number_input("Number of Variables", min_value=1, step=1)
#         num_constraints_ip = st.number_input("Number of Constraints", min_value=1, step=1)

#         c_ip_input = st.text_area("Enter Objective Coefficients (comma-separated)")
#         A_ip_input = st.text_area("Enter Constraint Matrix (comma-separated rows)")
#         b_ip_input = st.text_area("Enter Constraint Bounds (comma-separated)")

#         if st.button("Solve IP"):
#             try:
#                 c_ip = np.array([float(i) for i in c_ip_input.split(",")])
#                 A_ip = np.array([list(map(float, row.split(","))) for row in A_ip_input.splitlines()])
#                 b_ip = np.array([float(i) for i in b_ip_input.split(",")])
#                 result_ip = solve_integer_program(c_ip, A_ip, b_ip)

#                 st.write("Results:")
#                 st.write("Objective Value:", result_ip["Objective"])
#                 st.write("Variable Values:", result_ip["Variables"])
#                 st.write("Status:", result_ip["Status"])
#             except Exception as e:
#                 st.error(f"Error: {e}")

# Quadratic Programming
elif problem_type == "Quadratic Programming (QP)":
    if input_type == "Random Input":
        Q_random, c_random = generate_random_qp()
        
        if st.button("Solve QP"):
            try:
                bounds_qp = [(None, None)] * len(c_random)  # Example bounds
                
                def qp_objective(x):
                    return 0.5 * x @ Q_random @ x + c_random @ x
                
                constraints_qp = []  # Define any constraints if needed

                x0_qp = np.zeros(len(c_random))  # Initial guess
                
                result_qp = solve_non_linear_program(qp_objective,
                                                       x0_qp,
                                                       bounds=bounds_qp,
                                                       constraints=constraints_qp)

                # Display results
                st.write(f"Randomly Generated QP Problem:")
                st.write(f"Q Matrix:\n{Q_random}")
                st.write(f"Objective Coefficients:\n{c_random}")
                
                # Display results
                st.write(f"Results:")
                st.write(f"Objective Value: {result_qp.fun}")
                st.write(f"Variable Values: {result_qp.x}")

            except Exception as e:
                st.error(f"Error: {e}")
    
    else:
        num_vars_qp = st.number_input("Number of Variables", min_value=2, step=1)

        Q_input = st.text_area("Enter Q Matrix (comma-separated rows)")
        c_qp_input = st.text_area("Enter Objective Coefficients (comma-separated)")

        if st.button("Solve QP"):
            try:
                Q_matrix = np.array([list(map(float, row.split(","))) for row in Q_input.splitlines()])
                c_qp = np.array([float(i) for i in c_qp_input.split(",")])
                
                bounds_qp = [(None, None)] * len(c_qp)  # Example bounds
                
                def qp_objective(x):
                    return 0.5 * x @ Q_matrix @ x + c_qp @ x
                
                constraints_qp = []  # Define any constraints if needed

                x0_qp = np.zeros(len(c_qp))  # Initial guess
                
                result_qp = solve_non_linear_program(qp_objective,
                                                       x0_qp,
                                                       bounds=bounds_qp,
                                                       constraints=constraints_qp)

                # Display results
                st.write(f"Results:")
                st.write(f"Objective Value: {result_qp.fun}")
                st.write(f"Variable Values: {result_qp.x}")

            except Exception as e:
                st.error(f"Error: {e}")

# Non-Linear Programming
elif problem_type == "Non-Linear Programming (NLP)":
    if input_type == "Random Input":
        x0_nlp = generate_random_nlp()
        bounds_nlp = [(None, None) for _ in range(len(x0_nlp))]
        
        if st.button("Solve NLP"):
            try:
                constraints_nlp = []
                result_nlp = solve_non_linear_program(quadratic_objective,
                                                       x0_nlp,
                                                       bounds=bounds_nlp,
                                                       constraints=constraints_nlp)

                # Display results
                st.write(f"Randomly Generated Initial Guess: {x0_nlp}")
                st.write(f"Results:")
                st.write(f"Objective Value: {result_nlp.fun}")
                st.write(f"Variable Values: {result_nlp.x}")
                
                status_message_map = {
                    0: 'Optimization terminated successfully.',
                    1: 'Iteration limit reached.',
                    2: 'Optimization failed.'
                }
                
                status_message = status_message_map.get(result_nlp.status, 'Unknown status.')
                
                # Display optimization status
                if status_message == 'Optimization terminated successfully.':
                    st.success(status_message)
                else:
                    st.warning(status_message)

            except Exception as e:
                # Handle any errors that occur during optimization
                st.error(f"Error: {e}")

    else:
        num_vars_nlp = st.number_input("Number of Variables", min_value=2, step=1)

        # User-defined objective function input
        objective_function_input = st.text_area("Enter Objective Function (in terms of x0, x1, ...)")
        
        # User-defined constraints input
        constraints_input = st.text_area("Enter Constraints (one per line, format: lhs <= rhs)")

        if st.button("Solve NLP"):
            try:
                # Generate initial guess
                x0_nlp = np.zeros(num_vars_nlp)
                
                # Parse constraints
                constraints_nlp = []
                for line in constraints_input.splitlines():
                    if line.strip():
                        lhs, rhs = line.split("<=")
                        constraints_nlp.append({'type': 'ineq', 'fun': lambda x, lhs=lhs.strip(), rhs=float(rhs.strip()): eval(lhs) - rhs})

                # Define the objective function from user input
                def user_defined_objective(x):
                    return eval(objective_function_input)

                bounds_nlp = [(None, None) for _ in range(num_vars_nlp)]  # Example bounds

                result_nlp = solve_non_linear_program(user_defined_objective,
                                                       x0_nlp,
                                                       bounds=bounds_nlp,
                                                       constraints=constraints_nlp)

                # Display results
                st.write(f"Results:")
                st.write(f"Objective Value: {result_nlp.fun}")
                st.write(f"Variable Values: {result_nlp.x}")

            except Exception as e:
                st.error(f"Error: {e}")

# # Quadratic Programming
# elif problem_type == "Quadratic Programming (QP)":
#     if input_type == "Random Input":
#         Q_random, c_random = generate_random_qp()
        
#         if st.button("Solve QP"):
#             try:
#                 bounds_qp = [(None, None)] * len(c_random)  # Example bounds
                
#                 def qp_objective(x):
#                     return 0.5 * x @ Q_random @ x + c_random @ x
                
#                 constraints_qp = []  # Define any constraints if needed

#                 x0_qp = np.zeros(len(c_random))  # Initial guess
                
#                 result_qp = solve_non_linear_program(qp_objective,
#                                                        x0_qp,
#                                                        bounds=bounds_qp,
#                                                        constraints=constraints_qp)

#                 # Display results
#                 st.write(f"Randomly Generated QP Problem:")
#                 st.write(f"Q Matrix:\n{Q_random}")
#                 st.write(f"Objective Coefficients:\n{c_random}")
                
#                 # Display results
#                 st.write(f"Results:")
#                 st.write(f"Objective Value: {result_qp.fun}")
#                 st.write(f"Variable Values: {result_qp.x}")

#             except Exception as e:
#                 st.error(f"Error: {e}")
    
else: 
    pass
