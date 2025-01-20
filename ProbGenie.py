import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import norm, binom, poisson

# Function to visualize the normal distribution
def normal_distribution_simulation(mean, std_dev, num_samples):
    samples = np.random.normal(loc=mean, scale=std_dev, size=num_samples)
    x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 1000)
    pdf = norm.pdf(x, mean, std_dev)

    plt.figure(figsize=(10, 6))
    plt.hist(samples, bins=30, density=True, alpha=0.6, color='g', label="Random Samples")
    plt.plot(x, pdf, 'k-', lw=2, label="Normal Distribution PDF")
    plt.title(f"Normal Distribution (μ={mean}, σ={std_dev})")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    st.pyplot(plt)

# Function to visualize the binomial distribution
def binomial_distribution_simulation(n, p, num_samples):
    samples = np.random.binomial(n, p, num_samples)
    x = np.arange(0, n+1)
    pmf = binom.pmf(x, n, p)

    plt.figure(figsize=(10, 6))
    plt.hist(samples, bins=np.arange(n+2)-0.5, density=True, alpha=0.6, color='b', label="Random Samples")
    plt.plot(x, pmf, 'ro-', lw=2, label="Binomial Distribution PMF")
    plt.title(f"Binomial Distribution (n={n}, p={p})")
    plt.xlabel("Number of successes")
    plt.ylabel("Probability")
    plt.legend()
    st.pyplot(plt)

# Function to visualize the Poisson distribution
def poisson_distribution_simulation(lam, num_samples):
    samples = np.random.poisson(lam, num_samples)
    x = np.arange(0, np.max(samples)+1)
    pmf = poisson.pmf(x, lam)

    plt.figure(figsize=(10, 6))
    plt.hist(samples, bins=np.arange(np.max(samples)+2)-0.5, density=True, alpha=0.6, color='r', label="Random Samples")
    plt.plot(x, pmf, 'bo-', lw=2, label="Poisson Distribution PMF")
    plt.title(f"Poisson Distribution (λ={lam})")
    plt.xlabel("Number of occurrences")
    plt.ylabel("Probability")
    plt.legend()
    st.pyplot(plt)

# Function to visualize the Exponential distribution
def exponential_distribution_simulation(lam, num_samples):
    samples = np.random.exponential(1/lam, num_samples)
    x = np.linspace(0, np.max(samples), 1000)
    pdf = lam * np.exp(-lam * x)

    plt.figure(figsize=(10, 6))
    plt.hist(samples, bins=30, density=True, alpha=0.6, color='purple', label="Random Samples")
    plt.plot(x, pdf, 'k-', lw=2, label="Exponential Distribution PDF")
    plt.title(f"Exponential Distribution (λ={lam})")
    plt.xlabel("Time between events")
    plt.ylabel("Density")
    plt.legend()
    st.pyplot(plt)

# Main function to handle user interaction
def main():
    st.title("Probability Distribution Simulation Tool")
    
    # Select distribution type
    distribution_choice = st.radio("Choose Distribution", ("Normal Distribution", "Binomial Distribution", "Poisson Distribution", "Exponential Distribution"))
    
    # For Normal Distribution
    if distribution_choice == "Normal Distribution":
        st.header("Normal Distribution Simulation")
        if 'mean' not in st.session_state:
            st.session_state.mean = 0
            st.session_state.std_dev = 1
            st.session_state.num_samples = 1000

        mean = st.number_input("Enter the mean (μ)", value=st.session_state.mean)
        std_dev = st.number_input("Enter the standard deviation (σ)", value=st.session_state.std_dev)
        num_samples = st.number_input("Enter the number of samples", min_value=1, value=st.session_state.num_samples)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Calculate"):
                normal_distribution_simulation(mean, std_dev, num_samples)

        with col2:
            if st.button("Random"):
                st.session_state.mean = np.random.uniform(-10, 10)
                st.session_state.std_dev = np.random.uniform(1, 10)
                st.session_state.num_samples = np.random.randint(100, 10000)
                normal_distribution_simulation(st.session_state.mean, st.session_state.std_dev, st.session_state.num_samples)

    # For Binomial Distribution
    elif distribution_choice == "Binomial Distribution":
        st.header("Binomial Distribution Simulation")
        if 'n' not in st.session_state:
            st.session_state.n = 10
            st.session_state.p = 0.5
            st.session_state.num_samples = 1000

        n = st.number_input("Enter the number of trials (n)", min_value=1, value=st.session_state.n)
        p = st.number_input("Enter the probability of success (p)", min_value=0.0, max_value=1.0, value=st.session_state.p)
        num_samples = st.number_input("Enter the number of samples", min_value=1, value=st.session_state.num_samples)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Calculate"):
                binomial_distribution_simulation(n, p, num_samples)

        with col2:
            if st.button("Random"):
                st.session_state.n = np.random.randint(5, 20)
                st.session_state.p = np.random.uniform(0.1, 0.9)
                st.session_state.num_samples = np.random.randint(100, 10000)
                binomial_distribution_simulation(st.session_state.n, st.session_state.p, st.session_state.num_samples)

    # For Poisson Distribution
    elif distribution_choice == "Poisson Distribution":
        st.header("Poisson Distribution Simulation")
        if 'lam' not in st.session_state:
            st.session_state.lam = 5.0  # Ensure this is a float
            st.session_state.num_samples = 1000

        lam = st.number_input("Enter the average number of occurrences (λ)", min_value=0.0, value=float(st.session_state.lam))  # Cast value to float
        num_samples = st.number_input("Enter the number of samples", min_value=1, value=st.session_state.num_samples)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Calculate"):
                poisson_distribution_simulation(lam, num_samples)

        with col2:
            if st.button("Random"):
                st.session_state.lam = np.random.uniform(1.0, 20.0)  # Ensure this is a float
                st.session_state.num_samples = np.random.randint(100, 10000)
                poisson_distribution_simulation(st.session_state.lam, st.session_state.num_samples)

    # For Exponential Distribution
    elif distribution_choice == "Exponential Distribution":
        st.header("Exponential Distribution Simulation")
        if 'lam' not in st.session_state:
            st.session_state.lam = 1.0  # Ensure this is a float
            st.session_state.num_samples = 1000

        lam = st.number_input("Enter the rate parameter (λ)", min_value=0.0, value=float(st.session_state.lam))  # Cast value to float
        num_samples = st.number_input("Enter the number of samples", min_value=1, value=st.session_state.num_samples)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Calculate"):
                exponential_distribution_simulation(lam, num_samples)

        with col2:
            if st.button("Random"):
                st.session_state.lam = np.random.uniform(0.1, 5.0)  # Ensure this is a float
                st.session_state.num_samples = np.random.randint(100, 10000)
                exponential_distribution_simulation(st.session_state.lam, st.session_state.num_samples)


# Run the program
if __name__ == "__main__":
    main()
