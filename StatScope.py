import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

class StatisticalAnalysisTool:
    def __init__(self):
        self.df = None

    # Load data (CSV file or manually input)
    def load_data(self, file_path=None):
        if file_path:
            self.df = pd.read_csv(file_path)
        else:
            st.error("No file path provided.")
    
    # Generate random data for simulation
    def generate_random_data(self):
        # Random data: 3 columns with random numbers
        self.df = pd.DataFrame({
            'A': np.random.randn(100),
            'B': np.random.randn(100),
            'C': np.random.randn(100)
        })
        st.success("Random data generated.")
    
    # Display DataFrame
    def show_dataframe(self):
        if self.df is not None:
            st.write(self.df.head())
        else:
            st.error("No data loaded.")

    # Calculate basic descriptive statistics
    def descriptive_statistics(self):
        if self.df is not None:
            st.write("Descriptive Statistics:")
            st.write(self.df.describe())
        else:
            st.error("Data not loaded.")
    
    # Perform t-test (independent two-sample t-test)
    def t_test(self, group1, group2):
        if self.df is not None:
            group1_data = self.df[group1]
            group2_data = self.df[group2]
            t_stat, p_val = stats.ttest_ind(group1_data, group2_data)
            st.write(f"T-test result: t-statistic = {t_stat}, p-value = {p_val}")
        else:
            st.error("Data not loaded.")
    
    # Perform ANOVA (Analysis of Variance)
    def anova(self, *groups):
        if self.df is not None:
            group_data = [self.df[group] for group in groups]
            f_stat, p_val = stats.f_oneway(*group_data)
            st.write(f"ANOVA result: F-statistic = {f_stat}, p-value = {p_val}")
        else:
            st.error("Data not loaded.")
    
    # Visualize histogram of a column
    def plot_histogram(self, column):
        if self.df is not None:
            fig, ax = plt.subplots()
            self.df[column].hist(bins=10, alpha=0.7, ax=ax)
            ax.set_title(f"Histogram of {column}")
            ax.set_xlabel(column)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
        else:
            st.error("Data not loaded.")
    
    # Visualize box plot of a column
    def plot_boxplot(self, column):
        if self.df is not None:
            fig, ax = plt.subplots()
            sns.boxplot(x=self.df[column], ax=ax)
            ax.set_title(f"Boxplot of {column}")
            st.pyplot(fig)
        else:
            st.error("Data not loaded.")
    
    # Compute correlation matrix
    def correlation_matrix(self):
        if self.df is not None:
            corr = self.df.corr()
            st.write("Correlation Matrix:")
            st.write(corr)
            # Create the figure and axis explicitly
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
            ax.set_title("Correlation Matrix")
            st.pyplot(fig)
        else:
            st.error("Data not loaded.")


# Streamlit App Interface
def main():
    st.title("Statistical Analysis Tool")

    tool = StatisticalAnalysisTool()

    option = st.selectbox(
        "Choose how you want to input the data",
        ["Random Data", "Enter Data Manually", "Upload CSV"]
    )

    if option == "Random Data":
        tool.generate_random_data()
        tool.show_dataframe()
    elif option == "Enter Data Manually":
        st.write("Enter column names and data below.")
        num_rows = st.number_input("Number of rows", min_value=1, value=5)
        columns = st.text_input("Enter column names (comma separated)", "A,B,C").split(',')
        data = []
        for i in range(num_rows):
            row = [st.number_input(f"Row {i+1} - {col}", value=0.0) for col in columns]
            data.append(row)
        
        tool.df = pd.DataFrame(data, columns=columns)
        tool.show_dataframe()

    elif option == "Upload CSV":
        st.write(
            """
            **Instructions for uploading CSV:**

            1. **Ensure your CSV file contains column headers** with descriptive names (e.g., 'Age', 'Salary', 'Height', etc.).
            2. Each row should represent a single observation or data entry.
            3. The dataset should include numerical or categorical data in each column.
            4. Please ensure there are no missing values, or handle them before uploading (empty cells will be treated as NaN).
            5. The CSV file should be properly formatted and free of extra text or non-tabular data.

            **Example CSV Format:**

            ```
            Age,Salary,Height
            25,50000,5.8
            30,60000,6.0
            22,40000,5.5
            ```

            Once the file is uploaded, the tool will automatically process it and display the first few rows.
            """
        )
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            tool.load_data(uploaded_file)
            tool.show_dataframe()

    if tool.df is not None:
        st.subheader("Statistical Analysis")

        analysis_option = st.selectbox(
            "Choose an analysis to perform",
            ["Descriptive Statistics", "T-Test", "ANOVA", "Plot Histogram", "Plot Boxplot", "Correlation Matrix"]
        )

        if analysis_option == "Descriptive Statistics":
            tool.descriptive_statistics()
        elif analysis_option == "T-Test":
            group1 = st.selectbox("Select the first group column:", tool.df.columns)
            group2 = st.selectbox("Select the second group column:", tool.df.columns)
            tool.t_test(group1, group2)
        elif analysis_option == "ANOVA":
            selected_columns = st.multiselect("Select columns for ANOVA:", tool.df.columns)
            if len(selected_columns) > 1:
                tool.anova(*selected_columns)
            else:
                st.error("Please select at least two columns for ANOVA.")
        elif analysis_option == "Plot Histogram":
            column = st.selectbox("Select column for histogram:", tool.df.columns)
            tool.plot_histogram(column)
        elif analysis_option == "Plot Boxplot":
            column = st.selectbox("Select column for boxplot:", tool.df.columns)
            tool.plot_boxplot(column)
        elif analysis_option == "Correlation Matrix":
            tool.correlation_matrix()

if __name__ == "__main__":
    main()
