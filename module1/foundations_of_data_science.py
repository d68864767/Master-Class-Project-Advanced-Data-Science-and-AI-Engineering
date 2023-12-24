```python
# Module 1: Foundations of Data Science

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class DataScienceFoundations:
    def __init__(self, data):
        self.data = pd.read_csv(data)

    def data_cleaning(self):
        # Remove duplicates
        self.data.drop_duplicates(inplace=True)

        # Remove null values
        self.data.dropna(inplace=True)

        return self.data

    def exploratory_data_analysis(self):
        # Summary statistics
        summary = self.data.describe()

        # Correlation matrix
        correlation = self.data.corr()

        return summary, correlation

    def data_preprocessing(self):
        # Normalize the data
        normalized_data = (self.data - self.data.min()) / (self.data.max() - self.data.min())

        return normalized_data

    def statistical_analysis(self):
        # Hypothesis testing
        t_statistic, p_value = stats.ttest_1samp(self.data, 0)

        return t_statistic, p_value

if __name__ == "__main__":
    ds = DataScienceFoundations('data.csv')
    cleaned_data = ds.data_cleaning()
    summary, correlation = ds.exploratory_data_analysis()
    normalized_data = ds.data_preprocessing()
    t_statistic, p_value = ds.statistical_analysis()

    print("Cleaned Data:\n", cleaned_data)
    print("Summary Statistics:\n", summary)
    print("Correlation Matrix:\n", correlation)
    print("Normalized Data:\n", normalized_data)
    print("T-statistic:\n", t_statistic)
    print("P-value:\n", p_value)
```
