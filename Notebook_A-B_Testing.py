#!/usr/bin/env python
# coding: utf-8

# pip install pandas-profiling

# In[6]:


# Basics
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set()
import sklearn

# Advance
from pandas_profiling import ProfileReport
from scipy.stats import f_oneway
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import ttest_ind


# ### Data Loading and Data Profile
# 
# Load the *WA_Marketing-Campaign.csv* into a pandas data frame and create data profile report.
# 

# In[3]:


df = pd.DataFrame()
df = pd.read_csv("/Users/omerfaruk/Desktop/DS Projects/Food Market A:B Testing/WA_Marketing-Campaign.csv")
df


# In[4]:


df.info()


# In[5]:


profile = ProfileReport(df, title="Food Market Dataset Report")
profile.to_notebook_iframe()


# ### 1. ANOVA

# In[7]:


# Perform an ANOVA test to compare sales between the three promotions
# Group the sales data by promotion
promotion_groups = df.groupby('Promotion')['SalesInThousands'].apply(list)

# Perform one-way ANOVA
anova_test_stat, anova_p_value = f_oneway(*promotion_groups)

# Print the results
print("ANOVA Test Results")
print(f"Test Statistic: {anova_test_stat:.4f}")
print(f"p-value: {anova_p_value:.4f}")

# Interpretation
if anova_p_value < 0.05:
    print("The differences in sales between promotions are statistically significant.")
else:
    print("No statistically significant difference in sales between promotions.")


# ### ANOVA Results
# * Test Statistic: 21.9535
# * p-value: 0.0000 (statistically significant)
# #### Interpretation:
# 
# * There are significant differences in sales between the three promotion groups.
# * This warrants further investigation to determine which promotions perform better.

# ### 2. Uplift Modeling

# In[8]:


# Prepare data for uplift modeling
df['Promotion'] = df['Promotion'].astype(str)  # Ensure 'Promotion' is treated as categorical
df['Treatment'] = df['Promotion'].apply(lambda x: 1 if x != '1' else 0)  # Baseline: Promotion 1

# Target transformation: Higher sales = success (binary outcome for uplift modeling)
df['Success'] = (df['SalesInThousands'] > df['SalesInThousands'].median()).astype(int)

# Features and labels
X = df[['MarketSize', 'AgeOfStore', 'Promotion', 'week']]
X = pd.get_dummies(X, drop_first=True)  # One-hot encoding for categorical variables
y = df['Success']
treatment = df['Treatment']

# Model: Random Forest Classifier
uplift_model = RandomForestClassifier(random_state=42)
uplift_model.fit(X, treatment * y)

# Predict uplift scores
uplift_scores = uplift_model.predict_proba(X)[:, 1] - uplift_model.predict_proba(X)[:, 0]

# Evaluate performance (ROC-AUC)
auc = roc_auc_score(treatment, uplift_scores)
print(f"Uplift Model AUC: {auc:.4f}")


# ### Uplift Modeling Results
# * AUC: 0.8619
# #### Interpretation:
# * The uplift model's AUC score indicates good predictive performance in distinguishing treatment effects.
# * This suggests that the model is effective in identifying which promotion increases sales most effectively.

# ### 3. Bootstrapping

# In[9]:


# Bootstrapping to compare mean sales between promotions
np.random.seed(42)

# Function to perform bootstrap sampling and calculate mean differences
def bootstrap_means(data, group_col, target_col, n_bootstrap=1000):
    groups = data[group_col].unique()
    means = {group: [] for group in groups}
    for _ in range(n_bootstrap):
        for group in groups:
            sample = data[data[group_col] == group][target_col].sample(frac=1, replace=True)
            means[group].append(sample.mean())
    return means

# Perform bootstrap
bootstrap_results = bootstrap_means(df, group_col='Promotion', target_col='SalesInThousands')

# Calculate mean differences between groups
mean_differences = {}
groups = list(bootstrap_results.keys())
for i in range(len(groups)):
    for j in range(i + 1, len(groups)):
        diff = np.mean(bootstrap_results[groups[i]]) - np.mean(bootstrap_results[groups[j]])
        mean_differences[f"{groups[i]} vs {groups[j]}"] = diff

# Print mean differences
print("Bootstrapping Results")
for pair, diff in mean_differences.items():
    print(f"{pair}: Mean Difference = {diff:.4f}")


# ### Bootstrapping Results
# #### Mean Differences:
# * Promotion 3 vs. Promotion 2: +7.9477
# * Promotion 3 vs. Promotion 1: -2.7430
# * Promotion 2 vs. Promotion 1: -10.6907
# #### Interpretation:
# * Promotion 3 has the highest mean sales compared to Promotion 2.
# * However, Promotion 3 performs slightly worse than Promotion 1, but the difference is relatively small.
# * Promotion 2 has the lowest mean sales overall.

# ### Next Steps
# To solidify this decision:
# 
# 1. We can perform pairwise t-tests or confidence interval analysis to confirm the differences between specific promotions.
# 2. Then, reassess promotional costs and customer engagement metrics to ensure Promotion 3 aligns with business objectives.
# 

# ### Pairwise T-Tests and Confidence Intervals

# In[10]:


# Perform pairwise t-tests between promotions
promotions = df['Promotion'].unique()
pairwise_results = {}

for i in range(len(promotions)):
    for j in range(i + 1, len(promotions)):
        group1 = df[df['Promotion'] == promotions[i]]['SalesInThousands']
        group2 = df[df['Promotion'] == promotions[j]]['SalesInThousands']
        t_stat, p_value = ttest_ind(group1, group2, equal_var=False)  # Welch's t-test
        pairwise_results[f"{promotions[i]} vs {promotions[j]}"] = {'t_stat': t_stat, 'p_value': p_value}

# Print pairwise t-test results
print("Pairwise T-Test Results")
for pair, results in pairwise_results.items():
    print(f"{pair}: t-stat = {results['t_stat']:.4f}, p-value = {results['p_value']:.4f}")

# Confidence interval calculation
def bootstrap_confidence_interval(group1, group2, n_bootstrap=1000, alpha=0.05):
    diffs = []
    for _ in range(n_bootstrap):
        sample1 = np.random.choice(group1, size=len(group1), replace=True)
        sample2 = np.random.choice(group2, size=len(group2), replace=True)
        diffs.append(sample1.mean() - sample2.mean())
    lower = np.percentile(diffs, 100 * (alpha / 2))
    upper = np.percentile(diffs, 100 * (1 - alpha / 2))
    return lower, upper

# Calculate and print confidence intervals
print("\nConfidence Intervals for Mean Differences")
for i in range(len(promotions)):
    for j in range(i + 1, len(promotions)):
        group1 = df[df['Promotion'] == promotions[i]]['SalesInThousands'].values
        group2 = df[df['Promotion'] == promotions[j]]['SalesInThousands'].values
        ci_lower, ci_upper = bootstrap_confidence_interval(group1, group2)
        print(f"{promotions[i]} vs {promotions[j]}: CI = [{ci_lower:.4f}, {ci_upper:.4f}]")


# ### Pairwise T-Test Results:
# 1. Promotion 3 vs Promotion 2:
# 
# * t-stat = 4.8814, p-value = 0.0000
# * Interpretation: The difference in sales between Promotion 3 and Promotion 2 is statistically significant (p-value < 0.05). Promotion 3 has a significantly higher sales outcome compared to Promotion 2.
# 
# 2. Promotion 3 vs Promotion 1:
# 
# * t-stat = -1.5560, p-value = 0.1206
# * Interpretation: The difference in sales between Promotion 3 and Promotion 1 is not statistically significant (p-value > 0.05). This means the sales outcomes for Promotion 3 and Promotion 1 are similar, and we cannot reject the null hypothesis that there is no difference in sales between them.
# 
# 3. Promotion 2 vs Promotion 1:
# 
# * t-stat = -6.4275, p-value = 0.0000
# * Interpretation: The difference in sales between Promotion 2 and Promotion 1 is statistically significant (p-value < 0.05). Promotion 1 performs better than Promotion 2, with a significant difference in sales.

# ### Confidence Interval Results:
# 1. Promotion 3 vs Promotion 2:
# 
# * CI = [4.8809, 11.1622]
# * Interpretation: The confidence interval for the mean difference between Promotion 3 and Promotion 2 does not include zero. This confirms that Promotion 3 significantly outperforms Promotion 2 in terms of sales.
# 
# 2. Promotion 3 vs Promotion 1:
# 
# * CI = [-6.0149, 0.7421]
# * Interpretation: The confidence interval includes zero, meaning the difference in sales between Promotion 3 and Promotion 1 is not statistically significant. Therefore, there's no clear winner between these two promotions.
# 
# 3. Promotion 2 vs Promotion 1:
# 
# * CI = [-14.1082, -7.5414]
# * Interpretation: The confidence interval does not include zero, indicating that Promotion 1 outperforms Promotion 2 significantly, as the mean difference is negative, and Promotion 1 has higher sales.

# ## Final Conclusion:
# * Promotion 3 stands out as the best-performing promotion when compared to Promotion 2.
# * Promotion 1 performs similarly to Promotion 3, but the difference is not statistically significant.
# * Promotion 2 is the least effective promotion, performing significantly worse than both Promotion 3 and Promotion 1.
# 

# In[ ]:




