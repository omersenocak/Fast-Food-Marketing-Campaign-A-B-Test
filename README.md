# About Dataset

A fast-food chain plans to add a new item to its menu. However, they are still undecided between three possible marketing campaigns for promoting the new product. In order to determine which promotion has the greatest effect on sales, the new item is introduced at locations in several randomly selected markets. A different promotion is used at each location, and the weekly sales of the new item are recorded for the first four weeks. 

This data set is provided by 'IBM Watson Analytics Marketing Campaign' and the project is available at Kaggle. Please use below link to visit project page:

https://www.kaggle.com/datasets/chebotinaa/fast-food-marketing-campaign-ab-test/data

# A/B Testing for Marketing Campaign Effectiveness

## Project Overview
This project evaluates the effectiveness of three different marketing campaigns for a fast-food chain using data collected from various locations. The primary goal is to identify which marketing campaign leads to the highest sales for a newly introduced menu item.

---

## Dataset Description
The dataset includes information from multiple markets and stores over four weeks. Each store used one of three promotions, and weekly sales were recorded. The dataset is stored in `WA_Marketing-Campaign.csv` and contains the following columns:

- **MarketID**: Unique identifier for the market.
- **MarketSize**: Size of the market (e.g., Small, Medium, Large).
- **LocationID**: Unique identifier for the store location.
- **AgeOfStore**: Age of the store in years.
- **Promotion**: Identifier for the type of promotion (1, 2, or 3).
- **week**: Week of the observation.
- **SalesInThousands**: Weekly sales figures for the store, in thousands.

---

## Key Steps and Analysis
### 1. Data Loading and Profiling
- The dataset is loaded and examined for completeness and structure.
- Generated an interactive data profile report using the `pandas-profiling` library to explore distributions, missing values, and correlations.

### 2. Statistical Analysis
- **ANOVA Test**: Conducted a one-way ANOVA to check if there are significant differences in sales across promotions.
- **Pairwise T-Tests**: Compared the sales between pairs of promotions to pinpoint where differences lie.
- **Confidence Interval Analysis**: Calculated bootstrapped confidence intervals for mean differences in sales.

### 3. Machine Learning for Uplift Modeling
- Built a Random Forest-based uplift model to evaluate how each promotion impacts sales.
- Calculated uplift scores and compared them across promotions using the ROC-AUC metric.

---

## Results
### Statistical Insights
- The ANOVA test revealed statistically significant differences in sales between promotions.
- Pairwise t-tests identified **Promotion 3** as the top-performing strategy compared to **Promotion 2**, with a mean difference of ~7.95 in sales.

### Uplift Model
- The Random Forest uplift model achieved an **AUC score of 0.8619**, indicating strong predictive performance in identifying which promotions positively impact sales.

---

## How to Use
### Prerequisites
- Python 3.x
- Required Libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scipy`
  - `sklearn`
  - `pandas-profiling`

### Steps to Run
1. Clone the repository and navigate to the project directory.
2. Ensure the dataset (`WA_Marketing-Campaign.csv`) is available in the project directory.
3. Open and run `Notebook_A-B_Testing.ipynb` in Jupyter Notebook or another compatible environment.

---

## Visualizations
The project includes visualizations for:
- Sales distribution across promotions.
- Pairwise comparisons using boxplots.
- Uplift model scores.

* Note: It will be available when you run 'profile = ProfileReport(df, title="Food Market Dataset Report")
profile.to_notebook_iframe()' 
---

## Conclusion
Based on the analyses:
- **Promotion 3** is the best-performing campaign, driving the highest sales compared to other promotions.
- Further investigation and refinement of Promotion 3 could optimize its effectiveness even more.

