# Customer Segmentation using RFM Analysis

## Business Problem

The company has access to transactional data but lacks visibility into customer behavior and value distribution.  
There is no clear identification of high-value customers, loyal segments, or customers at risk of churn, which limits the effectiveness of retention and marketing strategies.

## Objective

The goal of this analysis is to segment customers based on their purchasing behavior and identify key groups that drive revenue, show growth potential, or require reactivation.

---

## Dataset

The dataset contains transactional data from an online retail store, including:

- Invoice information  
- Product details  
- Quantity and price  
- Customer ID  
- Country  

---

## Methodology

### 1. Data Preparation
- Removed missing customer IDs  
- Filtered out cancelled transactions  
- Removed non-positive quantity and price values  
- Cleaned column names and date formats  

### 2. Feature Engineering
Created key metrics:
- **Recency** – days since last purchase  
- **Frequency** – number of transactions  
- **Monetary** – total revenue per customer  

### 3. RFM Segmentation
Customers were scored and grouped into segments:
- Champions  
- Loyal Customers  
- Frequent Customers  
- Big Spenders  
- Recent Customers  
- At Risk  
- Others  

---

## Key Insights

- A small group of customers (Champions and Loyal Customers) generates a disproportionately large share of total revenue.  
- A large portion of the customer base contributes relatively little revenue despite high volume.  
- "At Risk" customers represent the biggest segment but have low current value, indicating churn risk.  
- Mid-value segments show potential for growth through targeted engagement.
- These findings highlight the importance of segment-based strategies and demonstrate how customer data can be used to support targeted business decisions.
---

## Visualizations

### Customer Segments – Recency vs Frequency
Bubble chart showing engagement patterns across segments.

### Revenue vs Customer Share
Comparison of how many customers are in each segment versus how much revenue they generate.

---

## Business Recommendations

- Focus retention strategies on high-value customers (Champions, Loyal Customers)  
- Implement reactivation campaigns for At Risk customers  
- Increase engagement of Recent and Frequent customers  
- Allocate marketing resources based on segment value rather than customer volume  

---

## Tools Used

- Python (Pandas, NumPy, Matplotlib)  
- Jupyter Notebook  

---

## Project Structure

Retail.ipynb
Retail 2009-10.csv
README.md