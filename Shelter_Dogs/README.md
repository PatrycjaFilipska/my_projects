# Shelter Dogs — Exploratory Data Analysis

## Overview

This project explores a dataset of shelter dogs available for adoption in Hungary.  
The dataset includes demographic, physical, and behavioral characteristics such as age, breed, size, coat, and social traits.

The goal of the analysis is to better understand the structure of the shelter population and identify key patterns in dog characteristics.

---

## Dataset

Source: https://www.kaggle.com/datasets/jmolitoris/adoptable-dogs/data  

The dataset contains information about nearly 3,000 dogs, including:

- demographic features (age, sex, breed)
- physical characteristics (size, color, coat)
- behavioral traits (interaction with people, children, other animals)
- additional attributes related to shelter records

---

## Methodology

The analysis follows a structured exploratory approach:

### 1. Data Preparation
- handling missing values  
- cleaning and selecting relevant features  

### 2. Age Distribution
- analysis of dominant age groups  

### 3. Physical Characteristics
- size distribution  
- coat and color analysis  

### 4. Social Behavior
- interaction with people and children  

### 5. Social Compatibility
- relationships between behavioral traits  

### 6. Breed Distribution
- identification of dominant breeds  

---

## Key Insights

- The shelter population is concentrated within specific age groups, rather than evenly distributed across all life stages.
- Most dogs are friendly toward people, but compatibility with children is more selective and varies across individuals.
- Behavioral traits are not fully aligned — dogs that like people do not always like children.
- Physical characteristics are highly imbalanced: short-coated and medium-to-large dogs dominate the dataset.
- A small number of colors and breeds account for a large share of the population, while many categories appear rarely.
- The dataset shows strong concentration effects, which may influence interpretation and limit generalization.

---

## Conclusions

The analysis highlights that shelter dog populations are shaped by a limited number of dominant physical and behavioral profiles.

No single characteristic is sufficient to assess suitability for adoption. Instead, a combination of traits should be considered when evaluating dogs and matching them with potential adopters.

---

## Tools & Libraries

- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  

---

## Project Structure

- Shelter_Dogs.ipynb
- README.md