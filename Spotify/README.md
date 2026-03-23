# Spotify Data Analysis

## Overview

This project explores Spotify track data to understand what factors influence song popularity.  
The analysis focuses on genre distribution, lyrical themes, artist dominance, and relationships between audio features and track success.

---

## Business Problem

Understanding what drives music popularity is critical for streaming platforms, record labels, and marketing teams.  
Without data-driven insights, decisions about promotion, content strategy, and recommendations rely on assumptions rather than evidence.

---

## Objective

The goal of this project is to analyze track-level data and identify patterns that explain:

- which genres dominate the market  
- how popularity is distributed across tracks and artists  
- whether audio features influence success  
- how lyrical themes are structured  

---

## Dataset

Source: Spotify dataset (Kaggle)

The dataset includes:

- track popularity scores  
- artist and genre information  
- audio features (e.g. energy, danceability, valence)  
- LDA-based lyrical topics  
- chart presence indicators  

---

## Methodology

### 1. Data Preparation
- removed duplicates and invalid records  
- standardized column names and data types  
- filtered missing values  

### 2. Feature Engineering
- created popularity segments using quartiles  
- prepared variables for comparative analysis  

### 3. Exploratory Data Analysis
- analyzed genre distribution and dataset structure  
- examined lyrical topic distribution (LDA topics)  
- identified most dominant artists and tracks  

### 4. Feature Analysis
- compared audio features across popularity levels  
- performed correlation analysis  
- evaluated relationships between features and success  

---

## Key Insights

- The dataset shows strong concentration effects across genres, artists, and tracks.
- A small number of genres dominate the landscape, which may bias popularity patterns.
- Popularity is highly skewed, with a limited number of tracks and artists driving most of the engagement.
- Audio features such as energy and danceability show only weak relationships with popularity.
- Lyrical themes are unevenly distributed, with a few dominant topics across the dataset.
- No single feature explains success — popularity appears to be driven by multiple factors.

---

## Business Implications

- Popularity is not determined by a single track characteristic, but by a combination of factors.
- Strong artist presence and exposure likely play a major role in success.
- Genre dominance should be considered when interpreting trends and building recommendation systems.
- Data-driven segmentation can support more targeted marketing and content strategies.

---

## Tools Used

- Python (Pandas, NumPy, Matplotlib, Seaborn)
- Jupyter Notebook

---

## Project Structure

- Spotify.ipynb
- README.md