# Hitting vs Pitching vs Fielding vs Baserunning
## Content
1. Intro
2. Metadata
3. Data Cleaning
4. EDA (Exploratory Data Analysis)
5. Feature Scaling
6. Random Forest Regressoin
7. Feature Importance
8. Cross-era Comparison
9. Multiple Linear Regression
10. Conclusion

### 1. Intro
Baseball is a complicated sport that consists of many different factors such as player skills, team chemistry, health, money, weather and so on (*even luck as well*). And these various factors can be broken down as follows: what we can measure and predict, and what we cannot. For example, we can easily measure skills and team payrolls, while it's relatively hard (may be impossible) to accurately measure weather and luck.

Then one of the most reasonable questions raised by baseball operators and fans would be **How much of each factor contributes a team's winning percentage?**
In this analysis, I focused on what I can predict: **players' pure skills**. The reason I excluded other factors is simple. *They're either unavailable or unmeasurable*. Team payroll data for old teams (1870s ~ 1950s) are unavailable, and team chemistry, impacts of weather on baseball or luck are unmeasaurable.

Players' pure skills can further be broke down into four different aspects: **Hitting**, **Pitching**, **Fielding**, and **Baserunning**. To find the percentage contributions of these factors to a team's winning percentage, I ran a random forest regression model and used an impurity-based feature importance method.
