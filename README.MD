
## <div class="header2">Structure </div>
- data/
  - raw/
    - cc.csv
- src/
  - `__`init__.py
  - abtest.py
- notebooks/
  - abtest
- requirements.txt
- README.md


# A/B Testing and Binomial Analysis for Marketing Campaign Evaluation

## Problem Description

Marketing companies aim to run successful campaigns, but the market is complex, and several options can work. Therefore, they often conduct A/B tests, a randomized experimentation process in which two or more versions of a variable (web page, page element, banner, etc.) are shown to different segments of people simultaneously to determine which version has the maximum impact and drives business metrics.

Companies seek to answer two fundamental questions:

1. Would the campaign be successful?
2. If the campaign was successful, how much of that success could be attributed to the ads?

With the second question in mind, it is common to conduct an A/B test. The majority of people will be exposed to ads (the experimental group), while a small portion of people (the control group) will see a Public Service Announcement (PSA) (or nothing) in the exact size and place where the ad would normally appear.

The goal of this dataset is to analyze the groups, determine if the ads were successful, calculate the company's revenue potential from the ads, and assess whether the difference between the groups is statistically significant.

## Possible Solutions

1. **A/B Testing:** Evaluate the performance of marketing campaigns by comparing conversion rates between the experimental group (exposed to ads) and the control group (exposed to PSA or nothing).

2. **Binomial Testing:** Analyze the effectiveness of ads through binomial tests, allowing the assessment of conversion proportions and the identification of significant differences.

3. **Sample Size Determination:** Use statistical methods to calculate the minimum sample size needed to ensure adequate statistical power in the tests.

## Notebook Structure

1. **Introduction:**
   - Problem description and objectives.
   
2. **Data Exploration:**
   - Initial analysis of the dataset to understand its structure and characteristics.
   
3. **A/B Testing:**
   - Implementation of A/B testing to compare conversion rates between groups.
   
4. **Binomial Tests:**
   - Application of binomial tests to assess the effectiveness of ads.
   
5. **Sample Size Determination:**
   - Calculation of the minimum sample size to ensure statistically robust results.
   
6. **Conclusions:**
   - Summary of findings and recommendations based on the obtained results.

## Additional Considerations

- Inclusion of graphical visualizations to facilitate the interpretation of results.
- Detailed documentation of the steps and codes used.
- Reserved space for the inclusion of relevant images, performance charts, etc.

**Conclusions:**

# A/B Test - Conclusion

Based on the results of the conducted A/B test, we can draw the following conclusions:

- **Significant Difference:** We observed a statistically significant difference in conversion rates between groups A (control) and B (experimental).
  
- **Observed Effect:** The `d_hat` value was calculated at approximately 0.0077, indicating a small but statistically significant advantage for group B.

- **Statistical Evidence:** The associated p-value was practically zero, reinforcing the statistical evidence of the observed difference.

- **High Statistical Power:** The test exhibited high statistical power, reaching 1.0, indicating its ability to detect the difference when it truly exists.

- **Minimum Sample Size:** The minimum sample size was calculated with two_sided equal to True, resulting in 5631.79, and with two_sided equal to False, resulting in 4436.16.

These results suggest that the introduction of ads had a significant impact on conversion rates compared to the control group.
