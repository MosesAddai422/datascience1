#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 14:11:39 2024

@author: mosesodeiaddai
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

#This project constitutes the use of chi-square for A/B testing in evaluating effectiveness of an ad 

#getting data
#If "ad" the person saw the advertisement(test group), if "psa" they only saw the public service announcement(control group)
data = pd.read_csv("/Users/mosesodeiaddai/Downloads/marketing_AB.csv")

#summary stats
summary_stats = data.groupby('test group').agg(
    conversions=('converted', 'sum'),
    total=('converted', 'count')
).reset_index()
summary_stats['conversion_rate'] = summary_stats['conversions'] / summary_stats['total']
print(summary_stats)

#implementing Chi-square for independence
conti_tb = np.array([
    [summary_stats.loc[0, 'conversions'], summary_stats.loc[0, 'total'] - summary_stats.loc[0, 'conversions']],
    [summary_stats.loc[1, 'conversions'], summary_stats.loc[1, 'total'] - summary_stats.loc[1, 'conversions']]
])

#getting chi-square results
chisq, pval, dof, ex = chi2_contingency(conti_tb)
print(f"Chi-square test statistic: {chisq}")
print(f"P-value: {pval}")

#visualizing results 
plt.bar(summary_stats['test group'], summary_stats['conversion_rate'], color=['blue', 'orange'])
plt.title('Conversion Rates for A/B Test')
plt.xlabel('Groups')
plt.ylabel('Conversion Rate')
plt.show()

#Given pval is less than than significance level of 5%, we can say that the impact of the ad on 
#conversion is statistically significant. 