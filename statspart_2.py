# 1. Z-test for Comparing a Sample Mean to a Known Population Mean
sample_data = np.array([10, 12, 14, 16, 18])
sample_mean = np.mean(sample_data)
population_mean = 15
sample_size = len(sample_data)
sample_std = np.std(sample_data, ddof=1)

# Z-test
z_score = (sample_mean - population_mean) / (sample_std / np.sqrt(sample_size))
p_value = stats.norm.sf(abs(z_score)) * 2  # Two-tailed test
print(f"1. Z-Test - Z-Score: {z_score}, P-Value: {p_value}")
if p_value < 0.05:
    print("   Reject the null hypothesis")
else:
    print("   Fail to reject the null hypothesis")

# 2. Simulate Random Data for Hypothesis Testing and Calculate the P-value
np.random.seed(0)
random_data = np.random.normal(loc=10, scale=2, size=100)
random_sample_mean = np.mean(random_data)
random_sample_std = np.std(random_data, ddof=1)
z_score_random = (random_sample_mean - population_mean) / (random_sample_std / np.sqrt(len(random_data)))
p_value_random = stats.norm.sf(abs(z_score_random)) * 2  # Two-tailed test
print(f"2. Simulated Data - Z-Score: {z_score_random}, P-Value: {p_value_random}")

# 3. One-Sample Z-Test for Sample Mean Comparison with Population Mean
z_score_one_sample = (sample_mean - population_mean) / (sample_std / np.sqrt(sample_size))
p_value_one_sample = stats.norm.sf(abs(z_score_one_sample)) * 2
print(f"3. One-Sample Z-Test - Z-Score: {z_score_one_sample}, P-Value: {p_value_one_sample}")

# 4. Two-Tailed Z-Test and Decision Region Visualization
z_score_two_tailed = (sample_mean - population_mean) / (sample_std / np.sqrt(sample_size))
p_value_two_tailed = stats.norm.sf(abs(z_score_two_tailed)) * 2
print(f"4. Two-Tailed Z-Test - Z-Score: {z_score_two_tailed}, P-Value: {p_value_two_tailed}")
x = np.linspace(-4, 4, 1000)
y = stats.norm.pdf(x)

plt.plot(x, y, label="Standard Normal Distribution")
plt.fill_between(x, y, where=(x > 1.96) | (x < -1.96), color='red', alpha=0.5, label="Reject Region")
plt.axvline(x=z_score_two_tailed, color='green', linestyle='--', label="Z-Score")
plt.title("Two-Tailed Z-Test")
plt.legend()
plt.show()

# 5. Type 1 and Type 2 Errors Function
def type1_type2_error(sample_data, population_mean, alpha=0.05):
    z_score = (np.mean(sample_data) - population_mean) / (np.std(sample_data, ddof=1) / np.sqrt(len(sample_data)))
    p_value = stats.norm.sf(abs(z_score)) * 2
    type1_error = p_value < alpha  # False Positive (Rejecting a true null hypothesis)
    type2_error = p_value >= alpha  # False Negative (Failing to reject a false null hypothesis)
    return type1_error, type2_error

type1, type2 = type1_type2_error(sample_data, population_mean)
print(f"5. Type 1 Error: {type1}, Type 2 Error: {type2}")

# 6. Independent T-Test
group1 = np.array([23, 21, 18, 24, 25])
group2 = np.array([30, 29, 27, 32, 31])
t_stat, p_value_t_test = stats.ttest_ind(group1, group2)
print(f"6. Independent T-Test - T-Statistic: {t_stat}, P-Value: {p_value_t_test}")
if p_value_t_test < 0.05:
    print("   Reject the null hypothesis")
else:
    print("   Fail to reject the null hypothesis")

# 7. Paired Sample T-Test
before = np.array([10, 15, 12, 18, 11])
after = np.array([12, 17, 13, 19, 14])
t_stat_paired, p_value_paired = stats.ttest_rel(before, after)
print(f"7. Paired Sample T-Test - T-Statistic: {t_stat_paired}, P-Value: {p_value_paired}")
if p_value_paired < 0.05:
    print("   Reject the null hypothesis")
else:
    print("   Fail to reject the null hypothesis")

# 8. Simulate Data for Z-Test and T-Test, then Compare Results
sample_data_z = np.random.normal(10, 2, 100)
z_score_test = (np.mean(sample_data_z) - population_mean) / (np.std(sample_data_z, ddof=1) / np.sqrt(len(sample_data_z)))
p_value_z_test = stats.norm.sf(abs(z_score_test)) * 2
t_stat_test, p_value_t_test_sim = stats.ttest_1samp(sample_data_z, population_mean)

print(f"8. Z-Test vs T-Test Comparison:")
print(f"   Z-Test - Z-Score: {z_score_test}, P-Value: {p_value_z_test}")
print(f"   T-Test - T-Statistic: {t_stat_test}, P-Value: {p_value_t_test_sim}")

# 9. Confidence Interval for a Sample Mean
def confidence_interval(sample_data, confidence_level=0.95):
    sample_mean = np.mean(sample_data)
    sample_std = np.std(sample_data, ddof=1)
    sample_size = len(sample_data)
    margin_of_error = stats.t.ppf((1 + confidence_level) / 2, df=sample_size - 1) * (sample_std / np.sqrt(sample_size))
    return sample_mean - margin_of_error, sample_mean + margin_of_error

ci_lower, ci_upper = confidence_interval(sample_data)
print(f"9. Confidence Interval for Sample Mean: ({ci_lower}, {ci_upper})")

# 10. Margin of Error for a Given Confidence Level Using Sample Data
sample_data = np.array([12, 14, 16, 18, 20, 22, 24, 26, 28, 30])
sample_mean = np.mean(sample_data)
sample_std = np.std(sample_data, ddof=1)
sample_size = len(sample_data)
confidence_level = 0.95
z_score = stats.norm.ppf((1 + confidence_level) / 2)
margin_of_error = z_score * (sample_std / np.sqrt(sample_size))
print(f"Margin of Error: {margin_of_error}")

# 11. Implement Bayesian Inference Using Bayes' Theorem
def bayes_theorem(P_B_given_A, P_A, P_B):
    return (P_B_given_A * P_A) / P_B

P_A = 0.1
P_B_given_A = 0.8
P_B = 0.2
P_A_given_B = bayes_theorem(P_B_given_A, P_A, P_B)
print(f"Posterior Probability (P(A|B)): {P_A_given_B}")

# 12. Chi-Square Test for Independence Between Two Categorical Variables
observed = np.array([[10, 20], [20, 30]])
chi2_stat, p_value, dof, expected = stats.chi2_contingency(observed)
print(f"Chi-Square Statistic: {chi2_stat}")
print(f"P-Value: {p_value}")
print(f"Degrees of Freedom: {dof}")
print(f"Expected Frequencies: {expected}")

# 13. Calculate the Expected Frequencies for a Chi-Square Test Based on Observed Data
expected_frequencies = stats.chi2_contingency(observed)[3]
print(f"Expected Frequencies: \n{expected_frequencies}")

# 14. Goodness-of-Fit Test Using Python
observed_frequencies = np.array([50, 60, 40])
expected_frequencies = np.array([50, 50, 50])
chi2_stat, p_value = stats.chisquare(observed_frequencies, expected_frequencies)
print(f"Chi-Square Statistic: {chi2_stat}")
print(f"P-Value: {p_value}")

# 15. Simulate and Visualize the Chi-Square Distribution
df = 2
x = np.linspace(0, 10, 1000)
y = stats.chi2.pdf(x, df)
plt.plot(x, y)
plt.title(f"Chi-Square Distribution (df={df})")
plt.xlabel('x')
plt.ylabel('Density')
plt.show()

# 16. Implement an F-Test for Comparing the Variances of Two Random Samples
np.random.seed(0)
sample1 = np.random.normal(10, 2, 100)
sample2 = np.random.normal(12, 2.5, 100)
f_stat = np.var(sample1, ddof=1) / np.var(sample2, ddof=1)
dfn = len(sample1) - 1
dfd = len(sample2) - 1
p_value_f_test = 1 - stats.f.cdf(f_stat, dfn, dfd)
print(f"F-Statistic: {f_stat}")
print(f"P-Value: {p_value_f_test}")

# 17. Perform an ANOVA Test to Compare Means Between Multiple Groups
group1 = np.random.normal(10, 2, 50)
group2 = np.random.normal(12, 2, 50)
group3 = np.random.normal(14, 2, 50)
f_stat, p_value_anova = stats.f_oneway(group1, group2, group3)
print(f"ANOVA - F-Statistic: {f_stat}")
print(f"P-Value: {p_value_anova}")

# 18. Perform a One-Way ANOVA Test and Plot the Results
plt.boxplot([group1, group2, group3], labels=['Group 1', 'Group 2', 'Group 3'])
plt.title("One-Way ANOVA: Group Comparisons")
plt.ylabel("Value")
plt.show()

# 19. Check Assumptions for ANOVA (Normality, Independence, Equal Variance)
normality_p1 = normaltest(group1)[1]
normality_p2 = normaltest(group2)[1]
normality_p3 = normaltest(group3)[1]
variance_p = levene(group1, group2, group3)[1]
print(f"Group 1 Normality P-Value: {normality_p1}")
print(f"Group 2 Normality P-Value: {normality_p2}")
print(f"Group 3 Normality P-Value: {normality_p3}")
print(f"Equality of Variance P-Value: {variance_p}")

# 20. Perform a Two-Way ANOVA and Visualize the Results
factor1 = np.random.choice(['A', 'B'], size=100)
factor2 = np.random.choice(['X', 'Y'], size=100)
values = np.random.normal(10, 2, 100)
df = pd.DataFrame({'Factor1': factor1, 'Factor2': factor2, 'Values': values})
model = ols('Values ~ C(Factor1) + C(Factor2) + C(Factor1):C(Factor2)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)
sns.boxplot(x='Factor1', y='Values', hue='Factor2', data=df)
plt.title('Two-Way ANOVA Results')
plt.show()

# 21. Visualize the F-Distribution
dfn = 2
dfd = 5
x = np.linspace(0, 5, 1000)
y = stats.f.pdf(x, dfn, dfd)
plt.plot(x, y)
plt.title(f"F-Distribution (dfn={dfn}, dfd={dfd})")
plt.xlabel('x')
plt.ylabel('Density')
plt.show()

# 22. One-Way ANOVA and Boxplots for Group Means (Already Covered in Step 9)
plt.boxplot([group1, group2, group3], labels=['Group 1', 'Group 2', 'Group 3'])
plt.title("One-Way ANOVA: Group Comparisons")
plt.ylabel("Value")
plt.show()

# 23. Simulate Random Data and Perform Hypothesis Testing to Evaluate the Means
sample_data = np.random.normal(10, 2, 100)
t_stat, p_value_t_test = stats.ttest_1samp(sample_data, 10)
print(f"One-Sample T-Test - T-Statistic: {t_stat}")
print(f"P-Value: {p_value_t_test}")

# 24. Hypothesis Test for Population Variance Using a Chi-Square Distribution
sample_data = np.random.normal(10, 2, 100)
sample_var = np.var(sample_data, ddof=1)
n = len(sample_data)
chi2_stat = (n - 1) * sample_var / 4  # Assuming population variance is 4
p_value_chi2 = 1 - stats.chi2.cdf(chi2_stat, df=n - 1)
print(f"Chi-Square Test for Variance - Chi-Square Statistic: {chi2_stat}")
print(f"P-Value: {p_value_chi2}")

# 25. Z-Test for Comparing Proportions Between Two Datasets or Groups
successes1 = 50
total1 = 100
successes2 = 30
total2 = 80
p1 = successes1 / total1
p2 = successes2 / total2
pooled_p = (successes1 + successes2) / (total1 + total2)
z_stat = (p1 - p2) / np.sqrt(pooled_p * (1 - pooled_p) * (1 / total1 + 1 / total2))
p_value_z_test = stats.norm.sf(abs(z_stat)) * 2
print(f"Z-Test for Proportions - Z-Statistic: {z_stat}")
print(f"P-Value: {p_value_z_test}")

# 26. F-Test for Comparing the Variances of Two Datasets
f_stat = np.var(sample1, ddof=1) / np.var(sample2, ddof=1)
dfn = len(sample1) - 1
dfd = len(sample2) - 1
p_value_f_test = 1 - stats.f.cdf(f_stat, dfn, dfd)
print(f"F-Test for Variances - F-Statistic: {f_stat}")
print(f"P-Value: {p_value_f_test}")

# 17. Chi-Square Test for Goodness of Fit with Simulated Data
observed_frequencies = np.random.poisson(5, 100)
expected_frequencies = np.full_like(observed_frequencies, 5)
chi2_stat, p_value = stats.chisquare(observed_frequencies, expected_frequencies)
print(f"Chi-Square Goodness of Fit - Chi-Square Statistic: {chi2_stat}")
print(f"P-Value: {p_value}")