import numpy as np
import scipy.stats as scs
import matplotlib.pyplot as plt
# Function to calculate the pooled probability or p_hat
def pooled_probability_func(total_A, total_B, exp_ctrl, exp_var):
    return (exp_ctrl + exp_var) / (total_A + total_B)

# Function to calculate the pooled standard error
def pooled_standard_error_func(total_A, total_B, exp_ctrl, exp_var):
    # Calculate the pooled probability
    p_hat = pooled_probability_func(total_A, total_B, exp_ctrl, exp_var)
    # Calculate the standard error using the appropriate formula
    SE = np.sqrt(p_hat * (1 - p_hat) * (1 / total_A + 1 / total_B))
    return SE

# Function to calculate the z-value based on the significance level
def z_value(sig_level=0.05, two_tailed=True):
    # Create a standard normal distribution
    z_dist = scs.norm()
    if two_tailed:
        # If the test is two-tailed, divide the significance level by 2
        sig_level = sig_level / 2
        area = 1 - sig_level
    else:
        area = 1 - sig_level
    # Calculate the z-value corresponding to the desired area
    z = z_dist.ppf(area)
    return z

# Function to calculate the confidence interval
def confidence_interval(sample_mean=0, sample_std=1, sample_size=1, sig_level=0.05):
    # Get the z-value based on the significance level
    z = z_value(sig_level)
    # Calculate the limits of the confidence interval
    left = sample_mean - z * sample_std / np.sqrt(sample_size)
    right = sample_mean + z * sample_std / np.sqrt(sample_size)
    return (left, right)

# Function to plot the confidence interval lines on the graph
def plot_confidence_interval(ax, mu, s, sig_level=0.05, color='grey'):
    # Calculate the limits of the confidence interval
    left, right = confidence_interval(sample_mean=mu, sample_std=s, sig_level=sig_level)
    # Plot vertical lines on the graph representing the confidence interval
    ax.axvline(left, c=color, linestyle='--', alpha=0.5)
    ax.axvline(right, c=color, linestyle='--', alpha=0.5)

# Function to plot the normal distribution
def plot_normal_distribution(ax, mu, std, with_confidence_interval=False, sig_level=0.05, label=None):
    # Create a set of values "x" for the normal distribution
    x = np.linspace(mu - 12 * std, mu + 12 * std, 1000)
    # Calculate the probability values of the normal distribution
    y = scs.norm(mu, std).pdf(x)
    # Plot the normal distribution on the graph
    ax.plot(x, y, label=label)
    if with_confidence_interval:
        # If specified, plot the confidence interval lines on the graph
        plot_confidence_interval(ax, mu, std, sig_level=sig_level)

# Function to plot the null hypothesis (H0) distribution
def plot_null_hypothesis(ax, stderr):
    # Plot the normal distribution with mean zero and specified standard error
    plot_normal_distribution(ax, 0, stderr, label="H0 - A - Null Hypothesis")
    # Plot the confidence interval lines for H0
    plot_confidence_interval(ax, mu=0, s=stderr, sig_level=0.05)

# Function to plot the alternative hypothesis (H1) distribution
def plot_alternative_hypothesis(ax, stderr, d_hat):
    # Plot the normal distribution with mean "d_hat" and specified standard error
    plot_normal_distribution(ax, d_hat, stderr, label="H1 - B - Alternative Hypothesis")

# Function to fill the area representing the power of the test on the graph
def show_power_area(ax, d_hat, stderr, sig_level):
    # Calculate the limits of the confidence interval for H0
    left, right = confidence_interval(sample_mean=0, sample_std=stderr, sig_level=sig_level)
    # Create a set of values "x" to fill the area
    x = np.linspace(-12 * stderr, 12 * stderr, 1000)
    # Create the null hypothesis distribution
    null = ab_distribution(stderr, 'A')
    # Create the alternative hypothesis distribution
    alternative = ab_distribution(stderr, d_hat, 'B')
    # Fill the area representing the power of the test on the graph
    ax.fill_between(x, 0, alternative.pdf(x), color='green', alpha=0.25, where=(x > right))
    # Add text indicating the power of the test on the graph
    ax.text(-3 * stderr, null.pdf(0), f'power = {1 - alternative.cdf(right):.3f}', fontsize=12, ha='right', color='k')

# Function to create a normal distribution
def ab_distribution(stderr, d_hat=0, group_type='A'):
    if group_type == 'A':
        sample_mean = 0
    elif group_type == 'B':
        sample_mean = d_hat
    dist = scs.norm(sample_mean, stderr)
    return dist

# Function to calculate the p-value
def p_value(total_A, total_B, p_conversao_a1, p_conversao_b1):
    return scs.binom(total_A, p_conversao_a1).pmf(p_conversao_b1 * total_B)

# bcr = Baseline Conversion Rate, in short, it is the conversion rate of the Control group.
# Main function to perform the A/B test
def abplot_func(total_A, total_B, bcr, d_hat, sig_level=0.05, show_p_value=True, show_legend=True):
    # Create a graph
    fig, ax = plt.subplots(figsize=(14, 8))
    # Calculate the number of conversions in samples A and B
    exp_ctrl = bcr * total_A
    exp_var = (bcr + d_hat) * total_B
    # Calculate the pooled standard error
    stderr = pooled_standard_error_func(total_A, total_B, exp_ctrl, exp_var)
    # Plot the null hypothesis (H0) distribution
    plot_null_hypothesis(ax, stderr)
    # Plot the alternative hypothesis (H1) distribution
    plot_alternative_hypothesis(ax, stderr, d_hat)
    # Set the limits of the graph
    ax.set_xlim(-8 * stderr, 14 * stderr)
    # Fill the area representing the power of the test on the graph
    show_power_area(ax, d_hat, stderr, sig_level)
    if show_p_value:
        # If desired, calculate and display the p-value on the graph
        null = ab_distribution(stderr, 'control')
        p_value_result = p_value(total_A, total_B, bcr, bcr + d_hat)
        ax.text(3 * stderr, null.pdf(0), f'p-value = {p_value_result:.4f}', fontsize=14, ha='left')
    if show_legend:
        # If desired, display the legend on the graph
        plt.legend(loc="lower left", bbox_to_anchor=(0, 1.0))
    plt.xlabel('X Normally distributed, z-scores')
    plt.ylabel('Probability Density of Normal Distributions')
    plt.show()


##-----------------------------------------------------------------##-------------------------------------------------------------##-------------------------------------------------------------------##


# Function to find the minimum sample size
def amostra_min(total_A, 
                total_B, 
                p_conversao_a1, 
                p_conversao_b1, 
                power=0.8, 
                sig_level=0.05, 
                two_sided=False):
    d_hat = p_conversao_b1 - p_conversao_a1
    k = total_A / total_B
    
    # Normal distribution to determine z values
    standard_norm = scs.norm(0, 1)

    # Find the z value for statistical power
    Z_beta = standard_norm.ppf(power)
    
    # Find alpha z
    if two_sided == True:
        Z_alpha = standard_norm.ppf(1 - sig_level/2)
    else:
        Z_alpha = standard_norm.ppf(1 - sig_level)

    # Pooled probability
    pooled_prob = (p_conversao_a1 + p_conversao_b1) / 2

    # Minimum sample size
    min_N = (2 * pooled_prob * (1 - pooled_prob) * (Z_beta + Z_alpha)**2 / d_hat**2)    

    return min_N
