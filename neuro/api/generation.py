import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm


# 30k
# cpm: log normal, 0-700
# wpm: log normal
# acc: normal
# sat: logarithmic normal
# age: 10-70, log normal
def generate_data(count):
    for _ in range(count):
        cpm, wpm, accuracy, satisfaction, age = random.triangular()


def generate_lognormal_within_range(min_val, max_val, desired_median, num_samples=1000):
    # Calculate parameters for the log-normal distribution
    sigma = np.sqrt(2 * (np.log(desired_median) - np.log(min_val)) / np.log(max_val / min_val))
    mu = np.log(desired_median) - 0.5 * sigma ** 2

    # Generate log-normal random numbers
    lognormal_samples = np.random.lognormal(mean=mu, sigma=sigma, size=num_samples)

    # Clip the generated samples to fit within the specified range
    lognormal_samples = np.clip(lognormal_samples, min_val, max_val * random.randint(80, 95) // 100)
    return lognormal_samples.astype(int)


def plot_lognormal_distribution(lognormal_samples, min_val, max_val):
    # Generate a range of values for the x-axis
    x = np.linspace(min_val, max_val, 1000)

    # Plot histogram
    plt.hist(lognormal_samples, bins=30, density=True, alpha=0.5, color='b', label='Histogram')

    # Plot the probability density function (PDF) of the log-normal distribution
    sigma = np.std(np.log(lognormal_samples))
    mu = np.mean(np.log(lognormal_samples))
    pdf_values = lognorm.pdf(x, s=sigma, scale=np.exp(mu))
    plt.plot(x, pdf_values, 'r-', lw=2, label='Log-Normal PDF')

    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Log-Normal Distribution')
    plt.legend()


# Example usage
min_val = 10.0  # Minimum value
max_val = 700.0  # Maximum value
desired_median = 220.0  # Desired median
num_samples = 30000  # Number of random samples

lognormal_samples_cpm = generate_lognormal_within_range(min_val, max_val, desired_median, num_samples)
lognormal_samples_wpm = np.floor_divide(lognormal_samples_cpm, random.randint(5, 6))
print(lognormal_samples_cpm)
print(lognormal_samples_wpm)
# Plot the original log-normal distribution
plt.subplot(1, 2, 1)
plot_lognormal_distribution(lognormal_samples_cpm, min_val, max_val)
plt.title('Original Log-Normal Distribution')

# Plot the log-normal distribution after division by 5
plt.subplot(1, 2, 2)
plot_lognormal_distribution(lognormal_samples_wpm, min_val / 5, max_val / 5)
plt.title('Log-Normal Distribution (Divided by 5)')

plt.tight_layout()
plt.show()
