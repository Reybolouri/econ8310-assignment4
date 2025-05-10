import pandas as pd
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/dustywhite7/Econ8310/master/AssignmentData/cookie_cats.csv"
df = pd.read_csv(url)
print(df.head())
print("\nSummary statistics:")
print(df.describe(include='all'))


df['retention_1'] = df['retention_1'].astype(int)
df['retention_7'] = df['retention_7'].astype(int)

###### MODEL FOR 1-DAY RETENTION
with pm.Model() as model_day1:
    # Priors for each gate
    p30_1 = pm.Uniform('p30_1', 0, 1)
    p40_1 = pm.Uniform('p40_1', 0, 1)
    # Deterministic difference
    delta_1 = pm.Deterministic('delta_1', p40_1 - p30_1)
    # Likelihoods
    obs30_1 = pm.Bernoulli('obs30_1', p30_1,
                           observed=df.query("version=='gate_30'")['retention_1'])
    obs40_1 = pm.Bernoulli('obs40_1', p40_1,
                           observed=df.query("version=='gate_40'")['retention_1'])
    # Sampling
    trace_1 = pm.sample(2000, tune=1000, chains=2, random_seed=42, progressbar=False)

# Extract posterior draws
posterior_1 = trace_1.posterior.stack(draws=("chain","draw"))
p30_samps_1 = posterior_1.p30_1.values
p40_samps_1 = posterior_1.p40_1.values
delta_samps_1 = posterior_1.delta_1.values

# Summaries for Day-1
mean_delta1 = delta_samps_1.mean()
ci_delta1 = np.percentile(delta_samps_1, [2.5, 97.5])

print(f"Day-1 retention Δ (gate40 − gate30): {mean_delta1:.3%}")
print(f"95% CI: [{ci_delta1[0]:.3%}, {ci_delta1[1]:.3%}]\n")

# Plot posterior p30 vs p40 (Day-1)
plt.figure()
plt.hist(p30_samps_1, bins=50, density=True, alpha=0.5)
plt.hist(p40_samps_1, bins=50, density=True, alpha=0.5)
plt.xlabel("Retention probability")
plt.ylabel("Density")
plt.title("Posterior of p30 vs p40 (1-day)")
plt.show()

# Plot posterior Δ (Day-1)
plt.figure()
plt.hist(delta_samps_1, bins=50, density=True)

plt.xlabel("Δ = p40 − p30")
plt.ylabel("Density")
plt.title("Posterior of Δ (1-day)")
plt.show()

#### MODEL FOR 7-DAY RETENTION
with pm.Model() as model_day7:
    p30_7 = pm.Uniform('p30_7', 0, 1)
    p40_7 = pm.Uniform('p40_7', 0, 1)
    delta_7 = pm.Deterministic('delta_7', p40_7 - p30_7)
    obs30_7 = pm.Bernoulli('obs30_7', p30_7,
                           observed=df.query("version=='gate_30'")['retention_7'])
    obs40_7 = pm.Bernoulli('obs40_7', p40_7,
                           observed=df.query("version=='gate_40'")['retention_7'])
    trace_7 = pm.sample(2000, tune=1000, chains=2, random_seed=42, progressbar=False)

# Extract posterior draws
posterior_7 = trace_7.posterior.stack(draws=("chain","draw"))
p30_samps_7 = posterior_7.p30_7.values
p40_samps_7 = posterior_7.p40_7.values
delta_samps_7 = posterior_7.delta_7.values

# Summaries for Day-7
mean_delta7 = delta_samps_7.mean()
ci_delta7 = np.percentile(delta_samps_7, [2.5, 97.5])

print(f"Day-7 retention Δ (gate40 − gate30): {mean_delta7:.3%}")
print(f"95% CI: [{ci_delta7[0]:.3%}, {ci_delta7[1]:.3%}]\n")

# Plot posterior p30 vs p40 (Day-7)
plt.figure()
plt.hist(p30_samps_7, bins=50, density=True, alpha=0.5)
plt.hist(p40_samps_7, bins=50, density=True, alpha=0.5)
plt.xlabel("Retention probability")
plt.ylabel("Density")
plt.title("Posterior of p30 vs p40 (7-day)")
plt.show()

# Plot posterior Δ (Day-7)
plt.figure()
plt.hist(delta_samps_7, bins=50, density=True)
plt.xlabel("Δ = p40 − p30")
plt.ylabel("Density")
plt.title("Posterior of Δ (7-day)")
plt.show()