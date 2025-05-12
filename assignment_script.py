
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

RANDOM_SEED = 123
np.random.seed(RANDOM_SEED)

URL = "https://raw.githubusercontent.com/dustywhite7/Econ8310/master/AssignmentData/cookie_cats.csv"
df = pd.read_csv(URL)

print("── head of data ──")
print(df.head(), "\n")

print("── info() ──")
df.info()
print()

print("── version counts & mean retention ──")
print(df['version'].value_counts(), "\n")
print(df.groupby('version')[['retention_1','retention_7']].mean(), "\n")

# Function to run A/B analysis for any retention column.This returns (model, trace) for gate30 vs gate40 on `ret_co
def ab_model(ret_col, sampler="nuts"):
    A = df[df['version']=="gate_30"][ret_col].values
    B = df[df['version']=="gate_40"][ret_col].values

    with pm.Model() as model:
        #  Priors on true retention probabilities
        p_30 = pm.Uniform("p_30", 0, 1)
        p_40 = pm.Uniform("p_40", 0, 1)

        # Likelihood: each user's 0/1 outcome
        pm.Bernoulli("obs_30", p=p_30, observed=A)
        pm.Bernoulli("obs_40", p=p_40, observed=B)

        # Deterministic difference we care about
        delta = pm.Deterministic("delta", p_40 - p_30)

        # Inference
        if sampler == "nuts":
            trace = pm.sample(
                2000,
                tune=1000,
                chains=2,
                random_seed=RANDOM_SEED,
                return_inferencedata=True
            )
        else:
            step = pm.Metropolis()
            trace = pm.sample(
                2000,
                step=step,
                chains=2,
                random_seed=RANDOM_SEED,
                return_inferencedata=True
            )

    return model, trace

#  Fit both models
model1, trace1 = ab_model("retention_1", sampler="nuts")
model7, trace7 = ab_model("retention_7", sampler="nuts")

# Summarize posteriors
print("── 1-DAY retention posterior summary ──")
az_summary1 = az.summary(
    trace1, var_names=["p_30","p_40","delta"], hdi_prob=0.95
)

print(az_summary1, "\n")
delta1 = trace1.posterior.delta.values.flatten()
print("Pr(Δ₁ > 0) =", np.mean(delta1 > 0).round(3))

print("── 7-DAY retention posterior summary ──")
az_summary7 = az.summary(
    trace7, var_names=["p_30","p_40","delta"], hdi_prob=0.95
)
print(az_summary7, "\n")
delta7 = trace7.posterior.delta.values.flatten()
print("Pr(Δ₇ > 0) =", np.mean(delta7 > 0).round(3))

#  Plotting helper
def plot_posteriors(trace, title_suffix):
    p30 = trace.posterior.p_30.values.flatten()
    p40 = trace.posterior.p_40.values.flatten()
    d   = trace.posterior.delta.values.flatten()

    plt.figure(figsize=(12.5,10))

    ax1 = plt.subplot(311)
    ax1.hist(p30, bins=30, density=True, alpha=0.7)
    ax1.set_title(f"Posterior of $p_{{30}}$ ({title_suffix})")

    ax2 = plt.subplot(312)
    ax2.hist(p40, bins=30, density=True, alpha=0.7)
    ax2.set_title(f"Posterior of $p_{{40}}$ ({title_suffix})")

    ax3 = plt.subplot(313)
    ax3.hist(d, bins=30, density=True, alpha=0.7)

    ax3.set_title(f"Posterior of Δ = $p_{{40}}-p_{{30}}$ ({title_suffix})")

    plt.tight_layout()
    plt.show()

#  Make the plots
az.style.use("arviz-darkgrid")
plot_posteriors(trace1, "1-Day Retention")
plot_posteriors(trace7, "7-Day Retention")

# Extract posterior samples
p30_1 = trace1.posterior.p_30.values.flatten()
p40_1 = trace1.posterior.p_40.values.flatten()
p30_7 = trace7.posterior.p_30.values.flatten()
p40_7 = trace7.posterior.p_40.values.flatten()

# Overlay plot for 1-day retention
plt.figure(figsize=(8, 4))
plt.hist(p30_1, bins=30, density=True, alpha=0.5, label='p_30 (1-day)')
plt.hist(p40_1, bins=30, density=True, alpha=0.5, label='p_40 (1-day)')
plt.title('Posterior Distributions: Gate 30 vs Gate 40 (1-Day Retention)')
plt.xlabel('Retention Probability')
plt.ylabel('Density')
plt.legend()
plt.show()

# Overlay plot for 7-day retention
plt.figure(figsize=(8, 4))
plt.hist(p30_7, bins=30, density=True, alpha=0.5, label='p_30 (7-day)')
plt.hist(p40_7, bins=30, density=True, alpha=0.5, label='p_40 (7-day)')
plt.title('Posterior Distributions: Gate 30 vs Gate 40 (7-Day Retention)')
plt.xlabel('Retention Probability')
plt.ylabel('Density')
plt.legend()
plt.show()
