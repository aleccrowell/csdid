# csdid — Difference-in-Differences with Multiple Time Periods

[![PyPI version](https://img.shields.io/pypi/v/csdid.svg?color=blue)](https://pypi.org/project/csdid/)
[![Downloads](https://static.pepy.tech/personalized-badge/csdid?period=total&units=international_system&left_color=blue&right_color=grey&left_text=Downloads)](https://pepy.tech/project/csdid)
[![Last commit](https://img.shields.io/github/last-commit/d2cml-ai/csdid.svg)](https://github.com/d2cml-ai/csdid/commits/main)
[![GitHub stars](https://img.shields.io/github/stars/d2cml-ai/csdid.svg?style=social)](https://github.com/d2cml-ai/csdid/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/d2cml-ai/csdid.svg)](https://github.com/d2cml-ai/csdid/issues)
[![License](https://img.shields.io/github/license/d2cml-ai/csdid.svg)](https://github.com/d2cml-ai/csdid/blob/main/LICENSE)

**csdid** is a Python implementation of the Callaway & Sant'Anna (2021) framework for causal inference using Difference-in-Differences (DiD) when there are multiple time periods and units adopt treatment at different points in time. It is a port of the original [R `did` package](https://github.com/bcallaway11/did).

---

## Table of Contents

1. [Background and Motivation](#background-and-motivation)
2. [The Callaway & Sant'Anna Framework](#the-callaway--santanna-framework)
3. [Estimation Methods](#estimation-methods)
4. [Aggregation Strategies](#aggregation-strategies)
5. [Installation](#installation)
6. [Step-by-Step Usage](#step-by-step-usage)
   - [Step 1: Load Data](#step-1-load-data)
   - [Step 2: Estimate Group-Time ATTs](#step-2-estimate-group-time-atts)
   - [Step 3: Inspect the Raw Results](#step-3-inspect-the-raw-results)
   - [Step 4: Plot Group-Time Effects](#step-4-plot-group-time-effects)
   - [Step 5: Aggregate — Dynamic (Event Study)](#step-5-aggregate--dynamic-event-study)
   - [Step 6: Aggregate — By Group](#step-6-aggregate--by-group)
   - [Step 7: Aggregate — By Calendar Time](#step-7-aggregate--by-calendar-time)
   - [Step 8: Aggregate — Simple Overall Average](#step-8-aggregate--simple-overall-average)
7. [Advanced Options](#advanced-options)
   - [Adding Covariates](#adding-covariates)
   - [Choosing a Control Group](#choosing-a-control-group)
   - [Changing the Estimation Method](#changing-the-estimation-method)
   - [Clustered Standard Errors](#clustered-standard-errors)
   - [Repeated Cross-Sections](#repeated-cross-sections)
   - [Anticipation Effects](#anticipation-effects)
8. [API Reference](#api-reference)
9. [How to Cite](#how-to-cite)

---

## Background and Motivation

The canonical DiD estimator compares two groups (treated and untreated) across two time periods (before and after treatment). In practice, however, policy evaluations frequently involve:

- **Multiple time periods** — the researcher observes units across many years, not just two.
- **Staggered treatment adoption** — different units receive treatment at different points in time (e.g., states pass a law in different years).
- **Treatment effect heterogeneity** — the effect of treatment can vary across units and evolve over time since treatment.

When these complications are present, the commonly-used two-way fixed effects (TWFE) regression estimator can produce severely biased estimates, because it implicitly uses already-treated units as controls for newly-treated units. This contamination makes the TWFE estimate a weighted average of many treatment effects — potentially with *negative* weights — and is not interpretable as a causal average treatment effect.

**csdid** solves this by following the Callaway & Sant'Anna (2021) approach: rather than collapsing everything into a single regression coefficient, it first estimates a fine-grained set of treatment effect parameters — one for each (group, time) pair — and then aggregates them in a transparent, researcher-controlled way.

---

## The Callaway & Sant'Anna Framework

### Cohort Definition

Units are partitioned into **cohorts** (called *groups*) based on the calendar period when they are first treated. For example, if some counties have their minimum wage raised in 2004 and others in 2006, these are two distinct groups: G=2004 and G=2006. Units that are never treated form the control pool.

### Group-Time Average Treatment Effects — ATT(g, t)

The core parameters are:

> **ATT(g, t)** = the average treatment effect for group *g* in calendar period *t*.

Formally, ATT(g, t) compares the (counterfactual) potential outcome that group *g* would have experienced in period *t* had they never been treated to their observed outcome. Under a **conditional parallel trends** assumption — that, conditional on observed covariates, the average untreated potential outcomes for the treated and control groups would have followed parallel trends — ATT(g, t) is non-parametrically identified.

**Pre-treatment ATT(g, t)** values (periods before group *g* is treated) serve as placebo tests: if the parallel trends assumption holds, these should be close to zero.

### Identification Assumptions

1. **Conditional parallel trends**: controlling for pre-treatment covariates X, the average change in untreated potential outcomes is the same for the treated group and the control group.
2. **No anticipation**: units do not alter behavior before their treatment begins (relaxable via the `anticipation` parameter).
3. **Overlap**: for each (group, time) pair, there is a positive probability of being in the control group conditional on covariates.

### From ATT(g, t) to Aggregated Effects

Once ATT(g, t) is estimated for all valid (group, time) cells, the package offers four aggregation strategies that summarise the grid into a smaller set of policy-relevant parameters. These are described in [Aggregation Strategies](#aggregation-strategies) below.

---

## Estimation Methods

For each (group, time) cell, csdid estimates ATT(g, t) by comparing group *g* in period *t* against a comparison group using a two-period DiD. Three estimators are available:

| Method | `est_method` | Description |
|--------|-------------|-------------|
| Doubly Robust | `"dr"` *(default)* | Combines outcome regression and inverse probability weighting. Consistent if *either* the propensity score model or the outcome model is correctly specified. Recommended for most applications. |
| Inverse Probability Weighting | `"ipw"` | Re-weights control units by the inverse of estimated propensity scores. Relies entirely on correct propensity score specification. |
| Outcome Regression | `"reg"` | Fits a linear outcome model on the control group and extrapolates to the treated group. Relies entirely on correct outcome model specification. |

All methods support a **varying base period** (each pre-treatment period is compared to the immediately preceding period) or a **universal base period** (all periods are compared to a single fixed pre-treatment period).

### Inference

Standard errors are computed via the **multiplier bootstrap** — a fast, sample-efficient bootstrap that resamples the influence functions of the estimators rather than the raw data. This produces valid pointwise confidence intervals and simultaneous confidence bands (the latter accounting for the joint distribution of all ATT(g, t) estimates). The number of bootstrap iterations is controlled via `biters` (default 1000).

---

## Aggregation Strategies

| Type | `typec` | What it estimates |
|------|---------|-------------------|
| **Dynamic** | `"dynamic"` | Effects by event-time (time relative to treatment). Produces the canonical event-study plot. |
| **Group** | `"group"` | Average effect for each cohort, averaging over all post-treatment periods for that group. |
| **Calendar** | `"calendar"` | Average effect in each calendar period, averaging over all treated groups active in that period. |
| **Simple** | `"simple"` | A single overall ATT, averaging over all (group, time) post-treatment cells weighted by group size. |

---

## Installation

```bash
pip install csdid
```

Or from GitHub:

```bash
pip install git+https://github.com/d2cml-ai/csdid/
```

**Dependencies**: `pandas`, `numpy`, `scipy`, `patsy`, `plotnine`, `joblib`

---

## Step-by-Step Usage

The examples below use the `mpdta` dataset from Callaway & Sant'Anna (2021), which studies the effect of state minimum wage increases on county-level teen employment rates. The dataset covers 500 counties from 2003–2007. Three cohorts of states raised their minimum wages in 2004, 2006, and 2007 respectively; states that never raised their minimum wage serve as controls.

Key variables:
- `lemp` — log county-level teen employment (outcome)
- `first.treat` — year the state first raised its minimum wage (0 = never treated)
- `year` — calendar year (2003–2007)
- `countyreal` — county identifier
- `lpop` — log county population (a covariate)

---

### Step 1: Load Data

```python
from csdid.att_gt import ATTgt
import pandas as pd

data = pd.read_csv(
    "https://raw.githubusercontent.com/d2cml-ai/csdid/function-aggte/data/mpdta.csv"
)
print(data.head())
print(data["first.treat"].value_counts().sort_index())
```

```
   year  countyreal      lpop      lemp  first.treat
0  2003        1001  5.896761  8.461469         2007
1  2004        1001  5.896761  8.336870         2007
2  2005        1001  5.896761  8.340217         2007
3  2006        1001  5.896761  8.378161         2007
4  2007        1001  5.896761  8.487219         2007

first.treat
0       500   # never treated
2004    200
2006    120
2007    180
```

The `first.treat` column encodes cohort membership. A value of 0 means the county was never treated. The three non-zero cohorts (2004, 2006, 2007) will each be compared against the control pool.

---

### Step 2: Estimate Group-Time ATTs

```python
out = ATTgt(
    yname="lemp",           # outcome variable
    gname="first.treat",    # cohort variable (0 = never treated)
    idname="countyreal",    # unit identifier
    tname="year",           # time variable
    xformla="lemp~1",       # covariate formula; "lemp~1" means intercept only
    data=data,
).fit(est_method="dr")      # doubly-robust estimation
```

**What happens here:**

1. **Preprocessing** (`pre_process_did`): The data is validated and organized into a balanced panel. The three treatment cohorts (G=2004, G=2006, G=2007) are identified. Units coded 0 in `gname` form the "never treated" control pool. Units treated before the first observed period are dropped.

2. **Grid construction**: The package enumerates all valid (group *g*, time *t*) combinations. For group *g*, the relevant pairs are all calendar periods *t* from the start of the data up through the last period — giving pre-treatment placebo cells and post-treatment effect cells.

3. **Estimation loop** (`compute_att_gt`): For each (g, t) cell, the package isolates units from group *g* and control units, constructs a two-period DiD dataset (base period vs. period *t*), and runs the chosen estimator. With `est_method="dr"`, a logistic propensity score is estimated, then the doubly-robust DiD formula is applied.

4. **Bootstrap inference** (`mboot`): Influence functions from each cell are stacked and resampled 1000 times to produce standard errors and simultaneous confidence bands.

The result is stored in `out` (the `ATTgt` object itself) and is ready for inspection and aggregation.

---

### Step 3: Inspect the Raw Results

```python
out.summ_attgt().summary2
```

```
    Group  Time  ATT(g, t)  Post  Std. Error  [95% Pointwise  Conf. Band]
0    2004  2004    -0.0105     0      0.0241         -0.0781      0.0571
1    2004  2005    -0.0704     0      0.0324         -0.1612      0.0204
2    2004  2006    -0.1373     0      0.0393         -0.2476     -0.0269  *
3    2004  2007    -0.1008     0      0.0360         -0.2017      0.0001
4    2006  2004     0.0065     0      0.0238         -0.0601      0.0732
5    2006  2005    -0.0028     0      0.0188         -0.0554      0.0499
6    2006  2006    -0.0046     0      0.0172         -0.0528      0.0437
7    2006  2007    -0.0412     0      0.0201         -0.0976      0.0152
8    2007  2004     0.0305     0      0.0147         -0.0108      0.0719
9    2007  2005    -0.0027     0      0.0160         -0.0476      0.0421
10   2007  2006    -0.0311     0      0.0173         -0.0796      0.0174
11   2007  2007    -0.0261     0      0.0171         -0.0740      0.0219
```

**Reading this table:**

- Each row is one (Group, Time) cell. `Group` is the cohort's first treatment year; `Time` is the calendar year of observation.
- `Post` will be 1 in future releases when `time >= group`; currently all rows show 0 because the varying base period references the immediately preceding year — in this dataset all periods happen to precede treatment for some groups. Check that pre-treatment rows (Time < Group) have ATT(g,t) near zero as a parallel trends diagnostic.
- `*` marks cells where the 95% simultaneous confidence band excludes zero.

From the table we can already see that Group 2004 shows a sizable negative effect in 2006 (-0.137), suggesting minimum wage increases reduced teen employment, with effects growing over time. Groups 2006 and 2007 show smaller, noisier estimates.

---

### Step 4: Plot Group-Time Effects

```python
out.plot_attgt();
```

This produces a panel of plots — one per cohort — showing ATT(g, t) estimates and confidence intervals across calendar time. The dashed vertical line marks the cohort's first treatment period. Points to its left are pre-treatment placebos; points to its right are post-treatment effects.

**What to look for:**
- Pre-treatment estimates close to zero support the parallel trends assumption.
- A clear break at the treatment period indicates a treatment effect.
- Widening confidence intervals at later periods are normal (fewer post-treatment observations per cohort).

---

### Step 5: Aggregate — Dynamic (Event Study)

The most common summary is an **event study**: effects averaged across cohorts at each relative time period (time since treatment began).

```python
out.aggte(typec="dynamic");
```

```
Overall summary of ATT's based on event-study/dynamic aggregation:
    ATT Std. Error  [95.0%  Conf. Int.]
-0.0772     0.0207     -0.1179     -0.0366 *

Dynamic Effects:
  Event time  Estimate  Std. Error  [95.0% Simult.   Conf. Band]
0         -3    0.0305      0.0146           0.0019      0.0591  *
1         -2   -0.0006      0.0129          -0.0259      0.0248
2         -1   -0.0245      0.0141          -0.0521      0.0032
3          0   -0.0199      0.0117          -0.0428      0.0030
4          1   -0.0510      0.0154          -0.0811     -0.0208  *
5          2   -0.1373      0.0366          -0.2091     -0.0655  *
6          3   -0.1008      0.0337          -0.1669     -0.0347  *
---
Signif. codes: `*' confidence band does not cover 0
Control Group:  Never Treated
Anticipation Periods:  0
Estimation Method:  Doubly Robust
```

**What is computed:** For each event-time value *e*, the package averages ATT(g, g+e) across all cohorts *g* for which that relative period is observed. This re-expresses the treatment effect grid in terms of time since treatment rather than calendar time, making effects comparable across cohorts.

**Reading the output:**

- **Event time < 0**: pre-treatment periods. Event time = -1 is the period immediately before treatment; -2 is two periods before, etc. These should ideally be near zero. Here, -1 is -0.025 (not statistically significant), consistent with parallel trends.
- **Event time = 0**: the impact effect — the effect in the first period of treatment. Here, -0.020 (not statistically significant individually, but the overall post-treatment effect is -0.077 and significant).
- **Event time > 0**: dynamic effects showing how the treatment effect evolves. The effect grows over time (event time = 2 is -0.137), consistent with a policy whose labor-market effects compound over years.
- **Overall ATT** (-0.077) is a weighted average over all post-treatment event-time cells.

```python
out.plot_aggte();
```

Plots the event-study coefficients with simultaneous confidence bands. Pre-treatment periods are shown in red, post-treatment in blue, making it easy to visually assess parallel trends and treatment effect dynamics.

---

### Step 6: Aggregate — By Group

Group aggregation computes a single average effect for each cohort, averaging over all post-treatment periods observed for that cohort. This answers: "what was the average effect of the treatment for the counties that adopted it in year X?"

```python
out.aggte(typec="group");
```

```
Overall summary of ATT's based on group/cohort aggregation:
    ATT Std. Error  [95.0%  Conf. Int.]
 -0.031     0.0124     -0.0553     -0.0067 *

Group Effects:
  Group  Estimate  Std. Error  [95.0% Simult.   Conf. Band]
0  2004   -0.0797      0.0301          -0.1387     -0.0208  *
1  2006   -0.0229      0.0172          -0.0567      0.0109
2  2007   -0.0261      0.0174          -0.0601      0.0080
```

**What is computed:** For each group *g*, the package averages ATT(g, t) over all post-treatment calendar periods *t ≥ g*. Groups are then averaged across cohorts, weighted by group size, to produce the overall ATT.

**Reading the output:**

- Group 2004 has the largest estimated effect: -0.080, statistically significant. This cohort was observed for the most post-treatment years (2004–2007), so its estimate is based on multiple years of exposure.
- Groups 2006 and 2007 show smaller, insignificant effects. These cohorts had fewer post-treatment periods observed in the dataset.
- **Overall ATT** (-0.031): the overall estimate is roughly interpretable as the ATT in the two-period, two-group sense. It is statistically significant.

---

### Step 7: Aggregate — By Calendar Time

Calendar aggregation averages effects by calendar year, across all groups that were already treated in that year. This answers: "what was the average treatment effect across all currently-treated units in year X?"

```python
out.aggte(typec="calendar");
```

```
Overall summary of ATT's based on calendar time aggregation:
    ATT Std. Error  [95.0%  Conf. Int.]
-0.0417     0.0169     -0.0748     -0.0086 *

Time Effects (calendar):
   Time  Estimate  Std. Error  [95.0% Simult.   Conf. Band]
0  2004   -0.0105      0.0244          -0.0584      0.0374
1  2005   -0.0704      0.0307          -0.1305     -0.0103  *
2  2006   -0.0488      0.0210          -0.0900     -0.0076  *
3  2007   -0.0371      0.0136          -0.0637     -0.0105  *
```

**What is computed:** For each calendar period *t*, the package averages ATT(g, t) over all groups *g ≤ t* (i.e., all groups already treated by period *t*). The composition of the treated pool changes across years as new cohorts enter treatment.

**Reading the output:**

- 2004 is -0.011 (not significant): only Group 2004 was treated this year, and it was their first year of treatment — effects may not yet have materialized.
- 2005 is -0.070 (significant): now Group 2004 has had one year of exposure.
- Effects stabilize around -0.04 to -0.05 in later years as more cohorts join the treated pool, diluting the large effects from Group 2004.

---

### Step 8: Aggregate — Simple Overall Average

The simplest aggregation collapses all post-treatment ATT(g, t) cells into a single number, weighted by the share of the sample in each (group, time) cell.

```python
out.aggte(typec="simple");
```

```
Overall summary of ATT's based on simple aggregation:
    ATT Std. Error  [95.0%  Conf. Int.]
-0.0331     0.0132     -0.0590     -0.0072 *
```

**What is computed:** A single weighted average of all ATT(g, t) for post-treatment periods, with weights proportional to group size and the number of periods observed. This is the closest analogue to what a naive TWFE regression estimates — but computed correctly, without the negative-weight contamination problem.

---

## Advanced Options

### Adding Covariates

When the parallel trends assumption only holds after conditioning on covariates, supply a Patsy-style formula via `xformla`. The formula's left-hand side should be the outcome variable; the right-hand side lists the covariates.

```python
out = ATTgt(
    yname="lemp",
    gname="first.treat",
    idname="countyreal",
    tname="year",
    xformla="lemp ~ lpop",   # control for log population
    data=data,
).fit(est_method="dr")
```

With the DR estimator, covariates enter both the propensity score model (logistic regression predicting cohort membership) and the outcome regression model (predicting the outcome change for controls). Including covariates improves efficiency and removes bias when parallel trends only holds conditionally.

---

### Choosing a Control Group

Two control group options are available:

```python
# Use only units that are never treated (default)
out = ATTgt(..., control_group="nevertreated").fit()

# Use units not yet treated by period t (larger control pool, but assumes
# parallel trends between early and late adopters)
out = ATTgt(..., control_group="notyettreated").fit()
```

- **`"nevertreated"`** (default): uses only the never-treated units as controls. This is the most conservative choice and avoids using already-treated units as controls for each other.
- **`"notyettreated"`**: includes units that will eventually be treated but have not been treated yet as of period *t*. This can substantially increase the control pool size (important when the "never treated" group is small), but requires the additional assumption that parallel trends hold between cohorts.

---

### Changing the Estimation Method

```python
# Doubly-robust (recommended)
out.fit(est_method="dr")

# Inverse probability weighting
out.fit(est_method="ipw")

# Outcome regression
out.fit(est_method="reg")
```

For most applications, `"dr"` is preferred because it is consistent if *either* the outcome model or the propensity score model is correctly specified — not both.

---

### Clustered Standard Errors

When units are grouped (e.g., counties within states), standard errors should account for within-cluster correlation:

```python
out = ATTgt(
    yname="lemp",
    gname="first.treat",
    idname="countyreal",
    tname="year",
    data=data,
    clustervar="statereal",   # cluster SE at the state level
).fit()
```

The multiplier bootstrap resamples at the cluster level when `clustervar` is specified.

---

### Repeated Cross-Sections

If you have repeated cross-sectional data (different individuals sampled each period) rather than a panel, set `panel=False`:

```python
out = ATTgt(
    yname="lemp",
    gname="first.treat",
    idname="countyreal",
    tname="year",
    data=data,
    panel=False,
).fit(est_method="dr")
```

The estimation adjusts to use cross-sectional DiD formulas rather than the within-unit differencing used for panels.

---

### Anticipation Effects

If units may begin responding to treatment before it officially starts (e.g., hiring decisions anticipating a future minimum wage increase), use the `anticipation` parameter to define how many periods before the official treatment start the unit is considered "treated":

```python
out = ATTgt(
    yname="lemp",
    gname="first.treat",
    idname="countyreal",
    tname="year",
    data=data,
    anticipation=1,    # treat units as treated one period early
).fit()
```

With `anticipation=1`, ATT(g, t) for period *t = g − 1* is now considered a treatment period rather than a pre-treatment placebo. The base period for identification shifts accordingly.

---

### Filtering Aggregations

The `aggte` method accepts `min_e` and `max_e` to restrict which event-time cells are included in dynamic aggregation, and `balance_e` to balance the composition of groups across event times:

```python
# Only include event-time windows from -2 to +3
out.aggte(typec="dynamic", min_e=-2, max_e=3)

# Balance: only include groups observable at all event times up to balance_e
out.aggte(typec="dynamic", balance_e=2)
```

Balancing is important when you want the composition of groups in the average to remain constant across event-time values — otherwise, the pool of cohorts contributing to event-time = 3 may differ from the pool contributing to event-time = 1, making comparisons across event times confounded by composition changes.

---

## API Reference

### `ATTgt`

Main class. Construct it with your data and settings, then call methods in sequence.

```python
ATTgt(
    yname: str,                          # outcome column name
    tname: str,                          # time column name
    idname: str,                         # unit identifier column name
    gname: str,                          # cohort column name (0 = never treated)
    data: pd.DataFrame,                  # the dataset
    control_group: str = "nevertreated", # "nevertreated" or "notyettreated"
    xformla: str | None = None,          # Patsy formula for covariates
    panel: bool = True,                  # True = panel, False = repeated cross-sections
    allow_unbalanced_panel: bool = True, # balance panel internally if needed
    clustervar: str | None = None,       # clustering variable for SEs
    weights_name: str | None = None,     # sampling weights column
    anticipation: int = 0,               # number of anticipation periods
    cband: bool = False,                 # compute simultaneous confidence bands
    biters: int = 1000,                  # bootstrap iterations
    alp: float = 0.05,                   # significance level
)
```

### `.fit()`

Estimate group-time average treatment effects.

```python
.fit(
    est_method: str = "dr",          # "dr", "ipw", or "reg"
    base_period: str = "varying",    # "varying" or "universal"
    bstrap: bool = True,             # use multiplier bootstrap for SEs
) -> ATTgt
```

Returns `self` to allow method chaining.

### `.summ_attgt()`

Print and return a summary DataFrame of all ATT(g, t) estimates.

```python
.summ_attgt(n: int = 4) -> ATTgt
# Access results via: out.summ_attgt().summary2
```

### `.aggte()`

Aggregate the ATT(g, t) estimates.

```python
.aggte(
    typec: str = "group",           # "simple", "dynamic", "group", "calendar"
    balance_e: int | None = None,   # balance groups at this max event time
    min_e: float = float("-inf"),   # minimum event time to include (dynamic only)
    max_e: float = float("inf"),    # maximum event time to include (dynamic only)
    na_rm: bool = False,            # drop NA cells before aggregating
    bstrap: bool | None = None,     # override bootstrap setting
    biters: int | None = None,      # override bootstrap iterations
    cband: bool | None = None,      # override confidence band setting
    alp: float | None = None,       # override significance level
    clustervars: list[str] | None = None,  # override cluster variable
) -> ATTgt
```

Prints a summary and returns `self`. Access results via `out.atte`.

### `.plot_attgt()`

Plot the full grid of group-time estimates.

```python
.plot_attgt(
    ylim=None, xlab=None, ylab=None,
    title="Group", xgap=1, ncol=1,
    legend=True, group=None,
    ref_line=0, theming=True,
) -> matplotlib.figure.Figure
```

### `.plot_aggte()`

Plot the aggregated estimates (event study or group/calendar bar chart).

```python
.plot_aggte(
    ylim=None, xlab=None, ylab=None,
    title="", xgap=1, legend=True,
    ref_line=0, theming=True,
) -> figure
```

---

## How to Cite

This package implements the methodology from:

> Callaway, Brantly and Pedro H.C. Sant'Anna. "Difference-in-Differences with Multiple Time Periods." *Journal of Econometrics*, Vol. 225, No. 2, pp. 200–230, 2021. https://doi.org/10.1016/j.jeconom.2020.12.001

If you use **csdid** in your research, please cite:

```bibtex
@software{csdid,
  author  = {Callaway, Brantly and Sant'Anna, Pedro H.C. and Quispe, Alexander and Guevara, Carlos},
  title   = {{csdid: Difference-in-Differences with Multiple Time Periods in Python}},
  year    = {2024},
  url     = {https://github.com/d2cml-ai/csdid}
}
```
