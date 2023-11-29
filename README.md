# covid_and_stat_mech
Creating a situational awareness model for SARS-CoV-2 transmission in Washington state (January 2020 to March 2021).

This is a repository of Python 3.8 code associated with the preprint [*COVID-19 epidemiology as emergent behavior on a dynamic transmission forest*](https://arxiv.org/abs/2205.02150), 2022. In the paper, we create a stochastic process model of SARS-CoV-2 transmission, and we use it to estimate a variety of transmission statistics, from population prevalence to the clustering of cases as outbreaks. 

The main scripts are:
1. `PathogenesisGPRVis.py`, which makes the paper's first and second figures.
2. `PrevalenceModel.py`, which makes the paper's third figure and a serialized pandas dataframe used as input in subsequent scripts.
3. `ContactDistributionsVis.py`, which generates the paper's fourth figure.
4. `IndividualLevelDistributions.py`, which generates the full set of daily contact distributions and serializes some output for use in subsequent scripts.
5. `OutbreakInvestigation.py`, which makes Figure 5a from the paper.
6. `TransmissionForestVis.py`, which makes Figure 5b from the paper.
7. `ForestStats.py`, which makes Figure 5c from the paper.

## November 2023 update
We've added to this repository additional code associated with our follow-up preprint [*A generating function perspective on the transmission forest*](https://arxiv.org/abs/2311.16317), 2023. In that paper, we use the Washington state model above as an example to illustrate the application of a formal, generating function-based approach to transmission forest calculations.

The main scripts for this second paper are
1. `generating_functions\SampleTransmissionTrees.py`, which creates a large serialized set of sample trees for method validation.
2. `generating_functions\SamplingVis.py`, which makes the paper's first figure.
3. `generating_functions\ConstellationDists.py`, which makes the paper's second figure.
4. `generating_functions\FormalimVsSamples.py`, which makes the paper's third figure.


## building the Python environment
The Python environment is managed through [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) and described in `environment.yml`.  Within an Anaconda Prompt, type `conda env create python=3.8 -f environment.yml`. This will create an environment named `covidx`. 

If your conda installation is not at the default path or you'd prefer to create it elsewhere, change the `prefix` path in `environment.yml` before running the command above.

If you run into errors with the `Qt` platform plugin when trying to run scripts from the `covidx` environment, such as,

```
qt.qpa.plugin: Could not find the Qt platform plugin "windows" in ""
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.
```
here's the solution that worked for [me](https://github.com/famulare), based on [this issue reply](https://github.com/ContinuumIO/anaconda-issues/issues/1270#issuecomment-287607288):

Copy the content of `[Anaconda directory]\envs\covidx\Library\plugins\platforms` to `[Anaconda directory]\envs\covidx\platforms`. 
