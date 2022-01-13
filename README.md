# xFAIR: Better fairness via model-based rebalancing of protected attributes 


## Overview

- `Data` folder contains all datasets used in this paper.
- `Measure.py` contains functions to compute performance measures and fairness measures.
- `Example` folder contains our running examples of xFAIR on the Adult Income dataset.
- `xFAIR` contains the code of algorithm proposed in our paper.
- `Baseline` folder contains all the baselines used in the paper.  
    - `FairSMOTE` contains [Fair-SMOTE](https://arxiv.org/abs/2105.12195), which rebalances the protected attributes by generating additional
synthetic training data. In Fair-SMOTE, `Generate_Samples.py` is used to generate synthetic data.
    - `Reweighing` contains [Reweighing](https://link.springer.com/content/pdf/10.1007/s10115-011-0463-8.pdf), another benchmark used here,
 which assigns instance weights to training samples.
    - `Random` contains a naive baseline we have implemented in the spirit of procedural justice. It will
    random shuffle the protected attributes in training data, and is expected to reduce bias by obfuscating
    the protected information.

## Dataset Description

1. `Adult`: Adult Income dataset [available online](http://archive.ics.uci.edu/ml/datasets/Adult)

2. `Bank`: Bank Marketing dataset [available online](https://archive.ics.uci.edu/ml/datasets/bank+marketing)

3. `German`: German Credit dataset [available online](https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29)

4. `Default`: Default payment dataset [available online](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)

5. `COMPAS`: ProPublica Recidivism dataset [available online](https://github.com/propublica/compas-analysis)

6. `MEPS`: Medical Expenditure Panel Survey dataset [available online](https://meps.ahrq.gov/mepsweb/)

7. `Heart`: Cleveland Heart Disease dataset [available online](https://archive.ics.uci.edu/ml/datasets/heart+Disease)

To automate the experiment, we have processed the data into *`_processed.csv` files to drop invalid data, unify the name of dependent variables, 
and selected relevant features. To replicate our experiment, please use the processed data.

## Experiment

The file [runexp.py](runexp.py) reproduces the experiments of this paper. This file generates the graphs that let us answer the following
research questions. 

- RQ1: Can we provide interpretations on the cause of bias?

  Run `xFAIR` and set `verbose=True` to enable the presentation of interpretations. We offer either decision
  tree or logistic regression as the interpretaion/extrapolation model. 
  
- RQ2,RQ3: Can we use interpretaions to mitigate bias, and how is xFAIR compared to prior works?

  Run `xFAIR` and other methods in `Baseline` using the same random seeds. Each method returns the performance and
  fairness measures. Note that for `xFAIR` we do not record the Flip Rate since it will always be 0.

- RQ4: Is xFAIR faster than Fair-SMOTE?

   Record the runtime of the two methods on each dataset. In the paper, we repeated the experiment for 20 runs.
