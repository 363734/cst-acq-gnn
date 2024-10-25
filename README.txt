

Python version : 3.11.3



To run the experiment: Run the steps file in order
Each step generally generate a list of job configuration first,
then you need to run each other step for each jobs needed
Scripts are SLURM compatible

- Step 1: generate the dataset and stats on it
- Step 2: train the models
- Step 3: evaluate the models
- Step 4: use in constraint acquisition settings