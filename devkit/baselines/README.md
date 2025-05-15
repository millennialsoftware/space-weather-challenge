
# Baseline submissions
Participants will be submitting their trained algorithm with their code/workflow to generation predictions into the competition platform, in the programming language they prefer. Here we provide a Docker container and an example submission in Python, but participants are allowed to submit with other custom containers too. 

## Evaluation script
The `evaluation.py` script provides participants a simple way to assess the performance of the models they submit for the challenge. It employs metrics and evaluation techniques that are aligned with the challenge's objectives.

The toy datasets `ground_truth_toy.json` and `participant_toy.json` serve as simplified, 
example datasets for the challenge. These datasets are intended for initial 
testing and understanding of the simplified evaluation script and the baseline model.

### Example Usage
The `run_evaluator` function is the main entry point of the script and accepts the following arguments:

- `participant`: Path to the participant's CSV file.
- `ground_truth`: Path to the ground truth CSV file.

You can also run the script directly from the command line. For example:
```bash
python evaluation.py --participant=participant.json --ground_truth=ground_truth.json
```
This example assumes you have a `participant.csv` and `ground_truth.csv`. If no arguments are provided, the evaluation will be run for the toy datasets.

### Returns
The `score` function within the file returns the evaluation metrics as per the challenge guidelines (that is, the F2 and the RMSE). 
