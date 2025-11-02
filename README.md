hello-py
===

Setup instructions:

1. Clone the repository:
   ```
   git clone https://github.com/preferencemodel/hello-py.git
   ```

2. Navigate to the project directory:
   ```
   cd hello-py
   ```

3. Set up `ANTHROPIC_API_KEY` environment variable:
   ```
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

4. Run the agent:
   ```
   uv run main.py
   ```

## Execution Modes

The test suite supports both concurrent and sequential execution. 

To change modes, edit the `concurrent` parameter at the bottom of `main.py`:

```python
asyncio.run(main(concurrent=True))
asyncio.run(main(concurrent=False))
```

When running concurrently, results print as they complete (not in run order) for faster overall execution.

RL Task Solution Overview
===

### Background and Literature

This solution draws on recent advances in reasoning LLMs, particularly the RL training approaches used in models like DeepSeek. The key observation is that RL training works well for tasks with objectively measurable outcomes and multiple valid solution paths—which explains why reasoning models are typically trained on math and coding problems where answers can be verified automatically and large datasets are available.

For this work, I adapted a beginner-level machine learning competition from Kaggle: [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic/overview). This is a straightforward binary classification task where the goal is to predict outcomes based on provided features. The standard Kaggle workflow involves receiving training and test datasets, then submitting predictions in a `submission.csv` file with the required format.

### Implementation

The RL task mirrors the Kaggle competition structure. I split the original [train_data.csv](https://www.kaggle.com/competitions/spaceship-titanic/data) into training (85%) and test (15%) subsets while preserving the class distribution of the binary target variable (`Transported`), using `./data/get_splits.py`.

**Evaluation:**
The LLM receives the test subset without labels and must generate predictions. These are evaluated against the ground truth labels using F1 score as the reward signal. The grading logic is implemented in `./submission_grader.py`.

**Environment:**
The LLM operates in an isolated Docker container (`./docker_shell.py`) that provides shell access and a persistent Python session. 

The environment includes (`Dockerfile`):
- `train.csv` - labeled training data
- `test.csv` - unlabeled test data for predictions
- `sample_submission.csv` - template showing the required submission format

The Python environment is intentionally restricted to:
- `numpy`
- `pandas`
- `scikit-learn`

**Agent Capabilities:**
The LLM has three available tools:
1. `run_python_code(code: str)` - executes multiline Python code with persistent variables across calls
2. `exec(command: str)` - runs shell commands
3. `submit_answer_tool(answer: str)` - submits predictions via `submission.csv`

**Task Objective:**
The goal is to achieve an F1 score ≥ 0.81 on the held-out test set. This threshold was chosen based on the [competition leaderboard](https://www.kaggle.com/competitions/spaceship-titanic/leaderboard) and represents a challenging but achievable target (roughly top 50 performance).

The complete task specification, including restrictions and guidelines, is defined in the system prompt (`./sys_prompt.py`). While the model is instructed to complete the task in 7 steps, there's a hard limit of 10 steps enforced in `./main.py` to allow for solution iteration and refinement.

### Experiments and Results

I ran 10 independent trials, and the LLM successfully passed the threshold in 3 out of 10 attempts:

```bash 
============================================================
Test Results:
  Passed: 3/10
  Failed: 7/10
  Pass Rate: 30.0%
============================================================
```

**Emergent Solution Pattern:**
Interestingly, all runs converged on essentially the same core approach without explicit guidance. Every run extracted features from cabin information (deck, number, side) and passenger groups, created spending aggregations, handled missing values with mode/median imputation, and applied label encoding to categorical variables. All runs exclusively used tree-based ensemble methods (RandomForest and GradientBoosting) with similar hyperparameter ranges. This consistency suggests the model reliably identifies effective feature engineering strategies and appropriate algorithms for the task.

**What Made the Difference:**
The successful runs shared three distinguishing techniques. First, they trained multiple models with different random seeds, creating ensembles of 4-6 models rather than just 1-2. Second, they optimized the prediction threshold through grid search instead of using the default 0.5. Third, they created interaction features by multiplying age with spending variables. The most impactful discovery was threshold tuning—this alone could push F1 scores from around 0.80 to above 0.81, but it required going beyond the basic modeling pipeline to discover.

**Why Runs Failed:**
Failed runs typically stuck with single models or small 2-3 model ensembles, used the default prediction threshold without tuning, and only created additive features rather than multiplicative interactions. Several runs also showed signs of overfitting, achieving strong validation scores that didn't transfer to test performance. The fact that failed runs consistently plateaued around 0.80 F1 suggests the model reliably reaches a good baseline with standard techniques, but needs to discover more advanced strategies—particularly ensemble diversity and threshold optimization—to cross the target threshold.

The complete log from the test run is available in `./run_log.txt`.