# Reinforcement learning in an adaptive GridWorld Environment
This study compares the model-free algorithm Q-Learning with the model-based algorithm Dyna-Q and variants thereof in a dynamic GridWorld environment. A novel adaptation of Dyna-Q incorporating a `forgetfulness' mechanism is proposed to minimise reliance on out-of-date model information whilst enhancing stability in settings with shifting high penalty states. Results reveal a trade-off between exploration and stability, with the proposed method achieving a better balance than competing algorithms.

## Dependencies
To run code in this repository, run `pip install -r requirements.txt` in terminal. This code is built using python version 3.11.11.

## Files
- `agents.py`: Sets up Q-learning, Dyna-Q, and Dyna-Q + agents.
- `environment.py`: Sets up adaptive GridWorld environment with mechanism for changing environment.
- `simulator.py`: Sets up actual simulation mechanics, metric evaluation and plotting.
- `experiments.ipynb`: Notebook of experiments and comparisons of methods.

## Output
- `images` folder shows the images saved in `experiments.ipynb`.
- `results.csv` contains the macro-replication results.
