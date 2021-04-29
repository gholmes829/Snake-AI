# Snake-AI
Train, manage, and control your own AI snakes playing an incredibly classic game! The training uses genetic algorithms and employs multiprocessing to speed up the process. Read below to learn about the different types of AI!

## Get started:
* Download Python 3.8+ `https://www.python.org/downloads/`
* Clone repository `git clone https://github.com/gholmes829/Snake-AI.git`
* Install dependencies `python -m pip install -r requirements.txt`
* Run with `python __main__.py`

## Command Line Arguments:
* Classic player control `python __main__.py -player`
* Select an AI to play `python __main__.py -ai`
* Watch a saved game `python __main__.py -saved`
* Train AI snakes `python __main__.py -train`

## Types of AI:
* Neural Network, uses a deep neural network to output left, straight, or right
* Pathfinding, uses an A* algorithm with backup floodfill to find direct paths to food
* Floodfill, moves towards food but always tries to move in direction resulting in most open spaces
* Cycle, calculates hamiltonian cycle approximation for path
* Multi, specialized neural network that chooses between pathfinding, cycle, and neural network as game progresses
* Hierarchical, specialized neural network chooses between three different neural networks as game progresses

## Notes:
* Change settings in `core\settings.py` including map size, target frame rate, default architectures and more
* Cycle AI can consistently win game if map size is square, even, and snake spawns in top row horizontally
* Hierarchical is the best overall AI in terms of both performance and consistency
* My personal best score is like 25 or something which is why I needed to make the AI
