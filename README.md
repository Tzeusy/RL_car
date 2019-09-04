## Description
The OpenAI Car Racing gym environment is a reinforcement learning (RL) task in which a car is trained to navigate a randomly generated racetrack. It is a continuous control image-based task, and its complexity makes it one of the more challenging environments to develop RL agents for.

In our 50.021 Artificial Intelligence project, we investigated differing approaches to creating an effective generalizable model, and an interactive GUI for user interpretation of model and parameter efficacy. Approaches taught during the AI class such as Double Deep Q-Learning, Policy Gradient, Advantage Actor-Critic (A2C), and Imitation Learning are used. We analyze the accompanying results, and build a system for contrasting efficacy of models. This application allows human players to observe in real-time the impact of parameter changes or imitation-based suggested movements.

## Requirements
* Python 3.6+
* Relevant Python packages  
  `pip3 install -r requirements.txt`

## Instructions
### Original
Run `car.py`

- You can use controls during the rendered episodes by using the arrow keys. This will override model-derived suggestions for the car in environment steps. If this is done, if the model-derived suggestion is different from user input, the model receives a negative penalty (similar to imitation learning). If no input is received, model proceeds with learning via Double Deep Q Learning, rewards derived from the OpenAI gym environment. Vary parameters as desired (or scale of rewards to change weightages as necessary)
- Different storage buffers are used for user-input and model-input state changes. This allows the user to, for example, train for only the first few episodes, and have these inputs be stored in memory for the duration of the program running, while the model cycles through its FIFO memory.

### With UI
Run `ui.py`  

Use the UI to set your desired hyperparameters, then hit the 'Start' button.

Like with `car.py`, you can use controls during the rendered episodes using the arrow keys. The first window that appears is the window that captures the keypresses.

## Notes
- When manual controlling, don't hold down forward + left/right at the same time. The car is extremely prone to drifting
  - Unless you want to train the car to drift like in Initial D, then by all means go for it, but you gotta be a pretty consistent drifter to train the car well
- Best strategy I've found so far is to maintain a moderate speed throughout, manages turns a lot more easily
