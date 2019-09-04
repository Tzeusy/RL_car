# Description

The OpenAI Car Racing gym environment is a reinforcement learning (RL) task in which a car is trained to navigate a randomly generated racetrack. It is a continuous control image-based task, and its complexity makes it one of the more challenging environments to develop RL agents for.

In our 50.021 Artificial Intelligence project, we investigated differing approaches to creating an effective generalizable model, and an interactive GUI for user interpretation of model and parameter efficacy. Approaches taught during the AI class such as Double Deep Q-Learning, Policy Gradient, Advantage Actor-Critic (A2C), and Imitation Learning are used. We analyze the accompanying results, and build a system for contrasting efficacy of models.

Our main results found that in such a complex continuous environment, human input as a prior for the model is essential in speeding up learning. This application allows human players to observe in real-time the impact of parameter changes or imitation-based suggested movements.

# User Input

The user is able to use controls during the rendered episodes by using the arrow keys. This will override model-derived suggestions for the car in environment steps. If this is done, if the model-derived suggestion is different from user input, the model receives a negative penalty (similar to imitation learning). If no input is received, model proceeds with learning via Double Deep Q-Learning, rewards derived from the OpenAI gym environment. In order to input suggestions to the model, click on the black window corresponding to the player control car. While this window is in focus, arrow key presses will indicate suggestions to the car to perform a particular action (steer left/right, accelerate, or brake).

Within the initial UI presented, the user can adjust the hyperparameters for training. The available hyperparameters are:

- LR (learning rate): The learning rate to be used for model optimization. A higher learning rate may allow models to learn faster, but may also result in divergence.
- Epsilon: The initial value of epsilon, and the final value of epsilon (that it decays to over several episodes). Epsilon represents the probability of choosing a random action (epsilon greedy policy), and helps in model training.
- Replay memory size: The number of experiences to store to use for learning. A higher value may make learning more efficient and converge faster, but may not work for off-policy methods.
- Imitation reward: The amount of reward to provide if a car follows a player suggestions. A higher value makes the model learn more closely to user suggestions.
- Kernel size: The kernel size of each convolutional layer in the neural network. A higher value indicates higher receptive field, which may let the model learn larger patterns in the image input.
- No. of conv layers: The number of convolutional layers in the neural network. A higher value indicates a deeper network, which may be able to learn more complex policies but may also take longer to train.


