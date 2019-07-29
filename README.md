## Description
Implementation of reinforcement learning for the OpenAI gym's Car-racing environment (https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py)

## Requirements
Install box2d and gym, along with pytorch and the usual data science stack (numpy, matplotlib)

## Instructions
Run car.py

- You can use controls during the rendered episodes by using the arrow keys. This will override model-derived suggestions for the car in environment steps. If this is done, if the model-derived suggestion is different from user input, the model receives a negative penalty (similar to imitation learning). If no input is received, model proceeds with learning via Double Deep Q Learning, rewards derived from the OpenAI gym environment. Vary parameters as desired (or scale of rewards to change weightages as necessary)
- Different storage buffers are used for user-input and model-input state changes. This allows the user to, for example, train for only the first few episodes, and have these inputs be stored in memory for the duration of the program running, while the model cycles through its FIFO memory.
