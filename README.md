## Requirements
Install box2d and gym, along with pytorch and the usual data science stack (numpy, matplotlib)

## Instructions
Run car.py

- You can use controls during the rendered episodes by using the arrow keys. This will override model-derived suggestions for the car in environment steps. If this is done, if the model-derived suggestion is different from user input, the model receives a negative penalty (similar to imitation learning). If no input is received, model proceeds with learning via Double Deep Q Learning. Vary parameters as desired (or scale of rewards to change weightages as necessary)
