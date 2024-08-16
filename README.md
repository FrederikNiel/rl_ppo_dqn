# Gymnasium/SKRL based Deep Reinforcement learning

![Deep Reinforcement Learning](/images_and_videos/deep_RL.png?raw=true "Image of Cartpole training in gym envioment")


---
## Table of Contents
1. [Gym: Deep Reinforcement Learning DQN & PPO for Cartpole](#-Gym:-Deep-Reinforcement-Learning-DQN-&-PPO-for-Cartpole)
2.  [SKRL: Deep Reinforcement Learning Multi agent PPO for Franka Emika](#-SKRL:-Deep-Reinforcement-Learning-Multi-agent-PPO-for-Franka-Emika)

---
## Gym: Deep Reinforcement Learning DQN & PPO for Cartpole
### Guide for running
1. **Make Python >=3.6 virtual environment e.g. with Anaconda/virtualenv**
If anaconda is installed then:
```
#Creating a Virtual enviroment in this excercise Python 3.12.12 was used
conda create -n drl python=3.12.12
#Activate the enviroment
conda activate drl
```

2. **Install Stable-Baselines3**
    A list of the dependencies used for the assignment are described in [dependencies.yaml](https://github.com/FrederikNiel/rl_ppo_dqn/blob/688edd8be6ad5b110dd53cb73c18c928560da5e0/gym_DQN_PPO/dependencies.yaml) for a quick install:
    ```
    pip install -r src/gym_DQN_PPO/requirements.txt
    ```
3. **Train the agent**
    1. The two models that are trained are the off-policy DQN [*Cartpole_DQN.py*](https://github.com/FrederikNiel/rl_ppo_dqn/blob/688edd8be6ad5b110dd53cb73c18c928560da5e0/gym_DQN_PPO/Cartpole_dqn.py) and the on-policy [*Cartpole_PPO.py*](https://github.com/FrederikNiel/rl_ppo_dqn/blob/688edd8be6ad5b110dd53cb73c18c928560da5e0/gym_DQN_PPO/Cartpole_ppo.py). To train the models, make sure your command prompt is using adminstrator mode in windows or you have root acces in the linux terminal and run the following commands for training the DQN and PPO models.

    ```
    python3 src/gym_DQN_PPO/Cartpole_dqn.py
    python3 src/gym_DQN_PPO/Cartpole_ppo.py
    ```
    When the training is done a Pygame should appear with the following simulations of the trained cartpole.

    | PPO Model                                            | DQN model                                            |
    | ---------------------------------------------------- | ---------------------------------------------------- |
    | ![alt "PPO-Model"](/images_and_videos/PPO_Model.gif) | ![alt "DQN-Model"](/images_and_videos/DQN_model.gif) |


4. **Observe the training with TensorBoard**
The TensorBoard was recorded in the Database weights and biases(Wandb). Wandb is a online database where the training informations from the scripts is recorded and can be observed by the entire team instead of having the TensorBoard on a single computer. If you want access to the weights and biases team write a email to asbjorn2625@gmail.com.


    When setting up Wandb with stable-baseline3, an initilisation of the project is needed:

    ```
    run = wandb.init(
        project="lecture9-cartpole-ppo",
        config=config,
        sync_tensorboard=True,  
    )
    ```
    Since Tensorboard is initialised through the parameter tensorboard_log, only a callback to the wandb api after each learninng step is needed.

    Before running [`Cartpole_DQN.py`](https://github.com/FrederikNiel/rl_ppo_dqn/blob/688edd8be6ad5b110dd53cb73c18c928560da5e0/gym_DQN_PPO/Cartpole_dqn.py) / [`Cartpole_PPO.py`](https://github.com/FrederikNiel/rl_ppo_dqn/blob/688edd8be6ad5b110dd53cb73c18c928560da5e0/gym_DQN_PPO/Cartpole_ppo.py), run the command:

    ```
    wandb login
    ```

    Login with you wandb credentials from [Wandb](https://wandb.ai/home), and copy/paste the API-key into the terminal, found under the personal-tab in the dropdown menu:

    ![wandb api key](/images_and_videos/WandbAPIKEY.png)

    When tracking the training process through Wandb, vscode-Windows should be run in administrator mode.
    Wandb projects can be found under project: [wandb.ai/YOURNAME/projects](https://wandb.ai/fsni/projects), where in an overwiew of former and active runs can be found. 

    ![wandb runs](/images_and_videos/wandbruns.png)    

    


### Training Parameters
| PPO model                                                         | DQN Model                                                         |
| ----------------------------------------------------------------- | ----------------------------------------------------------------- |
| ![alt Training PPO](/images_and_videos/Train_tensorboard_PPO.png) | ![alt Training DQN](/images_and_videos/Train_tensorboard_DQN.png) |
### Rollout Parameters
| PPO Model                                                          | DQN Model                                                          |
| ------------------------------------------------------------------ | ------------------------------------------------------------------ |
| ![alt Rollout PPO](/images_and_videos/Rollout_tensorboard_PPO.png) | ![alt Rollout DQN](/images_and_videos/Rollout_tensorboard_DQN.png) |








- DQN:
Deep Q-Networks is an off-policy reinforcement learning algorithm discrete action spaces. It utilizes a Deep Neural Network to approximate the Q-function, which estimates the expected future rewards for taking a specific action in a given state. The objective of DQN is value-based, focusing on learning the value of different actions in each state. During training, DQN employs techniques such as experience replay and target networks to stabilize learning. Experience replay involves storing past experiences in a replay buffer and sampling from it randomly, while target networks help mitigate the issue of overestimating Q-values by using a separate network for target Q-values. These techniques contribute to the stability and efficiency of the learning process.

- PPO:
Proximal Policy Optimization is an on-policy reinforcement learning algorithm suitable for environments with continuous action spaces. Unlike DQN, which focuses on learning the value function, PPO directly learns a policy to select actions. It samples actions from a probability distribution defined by a policy network, which outputs the probability of taking each action given the current state. The objective of PPO is policy-based, aiming to improve the policy directly without explicitly estimating the value function. During training, PPO utilizes a clipped surrogate objective to prevent large policy updates and ensure stability. It adapts the learning rate based on advantage estimates to improve sample efficiency. These mechanisms contribute to the effectiveness of PPO in learning complex policies for continuous action spaces.


- Cartpole Enviroment:
Cartpole is episodic task (ends after 500ms || angle>±12 degrees || ± 2.4 units from centre) 

  As seen  from the two GIF examples, the PPO converges quicker than DQN and seem better. But DQN is more exhaustive, since it has a buffer for its policy, if you were to spawn a PPO and DQN in the corner of the statesapce a well trained DQN will propably balance better.

  For DQN Hyperparameters:
https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html

  For PPO Hyperparameters:
https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

---
## SKRL: Deep Reinforcement Learning Multi agent PPO for Franka Emika

1. **Make Python ==3.8 virtual environment.**
2. **Use panda-gym simulation.**
- Inorder to install we need to setup a Virtual Anaconda/Miniconda enviroment running Python 3.8:


    Download and run the newest Linux installer:
    ```
    curl -sL \
      "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" > \
      "Miniconda3.sh"
    ```

    ```
    bash Miniconda3.sh
    export PATH="/home/$USER/miniconda3/bin:$PATH"

    
    ```

    Create the Python enviroment:
    ```
    conda create -n skrl python=3.8.18
    ```

    In the conda enviroment panda-gym is installed according to [SKRL_PPO_multi_agent/dependencies.yaml](https://github.com/FrederikNiel/rl_ppo_dqn/blob/688edd8be6ad5b110dd53cb73c18c928560da5e0/SKRL_PPO_multi_agent/dependencies.yaml):
    ```
    conda activate skrl
    ```

    Install the following depencies:

    ```
    pip install stable-baselines3
    pip install tensorboard
    pip install gymnasium[classic-control]
    pip install wandb
    pip install panda-gym
    pip install skrl["torch"]
    ```
    Or simply install the depencies through requirements.txt:

    ```
    pip install -r src/SKRL_PPO_multi_agent/requirements.txt
    ```
  
3. **Test current implementation in Stable-Baselines3.**

  For running the pybullet PPO implementiation, run: [SKRL_PPO_multi_agent/rl_panda_gym_pybullet_example/panda_reach_train_agent.py](https://github.com/FrederikNiel/rl_ppo_dqn/blob/688edd8be6ad5b110dd53cb73c18c928560da5e0/SKRL_PPO_multi_agent/rl_panda_gym_pybullet_example/panda_reach_train_agent.py):

  Ensure you use the skrl Python enviroment.

```
config = {
    "policy_type": "MultiInputPolicy",
    "total_timesteps": 200000,
    "env_id": gym.make( env_name, render_mode=render_mode, reward_type=reward_type, control_type=control_type),
    "learning_rate": 0.001,
    "batch_size": 64,
    "pi_hidden_units": [64, 64],
    "vf_hidden_units": [64, 64],
    "date_string": datetime.now().strftime("%d%m%Y")
    
    
}
```
With the pybullet implementation some parameters are set. The reward type dense is set so the agent will get rewarded during training, wich will make the policy reach acceptable results way quicker, than a sparse reward type, where the agent is only rewarded after an episode. It will simply take too long until the agent can find a "viable" direction.

From the PandaReach-v3 environment the policy can either be trained to have joint values as actions or cartesian end-effector pose. Where the PPO agent has to learn how to reach a box.

A batch size is set, so the agent can train multiple times on the same data, as PPO is very data intensive.

Two neural networks are trained on the data:
A Policy network (actor) with current states as input, and a probability distribution of actions given the input state. Which optimises the reward, and functions like reward for the recently learned actions.

Another Value function network (critic) takes states as input and learned an estimated value function. This functions almost like an average of rewards for the agent.

These two estimations lay the foundations for advantage, where the normalised value function, is subtracted to the policy estimation. So we figure out how much the newly learned action would improve the general policy.

The agent can then update the policy for actions with higher advantages. This is done through general policy gradient loss optimisation. Where the gradient increases the probability of taking and action that lead to higher expected reward, if advantage is positive. The probability of an action will be decreased if the advantage is negative.

So the action probabilities will eventually approximate the optimal route to highest reward (the fastest movement to touch the box).

PPO utilises clipping for avoiding updating the policy too much. A probability ratio for comparing the agent has changed its policy. In order to avoid high probability ratios, the PPO uses clipping. Where large updates are capped either by the maximum or minimum value of the clipping ratio. This ensures stability, because no unexpected updates will completely change the general policy.




- For general overwiew of PPO and its differences to DQN including hyperparametes, see Gym Cartpole example:


The saved pybullet implemnted policy can be run with:
[SKRL_PPO_multi_agent/rl_panda_gym_pybullet_example/panda_reach_test_agent.py](https://github.com/FrederikNiel/rl_ppo_dqn/blob/688edd8be6ad5b110dd53cb73c18c928560da5e0/SKRL_PPO_multi_agent/rl_panda_gym_pybullet_example/panda_reach_test_agent.py)

Where the saved model can be included:
```
model = PPO.load(
    "./rl-trained-agents/PandaReach_PPO_v3_ee_model.zip",
    # "./rl-trained-agents/PandaReach_PPO_v3_joints_model.zip",
    print_system_info=True,
    device="auto",
)
```


4. **Implement PPO agent in skrl and Gymnasium interface.**

- For training the setup run: 
[SKRL_PPO_multi_agent/torch_gymnasium_pendulum_ppo.py](https://github.com/FrederikNiel/rl_ppo_dqn/blob/688edd8be6ad5b110dd53cb73c18c928560da5e0/SKRL_PPO_multi_agent/torch_gymnasium_pendulum_ppo.py)

  If on Windows:
```
try:
    env = gym.vector.make("PandaReach-v3", num_envs=1, asynchronous=False, render_mode="human") 
except (gym.error.DeprecatedEnv, gym.error.VersionNotFound) as e:
    env_id = [spec for spec in gym.envs.registry if spec.startswith("PandaReach-v")][0]
    print("PandaReach-v3 not found. Trying {}".format(env_id))
    env = gym.vector.make(env_id, num_envs=1, asynchronous=False)
env = wrap_env(env)
```

  On Linux it is possible to train multple agent for updating the policy, at the same time:

```
try:
    env = gym.vector.make("PandaReach-v3", num_envs=4, asynchronous=True, render_mode="human") 

```
½
Four PPO agents training Panda Reach object for the same policy, through skrl implementation:

!["An execution of multiple agents training"](/images_and_videos/Lec10Runs-ezgif.com-crop.gif)


5. **Track experiments in tensorboard or wandb.**
  - The Wandb API setup follows the same procedure as gym Cartpole example, for the two Stable-Baselines3 implementations.
  - For the skrl implementation two additional configrations has to be added:
```
from wandb.integration.sb3 import WandbCallback

cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 1024  
cfg["learning_epochs"] = 10
.
.
.
cfg["experiment"]["checkpoint_interval"] = 5000
cfg["experiment"]["directory"] = "runs/torch/PandaReach-v3"
# We try to use wandb callback:
cfg["experiment"]["wandb"] = True
cfg["experiment"]["wandb_kwargs"] = {"project": "lecture10_panda_deep_ppo", "entity": "omtp_rob"}

```





