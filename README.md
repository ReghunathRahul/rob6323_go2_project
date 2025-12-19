# ROB6323 Go2 Project — Isaac Lab

This repository is the starter code for the NYU Reinforcement Learning and Optimal Control project in which students train a Unitree Go2 walking policy in Isaac Lab starting from a minimal baseline and improve it via reward shaping and robustness strategies. Please read this README fully before starting and follow the exact workflow and naming rules below to ensure your runs integrate correctly with the cluster scripts and grading pipeline.

## Repository policy

- Fork this repository and do not change the repository name in your fork.  
- Your fork must be named rob6323_go2_project so cluster scripts and paths work without modification.

### Prerequisites

- **GitHub Account:** You must have a GitHub account to fork this repository and manage your code. If you do not have one, [sign up here](https://github.com/join).

### Links
1.  **Project Webpage:** [https://machines-in-motion.github.io/RL_class_go2_project/](https://machines-in-motion.github.io/RL_class_go2_project/)
2.  **Project Tutorial:** [https://github.com/machines-in-motion/rob6323_go2_project/blob/master/tutorial/tutorial.md](https://github.com/machines-in-motion/rob6323_go2_project/blob/master/tutorial/tutorial.md)

## Connect to Greene

- Connect to the NYU Greene HPC via SSH; if you are off-campus or not on NYU Wi‑Fi, you must connect through the NYU VPN before SSHing to Greene.  
- The official instructions include example SSH config snippets and commands for greene.hpc.nyu.edu and dtn.hpc.nyu.edu as well as VPN and gateway options: https://sites.google.com/nyu.edu/nyu-hpc/accessing-hpc?authuser=0#h.7t97br4zzvip.

## Clone in $HOME

After logging into Greene, `cd` into your home directory (`cd $HOME`). You must clone your fork into `$HOME` only (not scratch or archive). This ensures subsequent scripts and paths resolve correctly on the cluster. Since this is a private repository, you need to authenticate with GitHub. You have two options:

### Option A: Via VS Code (Recommended)
The easiest way to avoid managing keys manually is to configure **VS Code Remote SSH**. If set up correctly, VS Code forwards your local credentials to the cluster.
- Follow the [NYU HPC VS Code guide](https://sites.google.com/nyu.edu/nyu-hpc/training-support/general-hpc-topics/vs-code) to set up the connection.

> **Tip:** Once connected to Greene in VS Code, you can clone directly without using the terminal:
> 1. **Sign in to GitHub:** Click the "Accounts" icon (user profile picture) in the bottom-left sidebar. If you aren't signed in, click **"Sign in with GitHub"** and follow the browser prompts to authorize VS Code.
> 2. **Clone the Repo:** Open the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P`), type **Git: Clone**, and select it.
> 3. **Select Destination:** When prompted, select your home directory (`/home/<netid>/`) as the clone location.
>
> For more details, see the [VS Code Version Control Documentation](https://code.visualstudio.com/docs/sourcecontrol/intro-to-git#_clone-a-repository-locally).

### Option B: Manual SSH Key Setup
If you prefer using a standard terminal, you must generate a unique SSH key on the Greene cluster and add it to your GitHub account:
1. **Generate a key:** Run the `ssh-keygen` command on Greene (follow the official [GitHub documentation on generating a new SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent#generating-a-new-ssh-key)).
2. **Add the key to GitHub:** Copy the output of your public key (e.g., `cat ~/.ssh/id_ed25519.pub`) and add it to your account settings (follow the [GitHub documentation on adding a new SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)).

### Execute the Clone
Once authenticated, run the following commands. Replace `<your-git-ssh-url>` with the SSH URL of your fork (e.g., `git@github.com:YOUR_USERNAME/rob6323_go2_project.git`).
```
cd $HOME
git clone <your-git-ssh-url> rob6323_go2_project
```
*Note: You must ensure the target directory is named exactly `rob6323_go2_project`. This ensures subsequent scripts and paths resolve correctly on the cluster.*
## Install environment

- Enter the project directory and run the installer to set up required dependencies and cluster-side tooling.  
```
cd $HOME/rob6323_go2_project
./install.sh
```
Do not skip this step, as it configures the environment expected by the training and evaluation scripts. It will launch a job in burst to set up things and clone the IsaacLab repo inside your greene storage. You must wait until the job in burst is complete before launching your first training. To check the progress of the job, you can run `ssh burst "squeue -u $USER"`, and the job should disappear from there once it's completed. It takes around **30 minutes** to complete. 
You should see something similar to the screenshot below (captured from Greene):

![Example burst squeue output](docs/img/burst_squeue_example.png)

In this output, the **ST** (state) column indicates the job status:
- `PD` = pending in the queue (waiting for resources).
- `CF` = instance is being configured.
- `R`  = job is running.

On burst, it is common for an instance to fail to configure; in that case, the provided scripts automatically relaunch the job when this happens, so you usually only need to wait until the job finishes successfully and no longer appears in `squeue`.

## What to edit

- In this project you'll only have to modify the two files below, which define the Isaac Lab task and its configuration (including PPO hyperparameters).  
  - source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/rob6323_go2_env.py  
  - source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/rob6323_go2_env_cfg.py
PPO hyperparameters are defined in source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/agents/rsl_rl_ppo_cfg.py, but you shouldn't need to modify them.

## How to edit

- Option A (recommended): Use VS Code Remote SSH from your laptop to edit files on Greene; follow the NYU HPC VS Code guide and connect to a compute node as instructed (VPN required off‑campus) (https://sites.google.com/nyu.edu/nyu-hpc/training-support/general-hpc-topics/vs-code). If you set it correctly, it makes the login process easier, among other things, e.g., cloning a private repo.
- Option B: Edit directly on Greene using a terminal editor such as nano.  
```
nano source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/rob6323_go2_env.py
```
- Option C: Develop locally on your machine, push to your fork, then pull changes on Greene within your $HOME/rob6323_go2_project clone.

> **Tip:** Don't forget to regularly push your work to github

## Launch training

- From $HOME/rob6323_go2_project on Greene, submit a training job via the provided script.  
```
cd "$HOME/rob6323_go2_project"
./train.sh
```
- Check job status with SLURM using squeue on the burst head node as shown below.  
```
ssh burst "squeue -u $USER"
```
Be aware that jobs can be canceled and requeued by the scheduler or underlying provider policies when higher-priority work preempts your resources, which is normal behavior on shared clusters using preemptible partitions.

## Where to find results

- When a job completes, logs are written under logs in your project clone on Greene in the structure logs/[job_id]/rsl_rl/go2_flat_direct/[date_time]/.  
- Inside each run directory you will find a TensorBoard events file (events.out.tfevents...), neural network checkpoints (model_[epoch].pt), YAML files with the exact PPO and environment parameters, and a rollout video under videos/play/ that showcases the trained policy.  

## Download logs to your computer

Use `rsync` to copy results from the cluster to your local machine. It is faster and can resume interrupted transfers. Run this on your machine (NOT on Greene):

```
rsync -avzP -e 'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null' <netid>@dtn.hpc.nyu.edu:/home/<netid>/rob6323_go2_project/logs ./
```

*Explanation of flags:*
- `-a`: Archive mode (preserves permissions, times, and recursive).
- `-v`: Verbose output.
- `-z`: Compresses data during transfer (faster over network).
- `-P`: Shows progress bar and allows resuming partial transfers.

## Visualize with TensorBoard

You can inspect training metrics (reward curves, loss values, episode lengths) using TensorBoard. This requires installing it on your local machine.

1.  **Install TensorBoard:**
    On your local computer (do NOT run this on Greene), install the package:
    ```
    pip install tensorboard
    ```

2.  **Launch the Server:**
    Navigate to the folder where you downloaded your logs and start the server:
    ```
    # Assuming you are in the directory containing the 'logs' folder
    tensorboard --logdir ./logs
    ```

3.  **View Metrics:**
    Open your browser to the URL shown (usually `http://localhost:6006/`).

## Debugging on Burst

Burst storage is accessible only from a job running on burst, not from the burst login node. The provided scripts do not automatically synchronize error logs back to your home directory on Greene. However, you will need access to these logs to debug failed jobs. These error logs differ from the logs in the previous section.

The suggested way to inspect these logs is via the Open OnDemand web interface:

1.  Navigate to [https://ood-burst-001.hpc.nyu.edu](https://ood-burst-001.hpc.nyu.edu).
2.  Select **Files** > **Home Directory** from the top menu.
3.  You will see a list of files, including your `.err` log files.
4.  Click on any `.err` file to view its content directly in the browser.

> **Important:** Do not modify anything inside the `rob6323_go2_project` folder on burst storage. This directory is managed by the job scripts, and manual changes may cause synchronization issues or job failures.

## Project scope reminder

- The assignment expects you to go beyond velocity tracking by adding principled reward terms (posture stabilization, foot clearance, slip minimization, smooth actions, contact and collision penalties), robustness via domain randomization, and clear benchmarking metrics for evaluation as described in the course guidelines.  
- Keep your repository organized, document your changes in the README, and ensure your scripts are reproducible, as these factors are part of grading alongside policy quality and the short demo video deliverable.

## Resources

- [Isaac Lab documentation](https://isaac-sim.github.io/IsaacLab/main/source/setup/ecosystem.html) — Everything you need to know about IsaacLab, and more!
- [Isaac Lab ANYmal C environment](https://github.com/isaac-sim/IsaacLab/tree/main/source/isaaclab_tasks/isaaclab_tasks/direct/anymal_c) — This targets ANYmal C (not Unitree Go2), so use it as a reference and adapt robot config, assets, and reward to Go2.
- [DMO (IsaacGym) Go2 walking project page](https://machines-in-motion.github.io/DMO/) • [Go2 walking environment used by the authors](https://github.com/Jogima-cyber/IsaacGymEnvs/blob/e351da69e05e0433e746cef0537b50924fd9fdbf/isaacgymenvs/tasks/go2_terrain.py) • [Config file used by the authors](https://github.com/Jogima-cyber/IsaacGymEnvs/blob/e351da69e05e0433e746cef0537b50924fd9fdbf/isaacgymenvs/cfg/task/Go2Terrain.yaml) — Look at the function `compute_reward_CaT` (beware that some reward terms have a weight of 0 and thus are deactivated, check weights in the config file); this implementation includes strong reward shaping, domain randomization, and training disturbances for robust sim‑to‑real, but it is written for legacy IsaacGym and the challenge is to re-implement it in Isaac Lab.
- **API References**:
    - [ArticulationData (`robot.data`)](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.assets.html#isaaclab.assets.ArticulationData) — Contains `root_pos_w`, `joint_pos`, `projected_gravity_b`, etc.
    - [ContactSensorData (`_contact_sensor.data`)](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sensors.html#isaaclab.sensors.ContactSensorData) — Contains `net_forces_w` (contact forces).

---
Students should only edit README.md below this line.

## Project Solution
| Authors              | Email                |
|-------------------|----------------------|
| **Long Quang**    | lq2146@nyu.edu       |
| **Rahul Reghunath** | rr4660@nyu.edu     |
| **Sourabh Misal** | srm9726@nyu.edu      |

<sup>
Authors are with New York University - Tandon School of Engineering
<br> 6 MetroTech Center, Brooklyn, NY. 
<br> All have contributed equally to this work.
</sup>

[Code](https://github.com/ReghunathRahul/rob6323_go2_project) | [Paper](#)

## Repository layout
- `source/rob6323_go2/`: Python package that defines the Go2 tasks, environment configuration, actuator models, and PPO hyperparameters.
  - `rob6323_go2/tasks/**/rob6323_go2/*_go2_env.py`: Task logic (observations, rewards, termination, friction randomization, gait clock, debug visualization).
  - `rob6323_go2/tasks/**/rob6323_go2/*_go2_env_cfg.py`: Environment, terrain, actuator, and reward-scale configuration.
  - `rob6323_go2/tasks/**/rob6323_go2/agents/rsl_rl_ppo_cfg.py`: PPO network architecture, learning rates, iteration limits, and clip settings.
- `scripts/`: Helper entry points (e.g., `scripts/rsl_rl/train.py`, `scripts/rsl_rl/play.py`).
- `train.sh` / `train.slurm`: Submit RL training + evaluation to HPC queue.
- `tests.sh` / `tests.slurm`: Submit pytests to HPC queue.
- `docs/report`: Project report.
- `tutorial/`: Upstream tutorial material.

## Baseline implementation summary
- **Observations:** Base linear/angular velocity in the body frame, projected gravity, commanded velocities, joint position offsets and velocities, previous actions, and four gait clock sinusoid inputs.
- **Actions & control:** Actions are joint position offsets scaled by `action_scale` and tracked by a PD controller with per-joint `Kp`/`Kd` and torque limits. A friction model adds stiction and viscous losses before clamping torques.
- **Reward shaping:**
  - Exponential tracking of commanded planar velocity and yaw rate.
  - Action-rate penalty using first- and second-order differences over the last three actions.
  - Raibert foot-placement penalty comparing desired vs. measured footsteps from a sinusoidal gait clock.
  - Penalties on roll/pitch (projected gravity), vertical velocity, joint velocities, and roll/pitch angular velocity.
  - Swing-foot clearance bonus and stance contact-force tracking.
- **Termination:** Episode ends on base-ground collision, upside-down orientation, minimum base height, or time-out.
- **Randomization:** Commands are resampled each episode; actuator stiction/viscous friction are randomized within configured ranges; resets jitter start times to avoid synchronous rollouts.

### Task-specfic training
Launch PPO training and automated evaluation from Greene. You may override `TASK_NAME` or pass extra flags to `scripts/rsl_rl/train.py` via `train.sh`.
```bash
cd ~/rob6323_go2_project
TASK_NAME=Template-Rob6323-Go2-<Direct|Backflip|RobustLocomotion|Bipedal>-v0 ./train.sh --experiment_name go2_<flat_direct|backflip|robust_localmotion|bipedal> --max_iterations 1000
```

## Running tests
Submit the provided automated checks to Burst:
```bash
cd ~/rob6323_go2_project
./tests.sh
```

## Training Results and Analysis (Weights & Biases)

All metrics below are logged using Weights & Biases during PPO training on NYU Greene. Each curve corresponds to a separate training run with different random seeds or task configurations.

---

## Training Results and Analysis (Weights & Biases)

All metrics below are logged using Weights & Biases during PPO training on NYU Greene.

---

### Mean Episode Length
![Mean Episode Length](docs/img/train_mean_episode_length.png)

**Interpretation:**  
The steady increase and saturation near the episode horizon indicates that the policy learns to remain upright and stable for longer durations. Episodes increasingly terminate due to time-out rather than failure, reflecting improved gait stability and balance.

---

### Mean Episode Reward
![Mean Episode Reward](docs/img/train_mean_reward.png)

**Interpretation:**  
The rising reward curve corresponds to improved velocity tracking, smoother actions, and better gait coordination. Flat or negative curves indicate failed runs that do not discover stable locomotion.

---

### Learning Rate Schedule
![Learning Rate](docs/img/loss_learning_rate.png)

**Interpretation:**  
Early learning-rate spikes correspond to exploratory PPO updates. The gradual reduction indicates convergence as policy updates become smaller and more stable.

---

### Episode Termination: Time-Out
![Termination Time-Out](docs/img/episode_termination_timeout.png)

**Interpretation:**  
As training progresses, a larger fraction of episodes terminate due to time-out instead of instability. This confirms successful stabilization of the locomotion policy.

---

### Velocity Tracking Reward
![Velocity Tracking Reward](docs/img/track_lin_vel_xy_exp.png)

**Interpretation:**  
The rapid rise and plateau demonstrate that the policy quickly learns to track commanded planar velocities accurately and consistently over long horizons.

---

### Action Smoothness Reward
![Action Smoothness Reward](docs/img/action_smoothness.png)

**Interpretation:**  
The decreasing penalty magnitude over training indicates reduced joint command chatter and smoother actuation, validating the effectiveness of first- and second-order action-rate regularization.

---

### Backflip Task: Takeoff Impulse Reward
![Takeoff Impulse Reward](docs/img/takeoff_impulse.png)

**Interpretation:**  
For the backflip task, the increasing takeoff impulse reward shows that the policy learns to generate sufficient vertical impulse and coordinated motion required for liftoff.

---

## Summary

Together, these plots demonstrate:
- Stable PPO convergence
- Improved gait stability and episode survivability
- Effective velocity tracking
- Reduced action chattering
- Successful task-specific learning (e.g., backflip takeoff)

These results validate the reward design, actuator modeling, and training pipeline implemented in the repository.
