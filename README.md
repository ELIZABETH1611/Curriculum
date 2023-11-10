# Curri7dofculum_air_hockey
Please, when you download it, execute the following commands to install everything =)

cd Curriculum_air_hockey
conda create -y -n challenge python=3.8
conda activate challenge
cd mushroom-rl
pip install --no-use-pep517 -e .[all]
cd ..
cd air_hockey_challenge
pip install -r requirements.txt
pip install -e .
cd ..
pip install experiment_launcher
pip install dm_control


#########
## Inside of " air_hockey_challenge/Test_Pascal" folder you will find the scripts for planar and Iiwa.

The following parameters will let you to define whether you want the pack to have velocity or not, the width of the target and whether you want curriculum or RL. Now all the test are define as CL with velocity

random_init      = True    # True: Distance Logarithm for Curriculum  False: RL algorithm
velocity         = True    # True: Puck with velocity   False: Puck Static
width_goal       = 0.25
   
