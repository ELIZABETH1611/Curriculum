import numpy as np
import torch
from mushroom_rl.utils.spaces import Box
import math
from air_hockey_challenge.environments.planar.single import AirHockeySingle
from air_hockey_challenge.environments.position_control_wrapper import PositionControlPlanar

from air_hockey_challenge.utils.kinematics import forward_kinematics
 
class AirHockeyCurriculum(AirHockeySingle):
    """
    Class for the air hockey hitting task using CL.
    """

    def __init__(self, gamma=0.99, horizon=500, moving_init=False, random_init=False, CL_velocity= False, viewer_params={}):
        """
        Constructor
        Args:
            moving_init(bool, False): If true, initialize the puck with inital velocity.
        """
        super().__init__(gamma=gamma, horizon=horizon, viewer_params=viewer_params)
        self.horizon = horizon
        self.random_init = random_init
        self.moving_init = moving_init
        self.hit_range   = np.array([[-0.65, -0.65], [-0.05, 0.05]])  #  Puck Static
        self.hit_range_2 = np.array([0.0,  0.0])  # Puck with Speed 
        self.sparce      = False
        self.velocity_cl = False
        self.Idx=0
        self.target_range = np.array([-0.35, 0.35])  # Target X Frame
        self.init_velocity_range = (0, 0.5)  # Table Frame
        self.init_ee_range = np.array([[0.60, 1.25], [-0.4, 0.4]])  # Robot Frame
        self.gamma=gamma
        self.CL_velocity = CL_velocity
        self.goal_accomplished = False
        self.goal_non_accomplished = False
        self.check_absorbing = False
        self.goal = np.array([0.945, 0])
        self.episode_steps = 0
        self.set_step_puck=0
        self.split=30
        self.split_2=30
        self.high_punishment=False
        self.high_punishment_2=False
        self.target_y=0.0
        self.mask_inf = 0
        self.mask_sup = 5
        self.puck_theta = 0.0
        self.last_idx = 0.0
        self.sen_theta = 0.0
        self.cos_theta = 0.0
        self.target_xi =0.0
        self.target_xs =0.0
        self.extra =0.0
        self.double=False
        self.similar=False
        self.decayed_target_entropy =-3
        self.initial_value = -self.env_info['table']['length'] / 2 + 4 * (self.env_info['puck']['radius']+self.env_info['mallet']['radius'])+0.35          
        self.Reduce_sample=self.reduce_sample(0.15,0.945,self.split)

    def decay_target_entropy(self,task):
        '''
        Reduce the entropy from -3 to -9
        Each time that change of task, reduce the entropy 
        Args:
            task: The current task
        Returns:
            decayed_target_entropy : The new value of the entropy
        '''
        if self.split-1 > task > 0:
            initial_target_entropy = -3
            decay_steps = self.split  
            decay_amount = 6.0  
            self.decayed_target_entropy = initial_target_entropy - (decay_amount / decay_steps) * (task)
        return self.decayed_target_entropy


    def _modify_mdp_info(self, mdp_info):
        '''
        Same as the original 
        Args:
            Goal:  Min and max Position of the goal
            Theta:  Min and max between the mallet and the puck
        Returns:
            All the mdp_info
        '''
        mdp_info = super()._modify_mdp_info(mdp_info)
        obs_low = np.concatenate([mdp_info.observation_space.low, [-0.6,-0.4,-1,-1]])
        obs_high = np.concatenate([mdp_info.observation_space.high, [1.,0.4,1,1]])
        mdp_info.observation_space = Box(obs_low, obs_high)
        return mdp_info
        
    
    def _modify_observation(self, obs):
        '''
        Add the extra infomation 
        Args:
            Goal: Position of the goal
            Theta: Angle between the mallet and the puck
        Returns:
            All the mdp_info
        '''
        obs =  super()._modify_observation(obs)
        obs = np.concatenate([obs, [self.goal[0],self.goal[1], self.sen_theta,self.cos_theta]])
        return obs    
 
  
 
    def setup(self, state=None):
        '''
        Add the extra infomation 
        Args:
            Goal: Position of the goal
            Theta: Angle between the mallet and the puck
        Returns:
            All the mdp_info
        '''      
        puck_pos=[]
        puck_pos.append(np.random.uniform(self.hit_range[0][0],self.hit_range[0][1],size=1)[0])
        puck_pos.append(np.random.uniform(self.hit_range[1][0],self.hit_range[1][1],size=1)[0])

        if self.random_init:
            self.goal[0],self.goal[1]=self.curriculum()
        else:
            self.goal[0]=(np.random.uniform(0.15,0.945,size=1)[0])
            self.goal[1]=(np.random.uniform(self.target_range[0],self.target_range[1],size=1)[0])

        self._model.site_pos[2]  = np.concatenate((self.goal,0.0), axis=None)

        self.set_step_puck= self.episode_steps

        if self.moving_init:
            self.extra= np.random.uniform(self.hit_range_2[0],self.goal[0]-0.06)
            puck_vel = np.zeros(3)
            puck_pos[0]= -0.25
            pos_y         = np.random.uniform(-0.35,0.35)
            pos_x         = np.random.uniform(-0.65,-0.35)
            puck_vel[:2]  = self.velocity([puck_pos[0]+self.extra,puck_pos[1]],[pos_x,pos_y])
            puck_vel[2]   = np.zeros(1)
            

            self._write_data("puck_x_vel", puck_vel[0])
            self._write_data("puck_y_vel", puck_vel[1])
            self._write_data("puck_yaw_vel", puck_vel[2])
        
        puck_pos[0]=puck_pos[0]+self.extra

        self._write_data("puck_x_pos", puck_pos[0])
        self._write_data("puck_y_pos", puck_pos[1])
        super(AirHockeyCurriculum, self).setup(state)
   
    def velocity(self, puck_ini, puck_end):
        '''
        Set the velocity of the puck
        Args:
            puck_ini    Start Position of the puck
            puck_end    End Position of the puck
        Return:
                Velocity  Vx,Vy
        '''
        vector=np.asarray(puck_ini[0:2])-np.asarray(puck_end[0:2])
        d=np.linalg.norm(vector)
        # t= np.random.uniform(0.3,0.6)
        t= np.random.uniform(0.5,0.7)
        velocidad_lineal_x = - d / t
        velocity=(vector/d)*velocidad_lineal_x
        return velocity
    
    def curriculum(self):
        '''
        Defines how the target sampling area is reduced (y-axis)
        Sample Goal each episode 
        '''
        self.last_idx =self.Idx 
        target_y= np.random.uniform(self.target_range[0],self.target_range[1])
        target_x= np.random.uniform(self.target_xi,self.target_xs)
        return target_x,target_y



    def reduce_sample(self,start, end, num_steps):
        '''
        Aplied logarithm function to sample the positions of the task in x-axis
        '''
        log_space = np.linspace(0, 1, num_steps)
        factor=self.split_2
        positiones=(np.log(factor*log_space+1)/np.log(factor+1))*(end-start)+start
        return positiones
    
    def reward(self, state, action, next_state, absorbing):
        '''
        Reward
        Sparce:
                True    - Activate Sparce reward [-10,0,10]
                False   - Activate Dense Reward
        '''
        r = 0
        puck_pos, puck_vel = self.get_puck(next_state)
        puck_pos_initial =  np.array([-0.65, -0.35])
        final_goal_pos   =  np.array([0.974, 0])
        self.episode_steps += 1

        if self.goal_accomplished:
           if puck_pos[:1] >= (self.goal[:1] -self.env_info['puck']['radius']) and \
                np.linalg.norm(puck_pos[1:2]-self.goal[1:2])<= ((self.env_info['table']['goal_width']/2.0)-self.env_info['puck']['radius']):    
            if self.sparce:
                r=10
            else:   
                r = 0
        elif self.check_absorbing:
            if self.sparce:
                r=-10
            else:   
                r=-60#60           
        elif self.high_punishment:
            if self.sparce:
                r=-10
            else:   
                r = -((np.linalg.norm(puck_pos[:2] - self.goal))/(np.linalg.norm(puck_pos_initial - final_goal_pos)))     
                r=(r/(1-self.gamma))
            self.high_punishment=False
        elif self.high_punishment_2:
            if self.sparce:
                r=-10
            else:   
                if self.similar: 
                    r = -((np.linalg.norm(puck_pos[:2] - [-0.97,0.0]))/(np.linalg.norm(puck_pos_initial - final_goal_pos)))     
                elif self.double:
                    r = -((np.linalg.norm(puck_pos[:2] - self.goal))/(np.linalg.norm(puck_pos_initial - final_goal_pos)))     
                else:
                    r = -((np.linalg.norm(puck_pos[:2] - self.goal))/(np.linalg.norm(puck_pos_initial - final_goal_pos)))/2   
                r=(r/(1-self.gamma))
            self.high_punishment_2=False
        else:
            if self.sparce:
                r=0                
            else:   
                r = -((np.linalg.norm(puck_pos[:2] - self.goal))/(np.linalg.norm(puck_pos_initial - final_goal_pos)))
        return r
    
    def is_absorbing(self, obs):
        '''
        Absrobing states
        '''

        boundary = np.array([self.env_info['table']['length'], self.env_info['table']['width']]) / 2
        puck_pos, puck_vel = self.get_puck(obs)

        q_pos, q_vel = self.get_joints(obs)

        x_pos, rot_mat = forward_kinematics(self.env_info['robot']['robot_model'],
                                             self.env_info['robot']['robot_data'], q_pos)
        ee_pos = x_pos+self.env_info['robot']['base_frame'][0][:3,3]
        
        puck_theta=math.atan((puck_pos[0]-ee_pos[0])/(puck_pos[1]-ee_pos[1]))
        if 0.0>puck_theta>=-np.pi/2:
            puck_theta=puck_theta+np.pi
        else:
            puck_theta=puck_theta
        self.sen_theta=math.sin(puck_theta)
        self.cos_theta=math.cos(puck_theta)

        puck_out = np.any(np.abs(puck_pos[:2])>boundary)
        mallet = np.any(np.abs(np.asarray(ee_pos[:2]))>boundary)
        reach_goal_x = puck_pos[:1] >=(self.goal[:1]-self.env_info['puck']['radius'])
        reach_goal_y = np.linalg.norm(puck_pos[1:2]-self.goal[1:2]) <=  ((self.env_info['table']['goal_width']/2.0)-self.env_info['puck']['radius'])
        out_goal_y = np.linalg.norm(puck_pos[1:2]-self.goal[1:2]) > ((self.env_info['table']['goal_width']/2.0)-self.env_info['puck']['radius'])
        out_goal_x = puck_pos[:1] >= (self.env_info['table']['length']/2 - 2.0*self.env_info['puck']['radius'])
        velocity = np.linalg.norm(puck_vel[:2]) > 100
        out_oposite_x = puck_pos[:1] <= (-0.974+2*self.env_info['puck']['radius'])


        if reach_goal_x and reach_goal_y:
            self.goal_accomplished     = True
            self.goal_non_accomplished = False
            self.check_absorbing       = False
            return True

        elif out_goal_y and (out_goal_x or reach_goal_x):
            self.goal_accomplished     = False
            self.goal_non_accomplished = True
            self.check_absorbing       = False
            self.high_punishment       =True
            return True

        elif mallet or velocity or puck_out:
            self.goal_non_accomplished = True
            self.goal_accomplished = False
            self.check_absorbing= True
            return True
        
        elif  out_oposite_x:
            self.goal_accomplished     = False
            self.goal_non_accomplished = True
            self.check_absorbing       = False
            self.high_punishment_2       =True
            return True
        
        else:
            self.goal_accomplished = False
            if (self.episode_steps-self.set_step_puck==self.horizon):
                self.goal_non_accomplished = True
            else:
                self.goal_non_accomplished = False
            self.check_absorbing= False
            return False
                  
    def set_puck(self, inc_x=0.0):
        '''
        Set the new range where the puck will be sampled
        Args:
            inc_x   - size of the increment in x-axis [-0.65,-0.25.]
            inc_y   - size of the increment in y-axis [-0.35,0.35.]
            
        Returns:
            hit_range 
        '''
        inc_y = inc_x/2
        if inc_x>0.40 :
            inc_x = 0.40  

        if inc_y>0.30 :
            inc_y = 0.30  
        
        if self.random_init:
            self.hit_range = np.array([[-0.65, -0.65+inc_x], [-0.05-inc_y, 0.05+inc_y]])
        else:
            self.hit_range = np.array([[-0.65, -0.25], [-0.35, 0.35]])

        self._model.site_pos[3]  = np.concatenate(((self.hit_range[0][1]+self.hit_range[0][0])/2.0,0.0,0.005), axis=None)
        self._model.site_size[3] = np.asarray([np.abs(self.hit_range[0][0]-self.hit_range[0][1])/2.0,np.abs(self.hit_range[1][1]-self.hit_range[1][0])/2.0,  -0.001])
        self._model.site_rgba[3] = np.asarray([0.0,30, 30, 0.3 ])
        return self.hit_range 
    

    def get_distribution(self,value=0,value_2=0):
        '''
        Argument:
            value:     - Number of the current task 
            value_2:   - Number of the last task
        random_init:
            True:    - Define the new range for the positions of the target in x-axis
                     - Define the increment of the area where the puck will be sample (x,y)
                     - length:    Total distance from the first task and the last one 
                     - Length_dist:  Size of the step that the area of the puck will increment in each task

            False:   - Define the increment of the area where the puck will be sample (x,y)
                     - The position of the Goal is static 

        Return:
            Position of the goal and size of the step 
        '''
        length        = self.env_info['table']['length'] - 4 * (self.env_info['puck']['radius']+self.env_info['mallet']['radius']) -0.35
        Length_dist   = length/self.split
        length_2      = 0.25 /self.split

        if self.random_init:
            self.Idx+=1
            if value>self.split:
                value = self.split

            if value < self.split:
                self.target_xi= self.Reduce_sample[value]
                self.target_xs= 0.15
                target_X=np.random.uniform(self.target_xi,self.target_xs)   
            else:
                target_X=np.random.uniform(self.target_xi,self.target_xs) 
            if self.goal[0]>0.945:
                self.goal=0.945
            self.goal     = np.array([target_X,self.target_y])
        
        if self.velocity_cl and value<(self.split-1) :
            self.hit_range_2[1]= self.hit_range_2[0]+(length_2*value)
        else:
            self.hit_range_2= [0.0, 1.15]
        self.set_puck(value*Length_dist)
        return self.goal
 
    def _create_info_dictionary(self, obs):
        '''
        Create dictionary to save when the task was accomplished or not 
        '''
        constraints = {}
        constraints["success_task"]=np.array([self.goal_accomplished])
        constraints["fail_task"]=np.array([self.goal_non_accomplished])
        return constraints
    

    
class AirHockeyCurriculumPosition(PositionControlPlanar, AirHockeyCurriculum):
    pass


if __name__ == '__main__':
    env = AirHockeyCurriculum(moving_init=False)

    env.reset()
    env.render()
    R = 0.
    J = 0.
    gamma = 1.
    steps = 0
    while True:
        action = np.zeros(3)

        observation, reward, done, info = env.step(action)
        env.render()

        gamma *= env.info.gamma
        J += gamma * reward
        R += reward
        steps += 1
        if done or steps > env.info.horizon:
            print("J: ", J, " R: ", R)
            R = 0.
            J = 0.
            gamma = 1.
            steps = 0
            env.reset()
