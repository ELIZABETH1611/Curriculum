import numpy as np
from mushroom_rl.utils.spaces import Box
import math
from air_hockey_challenge.environments.planar.single import AirHockeySingle
from air_hockey_challenge.environments.position_control_wrapper import PositionControlPlanar

from air_hockey_challenge.utils.kinematics import forward_kinematics

class AirHockeyCurriculum(AirHockeySingle):
    """
    Class for the air hockey hitting task using CL.
    """

    def __init__(self, gamma=0.99, horizon=120, moving_init=False, random_init=False, CL_velocity= False, viewer_params={}):
        """
        Constructor
        Args:
            moving_init(bool, False): If true, initialize the puck with inital velocity.
        """
        super().__init__(gamma=gamma, horizon=horizon, viewer_params=viewer_params)

        self.moving_init = moving_init
        hit_width = self.env_info['table']['width'] / 2 - self.env_info['puck']['radius'] - \
                    self.env_info['mallet']['radius'] * 2
        self.hit_range = np.array([[-0.65, -0.65], [-0.05, 0.05]])  # Table Frame
        self.Idx=0
        self.target_width = np.array([-0.35, 0.35])  # Target X Frame
        self.env_info['table']['goal_width']= 0.7
        self.init_velocity_range = (0, 0.5)  # Table Frame
        self.init_ee_range = np.array([[0.60, 1.25], [-0.4, 0.4]])  # Robot Frame
        self.width_target =2
        self.gamma=gamma
        self.random_init = random_init
        self.range_velocity_range = 0.01
        self.CL_velocity = CL_velocity
        self.goal_accomplished = False
        self.goal_non_accomplished = False
        self.check_absor = False
        self.goal = np.array([self.env_info['table']['width'] / 2 - self.env_info['puck']['radius'], 0])
        self.episode_steps = 0
        self.set_step_puck=0
        self.split=50
        self.Add_rp=False
        self.target_y=0.0
        self.puck_theta = 0.0
        self.last_idx = 0.0

    def _modify_mdp_info(self, mdp_info):
        """
        Modify mdp_info to add information to state (target_x, target_y)
        """
        mdp_info = super()._modify_mdp_info(mdp_info)
        obs_low = np.concatenate([mdp_info.observation_space.low, [-0.6,0.35]])
        obs_high = np.concatenate([mdp_info.observation_space.high, [1.,0.13]])
        mdp_info.observation_space = Box(obs_low, obs_high)
        return mdp_info
        
    
    def _modify_observation(self, obs):
        """
        Add information to state (target_x, target_y)
        """
        obs =  super()._modify_observation(obs)
        obs = np.concatenate([obs, [self.goal[0],(self.env_info['table']['goal_width'])/2]])
        return obs
        

    def setup(self, state=None):
        """
        Write data and set distribution of the puck and define range of hit(puck) and target (sampled)
        """
        if self.random_init:
            puck_pos=[]
            puck_pos.append(np.random.uniform(self.hit_range[0][0],self.hit_range[0][1],size=1)[0])
            puck_pos.append(np.random.uniform(self.hit_range[1][0],self.hit_range[1][1],size=1)[0])
        
            reduction=((1*(0.35-(self.env_info['table']['goal_width']/2)))/50)*self.Idx
            if 23>self.Idx>0 and self.Idx!=self.last_idx:
                self.target_width = np.array([-0.35+(0.25/2)+ reduction, 0.35-(0.25/2)-reduction])
                self.last_idx =self.Idx  # Target X Frame
            elif  self.Idx>=23 and self.Idx!=self.last_idx:
                self.target_width = np.array([-0.25/2, 0.25/2])  # Target X Frame
                self.last_idx =self.Idx  # Target X Frame
  
            self.env_info['table']['goal_width']= np.linalg.norm(self.target_width[0]-self.target_width[1])  
            self.target_range = np.array([0.0,0.0])

            target_y= np.random.uniform(self.target_range[0],self.target_range[1])
            target_x=np.random.uniform(self.target_xinf,self.target_xsup)
        else:
            puck_pos = np.random.rand(2) * (self.hit_range[:, 1] - self.hit_range[:, 0]) + self.hit_range[:, 0]

        self._write_data("puck_x_pos", puck_pos[0])
        self._write_data("puck_y_pos", puck_pos[1])

        self.set_distribution_target(target_x,target_y) 

        self.set_step_puck = self.episode_steps

        if self.moving_init:
            puck_vel = np.zeros(3)
            if self.CL_velocity:
                lin_vel = np.random.uniform(self.init_velocity_range,self.range_velocity_range)
                angle = np.random.uniform(-0.7, 0.7)
                puck_vel[2] =  np.zeros(1)
            else:
                lin_vel = np.random.uniform(self.init_velocity_range[0], self.init_velocity_range[1])
                angle = np.random.uniform(-np.pi / 2 - 0.1, np.pi / 2 + 0.1)
                puck_vel[2] = np.random.uniform(-2, 2, 1)
           
            puck_vel[0] = -np.cos(angle) * lin_vel
            puck_vel[1] = np.sin(angle) * lin_vel
            
            self._write_data("puck_x_vel", puck_vel[0])
            self._write_data("puck_y_vel", puck_vel[1])
            self._write_data("puck_yaw_vel", puck_vel[2])

        super(AirHockeyCurriculum, self).setup(state)

    def reward(self, state, action, next_state, absorbing):
        r = 0
        puck_pos, puck_vel = self.get_puck(next_state)
        puck_pos_initial =  np.array([-0.65, -0.35])
        final_goal_pos =  np.array([(self.env_info['table']['length']/2.0)-self.env_info['puck']['radius'], 0])
        self.episode_steps += 1

        if self.goal_accomplished: # If the puck is in the opponent goal
            r = 0
            # print("GOAL !!!!!!",r)
        elif self.check_absor: # if the mallet is out
            r=-60
            # print("Absorbing  !!",r)
        elif self.Add_rp: #if the puck ends in a position parallel to the y-position of the target
            r = -((np.linalg.norm(puck_pos[:2] - self.goal))/(np.linalg.norm(puck_pos_initial - final_goal_pos)))     
            r = (r/(1-self.gamma))
            self.Add_rp=False
            # print("Absorbing parallel !!",r)
        else:
            r = -((np.linalg.norm(puck_pos[:2] - self.goal))/(np.linalg.norm(puck_pos_initial - final_goal_pos)))
 
        # print("reward ",r)      
        return r
    


    def is_absorbing(self, obs):
        boundary = np.array([self.env_info['table']['length'], self.env_info['table']['width']]) / 2
        puck_pos, puck_vel = self.get_puck(obs)

        # Add absorbing state when the mallet is outside the table using the ee position
        q_pos, q_vel = self.get_joints(obs)
        x_pos, rot_mat = forward_kinematics(self.env_info['robot']['robot_model'],
                                             self.env_info['robot']['robot_data'], q_pos)
        ee_pos = x_pos+self.env_info['robot']['base_frame'][0][:3,3]
        
        # self.puck_theta=math.atan((puck_pos[0]-ee_pos[0])/(puck_pos[1]-ee_pos[1]))
        # yaw = -math.atan2(rot_mat[0][0], rot_mat[1][0])
        # yaw1 = -math.atan2(rot_mat[0][1], rot_mat[1][0])
        # yaw2 = -math.atan2(rot_mat[0][1], rot_mat[1][1])
        # pitch = math.acos(rot_mat[2][2])
        # print(math.degrees(yaw),math.degrees(yaw1),math.degrees(yaw2))
        # print(math.degrees(self.puck_theta))

        puck_out = np.any(np.abs(puck_pos[:2])>boundary) # Check if the puck is outside the table
        mallet_out = np.any(np.abs(np.asarray(ee_pos[:2]))>boundary)  # Check if the mallet is outside the table

        # Check if the puck achieves the goal
        reach_goal_x = puck_pos[:1] >=(self.goal[:1]-self.env_info['puck']['radius'])
        reach_goal_y = np.linalg.norm(puck_pos[1:2]-self.goal[1:2]) <=  ((self.env_info['table']['goal_width']/2.0)-self.env_info['puck']['radius'])
        
        # Check if the puck is parallel to the y-target position 
        out_goal_y = np.linalg.norm(puck_pos[1:2]-self.goal[1:2]) > ((self.env_info['table']['goal_width']/2.0)-self.env_info['puck']['radius'])
        out_goal_x = puck_pos[:1] >= (self.env_info['table']['length']/2 - 1.5*self.env_info['puck']['radius'])

        # Check if the puck is moving too fast
        velocity = np.linalg.norm(puck_vel) > 100
        
        if reach_goal_x and reach_goal_y: # Succes task
            self.goal_accomplished = True
            self.goal_non_accomplished = False
            self.check_absor= False
            self.Add_rp = False
            return True
        elif out_goal_y and (out_goal_x or reach_goal_x):   # Puck ends parallel to the target 
            self.goal_accomplished = False
            self.goal_non_accomplished = True
            self.check_absor= False
            self.Add_rp=True
            return True
        elif mallet_out or velocity or puck_out:  
            self.goal_non_accomplished = True
            self.goal_accomplished = False
            self.check_absor= True
            self.Add_rp = False
            return True
        else:
            self.goal_accomplished = False
            if (self.episode_steps-self.set_step_puck==119):
                self.goal_non_accomplished = True
            else:
                self.goal_non_accomplished = False
            self.check_absor= False
            self.Add_rp = False
            return False

    def set_puck(self, inc_x=0.0):
        inc_y = inc_x/2
        if inc_x>0.40 :
            inc_x = 0.40 
        if inc_y>0.30 :
            inc_y = 0.30  

        self.hit_range = np.array([[-0.65, -0.65+inc_x], [-0.05-inc_y, 0.05+inc_y]])
      
        self._model.site_pos[3]  = np.concatenate(((self.hit_range[0][1]+self.hit_range[0][0])/2.0,0.0,0.005), axis=None)
        self._model.site_size[3] = np.asarray([np.abs(self.hit_range[0][0]-self.hit_range[0][1])/2.0,np.abs(self.hit_range[1][1]-self.hit_range[1][0])/2.0,  -0.001])
        self._model.site_rgba[3] = np.asarray([0.0,30, 30, 0.3 ])
        return self.hit_range 
    

    def get_distribution(self,value=0):
        if value>self.split:
            value = self.split
       
        # Calculate the lenght of the area where the target and puck will be sample
        length        = self.env_info['table']['length'] - 4 * (self.env_info['puck']['radius']+self.env_info['mallet']['radius']) -0.35
        Length_step   = length/self.split
        initial_value = - self.env_info['table']['length'] / 2 + 4 * (self.env_info['puck']['radius']+self.env_info['mallet']['radius'])+0.35          
       
        #Target
        if value < (self.split-1):
            self.target_xinf=initial_value+(value*Length_step)
            self.target_xsup=initial_value+((value-self.width_target)*Length_step)
            target_x=np.random.uniform(self.target_xinf,self.target_xsup)
        else:
            self.target_xinf=initial_value+(value*Length_step)
            self.target_xsup=self.target_xinf
            target_x=np.random.uniform(self.target_xinf,self.target_xsup)

        self.set_puck(value*Length_step)
        
        if value>30: # After task #30 We will star to reduce the area of sampling of the target 
            self.Idx+=1

        return self.goal
    
    def set_distribution_target(self,value_x=0,value_y=0):
        '''
        Set a new position of the target every time that the puck is set in a new position
        '''
        self.goal[0]=value_x
        self.goal[1]=value_y 
        self._model.site_pos[2]  = np.concatenate((self.goal,0.0), axis=None)
        self._model.site_size[2] = np.asarray([0.015,self.env_info['table']['goal_width']/2, 0.001])
        # print(self.goal,self.Idx,self.env_info['table']['goal_width'])
        # print()
        return self.goal

    def _create_info_dictionary(self, obs):
        '''
        Add information about success o fail task
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
