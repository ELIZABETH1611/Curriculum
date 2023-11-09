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
        self.target_range = np.array([-0.35+(self.env_info['table']['goal_width']/2), 0.35-(self.env_info['table']['goal_width']/2)])  # Target X Frame
        
        self.init_velocity_range = (0, 0.5)  # Table Frame
        self.init_ee_range = np.array([[0.60, 1.25], [-0.4, 0.4]])  # Robot Frame
        
        self.gamma=gamma
        self.random_init = random_init
        self.range_velocity_range = 0.01
        self.CL_velocity = CL_velocity
        self.goal_accomplished = False
        self.goal_non_accomplished = False
        self.check_absor = False
        self.TEST70 = False
        self.keep_absorbing=False
        self.new_reward=False
        self.goal = np.array([0.974, 0])
        self.step_r=False
        self.Original=False
        self.absorbing_bonus = False
        self.episode_steps = 0
        self.out_task=False
        self.kepp_training=False
        self.set_step_puck=0
        self.split=50
        self.Add_rp=False
        self.target_y=0.0
        self.mask_inf = 0
        self.mask_sup = 5
        self.puck_theta = 0.0
        # self.Ad_T = False
        # self.add_T=False

    def _modify_mdp_info(self, mdp_info):
        mdp_info = super()._modify_mdp_info(mdp_info)
        obs_low = np.concatenate([mdp_info.observation_space.low, [-0.6,-0.4,-np.pi]])
        obs_high = np.concatenate([mdp_info.observation_space.high, [1.,0.4,np.pi]])
        mdp_info.observation_space = Box(obs_low, obs_high)
        return mdp_info
        
    
    def _modify_observation(self, obs):
        obs =  super()._modify_observation(obs)
        obs = np.concatenate([obs, [self.goal[0],self.goal[1],self.puck_theta]])
        return obs
        

    def setup(self, state=None):
        # Initial position of the puck
        if self.random_init:
            # puck_pos = np.random.rand(2) * (self.hit_range[:, 1] - self.hit_range[:, 0]) + self.hit_range[:, 0]
            puck_pos=[]
            puck_pos.append(np.random.uniform(self.hit_range[0][0],self.hit_range[0][1],size=1)[0])
            puck_pos.append(np.random.uniform(self.hit_range[1][0],self.hit_range[1][1],size=1)[0])
            reduction=((2*(0.35-(self.env_info['table']['goal_width']/2)))/40)*self.Idx
            if 20>self.Idx>0:
                self.target_range = np.array([-0.35+(self.env_info['table']['goal_width']/2)+ reduction, 0.35-(self.env_info['table']['goal_width']/2)-reduction])  # Target X Frame
            elif self.Idx>=20:
                self.target_range=np.array([0.0,0.0])
            target_y= np.random.uniform(self.target_range[0],self.target_range[1])
            # print(target_y,self.target_range, self.Idx)
            # target_y= np.random.choice(np.linspace(self.target_range[0],self.target_range[1],round(Samples)+1))
            # print("hit_range check: ", self.hit_range)
            # print("puck_pos: ", puck_pos)
        else:
            puck_pos = np.random.rand(2) * (self.hit_range[:, 1] - self.hit_range[:, 0]) + self.hit_range[:, 0]

        # self.init_state = np.array([-0.9273, 0.9273, np.pi / 2])
      
        self._write_data("puck_x_pos", puck_pos[0])
        self._write_data("puck_y_pos", puck_pos[1])
        self.set_distribution(target_y)
        self.set_step_puck= self.episode_steps

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
        final_goal_pos =  np.array([0.974, 0])
        self.episode_steps += 1

        # If puck is out of bounds
        if self.goal_accomplished:
            # If puck is in the opponent goal
            if puck_pos[:1] >=self.goal[:1]  and \
                np.linalg.norm(puck_pos[1:2]-self.goal[1:2])<= self.env_info['table']['goal_width']/2.0:
                if  self.absorbing_bonus:
                    r = 0
                else:
                    r = 0
                # print("GOAL !!!!!!")
        elif self.check_absor and self.keep_absorbing==False:
            r = -30
            self.keep_absorbing = True           
            # print("Absorbing  !!")
        elif self.check_absor and self.keep_absorbing:
            r =-3
            # print("out")
        elif self.Add_rp:
            r = -((np.linalg.norm(puck_pos[:2] - self.goal))/(np.linalg.norm(puck_pos_initial - final_goal_pos)))     
            r=(r/(1-self.gamma))
            self.Add_rp=False
        else:
            r = -((np.linalg.norm(puck_pos[:2] - self.goal))/(np.linalg.norm(puck_pos_initial - final_goal_pos)))
            # pass

        # if (self.goal_accomplished == False) and (self.goal_non_accomplished== False) :
            # r = -((np.linalg.norm(puck_pos[:2] - self.goal))/(np.linalg.norm(puck_pos_initial - final_goal_pos)))
        # print("reward ",r)
        
        return r
    


    def is_absorbing(self, obs):
        boundary = np.array([self.env_info['table']['length'], self.env_info['table']['width']]) / 2
        # puck_pos = self.obs_helper.get_from_obs(obs, "puck_pos")
        # puck_vel = self.obs_helper.get_from_obs(obs, "puck_vel")
        puck_pos, puck_vel = self.get_puck(obs)

        # Add absorbing state when the mallet is outside the table using the ee position
        #q_pos = self.obs_helper.get_joint_pos_from_obs(obs)
        q_pos, q_vel = self.get_joints(obs)

        #x_pos, _ = forward_kinematics(self.robot_model, self.robot_data, q_pos)
        x_pos, rot_mat = forward_kinematics(self.env_info['robot']['robot_model'],
                                             self.env_info['robot']['robot_data'], q_pos)
        ee_pos = x_pos+self.env_info['robot']['base_frame'][0][:3,3]
        # yaw = -math.atan2(rot_mat[0][0], rot_mat[1][0])
        self.puck_theta=math.atan((puck_pos[0]-ee_pos[0])/(puck_pos[1]-ee_pos[1]))
        # yaw1 = -math.atan2(rot_mat[0][1], rot_mat[1][0])
        # yaw2 = -math.atan2(rot_mat[0][1], rot_mat[1][1])
        # pitch = math.acos(rot_mat[2][2])
        # print(math.degrees(yaw),math.degrees(yaw1),math.degrees(yaw2))
        print(math.degrees(self.puck_theta))
        # Translate to table space -1.12598753e+00  9.53749014e-01 -2.77555756e-17
        #ee_pos = x_pos + self.agents[0]["frame"][:3, 3] 

        # Check if the puck is outside the table
        puck_out = np.any(np.abs(puck_pos[:2])>boundary)

        # Check if the mallet is outside the table
        mallet = np.any(np.abs(np.asarray(ee_pos[:2]))>boundary)


        # Check if the puck achieves the goal
        reach_goal_x = puck_pos[:1] >=(self.goal[:1])
        # reach_goal_x = puck_pos[:1] >=(self.goal[:1] - 1*self.env_info['puck']['radius'])
        reach_goal_y = np.linalg.norm(puck_pos[1:2]-self.goal[1:2]) <=  self.env_info['table']['goal_width']/2.0
        
        # Check if the puck is out of the absorbing line 
        out_goal_y = np.linalg.norm(puck_pos[1:2]-self.goal[1:2]) > self.env_info['table']['goal_width']/2.0
        out_goal_x = puck_pos[:1] >= (self.env_info['table']['length']/2 - 2*self.env_info['puck']['radius'])
        
        # Check if the puck is moving too fast
        velocity = np.linalg.norm(puck_vel) > 100
        # reach_goal_y = np.linalg.norm(puck_pos[1:2]-self.goal[1:2])<= self.env_info['table']['goal_width']
        # print(puck_out,mallet)
        # Check if the puck is moving too fast

        if self.Original:
            if reach_goal_x and reach_goal_y:
                self.goal_accomplished = True
                self.goal_non_accomplished = False
                self.check_absor= False
                self.keep_absorbing = False                      
                return True

            elif out_goal_y and (out_goal_x or reach_goal_x):
                self.goal_accomplished = False
                self.goal_non_accomplished = True
                self.check_absor= False
                self.keep_absorbing = False 
                self.Add_rp=True
                return True

            elif mallet or velocity:
                self.goal_non_accomplished = True
                self.goal_accomplished = False
                self.check_absor= True
                if puck_out:
                    return True
                else:
                    return True
            else:
                self.goal_accomplished = False
                if (self.episode_steps-self.set_step_puck==119):
                    self.goal_non_accomplished = True
                else:
                    self.goal_non_accomplished = False
                # self.goal_non_accomplished = False
                self.keep_absorbing = False 
                self.check_absor= False
                return False
            
        elif  self.absorbing_bonus:
            if reach_goal_x and reach_goal_y:
                self.goal_accomplished = True
                self.goal_non_accomplished = False
                self.check_absor = False
                self.keep_absorbing = False  
                return True     
            elif out_goal_y and (out_goal_x or reach_goal_x):
                self.goal_accomplished = False
                self.goal_non_accomplished = True
                self.check_absor= False
                self.keep_absorbing = False
                self.Add_rp=True

                return True
            elif mallet or velocity:
                if self.out_task:
                    if (self.episode_steps-self.set_step_puck==119):
                        self.goal_non_accomplished = True
                    else:
                      self.goal_non_accomplished = False
                else:
                    self.goal_non_accomplished = False
                self.goal_accomplished = False
                self.check_absor= True

                if puck_out:
                    return True
                else:
                    return False
            else:
                self.goal_accomplished = False
                if self.out_task:
                    if (self.episode_steps-self.set_step_puck==119):
                        self.goal_non_accomplished = True
                    else:
                        self.goal_non_accomplished = False
                else:
                    if (self.episode_steps%self.info.horizon==0):
                        self.goal_non_accomplished = True
                    else:
                        self.goal_non_accomplished = False
                self.keep_absorbing = False 
                self.check_absor= False
                return False
                  
        else:
            if reach_goal_x and reach_goal_y:
                self.goal_accomplished = True
                self.goal_non_accomplished = False
                self.check_absor= False
                self.keep_absorbing = False                      
                return True
            elif out_goal_y and (out_goal_x or reach_goal_x):
                self.goal_accomplished = False
                self.goal_non_accomplished = True
                self.check_absor= False
                self.keep_absorbing = False 
                return True               
            elif mallet or velocity:
                if (self.episode_steps-self.set_step_puck==119):
                    self.goal_non_accomplished = True
                else:
                    self.goal_non_accomplished = False
                self.goal_accomplished = False
                self.check_absor= True
                if puck_out:
                    return True
                else:
                    return False
            else:
                self.goal_accomplished = False
                if (self.episode_steps-self.set_step_puck==119):
                    self.goal_non_accomplished = True
                else:
                    self.goal_non_accomplished = False
                self.keep_absorbing = False 
                self.check_absor= False
                return False
           

    def set_puck(self, inc_x=0.0):
        inc_y = inc_x/2
        inc_r = inc_y
        if inc_x>0.40 :
            inc_x = 0.40 #30
        if inc_y>0.30 :
            inc_y = 0.30  
        if inc_r>0.10 :
            inc_r = 0.10
        if self.kepp_training:
            self.hit_range = np.array([[-0.65, -0.25], [-0.35, 0.35]])
        else:
            self.hit_range = np.array([[-0.65, -0.65+inc_x], [-0.05-inc_y, 0.05+inc_y]])
            
        self._model.site_pos[3]  = np.concatenate(((self.hit_range[0][1]+self.hit_range[0][0])/2.0,0.0,0.005), axis=None)
        self._model.site_size[3] = np.asarray([np.abs(self.hit_range[0][0]-self.hit_range[0][1])/2.0,np.abs(self.hit_range[1][1]-self.hit_range[1][0])/2.0,  -0.001])
        self._model.site_rgba[3] = np.asarray([0.0,30, 30, 0.3 ])
        return self.hit_range 
    

    def get_distribution(self,value=0):
        
        if value>self.split:
            value = self.split
        length        = self.env_info['table']['length'] - 4 * (self.env_info['puck']['radius']+self.env_info['mallet']['radius']) -0.35
        Length_dist   = length/self.split
        initial_value = - self.env_info['table']['length'] / 2 + 4 * (self.env_info['puck']['radius']+self.env_info['mallet']['radius'])+0.35          
        if self.kepp_training:
            self.goal     = np.array([initial_value+(value*Length_dist),0.0])
        else:
            self.goal     = np.array([initial_value+(value*Length_dist),self.target_y])
        self.set_puck(value*Length_dist)
        if value>30:
            self.Idx+=1

        
        print(self.goal)
        return self.goal
    
    def set_distribution(self,value=0):
        self.goal[1]=value 
        print(self.goal)
        self._model.site_pos[2]  = np.concatenate((self.goal,0.0), axis=None)
        return self.goal

    def _create_info_dictionary(self, obs):
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
