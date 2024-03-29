import logging
import random
import numpy as np  
import gym 


logger=logging.getLogger(__name__)
class GridEnv(gym.Env):
    metadata={
        'render.modes':['human','rgb_array'],
        'video.frames_per_second':2

    }
    def __init__(self):
        #机器人初始化状态
        self.x=[140,220,300,380,460,140,300,460]
        self.y=[250,250,250,250,250,150,150,150]
        #终止状态
        self.terminate_states=dict()
        self.terminate_states[6]=1
        self.terminate_states[7]=1
        self.terminate_states[8]=1
        #状态空间
        self.states=[1,2,3,4,5,6,7,8]
        #动作空间
        self.actions=['n','e','w','s']
        #回报函数
        self.rewards=dict()
        self.rewards['1_s']=-1.0
        self.rewards['3_s']=1.0
        self.rewards['5_s']=-1.0
        #状态转移概率
        self.t=dict()
        self.t['1_e']=2
        self.t['1_s']=6
        self.t['2_w']=1
        self.t['2_e']=3
        self.t['3_w']=2
        self.t['3_e']=4
        self.t['3_s']=7
        self.t['4_w']=3
        self.t['4_e']=5
        self.t['5_w']=4
        self.t['5_s']=8

        #折扣因子
        self.gamma=0.8
        #显示器
        self.viewer=None
        #状态
        self.state=None
        #在类对象内部访问实例属性
        #获取终止状态
    #返回下一步状态、立即回报和状态转移概率
    def transform(self,state,action):
        #遍历动作空间，所以状态有可能不存在，设不存在的状态为-1
        s=-1
        r=0
        key='%i_%s'%(state,action)
        if key in self.rewards:
            r=self.rewards[key]
        if key in self.t:
            s=self.t[key]
        return self.t,s,r
            
    def getTerminal(self):
        return self.terminate_states
    #获取状态空间
    def getStates(self):
        return self.states
    #获取动作空间
    def getActions(self):
        return self.actions
    #获取折扣因子
    def getGamma(self):
        return self.gamma
    #定义reset()函数
    def reset(self):
        self.state=self.states[int(random.random()*len(self.states))]
        print("hello world")
        return self.state

    #定义step():扮演物理引擎的角色，物理引模拟环境中物体的运动规律
    def step(self,action):
        #系统当前状态
        state=self.state
        #判断当前状态是否处于终止状态
        if state in self.terminate_states:
            return state,0,True,{}
        #'定义的格式化字符串'%实际值
        #当定义的格式化字符串中包含两个以上占位符时，必须将所有实际值封装在元组中
        key='%i_%s'%(state,action)
        #状态转移
        if key in self.t:
            next_state=self.t[key]
        else:
            next_state=state
        #系统当前状态
        self.state=next_state
        is_terminal=False
        if next_state in self.terminate_states:
            is_terminal=True
        if key not in self.rewards:
            r=0.0
        else:
            r=self.rewards[key]
        return next_state,r,is_terminal,{}
    
    #定义render():扮演图像引擎的角色，图像引擎显示环境中物体的图像

    def render(self,mode='human',close=False):
        if close==True:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer=None
            return
        screen_width=600
        screen_height=400
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer=rendering.Viewer(screen_width,screen_height)
            #创建网格世界，一共11条直线
            self.line1=rendering.Line((100,300),(500,300))
            self.line2=rendering.Line((100,200),(500,200))
            self.line3=rendering.Line((100,100),(100,300))
            self.line4=rendering.Line((180,100),(180,300))
            self.line5=rendering.Line((260,100),(260,300))
            self.line6=rendering.Line((340,100),(340,300))
            self.line7=rendering.Line((420,100),(420,300))
            self.line8=rendering.Line((500,100),(500,300))
            self.line9=rendering.Line((100,100),(180,100))
            self.line10=rendering.Line((260,100),(340,100))
            self.line11=rendering.Line((420,100),(500,100))
            #创建死亡区域
            #画圆，半径为40
            self.kulo1=rendering.make_circle(40)
            #圆心为(140,150)
            #创建第一个骷髅
            self.circletrans=rendering.Transform((140,150))
            self.kulo1.add_attr(self.circletrans)
            self.kulo1.set_color(0,0,0)
            #创建第二个骷髅
            self.kulo2=rendering.make_circle(40)
            self.circletrans=rendering.Transform((460,150))
            self.kulo2.add_attr(self.circletrans)
            self.kulo2.set_color(0,0,0)
            #创建金币区域
            self.gold=rendering.make_circle(40)
            self.circletrans=rendering.Transform((300,150))
            self.gold.add_attr(self.circletrans)
            self.gold.set_color(1,0.9,0)
            #创建机器人
            self.robot=rendering.make_circle(30)
            self.robotrans=rendering.Transform()
            self.robot.add_attr(self.circletrans)
            self.robot.set_color(0.8,0.6,0.4)
            #设置颜色并将对象添加到几何中
            self.line1.set_color(0,0,0)
            self.line2.set_color(0,0,0)
            self.line3.set_color(0,0,0)
            self.line4.set_color(0,0,0)
            self.line5.set_color(0,0,0)
            self.line6.set_color(0,0,0)
            self.line7.set_color(0,0,0)
            self.line8.set_color(0,0,0)
            self.line9.set_color(0,0,0)
            self.line10.set_color(0,0,0)
            self.line11.set_color(0,0,0)
            self.viewer.add_geom(self.line1)
            self.viewer.add_geom(self.line2)
            self.viewer.add_geom(self.line3)
            self.viewer.add_geom(self.line4)
            self.viewer.add_geom(self.line5)
            self.viewer.add_geom(self.line6)
            self.viewer.add_geom(self.line7)
            self.viewer.add_geom(self.line8)
            self.viewer.add_geom(self.line9)
            self.viewer.add_geom(self.line10)
            self.viewer.add_geom(self.line11)
            self.viewer.add_geom(self.kulo1)
            self.viewer.add_geom(self.kulo2)
            self.viewer.add_geom(self.gold)
            self.viewer.add_geom(self.robot)
        if self.state is None:
            return None
        #设置机器人圆心坐标
        self.robotrans.set_translation(self.x[self.state-1],self.y[self.state-1])
        return self.viewer.render('rgb_array')


            
