Tricks from Slack:

1. I did the "take off" task. So Some things that were helpful to me in my reward function.  I heard of other students using trigonometric identities.  I wondered why. Well the angles we are given for the euler angles are in radians. But, the radian values themselves are the distance in radi of the arc off the axis not the straight line distance.  For that you need the sine or cosine of the radians.  The explanation for this is at Khan academy here: https://www.khanacademy.org/math/trigonometry/unit-circle-trig-func/modal/v/trigonometry-unit-circle-symmetry        I also found it helpful too look at other DDPG projects online (Like the DDPG Torcs project) to get input to my reward function.  Larger numbers in the hidden layers help a lot, as does paying attention to your soft update param.  Too small and you won't make changes fast enough, too big and the movements will be erratic.     Also, someone mentioned if you are doing the take off task, not to make the starting position 0.  Your agent can only make 3 or 4 steps before the Sim considers it "crashed" and ends the episode.  Start at 10 and make it go to 20 and your agent will have more time (steps) to learn.    When my agent finally started learning, it was at about 900 episodes in and I ran anywhere from 1500-2000 episodes per run.  The rewards plot was steadily increasing, but the x, y, z plot looks like it finally learned at about 900 episodes.  I also lowered the learning rate of the actor below the default adam rate.   Hope that helps.

2. I just had some first non-frustrating experience with my DDPG agent learning the OpenAI pendulum and thought I'd share it here in case someone wants to try their agent on a simple task first.
I've basically used the hyper params from the [DDPG paper](https://arxiv.org/pdf/1509.02971.pdf). Especially the large hidden layers (400 and 300 units) seemed to help a lot. Previously I was using much smaller networks sizes because it seemed intuitive regarding the small state and action spaces. So I encourage everyone to try an OpenAI task first as -- at least for me -- it really raises the spirits and helps overcome frustration.


3. Thanks for all the hints! Yes, I am taking care of the mod 2pi in the Euler angles. In debugging my status is now this: Constraining the rotor speeds to be all the same, the network can easily learn to take off. So I am focusing now on the problem of the rotor speeds going to maximum or minimum if not constrained. I really would like to understand why that happens. For debugging purposes I now let my quadcopter start at a very high position and my reward function is to punish for high angular velocities. I do not bother about hovering or whatever, I just would like to see that the network is able to train towards equal rotor speeds. My reward function should do that, shouldn't it? But however I manipulate the reward function and the hyperparameters, still the rotor speeds diverge and the angular velocities increase to ridiculous values. So either I am missing some fundamental understanding, or there is some bug in my code

4. #add small bias to allow for small euler angles
       euler_bias = 10
       Eulers_angle_penalty = abs(self.sim.pose[3:] - self.target_pos[3:]).sum() - euler_bias
       
       # Reward based on how close we are to our designated coordinate z. This should be our main objective
       z_reward = abs(self.sim.pose[2] - self.target_pos[2])
                     
       # Reward agent for minimaly straying or not moving from x,y axis (more stable take off)
       other_reward = abs(self.sim.pose[:2] - self.target_pos[:2]).sum()                    
                     
       penalties = (-.0003*(other_reward) - .0009*(z_reward) - .0003*(Eulers_angle_penalty))/3
         
       reward =   1 + penalties # penalties should be a negative number  # add 1 for every second flying

5. TORCS code
https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html
https://gist.github.com/ctmakro/2e0017f25b06177af539dd410b946428

6. https://mpatacchiola.github.io/blog/2017/08/14/dissecting-reinforcement-learning-6.html

7. https://keon.io/deep-q-learning/

8. https://blog.openai.com/faulty-reward-functions/

9. https://github.com/letsgogeeky/QuadCopter-RL
https://github.com/racersmith/RL_Quadcopter_2
https://github.com/delinhabit/nd101-quadcopter-RL
https://github.com/domangi/rl-quadcopter/blob/master/Quadcopter_Project.ipynb
