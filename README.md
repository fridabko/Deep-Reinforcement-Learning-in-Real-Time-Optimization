# Deep-Reinforcement-Learning-in-Real-Time-Optimization

This project was conducted as a part of the course the final project for the Chemical Engineering and Biotechnology Master in the course TKP4900 - “Chemical Process Technology, Master’s Thesis” at the Norwegian University of Science and Technology, NTNU. The project introduces reinforcement learning from a process control perspective and investigates how to apply model-free reinforcement learning in static real-time optimization. Deep deterministic policy gradient (DDPG) algorithm is used to solve the Williams-Otto reactor.

# Abstract from thesis

Reinforcement learning (RL) is a machine learning field attracting attention for its ability to solve complex problems. The fundamental idea is learning through trial-and-error, where the problem is formulated as a Markov decision process (MDP). It can be seen as an optimization tool, where the best decisions are chosen to fulfill a long-term goal. Deep RL is RL in combination with deep learning, for which high dimensional and continuous problems can be solved using RL. Model-free RL algorithms do not require any process model. A challenge is that they suffer from computational issues and sample inefficiency. 

Real-time optimization (RTO) ensures that process plant operation is continuously optimized to the economic optimum by solving a steady-state optimization problem. Developing an accurate process model to use in RTO can be challenging in chemical process plants. Therefore, the use of RL in RTO can eliminate the need for a process model. However, the steady-state RTO problem differs from a typical RL problem. Formulating it as an MDP is essential to enable the use of RL as an optimization method in RTO. 

This project contributes to a detailed description and discussion on how to formulate a steady-state optimization problem as an MDP from first principles, assuming minimal prior knowledge in RL. Further, it shows how to utilize the model-free RL algorithm deep deterministic policy gradient (DDPG) to solve the problem with the Python RL tool Stable Baselines 3. A version of RTO called modifier adaptation (MA) is introduced as an example of a conventional RTO alternative. It is used as a base case for comparison with the RL-RTO, and both schemes are applied to the Williams-Otto reactor as a case study for implementation. The results prove that the steady-state optimization problem can be solved using RL. However, RL-RTO faces challenges with sample efficiency and constraint violation that must be addressed for real-life implementation. Finally, measures to overcome the challenges and a discussion of RL-RTO's potential as a subject for further research are presented.
<img width="279" alt="WOrx" src="https://user-images.githubusercontent.com/94930940/172698624-824647a2-c1ff-4c9e-b139-9d0664427b80.png">

# Resources 

Open AI Gym was used to generate the reinforcement learning environment for the reactor and DDPG from Stable Baselines 3 were used. The algorithms TD3 and SAC was also tested on the environment. 

Running the files require that following installations:
Open AI Gym - https://robots.uc3m.es/installation-guides/install-openai-gym.html
Stable Baslines 3 - https://github.com/DLR-RM/stable-baselines3

