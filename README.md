# MaDRLAM
The folder consists of all source codes for the paper titled "Multi-Agent Deep Reinforcement Learning for Task Offloading in Group Distributed Manufacturing Systems."
The original manuscript will be available for public readers after its acceptance.

The key files of the codes are described as follows.
The file 'data2' covers the training data set and test data set used in each comparison experiment.
The file 'act_critic.py' covers the two agents proposed in the article (Task Selection Agent and Computing Node Selection Agent). The collaboration of the two agents is also represented in the file 'act_critic.py'. 
The file 'Datageneration.py' and the file 'seed.py' are used to generate training and testing datasets.
The file 'ceenv.py' corresponds to the Environment in Figure 1. The update and extraction of features are also integrated into the file 'ceenv.py'
The file 'params.py' defines some parameters of the model.
Run the file 'main.py' to train the model. 
After the model is trained, load the trained model parameters through the file 'vali.py' to test the training effect.
Due to the limination of the file size, "Data2" folder can be accessed via the following link. "https://www.dropbox.com/scl/fo/b3yywug4xr8xv0wj0jzvk/h?dl=0&rlkey=hxnrb3zwqac8zrnsmkdmhewgq"
