 ----------------- HOW TO USE -----------------

 the experiment.py is running RNN-LSTM model with one layered MLP.
 the data contains 1000 examples (0.8/0.2 train/dev) generated randomly

 optional parameters
 epochs -           default: 50
 learning rate -    default: 0.01

 command:
 python experiment.py
 python experiment.py num_epochs
 python experiment.py num_epochs learning_rate

 example:

 >> python experiment 40 0.005          # run 40 epochs with lr=0.005