# IEEE Transactions on Intelligent Transportation Systems (T-ITS)

## Environments

### Windows (10 or 11, both okay)
### anaconda=22.9.0
### python=3.10.12
### sumo=1.20.0 (but, latest version will work)
### tensorflow-gpu=2.9.1

### Check the requirements.txt    

## How to run

### 1. Install Anaconda, sumo, python and all the requirements.
> install the libraries(which are not listed on requirements.txt) imported in the python code.
### 2. Run the 2023_T_ITS_MLSAINT_More.py or 2023_T_ITS_only_SAINT_More.py
> simulation files (Gangnam4.add.xml, Gangnam4.net.xml, Gangnam4.rou.xml, Gangnam4.sumocfg) should be located in same directory with python files.
### 3. Running python file will automatically start the sumo simulator.


## Tips
### 1. If you want train model with your own data, use the code on _model/multi_tensor_big_data_adam.py_
### 2. You can refer the codes in _performance_ directory to draw some graphs. 
