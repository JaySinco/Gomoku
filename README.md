## Introduction
Gomoku implemention of reinforcement learning by AlphaZero methods using mxnet cpp-package.

## Compile
* On windows, compiled Mxnet shared libray is needed, edit CMakeLists.txt to include and link correct directories.  
* On linux, if you don't bother to compile Mxnet yourself, you can use python `pip install mxnet` to download mxnet, 
then edit build.sh, add library path to LD_LIBRARY_PATH.
* In any way, it is indispensible to include correct header. 
* Pay attention to your c++ compiler version, it must fully support c++11.

## Usage
Enter `gomoku <command>` to see subcommand help in detail.  
These are common Gomoku commands used in various situations:  
```bash
   config     Print global configure  
   train      Train model from scatch or parameter file  
   play       Play with trained model  
   benchmark  Benchmark between two mcts deep players  
```

## Demo
The model supplied has 8x8 board size, 64 filters, 3 residual blocks, 
trained on 1cpu for about 1.5 days, 9336 backward updates to the network.    
![image]( https://github.com/JaySinco/Gomoku/blob/master/play_against_ai.gif)  
Above shows a game played between human(first hand, represented by `x`) and AI(represented by `o`) 
using pretrained model with command `gomoku play 0 9336`.  
