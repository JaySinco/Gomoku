#!/bin/sh
source /opt/rh/devtoolset-7/enable
export LD_LIBRARY_PATH=/usr/local/lib/python3.6/site-packages/mxnet:$LD_LIBRARY_PATH
g++ -L/usr/local/lib/python3.6/site-packages/mxnet -Iinclude -w -std=c++11 -lmxnet -O3 -DNDEBUG src/game.cc src/network.cc src/mcts.cc src/train.cc src/main.cc -o gomoku
