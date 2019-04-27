#pragma once

#include "network.h"

int selfplay(std::shared_ptr<FIRNet> net, DataSet &dataset, int itermax);
void train(std::shared_ptr<FIRNet> net);
