#pragma once

#include "network.h"

template<typename T>
std::string param_file_name(T &suffix) {
	std::ostringstream filename;
	filename << "FIR-" << BOARD_MAX_COL << "x" << BOARD_MAX_ROW << "by" << FIVE_IN_ROW
		<< "_" << suffix << ".param";
	return filename.str();
}

int selfplay(std::shared_ptr<FIRNet> net, DataSet &dataset, int itermax);
void train(std::shared_ptr<FIRNet> net);
