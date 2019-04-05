#include <mxnet-cpp/MxNetCpp.h>
#include "engine.h"

class FIRNet {
	using Symbol = mxnet::cpp::Symbol;
	using Context = mxnet::cpp::Context;
	using NDArray = mxnet::cpp::NDArray;
	using Executor = mxnet::cpp::Executor;
public:
	int num_filter;
	int num_residual_block;
	const Context ctx;
	std::map<std::string, NDArray> args_map;
	Symbol plc, val, loss;
	NDArray data_predict, data_train, plc_label, val_label;
	Executor *plc_predict, *val_predict, *loss_train;
public:
	Symbol middle_layer(Symbol data);
	std::pair<Symbol, Symbol> plc_layer(Symbol data, Symbol label);
	std::pair<Symbol, Symbol> val_layer(Symbol data, Symbol label);
public:
	FIRNet(const std::string &param_file = "None", int filter=64, int res_block=5, int batch_size=8);
	~FIRNet();
	void save_parameters(const std::string &file_name);
	void forward(const State &state, float data[2 * BOARD_SIZE], 
		float value[1], std::vector<std::pair<Move, float>> &move_priors);
	void train_step();
};