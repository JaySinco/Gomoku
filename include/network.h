#include <mxnet-cpp/MxNetCpp.h>
#include "engine.h"

using namespace mxnet::cpp;

static Symbol dense_layer(const std::string &name, Symbol data,
	int num_hidden, const std::string &act_type);

static Symbol convolution_layer(const std::string &name, Symbol data,
	int num_filter, Shape kernel, Shape stride, Shape pad,
	bool use_act, bool use_bn);

static Symbol residual_layer(const std::string &name, Symbol data, 
	int num_filter);

static Symbol residual_block(const std::string &name, Symbol data,
	int num_block, int num_filter);

class FIRNet {
public:
	int num_filter;
	int num_residual_block;
	const Context ctx;
	std::map<std::string, NDArray> args_map;
	Symbol plc, val, ploss, vloss, loss;
public:
	Symbol middle_layer(Symbol data);
	std::pair<Symbol, Symbol> plc_layer(Symbol data, Symbol label);
	std::pair<Symbol, Symbol> val_layer(Symbol data, Symbol label);
public:
	FIRNet(const std::string &param_file = "None", int filter=64, int res_block=5, int batch_size=8);
	void save_parameters(const std::string &file_name);
	void forward(const State &state, float value[1], float policy[BOARD_SIZE]);
};