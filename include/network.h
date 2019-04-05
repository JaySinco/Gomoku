#include <mxnet-cpp/MxNetCpp.h>
#include "engine.h"

constexpr int BATCH_SIZE = 8;
constexpr float LEARNING_RATE = 2e-3;

class FIRNet {
	using Symbol = mxnet::cpp::Symbol;
	using Context = mxnet::cpp::Context;
	using NDArray = mxnet::cpp::NDArray;
	using Executor = mxnet::cpp::Executor;
	using Optimizer = mxnet::cpp::Optimizer;

	const Context ctx;
	std::map<std::string, NDArray> args_map;
	std::vector<std::string> loss_arg_names;
	Symbol plc, val, loss;
	NDArray data_predict, data_train, plc_label, val_label;
	Executor *plc_predict, *val_predict, *loss_train;
	Optimizer* optimizer;
public:
	FIRNet(const std::string &param_file = "None");
	~FIRNet();
	void save_parameters(const std::string &file_name);
	void forward(const State &state, float data[2 * BOARD_SIZE], 
		float value[1], std::vector<std::pair<Move, float>> &move_priors);
	void train_step(const float data[BATCH_SIZE * 2 * BOARD_SIZE], 
		const float p_label[BATCH_SIZE * BOARD_SIZE],
		const float v_label[BATCH_SIZE * 1]);
};
