#include "network.h"

using namespace mxnet::cpp;

static Symbol dense_layer(const std::string &name, Symbol data,
	int num_hidden, const std::string &act_type) {
	Symbol w(name + "_w"), b(name + "_b");
	Symbol out = FullyConnected("fc_" + name, data, w, b, num_hidden);
	if (act_type != "None")
		out = Activation("act_" + name, out, act_type);
	return out;
}

static Symbol convolution_layer(const std::string &name, Symbol data,
	int num_filter, Shape kernel, Shape stride, Shape pad,
	bool use_act, bool use_bn) {
	Symbol conv_w(name + "_w");
	Symbol conv_b(name + "_b");
	Symbol out = Convolution("conv_" + name, data,
		conv_w, conv_b, kernel, num_filter, stride, Shape(1, 1), pad);
	if (use_bn) {
		Symbol gamma(name + "_bn_gamma");
		Symbol beta(name + "_bn_beta");
		Symbol mmean(name + "_bn_mmean");
		Symbol mvar(name + "_bn_mvar");
		out = BatchNorm("bn_" + name, out, gamma, beta, mmean, mvar);
	}
	if (use_act)
		out = Activation("relu_" + name, out, "relu");
	return out;
}

static Symbol residual_layer(const std::string &name, Symbol data,
	int num_filter) {
	Symbol conv1 = convolution_layer(name + "_conv1_layer", data, num_filter,
		Shape(3, 3), Shape(1, 1), Shape(1, 1), true, false);
	Symbol conv2 = convolution_layer(name + "_conv2_layer", conv1, num_filter,
		Shape(3, 3), Shape(1, 1), Shape(1, 1), false, false);
	return Activation("relu_" + name, data + conv2, "relu");
}

static Symbol residual_block(const std::string &name, Symbol data,
	int num_block, int num_filter) {
	Symbol out = data;
	for (int i = 0; i < num_block; ++i)
		out = residual_layer(name + "_block" + std::to_string(i + 1), out, num_filter);
	return out;
}

Symbol FIRNet::middle_layer(Symbol data) {
	Symbol front_conv = convolution_layer("front_conv", data,
		num_filter, Shape(3, 3), Shape(1, 1), Shape(1, 1), true, false);
	return residual_block("middle", front_conv, num_residual_block, num_filter);
}

std::pair<Symbol, Symbol> FIRNet::plc_layer(Symbol data, Symbol label) {
	Symbol plc_conv = convolution_layer("plc_conv", data,
		2, Shape(1, 1), Shape(1, 1), Shape(0, 0), true, false);
	Symbol plc_logist_out = dense_layer("plc_logist_out", plc_conv, BOARD_SIZE, "None");
	Symbol plc_out = softmax("plc_out", plc_logist_out);
	Symbol plc_m_loss = elemwise_mul(label, log_softmax(plc_logist_out));
	Symbol plc_loss = mean(sum(plc_m_loss, dmlc::optional<Shape>(Shape(1))));
	return std::make_pair(plc_out, plc_loss);
}

std::pair<Symbol, Symbol> FIRNet::val_layer(Symbol data, Symbol label) {
	Symbol val_conv = convolution_layer("val_conv", data,
		1, Shape(1, 1), Shape(1, 1), Shape(0, 0), true, false);
	Symbol val_dense = dense_layer("val_dense", val_conv, num_filter, "relu");
	Symbol val_out = dense_layer("val_logist_out", val_dense, 1, "tanh");
	Symbol val_loss = mean(square(elemwise_sub("val_loss_sub", val_out, label)));
	return std::make_pair(val_out, val_loss);
}

FIRNet::FIRNet(const std::string &param_file, int filter, int res_block, int batch_size) :
		num_filter(filter), num_residual_block(res_block), ctx(Context::cpu()),
		data_predict(NDArray(Shape(1, 2, BOARD_MAX_ROW, BOARD_MAX_COL), ctx)),
		data_train(NDArray(Shape(batch_size, 2, BOARD_MAX_ROW, BOARD_MAX_COL), ctx)),
		plc_label(NDArray(Shape(batch_size, BOARD_SIZE), ctx)),
		val_label(NDArray(Shape(batch_size, 1), ctx)) {
	auto middle = middle_layer(Symbol::Variable("data"));
	auto plc_pair = plc_layer(middle, Symbol::Variable("plc_label"));
	auto val_pair = val_layer(middle, Symbol::Variable("val_label"));
	plc = plc_pair.first;
	val = val_pair.first;
	loss = plc_pair.second + val_pair.second;
	if (param_file != "None") {
		LOG(INFO) << "loading parameters from " << param_file << std::endl;
		NDArray::Load(param_file, nullptr, &args_map);
	}
	args_map["data"] = data_train;
	args_map["plc_label"] = plc_label;
	args_map["val_label"] = val_label;
	if (param_file == "None") {
		loss.InferArgsMap(ctx, &args_map, args_map);
		Xavier xavier = Xavier(Xavier::gaussian, Xavier::in, 2);
		for (auto &arg : args_map) {
			xavier(arg.first, &arg.second);
		}
	}
	loss_train = loss.SimpleBind(ctx, args_map);
	args_map["data"] = data_predict;
	plc_predict = plc.SimpleBind(ctx, args_map);
	val_predict = val.SimpleBind(ctx, args_map);
	args_map.erase("data");
	args_map.erase("plc_label");
	args_map.erase("val_label");
}

FIRNet::~FIRNet() {
	delete plc_predict;
	delete val_predict;
	delete loss_train;
}

void FIRNet::forward(const State &state, float data[2 * BOARD_SIZE],
		float value[1], std::vector<std::pair<Move, float>> &move_priors) {
	if (data == nullptr) {
		float temp[2 * BOARD_SIZE] = { 0.0f };
		data = temp;
	}
	state.fill_feature_array(data);
	NDArray sample(data, Shape(1, 2, BOARD_MAX_ROW, BOARD_MAX_COL), ctx);
	sample.CopyTo(&data_predict);
	plc_predict->Forward(false);
	val_predict->Forward(false);
	NDArray::WaitAll();
	const float *plc_ptr = plc_predict->outputs[0].GetData();
	for (const auto mv : state.get_options()) {
		move_priors.push_back(std::make_pair(mv, plc_ptr[mv.z()]));
	}
	value[0] = val_predict->outputs[0].GetData()[0];
}

void FIRNet::save_parameters(const std::string &file_name) {
	LOG(INFO) << "saving parameters into " << file_name << std::endl;
	NDArray::Save(file_name, args_map);
}