#pragma once

#include <iomanip>
#include <string>
#include <sstream>

#include "mxnet-cpp/MxNetCpp.h"
#include "game.hpp"

struct Player {
	Player() {}
	virtual void reset() = 0;
	virtual const std::string &name() const = 0;
	virtual Move play(const State &state) = 0;
	virtual ~Player() {};
};

inline Player &play(Player &p1, Player &p2, bool silent=true) {
	const std::map<Color, Player*> player_color {
		{Color::Black, &p1}, 
		{Color::White, &p2}, 
		{Color::Empty, nullptr}
	};
	auto game = State();
	p1.reset();
	p2.reset();
	int turn = 0;
	while (!game.over()) {
		auto player = player_color.at(game.current());
		auto act = player->play(game);
		game.next(act);
		++turn;
		if (!silent) std::cout << game << std::endl;
	}	
	auto winner = player_color.at(game.get_winner());
	if (!silent) std::cout << "winner: " 
		<< (winner == nullptr ? "no winner, even!" : winner->name()) << std::endl;
	return *winner;
}

inline void benchmark(Player &p1, Player &p2, int round=100, bool silent=true) {
	assert(round > 0);
	int p1win = 0, p2win = 0, even = 0;
	Player *temp = nullptr, *pblack = &p1, *pwhite = &p2;
	for (int i = 0; i < round; ++i) {
		temp = pblack, pblack = pwhite, pwhite = temp;
		Player *winner = &play(*pblack, *pwhite);
		if (winner == nullptr)
			++even;
		else if (winner == &p1)
			++p1win;
		else {
			assert(winner == &p2);	
			++p2win;
		}
		if (!silent) {
			std::cout << std::setfill('0') 
				<< "\rscore: total=" << std::setw(4) << i+1 << ", " 
				<< p1.name() << "=" << std::setw(4) << p1win << ", "
				<< p2.name() << "=" << std::setw(4) << p2win;
			std::cout.flush();
		}
	}
	if (!silent) { std::cout << std::endl; }
	float p1prob = float(p1win) / float(round);
	float p2prob = float(p2win) / float(round);
	float eprob = float(even) / float(round);
	std::cout << "benchmark player win probality: " << p1.name() << "=" << p1prob << ", "
		<< p2.name() << "=" << p2prob << ", even=" << eprob << ", sim=" << round << std::endl;
}

class RandomPlayer: public Player {
	std::string id;
public:
	RandomPlayer(const char *name) : id(name) {}
	void reset() override {}
	const std::string &name() const override { return id; }
	Move play(const State &state) override { return state.get_options()[0]; }
	~RandomPlayer() {};
};

class HumanPlayer: public Player {
	std::string id;
	bool get_move(int &row, int &col) {
		std::string line, srow;
		if (!std::getline(std::cin, line))
			return false;
		std::istringstream line_stream(line);
		if (!std::getline(line_stream, srow, ',') || !(line_stream >> col))
			return false;
		std::istringstream row_stream(srow);
		if (!(row_stream >> row))
			return false;
		return true;
	}
public:
	HumanPlayer(const char *name) : id(name) {}
	void reset() override {}
	const std::string &name() const override { return id; }
	Move play(const State &state) override {
		int col, row;
		while (true) {
			std::cout << state.current() << "(" << id << "): ";
			std::cout.flush();
			if (get_move(row, col)) {
				auto mv = Move(row, col);
				if (state.valid(mv))
					return mv;
			}
		}	
	}
	~HumanPlayer() {};		
};

class MCTSPurePlayer: public Player {
	std::string id;
	int itermax;
	float c_puct;
	MCTSNode *root;
	void swap_root(MCTSNode * new_root) {
		delete root;
		root = new_root;
	}
public:
	MCTSPurePlayer(const char *name, int itermax=10000, float c_puct=5.0f)
		: id(name), itermax(itermax), c_puct(c_puct){
		root = new MCTSNode(nullptr, 1.0f);
	}
	~MCTSPurePlayer() { delete root; }
	const std::string &name() const override { return id; }
	void reset() override {
		delete root;
		root = new MCTSNode(nullptr, 1.0f);
	}
	Move play(const State &state) override {
		if (!(state.get_last().z() == NO_MOVE_YET) && !root->is_leaf())
			swap_root(root->cut(state.get_last()));
		for (int i = 0; i < itermax; ++i) {
			State state_copied(state);
			MCTSNode *node = root;
			while (!node->is_leaf()) {
				auto move_node = node->select(c_puct);
				node = move_node.second;
				state_copied.next(move_node.first);
			}
			Color enemy_side = state_copied.current();
			Color winner = state_copied.get_winner();
			if (winner == Color::Empty) {
				int n_options = state_copied.get_options().size();
				if (n_options > 0) {
					std::vector<std::pair<Move, float>> move_priors;
					for (const auto mv : state_copied.get_options()) {
						move_priors.push_back(std::make_pair(mv, 1.0f/float(n_options)));
					}
					node->expand(move_priors);
				}
				winner = state_copied.next_rand_till_end();
			}
			float leaf_value;
			if (winner == enemy_side) 
				leaf_value = -1.0f;
			else if (winner == ~enemy_side) 
				leaf_value = 1.0f;
			else 
				leaf_value = 0.0f;;
			node->update_recursive(leaf_value);
		}
		Move act = root->most_visted();
		swap_root(root->cut(act));
		return act;
	}
};

//class MCTSDeepPlayer : public Player {
//	std::string id;
//	int itermax;
//	float c_puct;
//	MCTSNode *root;
//	void swap_root(MCTSNode * new_root) {
//		delete root;
//		root = new_root;
//	}
//public:
//	MCTSDeepPlayer(const char *name, int itermax = 10000, float c_puct = 5.0f)
//		: id(name), itermax(itermax), c_puct(c_puct) {
//		root = new MCTSNode(nullptr, 1.0f);
//	}
//	~MCTSDeepPlayer() { delete root; }
//}

class DeepNet {
public:
	using Symbol = mxnet::cpp::Symbol;
	using Shape = mxnet::cpp::Shape;
	using NDArray = mxnet::cpp::NDArray;
	using Executor = mxnet::cpp::Executor;
	using Context = mxnet::cpp::Context;

	int num_filter;
	int num_residual_block;
	int batch_size;

	const Context &ctx;
	std::map<std::string, NDArray> args_map;
	Executor *plc_eval, *val_eval, *loss_eval;

	DeepNet(int filter=64, int res_block=5, int batch_size=8) :
		    num_filter(filter), num_residual_block(res_block), batch_size(batch_size),
			ctx(Context::cpu()){

		auto middle = middle_layer(Symbol::Variable("data"));
		auto plc_pair = plc_layer(middle, Symbol::Variable("plc_label"));
		auto val_pair = val_layer(middle, Symbol::Variable("val_label"));
		Symbol plc = plc_pair.first;
		Symbol val = val_pair.first;
		Symbol loss = plc_pair.second + val_pair.second;

		args_map["data"] = NDArray(Shape(batch_size, 2, BOARD_MAX_ROW, BOARD_MAX_COL), ctx);
		args_map["plc_label"] = NDArray(Shape(batch_size), ctx);
		args_map["val_label"] = NDArray(Shape(batch_size, 1), ctx);
		loss.InferArgsMap(ctx, &args_map, args_map);

		plc_eval = plc.SimpleBind(ctx, args_map);
		val_eval = val.SimpleBind(ctx, args_map);
		loss_eval = loss.SimpleBind(ctx, args_map);
	}
	~DeepNet() {
		delete plc_eval;
		delete val_eval;
		delete loss_eval;
	}
	void load_params() {
	}
	Symbol middle_layer(Symbol data) {
		Symbol front_conv = convolution_layer("front_conv", data, 
			num_filter, Shape(3, 3), Shape(1, 1), Shape(1, 1), true, true);
		return residual_block("middle", front_conv, num_residual_block, num_filter);
	}
	std::pair<Symbol, Symbol> plc_layer(Symbol data, Symbol label) {
		Symbol plc_conv = convolution_layer("plc_conv", data,
			2, Shape(1, 1), Shape(1, 1), Shape(0, 0), true, true);
		Symbol plc_logist_out = dense_layer("plc_logist_out", plc_conv, BOARD_SIZE, "None");
		Symbol plc_out = softmax("plc_out", plc_logist_out);
		Symbol plc_loss = softmax_cross_entropy("plc_loss", plc_logist_out, label);
		return std::make_pair(plc_out, plc_loss);
	}
	std::pair<Symbol, Symbol> val_layer(Symbol data, Symbol label) {
		Symbol val_conv = convolution_layer("val_conv", data,
			1, Shape(1, 1), Shape(1, 1), Shape(0, 0), true, true);
		Symbol val_dense = dense_layer("val_dense", val_conv, num_filter, "relu");
		Symbol val_out = dense_layer("val_logist_out", val_dense, 1, "tanh");
		Symbol val_loss = sum(square(elemwise_sub("val_loss_sub", val_out, label)));
		return std::make_pair(val_out, val_loss);
	}
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
			Shape(3, 3), Shape(1, 1), Shape(1, 1), true, true);
		Symbol conv2 = convolution_layer(name + "_conv2_layer", conv1, num_filter,
			Shape(3, 3), Shape(1, 1), Shape(1, 1), false, true);
		return Activation("relu_" + name, data + conv2, "relu");
	}
	static Symbol residual_block(const std::string &name, Symbol data, 
		                         int num_block, int num_filter) {
		Symbol out = data;
		for (int i = 0; i < num_block; ++i)
			out = residual_layer(name + "_block" + std::to_string(i + 1), out, num_filter);
		return out;
	}
	static void display_shape(const NDArray &nd) {
		std::cout << "Shape(";
		auto shape = nd.GetShape();
		for (int i = 0; i < shape.size(); ++i) {
			if (i != 0) std::cout << ", ";
			std::cout << shape.at(i);
		}
		std::cout << ")";
	}
	static void display_dict(const std::string &name, std::map<std::string, NDArray> &dict) {
		std::cout << "******* " << name << " *******" << std::endl;
		for (const auto &pair : dict) {
			std::cout << pair.first << " => ";
			display_shape(pair.second);
			std::cout << std::endl;
		}
	}
};