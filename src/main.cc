#include <iostream>

#include "mcts.h"
#include "train.h"

#define EXIT_WITH_USAGE(usage)  { std::cout << usage; return -1; }

const char *usage = "usage: gomoku <command>\n\n"
					"These are common Gomoku commands used in various situations:\n\n"
					"   config     Print global configure\n"
					"   train      Train model from scatch or parameter file\n"
					"   play       Play with trained model\n"
	                "   benchmark  Benchmark between two mcts deep players\n";

const char *train_usage = "usage: gomoku train [net]\n"
                          "   [net]      Suffix of network parameter file name\n"
                          "              If given, continue to train model from last check-point\n";

const char *play_usage = "usage: gomoku play <color> <net> [itermax]\n"
                         "   <color>    [0] if human take first hand, [1] otherwise\n"
                         "   <net>      Suffix of network parameter file name\n"
                         "   [itermax]  itermax for mcts deep player\n"
                         "              If not given, default from global configure\n";

const char *benchmark_usage = "usage: gomoku benchmark <net1> <net2> [itermax]\n"
                              "   <net>      Suffix of network parameter file name\n"
	                          "   [itermax]  itermax for mcts deep player\n"
	                          "              If not given, default from global configure\n";

std::random_device global_random_device;
std::mt19937 global_random_engine(global_random_device());

int main(int argc, char *argv[]) {
	if (argc > 1 && strcmp(argv[1], "config") == 0) {
		show_global_cfg(std::cout);
		return 0;
	}

	if (argc > 1 && strcmp(argv[1], "train") == 0) {
		std::shared_ptr<FIRNet> net;
		if (argc == 2)
			net = std::make_shared<FIRNet>();
		else if (argc == 3)
			net = std::make_shared<FIRNet>(param_file_name(argv[2]));
		else
			EXIT_WITH_USAGE(train_usage);
		show_global_cfg(std::cout);
		net->show_param(std::cout);
		train(net);
		return 0;
	}

	if (argc > 1 && strcmp(argv[1], "play") == 0) {
		if (argc == 4 || argc == 5) {
			int itermax = TRAIN_DEEP_ITERMAX;
			if (argc == 5)
				itermax = std::atoi(argv[4]);
			if (itermax <= 0)
				EXIT_WITH_USAGE(play_usage);
			std::cout << "mcts_itermax=" << itermax << std::endl;
			auto p0 = HumanPlayer("human");
			auto net = std::make_shared<FIRNet>(param_file_name(argv[3]));
			auto p1 = MCTSDeepPlayer("mcts_deep", net, itermax, C_PUCT);
			if (strcmp(argv[2], "0") == 0)
				play(p0, p1, false);
			else if (strcmp(argv[2], "1") == 0)
				play(p1, p0, false);
			else
				EXIT_WITH_USAGE(play_usage);
		}
		else {
			EXIT_WITH_USAGE(play_usage);
		}
		return 0;
	}

	if (argc > 1 && strcmp(argv[1], "benchmark") == 0) {
		if (argc == 4 || argc == 5) {
			int itermax = TRAIN_DEEP_ITERMAX;
			if (argc == 5)
				itermax = std::atoi(argv[4]);
			if (itermax <= 0)
				EXIT_WITH_USAGE(benchmark_usage);
			std::cout << "mcts_itermax=" << itermax << std::endl;
			auto net1 = std::make_shared<FIRNet>(param_file_name(argv[3]));
			auto net2 = std::make_shared<FIRNet>(param_file_name(argv[3]));
			auto p1 = MCTSDeepPlayer("mcts_deep1", net1, itermax, C_PUCT);
			auto p2 = MCTSDeepPlayer("mcts_deep2", net2, itermax, C_PUCT);
			benchmark(p1, p2, 10, false);
		}
		else {
			EXIT_WITH_USAGE(benchmark_usage);
		}
		return 0;
	}

	EXIT_WITH_USAGE(usage);
}