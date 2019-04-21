#include <iostream>

#include "mcts.h"
#include "train.h"

std::random_device global_random_device;
std::mt19937 global_random_engine(global_random_device());

const char *usage = "usage: gomoku <command>\n\n"
					"These are common Gomoku commands used in various situations:\n\n"
					"   config     Print global configure\n"
					"   train      Train model from scatch\n"
					"   continue   Continue to train model from last check-point\n"
					"   play       Play with trained model, append [0] if human take first hand, [1] otherwise\n"
					"   benchmark  Benchmark between two players\n";

int main(int argc, char *argv[]) {
	switch (argc) {
	case 2:
		if (strcmp(argv[1], "config") == 0) {
			show_global_cfg(std::cout);
		}
		else if (strcmp(argv[1], "train") == 0) {
			show_global_cfg(std::cout);
			auto net = std::make_shared<FIRNet>();
			train(net);
		}
		else {
			std::cout << usage;
		}
		break;
	case 3:
		if (strcmp(argv[1], "continue") == 0) {
			show_global_cfg(std::cout);
			auto net = std::make_shared<FIRNet>(param_file_name(argv[2]));
			train(net);
		}
		else if (strcmp(argv[1], "play0") == 0) {
			auto net = std::make_shared<FIRNet>(param_file_name(argv[2]));
			auto p0 = HumanPlayer("human");
			auto p1 = MCTSDeepPlayer("mcts_deep", net, TRAIN_DEEP_ITERMAX, C_PUCT);
			play(p0, p1, false);
		}
		else if (strcmp(argv[1], "play1") == 0) {
			auto net = std::make_shared<FIRNet>(param_file_name(argv[2]));
			auto p0 = MCTSDeepPlayer("mcts_deep", net, TRAIN_DEEP_ITERMAX, C_PUCT);
			auto p1 = HumanPlayer("human");
			play(p1, p0, false);
		}
		else if (strcmp(argv[1], "benchmark") == 0) {
			auto net = std::make_shared<FIRNet>(param_file_name(argv[2]));
			auto p0 = MCTSPurePlayer("mcts_pure", TEST_PURE_ITERMAX, C_PUCT);
			auto p1 = MCTSDeepPlayer("mcts_deep", net, TRAIN_DEEP_ITERMAX, C_PUCT);
			benchmark(p0, p1, 10, false);
		}
		else {
			std::cout << usage;
		}
		break;
	default:
		std::cout << usage;
		break;
	}
}