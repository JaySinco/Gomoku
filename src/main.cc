#include <iostream>

#include "mcts.h"
#include "train.h"

#define EXIT_WITH_USAGE(usage)  { std::cout << usage; return -1; }

const char *usage =
    "usage: gomoku <command>\n\n"
    "These are common Gomoku commands used in various situations:\n"
    "   config     Print global configure\n"
    "   train      Train model from scatch or parameter file\n"
    "   play       Play with trained model\n"
    "   benchmark  Benchmark between two mcts deep players\n\n";

const char *train_usage =
    "usage: gomoku train <net>\n"
    "   <net>      verno of network(must >= 0), which is the suffix of parameter file basename\n"
    "              if equal to zero, train from scratch; otherwise continue to train model from last check-point\n\n";

const char *play_usage =
    "usage: gomoku play <color> <net> [itermax]\n"
    "   <color>    '0' if human take first hand, '1' otherwise\n"
    "              specially '-1' means let computer selfplay\n"
    "   <net>      verno of network(must > 0), which is the suffix of parameter file basename\n"
    "   [itermax]  itermax for mcts deep player\n"
    "              if not given, default from global configure\n\n";

const char *benchmark_usage =
    "usage: gomoku benchmark <net1> <net2> [itermax]\n"
    "   <net1>     verno of network to compare(must > 0), which is the suffix of parameter file basename\n"
    "   <net2>     see above\n"
    "   [itermax]  itermax for mcts deep player\n"
    "              if not given, default from global configure\n\n";

std::random_device global_random_device;
std::mt19937 global_random_engine(global_random_device());

int main(int argc, char *argv[]) {
    if (argc > 1 && strcmp(argv[1], "config") == 0) {
        show_global_cfg(std::cout);
        return 0;
    }

    if (argc > 1 && strcmp(argv[1], "train") == 0) {
        if (argc == 3) {
            long long verno = std::atoi(argv[2]);
            std::shared_ptr<FIRNet> net = std::make_shared<FIRNet>(verno);
            show_global_cfg(std::cout);
            net->show_param(std::cout);
            train(net);
            return 0;
        }
        EXIT_WITH_USAGE(train_usage);
    }

    if (argc > 1 && strcmp(argv[1], "play") == 0) {
        if (argc == 4 || argc == 5) {
            int itermax = TRAIN_DEEP_ITERMAX;
            if (argc == 5)
                itermax = std::atoi(argv[4]);
            std::cout << "mcts_itermax=" << itermax << std::endl;
            long long verno = std::atoi(argv[3]);
            auto net = std::make_shared<FIRNet>(verno);
            auto p1 = MCTSDeepPlayer(net, itermax, C_PUCT);
            if (strcmp(argv[2], "0") == 0) {
                auto p0 = HumanPlayer("human");
                play(p0, p1, false);
            }
            else if (strcmp(argv[2], "1") == 0) {
                auto p0 = HumanPlayer("human");
                play(p1, p0, false);
            }
            else if (strcmp(argv[2], "-1") == 0) {
                auto p0 = MCTSDeepPlayer(net, itermax, C_PUCT);
                play(p0, p1, false);
            }
            return 0;
        }
        EXIT_WITH_USAGE(play_usage);
    }

    if (argc > 1 && strcmp(argv[1], "benchmark") == 0) {
        if (argc == 4 || argc == 5) {
            int itermax = TRAIN_DEEP_ITERMAX;
            if (argc == 5)
                itermax = std::atoi(argv[4]);
            std::cout << "mcts_itermax=" << itermax << std::endl;
            long long verno1 = std::atoi(argv[2]);
            auto net1 = std::make_shared<FIRNet>(verno1);
            long long verno2 = std::atoi(argv[3]);
            auto net2 = std::make_shared<FIRNet>(verno2);
            auto p1 = MCTSDeepPlayer(net1, itermax, C_PUCT);
            auto p2 = MCTSDeepPlayer(net2, itermax, C_PUCT);
            benchmark(p1, p2, 10, false);
            return 0;
        }
        EXIT_WITH_USAGE(benchmark_usage);
    }

    EXIT_WITH_USAGE(usage);
}