#include <random>
#include <iostream>

constexpr int FIVE_IN_ROW = 4;
constexpr int BOARD_MAX_ROW = 6;
constexpr int BOARD_MAX_COL = 6;
constexpr int BOARD_SIZE = BOARD_MAX_ROW * BOARD_MAX_COL;
constexpr int INPUT_FEATURE_NUM = 2;
constexpr int NO_MOVE_YET = -1;
constexpr int BATCH_SIZE = 512;
constexpr int BUFFER_SIZE = 10000;
constexpr int EPOCH_PER_GAME = 5;
constexpr int PURE_MCTS_ITERMAX = 1000;
constexpr float LEARNING_RATE = 2e-3;
constexpr float WEIGHT_DECAY = 1e-4;
constexpr float DIRICHLET_ALPHA = 0.3;
constexpr bool DEBUG_MCTS_PROB = false;
constexpr bool DEBUG_TRAIN_DATA = false;
extern std::mt19937 global_random_engine;

inline void show_global_cfg(std::ostream &out) {
	out << "=== global configure ===" << "\ngame_mode=" << BOARD_MAX_ROW << "x" << BOARD_MAX_COL << "by" << FIVE_IN_ROW
		<< "\ninput_feature=" << INPUT_FEATURE_NUM << "\nbatch_size=" << BATCH_SIZE
		<< "\nbuffer_size=" << BUFFER_SIZE << "\nepoch_per_game=" << EPOCH_PER_GAME
		<< "\npure_mcts_itermax=" << PURE_MCTS_ITERMAX << "\nlearning_rate=" << LEARNING_RATE
		<< "\nweight_decay=" << WEIGHT_DECAY << "\ndirichlet_alpha=" << DIRICHLET_ALPHA << "\n" << std::endl;
}