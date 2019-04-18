#include <random>
#include <iostream>

constexpr int FIVE_IN_ROW = 4;
constexpr int BOARD_MAX_ROW = 6;
constexpr int BOARD_MAX_COL = 6;
constexpr int INPUT_FEATURE_NUM = 4;
constexpr int BATCH_SIZE = 512;
constexpr int BUFFER_SIZE = 10000;
constexpr int EPOCH_PER_GAME = 5;
constexpr int TEST_PURE_MCTS_ITERMAX = 1000;
constexpr int TRAIN_DEEP_MCTS_ITERMAX = 400;
constexpr int EXPLORE_STEP = 7;
constexpr float LEARNING_RATE = 1e-3;
constexpr float WEIGHT_DECAY = 1e-4;
constexpr float DIRICHLET_ALPHA = 0.15;
constexpr float NOISE_RATE = 0.25;
constexpr float C_PUCT_DEFAULT = 1.0;
constexpr bool RANDOM_OPENING = false;
constexpr bool DEBUG_MCTS_PROB = false;
constexpr bool DEBUG_TRAIN_DATA = false;

constexpr int BOARD_SIZE = BOARD_MAX_ROW * BOARD_MAX_COL;
constexpr int NO_MOVE_YET = -1;
extern std::mt19937 global_random_engine;

inline void show_global_cfg(std::ostream &out) {
	out << "=== global configure ===" << "\ngame_mode=" << BOARD_MAX_ROW << "x" << BOARD_MAX_COL << "by" << FIVE_IN_ROW
		<< "\ninput_feature=" << INPUT_FEATURE_NUM << "\nbatch_size=" << BATCH_SIZE
		<< "\nbuffer_size=" << BUFFER_SIZE << "\nepoch_per_game=" << EPOCH_PER_GAME
		<< "\nc_puct=" << C_PUCT_DEFAULT << "\ndirichlet_alpha=" << DIRICHLET_ALPHA
		<< "\nlearning_rate=" << LEARNING_RATE << "\nweight_decay=" << WEIGHT_DECAY
		<< "\nexplore_step=" << EXPLORE_STEP << "\nnoise_rate=" << NOISE_RATE
		<< "\ntest_pure_mcts_itermax=" << TEST_PURE_MCTS_ITERMAX
		<< "\ntrain_deep_mcts_itermax=" << TRAIN_DEEP_MCTS_ITERMAX
		 << "\n" << std::endl;
}