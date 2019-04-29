#pragma once

#include <random>
#include <iostream>

constexpr int FIVE_IN_ROW = 5;
constexpr int BOARD_MAX_COL = 8;
constexpr int BOARD_MAX_ROW = BOARD_MAX_COL;
constexpr int INPUT_FEATURE_NUM = 4; // self, opponent[[, lastmove], color]
constexpr int BATCH_SIZE = 512;
constexpr int BUFFER_SIZE = 10000;
constexpr int EPOCH_PER_GAME = 1;
constexpr int TEST_PURE_ITERMAX = 1000;
constexpr int TRAIN_DEEP_ITERMAX = 400;
constexpr int EXPLORE_STEP = 20;
constexpr int NET_NUM_FILTER = 64;
constexpr int NET_NUM_RESIDUAL_BLOCK = 3;
constexpr int LR_DROP_STEP1 = 2000;
constexpr int LR_DROP_STEP2 = 8000;
constexpr int LR_DROP_STEP3 = 10000;
constexpr float C_PUCT = 1.0;
constexpr float INIT_LEARNING_RATE = 2e-1;
constexpr float WEIGHT_DECAY = 1e-4;
constexpr float DIRICHLET_ALPHA = 0.3;
constexpr float NOISE_RATE = 0.2;
constexpr bool USE_BATCH_NORM = true;

constexpr int MINUTE_PER_LOG = 3;
constexpr int MINUTE_PER_SAVE = 30;
constexpr int MINUTE_PER_BENCHMARK = 15;
constexpr int COLOR_OCCUPY_SPACE = 1;
constexpr float BN_MVAR_INIT = 1.0f;

constexpr bool DEBUG_MCTS_PROB = false;
constexpr bool DEBUG_TRAIN_DATA = false;

constexpr int BOARD_SIZE = BOARD_MAX_ROW * BOARD_MAX_COL;
constexpr int NO_MOVE_YET = -1;
extern std::mt19937 global_random_engine;

inline void show_global_cfg(std::ostream &out) {
    out << "=== global configure ===" << "\ngame_mode=" << BOARD_MAX_ROW << "x" << BOARD_MAX_COL << "by" << FIVE_IN_ROW
        << "\ninput_feature=" << INPUT_FEATURE_NUM << "\nbatch_size=" << BATCH_SIZE
        << "\nbuffer_size=" << BUFFER_SIZE << "\nepoch_per_game=" << EPOCH_PER_GAME
        << "\nc_puct=" << C_PUCT << "\ndirichlet_alpha=" << DIRICHLET_ALPHA
        << "\ninit_learning_rate=" << INIT_LEARNING_RATE << "\nweight_decay=" << WEIGHT_DECAY
        << "\nlr_drop_step1=" << LR_DROP_STEP1 << "\nlr_drop_step2=" << LR_DROP_STEP2
        << "\nlr_drop_step3=" << LR_DROP_STEP3 << "\nuse_batch_norm=" << USE_BATCH_NORM
        << "\nexplore_step=" << EXPLORE_STEP << "\nnoise_rate=" << NOISE_RATE
        << "\nnet_num_filter=" << NET_NUM_FILTER << "\nnet_num_resudual_block=" << NET_NUM_RESIDUAL_BLOCK
        << "\ntest_pure_itermax=" << TEST_PURE_ITERMAX
        << "\ntrain_deep_itermax=" << TRAIN_DEEP_ITERMAX
        << "\n" << std::endl;
}