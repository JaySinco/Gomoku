#include <random>

constexpr int FIVE_IN_ROW = 5;
constexpr int BOARD_MAX_ROW = 8;
constexpr int BOARD_MAX_COL = 8;
constexpr int BOARD_SIZE = BOARD_MAX_ROW * BOARD_MAX_COL;
constexpr int NO_MOVE_YET = -1;

constexpr bool DEBUG_MCTS_MODE = false;

constexpr int BATCH_SIZE = 256;
constexpr int BUFFER_SIZE = 10240;
constexpr int EPOCH_PER_GAME = 40;
constexpr float LEARNING_RATE = 2e-4;
constexpr float WEIGHT_DECAY = 1e-4;

extern std::mt19937 global_random_engine;
