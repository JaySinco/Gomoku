#include <chrono>

#include "train.h"
#include "mcts.h"

int selfplay(std::shared_ptr<FIRNet> net, DataSet &dataset, int itermax) {
    State game;
    std::vector<SampleData> record;
    MCTSNode *root = new MCTSNode(nullptr, 1.0f);
    float ind = -1.0f;
    int step = 0;
    while (!game.over()) {
        ++step;
        ind *= -1.0f;
        SampleData one_step;
        *one_step.v_label = ind;
        game.fill_feature_array(one_step.data);
        MCTSDeepPlayer::think(itermax, C_PUCT, game, net, root, true);
        Move act = root->act_by_prob(one_step.p_label, step <= EXPLORE_STEP ? 1.0f : 1e-3);
        record.push_back(one_step);
        game.next(act);
        auto temp = root->cut(act);
        delete root;
        root = temp;
        if (DEBUG_TRAIN_DATA)
            std::cout << game << std::endl;
    }
    delete root;
    if (game.get_winner() != Color::Empty) {
        if (ind < 0)
            for (auto &step : record)
                (*step.v_label) *= -1;
    }
    else {
        for (auto &step : record)
            (*step.v_label) = 0.0f;
    }
    for (auto &step : record) {
        if (DEBUG_TRAIN_DATA)
            std::cout << step << std::endl;
        dataset.push_with_transform(&step);
    }
    return step;
}

bool trigger_timer(std::chrono::time_point<std::chrono::system_clock> &last, int per_minute) {
    auto now = std::chrono::system_clock::now();
    auto sec = std::chrono::duration_cast<std::chrono::seconds>(now - last).count();
    if (sec >= per_minute * 60) {
        last = now;
        return true;
    }
    return false;
}

void train(std::shared_ptr<FIRNet> net) {
    LOG(INFO) << "start training...";

    auto last_log = std::chrono::system_clock::now();
    auto last_save = std::chrono::system_clock::now();
    auto last_benchmark = std::chrono::system_clock::now();

    long long game_cnt = 0;
    float avg_turn = 0.0f;
    DataSet dataset;

    int test_itermax = TEST_PURE_ITERMAX;
    auto test_player = MCTSPurePlayer(test_itermax, C_PUCT);
    auto net_player = MCTSDeepPlayer(net, TRAIN_DEEP_ITERMAX, C_PUCT);

    for (;;) {
        ++game_cnt;
        int step = selfplay(net, dataset, TRAIN_DEEP_ITERMAX);
        avg_turn += (step - avg_turn) / float(game_cnt > 10 ? 10 : game_cnt);
        if (dataset.total() > BATCH_SIZE) {
            for (int epoch = 0; epoch < EPOCH_PER_GAME; ++epoch) {
                auto batch = new MiniBatch();
                dataset.make_mini_batch(batch);
                float loss = net->train_step(batch);
                if (trigger_timer(last_log, MINUTE_PER_LOG)) {
                    LOG(INFO) << "loss=" << loss << ", dataset_total=" << dataset.total() << ", update_cnt="
                        << net->verno() << ", avg_turn=" << avg_turn << ", game_cnt=" << game_cnt;
                }
                delete batch;
            }
        }
        if (trigger_timer(last_benchmark, MINUTE_PER_BENCHMARK)) {
            float lose_prob = 1 - benchmark(net_player, test_player, 10);
            LOG(INFO) << "benchmark 10 games against " << test_player.name() << ", lose_prob=" << lose_prob;
            if (lose_prob < 1e-3 && test_itermax < 15 * TEST_PURE_ITERMAX) {
                test_itermax += TEST_PURE_ITERMAX;
                test_player.set_itermax(test_itermax);
            }
        }
        if (trigger_timer(last_save, MINUTE_PER_SAVE)) {
            net->save_param();
        }
    }
}