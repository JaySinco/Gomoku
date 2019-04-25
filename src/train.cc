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

void train(std::shared_ptr<FIRNet> net) {
	LOG(INFO) << "start training...";

	long long update_cnt = 0;
	long long game_cnt = 0;
	float avg_turn = 0.0f;
	DataSet dataset;

	int test_itermax = TEST_PURE_ITERMAX;
	auto test_player = MCTSPurePlayer("pure_player", test_itermax, C_PUCT);
	auto net_player = MCTSDeepPlayer("net_player", net, TRAIN_DEEP_ITERMAX, C_PUCT);

	for (;;) {
		int step = selfplay(net, dataset, TRAIN_DEEP_ITERMAX);

		if (dataset.total() > BATCH_SIZE) {
			++game_cnt;
			constexpr int game_per_log = 3;
			avg_turn += (step - avg_turn) / float(game_cnt > game_per_log ? game_per_log : game_cnt);
			for (int epoch = 0; epoch < EPOCH_PER_GAME; ++epoch) {
				auto batch = new MiniBatch();
				dataset.make_mini_batch(batch);
				float loss = net->train_step(batch);
				++update_cnt;
				if (update_cnt % (game_per_log * EPOCH_PER_GAME) == 0) {
					LOG(INFO) << "loss=" << loss << ", dataset_total=" << dataset.total() << ", update_cnt="
						<< update_cnt << ", avg_turn=" << avg_turn << ", game_cnt=" << game_cnt;
				}
				delete batch;
			}
		}
		if (game_cnt > 0 && game_cnt % 15 == 0) {
			constexpr int sim_game = 10;
			float lose_prob = 1 - benchmark(net_player, test_player, sim_game);
			LOG(INFO) << "benchmark " << sim_game << " games against MCTSPurePlayer(itermax="
				<< test_itermax << "), lose_prob=" << lose_prob;
			if (lose_prob < 1e-3 && test_itermax < 15 * TEST_PURE_ITERMAX) {
				test_itermax += TEST_PURE_ITERMAX;
				test_player.reset_itermax(test_itermax);
			}
		}

		if (game_cnt > 0 && game_cnt % 30 == 0)
			net->save_parameters(param_file_name(game_cnt));
	}
}