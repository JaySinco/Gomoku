#include "mcts.h"

MCTSNode::~MCTSNode() {
	for (const auto &mn : children) 
		delete mn.second;
}

void MCTSNode::expand(const std::vector<std::pair<Move, float>> &set) {
	for (auto &mvp : set)
		children[mvp.first] = new MCTSNode(this, mvp.second);
}

MCTSNode *MCTSNode::cut(Move occurred) {
	auto citer = children.find(occurred);
	assert(citer != children.end());
	auto child = citer->second;
	children.erase(occurred);
	child->parent = nullptr;
	return child;
}

std::pair<Move, MCTSNode*> MCTSNode::select(float c_puct) const {
	std::pair<Move, MCTSNode*> picked(Move(NO_MOVE_YET), nullptr);
	float max_value = -1 * std::numeric_limits<float>::max();
	for (const auto &mn : children) {
		float value = mn.second->value(c_puct);
		if (value > max_value) {
			picked = mn;
			max_value = value;
		}
	}
	return picked;
}

Move MCTSNode::most_visted() const {
	int max_visit = -1 * std::numeric_limits<int>::max();
	Move act(NO_MOVE_YET);
	if (DEBUG_MCTS_MODE)
		std::cout << "(ROOT): " << *this << std::endl;
	for (const auto &mn : children) {
		if (DEBUG_MCTS_MODE)
			std::cout << mn.first << ": " << *mn.second << std::endl;
		auto vn = mn.second->visits;
		if (vn > max_visit) {
			act = mn.first;
			max_visit = vn;
		}
	}
	return act;
}

Move MCTSNode::act_by_prob(float mcts_move_priors[BOARD_SIZE], bool add_noise) const {
	float temp[BOARD_SIZE] = { 0.0f };
	if (mcts_move_priors == nullptr) {
		mcts_move_priors = temp;
	}
	float noise_added[BOARD_SIZE] = { 0.0f };
	float noise_sum = 0;
	std::normal_distribution<float> normal(0, 1);
	for (const auto &mn : children) {
		auto vn = mn.second->visits;
		mcts_move_priors[mn.first.z()] = float(vn) / float(visits);
		noise_added[mn.first.z()] = std::abs(normal(global_random_engine));
		noise_sum += noise_added[mn.first.z()];
	}
	float *move_priors = mcts_move_priors;
	float noised_move_priors[BOARD_SIZE];
	if (add_noise) {
		std::copy(mcts_move_priors, mcts_move_priors + BOARD_SIZE, noised_move_priors);
		for (int i = 0; i < BOARD_SIZE; ++i)
			noised_move_priors[i] = 0.75 * noised_move_priors[i] + 0.25 * noise_added[i] / noise_sum;
		move_priors = noised_move_priors;
	}
	float check_sum = 0;
	for (int i = 0; i < BOARD_SIZE; ++i)
		check_sum += move_priors[i];
	assert(check_sum > 0.98);
	std::discrete_distribution<int> discrete(move_priors, move_priors + BOARD_SIZE);
	return Move(discrete(global_random_engine));
}

void MCTSNode::update(float leafValue) {
	++visits;
	float delta = (leafValue - quality) / float(visits);
	quality += delta;
}

void MCTSNode::update_recursive(float leafValue) {
	if (parent != nullptr)
		parent->update_recursive(-1 * leafValue);
	update(leafValue);
}

float MCTSNode::value(float c_puct) const {
	assert(!is_root());
	float N = float(parent->visits);
	float n = float(1 + visits);
	return quality + (c_puct * prior * std::sqrt(N) / n);
}

std::ostream &operator<<(std::ostream &out, const MCTSNode &node) {
	return out << "MCTSNode(" << node.parent << "): "
		<< node.children.size() << " children, " << node.visits << " visits, "
		<< node.prior << " prior, " << node.quality << " quality";
}

MCTSPurePlayer::MCTSPurePlayer(const std::string &name, int itermax, float c_puct)
	: id(name), itermax(itermax), c_puct(c_puct) {
	root = new MCTSNode(nullptr, 1.0f);
}

void MCTSPurePlayer::reset() {
	delete root;
	root = new MCTSNode(nullptr, 1.0f);
}

Move MCTSPurePlayer::play(const State &state) {
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
					move_priors.push_back(std::make_pair(mv, 1.0f / float(n_options)));
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

MCTSDeepPlayer::MCTSDeepPlayer(const std::string &name, std::shared_ptr<FIRNet> nn, int itermax, float c_puct)
	: id(name), itermax(itermax), c_puct(c_puct), net(nn) {
	root = new MCTSNode(nullptr, 1.0f);
}

void MCTSDeepPlayer::reset() {
	delete root;
	root = new MCTSNode(nullptr, 1.0f);
}

void MCTSDeepPlayer::think(int itermax, float c_puct, const State &state, 
		std::shared_ptr<FIRNet> net, MCTSNode *root) {
	for (int i = 0; i < itermax; ++i) {
		State state_copied(state);
		MCTSNode *node = root;
		while (!node->is_leaf()) {
			auto move_node = node->select(c_puct);
			node = move_node.second;
			state_copied.next(move_node.first);
		}
		float leaf_value;
		Color winner = state_copied.get_winner();
		if (winner == Color::Empty) {
			std::vector<std::pair<Move, float>> net_move_priors;
			net->forward(state_copied, &leaf_value, net_move_priors);
			node->expand(net_move_priors);
			leaf_value *= -1;
		}
		else {
			Color enemy_side = state_copied.current();
			if (winner == enemy_side)
				leaf_value = -1.0f;
			else if (winner == ~enemy_side)
				leaf_value = 1.0f;
			else
				leaf_value = 0.0f;;
		}
		node->update_recursive(leaf_value);
	}
}

Move MCTSDeepPlayer::play(const State &state) {
	if (!(state.get_last().z() == NO_MOVE_YET) && !root->is_leaf())
		swap_root(root->cut(state.get_last()));
	think(itermax, c_puct, state, net, root);
	Move act = root->act_by_prob();
	swap_root(root->cut(act));
	return act;
}

void train_mcts_deep(std::shared_ptr<FIRNet> net, int itermax, float c_puct) {
	auto trainee = MCTSDeepPlayer("trainee", net);
	auto opponent = MCTSPurePlayer("opponent");
	LOG(INFO) << "training configuration: " << "itermax=" << itermax << ", c_puct=" << c_puct
		<< ", batch_size=" << BATCH_SIZE << ", epoch_per_game=" << EPOCH_PER_GAME
		<< ", buffer_size=" << BUFFER_SIZE << ", learning_rate=" << LEARNING_RATE;
	long long update_cnt = 0;
	long long game_cnt = 0;
	float avg_turn = 0.0f;
	DataSet dataset;
	for (;;) {
		++game_cnt;
		State game;
		MCTSNode *root = new MCTSNode(nullptr, 1.0f);
		std::vector<SampleData> record;
		float ind = -1.0f;
		int turn = 0;
		while (!game.over()) {
			++turn;
			ind *= -1.0f;
			SampleData one_step;
			*one_step.v_label = ind;
			game.fill_feature_array(one_step.data);
			MCTSDeepPlayer::think(itermax, c_puct, game, net, root);
			Move act = root->act_by_prob(one_step.p_label, true);
			record.push_back(one_step);
			game.next(act);
			auto temp = root->cut(act);
			delete root; 
			root = temp;
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
			dataset.push_with_transform(&step);
		}
		avg_turn += (turn - avg_turn) / float(game_cnt > 10 ? 10 : game_cnt);
		for (int epoch = 0; dataset.total() > BATCH_SIZE && epoch < EPOCH_PER_GAME; ++epoch) {
			MiniBatch batch;
			dataset.make_mini_batch(&batch);
			float loss = net->train_step(&batch);
			++update_cnt;
			if (update_cnt % (2 * EPOCH_PER_GAME) == 0) {
				LOG(INFO) << "loss=" << loss << ", dataset_total=" << dataset.total() << ", update_cnt="
					<< update_cnt << ", avg_turn=" << avg_turn << ", game_cnt=" << game_cnt;
			}
		}
		if (game_cnt % 100 == 0) {
			std::ostringstream filename;
			filename << "FIR-" << BOARD_MAX_COL << "x" << BOARD_MAX_ROW << "by" << FIVE_IN_ROW
				<< "_" << game_cnt << ".param";
			net->save_parameters(filename.str());
			float lose_rate = benchmark(opponent, trainee);
			LOG(INFO) << "benchmark against opponent, lose_rate=" << lose_rate;
		}
	}
}