#include <iomanip>

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

Move MCTSNode::act_by_most_visted() const {
    int max_visit = -1 * std::numeric_limits<int>::max();
    Move act(NO_MOVE_YET);
    if (DEBUG_MCTS_PROB)
        std::cout << "(ROOT): " << *this << std::endl;
    for (const auto &mn : children) {
        if (DEBUG_MCTS_PROB)
            std::cout << mn.first << ": " << *mn.second << std::endl;
        auto vn = mn.second->visits;
        if (vn > max_visit) {
            act = mn.first;
            max_visit = vn;
        }
    }
    return act;
}

Move MCTSNode::act_by_prob(float mcts_move_priors[BOARD_SIZE], float temp) const {
    float move_priors_buffer[BOARD_SIZE] = { 0.0f };
    if (mcts_move_priors == nullptr)
        mcts_move_priors = move_priors_buffer;
    std::map<int, float> move_priors_map;
    if (DEBUG_MCTS_PROB)
        std::cout << "(ROOT): " << *this << std::endl;
    float alpha = -1 * std::numeric_limits<float>::max();
    for (const auto &mn : children) {
        if (DEBUG_MCTS_PROB)
            std::cout << mn.first << ": " << *mn.second << std::endl;
        auto vn = mn.second->visits;
        move_priors_map[mn.first.z()] = 1.0f / temp * std::log(float(vn) + 1e-10);
        if (move_priors_map[mn.first.z()] > alpha)
            alpha = move_priors_map[mn.first.z()];
    }
    float denominator = 0;
    for (auto &mn : move_priors_map) {
        float value = std::exp(mn.second - alpha);
        move_priors_map[mn.first] = value;
        denominator += value;
    }
    for (auto &mn : move_priors_map) {
        mcts_move_priors[mn.first] = mn.second / denominator;
    }
    float check_sum = 0;
    for (int i = 0; i < BOARD_SIZE; ++i)
        check_sum += mcts_move_priors[i];
    assert(check_sum > 0.99);
    std::discrete_distribution<int> discrete(mcts_move_priors, mcts_move_priors + BOARD_SIZE);
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

void gen_ran_dirichlet(const size_t K, float alpha, float theta[]) {
    std::gamma_distribution<float> gamma(alpha, 1.0f);
    float norm = 0.0;
    for (size_t i = 0; i < K; i++) {
        theta[i] = gamma(global_random_engine);
        norm += theta[i];
    }
    for (size_t i = 0; i < K; i++) {
        theta[i] /= norm;
    }
}

void MCTSNode::add_noise_to_child_prior(float noise_rate) {
    auto noise_added = new float[children.size()];
    gen_ran_dirichlet(children.size(), DIRICHLET_ALPHA, noise_added);
    int prior_cnt = 0;
    for (auto &item : children) {
        item.second->prior = (1 - noise_rate) * item.second->prior + noise_rate * noise_added[prior_cnt];
        ++prior_cnt;
    }
    delete [] noise_added;
}

float MCTSNode::value(float c_puct) const {
    assert(!is_root());
    float N = float(parent->visits);
    float n = visits + 1;
    return quality + (c_puct * prior * std::sqrt(N) / n);
}

std::ostream &operator<<(std::ostream &out, const MCTSNode &node) {
    out << "MCTSNode(" << node.parent << "): "
        << std::setw(3) << node.children.size() << " children, ";
    if (node.parent != nullptr)
        out << std::setw(6) << std::fixed << std::setprecision(3)
            << float(node.visits) / float(node.parent->visits) * 100 << "% / ";
    out	<< std::setw(3) << node.visits << " visits, "
        << std::setw(6) << std::fixed << std::setprecision(3)
        << node.prior * 100 << "% prior, "
        << std::setw(6) << std::fixed << std::setprecision(3)
        << node.quality << " quality";
     return out;
}

MCTSPurePlayer::MCTSPurePlayer(int itermax, float c_puct)
    : itermax(itermax), c_puct(c_puct) {
    make_id();
    root = new MCTSNode(nullptr, 1.0f);
}

void MCTSPurePlayer::make_id() {
    std::ostringstream ids;
    ids << "mcts" << itermax;
    id = ids.str();
}

void MCTSPurePlayer::set_itermax(int n) {
    itermax = n;
    make_id();
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
        if (!state_copied.over()) {
            int n_options = state_copied.get_options().size();
            std::vector<std::pair<Move, float>> move_priors;
            for (const auto mv : state_copied.get_options()) {
                move_priors.push_back(std::make_pair(mv, 1.0f / float(n_options)));
            }
            node->expand(move_priors);
            winner = state_copied.next_rand_till_end();
        }
        float leaf_value;
        if (winner == enemy_side)
            leaf_value = -1.0f;
        else if (winner == ~enemy_side)
            leaf_value = 1.0f;
        else
            leaf_value = 0.0f;
        node->update_recursive(leaf_value);
    }
    Move act = root->act_by_most_visted();
    swap_root(root->cut(act));
    return act;
}

MCTSDeepPlayer::MCTSDeepPlayer(std::shared_ptr<FIRNet> nn, int itermax, float c_puct)
    : itermax(itermax), c_puct(c_puct), net(nn) {
    make_id();
    root = new MCTSNode(nullptr, 1.0f);
}

void MCTSDeepPlayer::make_id() {
    std::ostringstream ids;
    ids << "mcts" << itermax << "_net" << net->verno();
    id = ids.str();
}

void MCTSDeepPlayer::reset() {
    delete root;
    root = new MCTSNode(nullptr, 1.0f);
}

void MCTSDeepPlayer::think(int itermax, float c_puct, const State &state,
        std::shared_ptr<FIRNet> net, MCTSNode *root, bool add_noise_to_root) {
    if (add_noise_to_root)
        root->add_noise_to_child_prior(NOISE_RATE);
    for (int i = 0; i < itermax; ++i) {
        State state_copied(state);
        MCTSNode *node = root;
        while (!node->is_leaf()) {
            auto move_node = node->select(c_puct);
            node = move_node.second;
            state_copied.next(move_node.first);
        }
        float leaf_value;
        if (!state_copied.over()) {
            std::vector<std::pair<Move, float>> net_move_priors;
            net->forward(state_copied, &leaf_value, net_move_priors);
            node->expand(net_move_priors);
            leaf_value *= -1;
        }
        else {
            if (state_copied.get_winner() != Color::Empty)
                leaf_value = 1.0f;
            else
                leaf_value = 0.0f;
        }
        node->update_recursive(leaf_value);
    }
}

Move MCTSDeepPlayer::play(const State &state) {
    if (!(state.get_last().z() == NO_MOVE_YET) && !root->is_leaf())
        swap_root(root->cut(state.get_last()));
    think(itermax, c_puct, state, net, root);
    Move act = root->act_by_prob(nullptr, 1e-3);
    swap_root(root->cut(act));
    return act;
}
