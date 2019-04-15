#pragma once

#include <map>

#include "game.h"
#include "network.h"

class MCTSNode {
	friend std::ostream &operator<<(std::ostream &out, const MCTSNode &node);
	MCTSNode *parent;
	std::map<Move, MCTSNode*> children;
	int visits;
	float quality;
	float prior;
public:
	MCTSNode(MCTSNode *node_p, float prior_p) : parent(node_p), visits(0), quality(0), prior(prior_p) {}
	~MCTSNode();
	void expand(const std::vector<std::pair<Move, float>> &set);
	MCTSNode *cut(Move occurred);
	std::pair<Move, MCTSNode*> select(float c_puct) const;
	Move most_visted() const;
	Move act_by_prob(float mcts_move_priors[BOARD_SIZE], bool add_noise = false, float noise_rate = 0) const;
	void update(float leafValue);
	void update_recursive(float leafValue);
	float value(float c_puct) const;
	bool is_leaf() const { return children.size() == 0; }
	bool is_root() const { return parent == nullptr; }
};
std::ostream &operator<<(std::ostream &out, const MCTSNode &node);

class MCTSPurePlayer: public Player {
	std::string id;
	int itermax;
	float c_puct;
	MCTSNode *root;
	void swap_root(MCTSNode * new_root) { delete root; root = new_root; }
public:
	MCTSPurePlayer(const std::string &name, int itermax, float c_puct = 5.0f);
	~MCTSPurePlayer() { delete root; }
	const std::string &name() const override { return id; }
	void reset_itermax(int n) { itermax = n; }
	void reset() override;
	Move play(const State &state) override;

};

class MCTSDeepPlayer : public Player {
	std::string id;
	int itermax;
	float c_puct;
	MCTSNode *root;
	std::shared_ptr<FIRNet> net;
	void swap_root(MCTSNode * new_root) { delete root; root = new_root; }
public:
	MCTSDeepPlayer(const std::string &name, std::shared_ptr<FIRNet> nn, int itermax, float c_puct = 5.0f);
	~MCTSDeepPlayer() { delete root; }
	const std::string &name() const override { return id; }
	void reset() override;
	Move play(const State &state) override;
	static void think(int itermax, float c_puct, const State &state,
		std::shared_ptr<FIRNet> net, MCTSNode *root);
};

void train_mcts_deep(std::shared_ptr<FIRNet> net, int itermax = 400, float c_puct = 5.0f);