#pragma once

#include <map>
#include "engine.h"

constexpr bool DEBUG_MCTS_MODE = false;

class MCTSNode {
	friend std::ostream &operator<<(std::ostream &out, MCTSNode &node);
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
	void update(float leafValue);
	void update_recursive(float leafValue);
	float value(float c_puct) const;
	bool is_leaf() const { return children.size() == 0; }
	bool is_root() const { return parent == nullptr; }
};
std::ostream &operator<<(std::ostream &out, MCTSNode &node);

class MCTSPurePlayer: public Player {
	std::string id;
	int itermax;
	float c_puct;
	MCTSNode *root;
	void swap_root(MCTSNode * new_root) { delete root; root = new_root; }
public:
	MCTSPurePlayer(const char *name, int itermax = 10000, float c_puct = 5.0f);
	~MCTSPurePlayer() { delete root; }
	const std::string &name() const override { return id; }
	void reset() override;
	Move play(const State &state) override;
	
};

//class MCTSDeepPlayer : public Player {
//	std::string id;
//	int itermax;
//	float c_puct;
//	MCTSNode *root;
//	void swap_root(MCTSNode * new_root) {
//		delete root;
//		root = new_root;
//	}
//public:
//	MCTSDeepPlayer(const char *name, int itermax = 10000, float c_puct = 5.0f)
//		: id(name), itermax(itermax), c_puct(c_puct) {
//		root = new MCTSNode(nullptr, 1.0f);
//	}
//	~MCTSDeepPlayer() { delete root; }
//}
