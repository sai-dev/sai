/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2018 Gian-Carlo Pascutto
    Copyright (C) 2018 SAI Team

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "config.h"

#include <cassert>
#include <cstdio>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "UCTNode.h"
#include "FastBoard.h"
#include "FastState.h"
#include "GTP.h"
#include "GameState.h"
#include "Network.h"
#include "Random.h"
#include "Utils.h"

using namespace Utils;

UCTNode::UCTNode(int vertex, float score) : m_move(vertex), m_score(score) {
}

bool UCTNode::first_visit() const {
    return m_visits == 0;
}

SMP::Mutex& UCTNode::get_mutex() {
    return m_nodemutex;
}

bool UCTNode::create_children(std::atomic<int>& nodecount,
                              GameState& state,
			      float& value,
                              float& alpkt,
			      float& beta,
                              float min_psa_ratio) {
    // check whether somebody beat us to it (atomic)
    if (!expandable(min_psa_ratio)) {
        return false;
    }

    const auto to_move = state.board.get_to_move();
    const auto komi = state.get_komi();

    // acquire the lock
    LOCK(get_mutex(), lock);
    // no successors in final state
    if (state.get_passes() >= 2) {
        return false;
    }
    // check whether somebody beat us to it (after taking the lock)
    if (!expandable(min_psa_ratio)) {
        return false;
    }
    // Someone else is running the expansion
    if (m_is_expanding) {
        return false;
    }
    // We'll be the one queueing this node for expansion, stop others
    m_is_expanding = true;
    lock.unlock();

    const auto raw_netlist = Network::get_scored_moves(
        &state, Network::Ensemble::RANDOM_SYMMETRY, -1, cfg_symm_nonrandom);

    beta = m_net_beta = raw_netlist.beta;
    value = raw_netlist.value; // = m_net_value

    // DCNN returns value as side to move
    // our search functions evaluate from black's point of view
    if (state.board.white_to_move())
        value = 1.0f - value;

    if (is_mult_komi_net) {
        const auto result_extended = Network::get_extended(state, raw_netlist);
        m_net_alpkt = alpkt = result_extended.alpkt;
        m_eval_bonus = result_extended.eval_bonus;
        m_eval_base = result_extended.eval_base;
        m_agent_eval = result_extended.agent_eval;
        m_net_eval = result_extended.pi;

    } else {
        m_net_alpkt = -komi;
        m_eval_bonus = 0.0f;
        m_eval_base = 0.0f;
        m_net_eval = value;
        m_agent_eval = value;
    }


    std::vector<int> stabilizer_subgroup;

    for (auto i = 0; i < 8; i++) {
        if(i == 0 || (cfg_exploit_symmetries && state.is_symmetry_invariant(i))) {
            stabilizer_subgroup.emplace_back(i);
        }
    }

    std::vector<Network::ScoreVertexPair> nodelist;
    std::array<bool, BOARD_SQUARES> taken_already{};
    auto unif_law = std::uniform_real_distribution<float>{0.0, 1.0};

    auto legal_sum = 0.0f;
    for (auto i = 0; i < BOARD_SQUARES; i++) {
        const auto vertex = state.board.get_vertex(i);
        if (state.is_move_legal(to_move, vertex) && !taken_already[i]) {
            auto taken_policy = 0.0f;
            auto max_u = 0.0f;
            auto chosen_vertex = vertex;
            for (auto sym : stabilizer_subgroup) {
                const auto j_vertex = state.board.get_sym_move(vertex, sym);
                const auto j = state.board.get_index(j_vertex);
                if (!taken_already[j]) {
                    taken_already[j] = true;
                    taken_policy += raw_netlist.policy[j];

                    auto u = 0.0f;
                    if (cfg_symm_nonrandom) {
                        const auto p = state.board.get_xy(j_vertex);
                        u = p.first + 2.001 * p.second;
                    } else {
                        u = unif_law(Random::get_Rng());
                    }
                    if (u > max_u) {
                        max_u = u;
                        chosen_vertex = j_vertex;
                    }
                }
            }
            const auto warm_policy = std::pow(taken_policy,
                                              1.0f/cfg_policy_temp);
            nodelist.emplace_back(warm_policy, chosen_vertex);
            legal_sum += warm_policy;
        }
    }
    const auto warm_pass_policy = std::pow(raw_netlist.policy_pass,
                                           1.0f/cfg_policy_temp);
    nodelist.emplace_back(warm_pass_policy, FastBoard::PASS);
    legal_sum += warm_pass_policy;

    if (legal_sum > std::numeric_limits<float>::min()) {
        // re-normalize after removing illegal moves.
        for (auto& node : nodelist) {
            node.first /= legal_sum;
        }
    } else {
        // This can happen with new randomized nets.
        auto uniform_prob = 1.0f / nodelist.size();
        for (auto& node : nodelist) {
            node.first = uniform_prob;
        }
    }

    link_nodelist(nodecount, nodelist, min_psa_ratio);
    return true;
}

void UCTNode::link_nodelist(std::atomic<int>& nodecount,
                            std::vector<Network::ScoreVertexPair>& nodelist,
                            float min_psa_ratio) {
    assert(min_psa_ratio < m_min_psa_ratio_children);

    if (nodelist.empty()) {
        return;
    }

    // Use best to worst order, so highest go first
    std::stable_sort(rbegin(nodelist), rend(nodelist));

    LOCK(get_mutex(), lock);

    const auto max_psa = nodelist[0].first;
    const auto old_min_psa = max_psa * m_min_psa_ratio_children;
    const auto new_min_psa = max_psa * min_psa_ratio;
    if (new_min_psa > 0.0f) {
        m_children.reserve(
            std::count_if(cbegin(nodelist), cend(nodelist),
                [=](const auto& node) { return node.first >= new_min_psa; }
            )
        );
    } else {
        m_children.reserve(nodelist.size());
    }

    auto skipped_children = false;
    for (const auto& node : nodelist) {
        if (node.first < new_min_psa) {
            skipped_children = true;
        } else if (node.first < old_min_psa) {
            m_children.emplace_back(node.second, node.first);
            ++nodecount;
        }
    }

    m_min_psa_ratio_children = skipped_children ? min_psa_ratio : 0.0f;
    m_is_expanding = false;
}

const std::vector<UCTNodePointer>& UCTNode::get_children() const {
    return m_children;
}


int UCTNode::get_move() const {
    return m_move;
}

void UCTNode::virtual_loss() {
    m_virtual_loss += VIRTUAL_LOSS_COUNT;
}

void UCTNode::virtual_loss_undo() {
    m_virtual_loss -= VIRTUAL_LOSS_COUNT;
}

void UCTNode::clear_visits() {
    m_visits = 0;
    m_blackevals = 0;
}

void UCTNode::clear_children_visits() {
    for (const auto& child : m_children) {
        if(child.is_inflated()) {
            child.get()->clear_visits();
        }
    }
}

void UCTNode::update(float eval) {
    m_visits++;
    accumulate_eval(eval);
}

bool UCTNode::has_children() const {
    return m_min_psa_ratio_children <= 1.0f;
}

bool UCTNode::expandable(const float min_psa_ratio) const {
    return min_psa_ratio < m_min_psa_ratio_children;
}

float UCTNode::get_score() const {
    return m_score;
}

float UCTNode::get_eval_bonus() const {
    return m_eval_bonus;
}

float UCTNode::get_eval_bonus_father() const {
    return m_eval_bonus_father;
}

void UCTNode::set_eval_bonus_father(float bonus) {
    m_eval_bonus_father = bonus;
}

float UCTNode::get_eval_base() const {
    return m_eval_base;
}

float UCTNode::get_eval_base_father() const {
    return m_eval_base_father;
}

void UCTNode::set_eval_base_father(float bonus) {
    m_eval_base_father = bonus;
}

float UCTNode::get_net_eval() const {
    return m_net_eval;
}

float UCTNode::get_net_beta() const {
    return m_net_beta;
}

float UCTNode::get_net_alpkt() const {
    return m_net_alpkt;
}

void UCTNode::set_values(float value, float alpkt, float beta) {
    m_net_eval = value;
    m_net_alpkt = alpkt;
    m_net_beta = beta;
}

void UCTNode::set_progid(int id) {
    assert(m_progid == -1 && id >= 0);
    m_progid = id;
}

int UCTNode::get_progid() const {
    return m_progid;
}

void UCTNode::set_score(float score) {
    m_score = score;
}

int UCTNode::get_visits() const {
    return m_visits;
}

#ifndef NDEBUG
void UCTNode::set_urgency(float urgency,
                          float psa,
                          float q,
                          float den,
                          float num) {
    m_last_urgency = {urgency, psa, q, den, num};
}

std::array<float, 5> UCTNode::get_urgency() const {
    return m_last_urgency;
}
#endif

float UCTNode::get_eval(int tomove) const {
    // Due to the use of atomic updates and virtual losses, it is
    // possible for the visit count to change underneath us. Make sure
    // to return a consistent result to the caller by caching the values.
    auto virtual_loss = int{m_virtual_loss};
    auto visits = get_visits() + virtual_loss;
    assert(visits > 0);
    auto blackeval = get_blackevals();
    if (tomove == FastBoard::WHITE) {
        blackeval += static_cast<double>(virtual_loss);
    }
    auto score = static_cast<float>(blackeval / double(visits));
    if (tomove == FastBoard::WHITE) {
        score = 1.0f - score;
    }
    return score;
}

float UCTNode::get_net_eval(int tomove) const {
    if (tomove == FastBoard::WHITE) {
        return 1.0f - m_net_eval;
    }
    return m_net_eval;
}

float UCTNode::get_agent_eval(int tomove) const {
    if (tomove == FastBoard::WHITE) {
        return 1.0f - m_agent_eval;
    }
    return m_agent_eval;
}

double UCTNode::get_blackevals() const {
    return m_blackevals;
}

void UCTNode::accumulate_eval(float eval) {
    atomic_add(m_blackevals, double(eval));
}

UCTNode* UCTNode::uct_select_child(const GameState & currstate, bool is_root,
                                   int max_visits,
                                   std::vector<int> move_list,
                                   bool nopass) {
    LOCK(get_mutex(), lock);

    // Count parentvisits manually to avoid issues with transpositions.
    auto total_visited_policy = 0.0f;
    auto parentvisits = size_t{0};

    // fpu reduction is computed on parent agent_eval or on the
    // largest of the children which have already been visited,
    // whichever is larger.
    const auto color = currstate.get_to_move();
    auto max_eval = get_agent_eval(color);

    for (const auto& child : m_children) {
        if (child.valid()) {
            parentvisits += child.get_visits();
            if (child.get_visits() > 0) {
                const auto child_eval = child.get_eval(color);
                max_eval = std::max (max_eval, child_eval);
                total_visited_policy += child.get_score();
            }
        }
    }

    auto numerator = std::sqrt(double(parentvisits));
    auto fpu_reduction = 0.0f;
    // Lower the expected eval for moves that are likely not the best.
    // Do not do this if we have introduced noise at this node exactly
    // to explore more.
    if (!is_root || !cfg_noise) {
        fpu_reduction = cfg_fpu_reduction * std::sqrt(total_visited_policy);
    }

    // Estimated eval for unknown nodes = original parent NN eval - reduction
    auto fpu_eval = 0.5f;

    if ( !cfg_fpuzero ) {
	fpu_eval = std::max(0.0f, max_eval - fpu_reduction);
    }

    auto best = static_cast<UCTNodePointer*>(nullptr);
    auto best_value = std::numeric_limits<double>::lowest();

#ifndef NDEBUG
    auto b_psa = 0.0f;
    auto b_q = 0.0f;
    auto b_denom = 0.0f;
#endif

    for (auto& child : m_children) {
        if (!child.active()) {
            continue;
        }

        auto is_listed = false;
        for (auto& listed : move_list) {
            if (child.get_move() == listed) {
                is_listed = true;
                break;
            }
        }
        if (!is_listed && move_list.size() > 0) {
            continue;
        }

        const auto visits = child.get_visits();

        // If max_visits is specified, then stop choosing nodes that
        // already have enough visits. This guarantees that
        // exploration is wide enough and not too deep when doing fast
        // roll-outs in the endgame exploration.
        if (max_visits > 0 && visits >= max_visits) {
            continue;
        }

        auto winrate = fpu_eval;
        if (visits > 0) {
            winrate = child.get_eval(color);
        }
        auto psa = child.get_score();

        if (nopass && child.get_move() == FastBoard::PASS) {
            psa = 0.0;
            winrate -= 0.05;
        }

        // If the move to explore is a second pass, or a first pass
        // but -d was not specified, Tromp-Taylor score is checked.

        // If the position appears to be losing, then exploration
        // should be restricted as much as possible.  This is
        // particularly important when there are dead stones and the
        // position is actually winning, as in that case even a small
        // number if visits could change the average probability
        // significantly for the parent node.

        // If the position appears to be winning, one must consider
        // that maybe a larger victory could be achieved: if lambda is
        // positive the average winrate holds this information, while
        // T.T. score at official komi does not, so in this case we
        // only adjust fpu to help the learning in the first
        // generations.


        if (child.get_move() == FastBoard::PASS &&
            ( currstate.get_passes() >= 1 || !cfg_dumbpass ) ) {
            const auto score = ( color == FastBoard::BLACK ? 1.0 : -1.0 ) * 
                currstate.final_score();
            if (score < -0.001) {
                winrate = 0.0;
                psa = 0.0;
            } else if (visits == 0) {
                winrate = Utils::winner(score);
            }
        }

        auto denom = 1.0 + visits;
        auto puct = cfg_puct * psa * (numerator / denom);

        auto value = winrate + puct;
        assert(value > std::numeric_limits<double>::lowest());

        if (value > best_value) {
            best_value = value;
            best = &child;
#ifndef NDEBUG
	    b_psa = psa;
	    b_q = winrate;
	    b_denom = denom;
#endif
	}
    }

    assert(best != nullptr);
    if(best->get_visits() == 0) {
        best->inflate();
        best->get()->set_values(m_net_eval, m_net_alpkt, m_net_beta);
    }
#ifndef NDEBUG
    best->get()->set_urgency(best_value, b_psa, b_q,
                             b_denom, numerator);
#endif
    return best->get();
}

class NodeComp : public std::binary_function<UCTNodePointer&,
                                             UCTNodePointer&, bool> {
public:
    NodeComp(int color) : m_color(color) {};
    bool operator()(const UCTNodePointer& a,
                    const UCTNodePointer& b) {
        // if visits are not same, sort on visits
        if (a.get_visits() != b.get_visits()) {
            return a.get_visits() < b.get_visits();
        }

        // neither has visits, sort on prior score
        if (a.get_visits() == 0) {
            return a.get_score() < b.get_score();
        }

        // both have same non-zero number of visits
        return a.get_eval(m_color) < b.get_eval(m_color);
    }
private:
    int m_color;
};

void UCTNode::sort_children(int color) {
    LOCK(get_mutex(), lock);
    std::stable_sort(rbegin(m_children), rend(m_children), NodeComp(color));
}

class NodeCompByPolicy : public std::binary_function<UCTNodePointer&,
                                             UCTNodePointer&, bool> {
public:
    bool operator()(const UCTNodePointer& a,
                    const UCTNodePointer& b) {
        return a.get_score() < b.get_score();
    }
};

void UCTNode::sort_children_by_policy() {
    LOCK(get_mutex(), lock);
    std::stable_sort(rbegin(m_children), rend(m_children), NodeCompByPolicy());
}

UCTNode& UCTNode::get_best_root_child(int color) {
    LOCK(get_mutex(), lock);
    assert(!m_children.empty());

    auto ret = std::max_element(begin(m_children), end(m_children),
                                NodeComp(color));
    ret->inflate();
    return *(ret->get());
}

size_t UCTNode::count_nodes() const {
    auto nodecount = size_t{0};
    nodecount += m_children.size();
    for (auto& child : m_children) {
        if (child.get_visits() > 0) {
            nodecount += child->count_nodes();
        }
    }
    return nodecount;
}

void UCTNode::invalidate() {
    m_status = INVALID;
}

void UCTNode::set_active(const bool active) {
    if (valid()) {
        m_status = active ? ACTIVE : PRUNED;
    }
}

bool UCTNode::valid() const {
    return m_status != INVALID;
}

bool UCTNode::active() const {
    return m_status == ACTIVE;
}

UCTNode* UCTNode::select_child(int move) {
    auto selected = static_cast<UCTNodePointer*>(nullptr);

    for (auto& child : m_children) {
        if (child.get_move() == move) {
            selected = &child;
            selected->inflate();
            return selected->get();
        }
    }
    return nullptr;
}

void UCTNode::get_subtree_alpkts(std::vector<float> & vector,
                                 int passes,
                                 bool is_tromptaylor_scoring) const {
    auto children_visits = 0;

    vector.emplace_back(get_net_alpkt());
    for (auto& child : m_children) {
        const auto child_visits = child.get_visits();
        if (child_visits > 0) {
            const auto pass = (child.get_move() == FastBoard::PASS) ? 1 : 0;
            child->get_subtree_alpkts(vector, ++passes * pass,
                                      is_tromptaylor_scoring);
                       children_visits += child_visits;
        }
    }

    const auto missing_nodes = get_visits() - children_visits - 1;
    if (missing_nodes > 0 && is_tromptaylor_scoring) {
        const std::vector<float> rep(missing_nodes, get_net_alpkt());
        vector.insert(vector.end(), std::begin(rep), std::end(rep));
    }

    return;
}

float UCTNode::estimate_alpkt(int passes,
                              bool is_tromptaylor_scoring) const {
    std::vector<float> subtree_alpkts;

    get_subtree_alpkts(subtree_alpkts, passes, is_tromptaylor_scoring);

    return Utils::median(subtree_alpkts);
}
