/*
    This file is part of SAI, which is a fork of Leela Zero.
    Copyright (C) 2017-2019 Gian-Carlo Pascutto and contributors
    Copyright (C) 2018-2019 SAI Team

    SAI is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    SAI is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with SAI.  If not, see <http://www.gnu.org/licenses/>.

    Additional permission under GNU GPL version 3 section 7

    If you modify this Program, or any covered work, by linking or
    combining it with NVIDIA Corporation's libraries from the
    NVIDIA CUDA Toolkit and/or the NVIDIA CUDA Deep Neural
    Network library and/or the NVIDIA TensorRT inference library
    (or a modified version of those libraries), containing parts covered
    by the terms of the respective license agreement, the licensors of
    this Program grant you additional permission to convey the resulting
    work.
*/

#ifndef UCTNODE_H_INCLUDED
#define UCTNODE_H_INCLUDED

#include "config.h"

#include <atomic>
#include <memory>
#include <vector>
#include <tuple>
#include <cassert>
#include <cstring>

#include "GameState.h"
#include "Network.h"
#include "SMP.h"
#include "UCTNodePointer.h"
#include "UCTSearch.h"

struct UCTStats {
    float alpkt_tree;
    float beta_median;
    float azwinrate_avg;
};

class SearchResult;

class UCTNode {
public:
    // When we visit a node, add this amount of virtual losses
    // to it to encourage other CPUs to explore other parts of the
    // search tree.
    static constexpr auto VIRTUAL_LOSS_COUNT = 3;
    // Defined in UCTNode.cpp
    explicit UCTNode(int vertex, float policy);
    UCTNode() = delete;
    ~UCTNode() = default;

    bool create_children(Network & network,
                         std::atomic<int>& nodecount,
                         GameState& state, float& value, float& alpkt,
                                     float& beta,
                         float min_psa_ratio = 0.0f);

    const std::vector<UCTNodePointer>& get_children() const;
    void sort_children_by_policy();
    void sort_children(int color, float lcb_min_visits);
    UCTNode& get_best_root_child(int color);
    UCTNode* uct_select_child(const GameState & currstate, bool is_root,
                              int max_visits,
                              const std::vector<int> & move_list,
                              bool nopass = false);

    size_t count_nodes_and_clear_expand_state();
    bool first_visit() const;
    bool has_children() const;
    bool expandable(const float min_psa_ratio = 0.0f) const;
    void invalidate();
    void set_active(const bool active);
    bool valid() const;
    bool active() const;
    double get_blackevals() const;
    int get_move() const;
    int get_visits() const;
    int get_denom() const;
    float get_policy() const;
    void set_policy(float policy);
    float get_eval_variance(float default_var = 0.0f) const;
    float get_eval(int tomove = FastBoard::BLACK) const;
    float get_raw_eval(int tomove, int virtual_loss = 0) const;
    float get_net_pi(int tomove = FastBoard::BLACK) const;
    void set_values(float value, float alpkt, float beta);
    bool low_visits_child(UCTNode* const child) const;
#ifdef USE_EVALCMD
    void set_progid(int id);
    std::vector<int>& get_progid();
#endif
#ifndef NDEBUG
    void set_urgency(float urgency, float psa, float q,
                     float num, float den);
    std::array<float, 5> get_urgency() const;
#endif
    void virtual_loss();
    void virtual_loss_undo();
    float update(const SearchResult &result, bool forced=false);
    float get_eval_lcb(int color) const;

    // Defined in UCTNodeRoot.cpp, only to be called on m_root in UCTSearch
    FastState::move_flags_t
      randomize_first_proportionally(int color, bool is_blunder_allowed);

    void prepare_root_node(Network & network, int color,
                           std::atomic<int>& nodecount,
                           GameState& state,
                           bool fast_roll_out = false,
                           bool verbose = true);
    bool get_children_visits(const GameState& state, const UCTNode& root,
                             std::vector<float> & probabilities,
                             bool standardize = true);

    UCTNode* get_first_child() const;
    UCTNode* get_second_child() const;
    UCTNode* get_nopass_child(FastState& state) const;
    std::unique_ptr<UCTNode> find_child(const int move);
    void inflate_all_children();
    UCTNode* select_child(int move);
    float estimate_alpkt(int passes, bool is_tromptaylor_scoring = false) const;
    float get_beta_median() const;
    float get_azwinrate_avg() const;
    UCTStats get_uct_stats() const;
    void update_quantile(std::atomic<float> &old_quantile, float old_gxgp_sum,
                         float old_gp_sum, float parameter, int new_visits,
                         float avg_pi, float new_alpkt, float new_beta);
    void update_all_quantiles(float new_alpkt, float new_beta);
    std::tuple<float, float, float> score_stats() const;
    void clear_expand_state();

    float get_avg_pi(int tomove = FastBoard::BLACK) const;
    float get_net_alpkt() const { return m_net_alpkt; }
    float get_net_beta() const { return m_net_beta; }
    float get_lambda() const { return m_lambda; }
    float get_mu() const { return m_mu; }
    float get_quantile_lambda(int tomove = FastBoard::BLACK) const;
    float get_quantile_mu(int tomove = FastBoard::BLACK) const;
    float get_quantile_one() const { return m_quantile_one; }
    float get_father_quantile_lambda() const { return m_father_quantile_lambda; }
    float get_father_quantile_mu() const { return m_father_quantile_mu; }
    void set_father_quantiles(const UCTNode* father) {
        m_father_quantile_lambda = father->get_quantile_lambda();
        m_father_quantile_mu = father->get_quantile_mu();
    }
    StateEval state_eval() const;
    void set_lambda_mu();
    AgentEval get_agent_eval() const {
        return {m_lambda, m_mu, m_quantile_lambda, m_quantile_mu, -m_quantile_one};
    }

private:
    enum Status : char {
        INVALID, // superko
        PRUNED,
        ACTIVE
    };
    void link_nodelist(std::atomic<int>& nodecount,
                       std::vector<Network::PolicyVertexPair>& nodelist,
                       float min_psa_ratio);
    void accumulate_eval(float eval);
    void kill_superkos(const GameState& state);
    void dirichlet_noise(float epsilon, float alpha);
    void get_subtree_alpkts(std::vector<float> & vector, int passes,
                            bool is_tromptaylor_scoring) const;
    void get_subtree_betas(std::vector<float> & vector) const;
    void az_sum_recursion(float& sum, size_t& n) const;
    void update_gxx_sums(std::atomic<float> &old_gxgp_sum,
                         std::atomic<float> &old_gp_sum,
                         float old_quantile,
                         float new_alpkt, float new_beta);

    // Note : This class is very size-sensitive as we are going to create
    // tens of millions of instances of these.  Please put extra caution
    // if you want to add/remove/reorder any variables here.

    // Move
    std::int16_t m_move;
    // UCT
    std::atomic<std::int16_t> m_virtual_loss{0};
    std::atomic<int> m_visits{0};
    // number of forced moves visited after this node, to be
    // subtracted from visits in the denominator of psa
    std::atomic<int> m_forced{0};
    // UCT eval
    float m_policy;
    // Original net eval for this node (not children, black's pov).
    float m_net_pi{0.5f};

#ifdef USE_EVALCMD
    std::vector<int> m_progid; // progressive unique identifier,
                               // typically it is just one integer,
                               // but a second pass can be visited
                               // more than once and in that case the
                               // vector is used
#endif
#ifndef NDEBUG
    std::array<float, 5> m_last_urgency;
#endif

    // Variable used for calculating variance of evaluations.
    // Initialized to small non-zero value to avoid accidental zero variances
    // at low visits.
    std::atomic<float> m_squared_eval_diff{1e-4f};
    std::atomic<double> m_blackevals{0.0};
    std::atomic<Status> m_status{ACTIVE};
    std::atomic<float> m_pi_sum{0.0f};

    // m_expand_state acts as the lock for m_children.
    // see manipulation methods below for possible state transition
    enum class ExpandState : std::uint8_t {
        // initial state, no children
        INITIAL = 0,

        // creating children.  the thread that changed the node's state to
        // EXPANDING is responsible of finishing the expansion and then
        // move to EXPANDED, or revert to INITIAL if impossible
        EXPANDING,

        // expansion done.  m_children cannot be modified on a multi-thread
        // context, until node is destroyed.
        EXPANDED,
    };
    std::atomic<ExpandState> m_expand_state{ExpandState::INITIAL};

    // Tree data
    std::atomic<float> m_min_psa_ratio_children{2.0f};
    std::vector<UCTNodePointer> m_children;

    // m_expand_state manipulation methods
    // INITIAL -> EXPANDING
    // Return false if current state is not INITIAL
    bool acquire_expanding();

    // EXPANDING -> DONE
    void expand_done();

    // EXPANDING -> INITIAL
    void expand_cancel();

    // wait until we are on EXPANDED state
    void wait_expanded();

    float m_net_alpkt{0.0f}; // alpha + \tilde k
    float m_net_beta{1.0f};
    float m_lambda{0.0f};
    float m_mu{0.0f};

    // should be equal to m_visits in single threading
    std::atomic<int> m_quantile_updates{0};

    std::atomic<float> m_quantile_lambda{0.0f}; // x bar
    std::atomic<float> m_quantile_mu{0.0f}; // x base
    std::atomic<float> m_quantile_one{0.0f}; // quantile for parameter = 1, equals -alpkt
    std::atomic<float> m_gxgp_sum_lambda{0.0f};
    std::atomic<float> m_gxgp_sum_mu{0.0f};
    std::atomic<float> m_gxgp_sum_one{0.0f};
    std::atomic<float> m_gp_sum_lambda{0.0f};
    std::atomic<float> m_gp_sum_mu{0.0f};
    std::atomic<float> m_gp_sum_one{0.0f};
    std::atomic<float> m_father_quantile_lambda{0.0f}; // x bar of father node
    std::atomic<float> m_father_quantile_mu{0.0f}; // x base of father node
};

#endif
