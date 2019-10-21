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
    float alpkt_online_median;
    float beta_median;
    float azwinrate_avg;
};

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
    float get_policy() const;
    void set_policy(float policy);
    float get_eval_variance(float default_var = 0.0f) const;
    float get_eval(int tomove) const;
    float get_raw_eval(int tomove, int virtual_loss = 0) const;
    float get_net_eval(int tomove) const;
    float get_agent_eval(int tomove) const;
    float get_eval_bonus() const;
    float get_eval_bonus_father() const;
    void set_eval_bonus_father(float bonus);
    float get_eval_base() const;
    float get_eval_base_father() const;
    void set_eval_base_father(float bonus);
    float get_net_eval() const;
    float get_net_beta() const;
    float get_net_alpkt() const;
    float get_alpkt_online_median() const;
    void set_values(float value, float alpkt, float beta);
    bool low_visits_child(UCTNode* const child) const;
#ifdef USE_EVALCMD
    void set_progid(int id);
    int get_progid() const;
#endif
#ifndef NDEBUG
    void set_urgency(float urgency, float psa, float q,
                     float num, float den);
    std::array<float, 5> get_urgency() const;
#endif
    void virtual_loss();
    void virtual_loss_undo();
    void clear_visits();
    void clear_children_visits();
    void update(float eval);
    float get_eval_lcb(int color) const;

    // Defined in UCTNodeRoot.cpp, only to be called on m_root in UCTSearch
    std::tuple<bool,std::vector<int>>
      randomize_first_proportionally(int color, bool is_blunder_allowed);

    void prepare_root_node(Network & network, int color,
                           std::atomic<int>& nodecount,
                           GameState& state,
                           bool fast_roll_out = false);
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
    void update_alpkt_median(float new_alpkt_value);

    void clear_expand_state();
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

    // Note : This class is very size-sensitive as we are going to create
    // tens of millions of instances of these.  Please put extra caution
    // if you want to add/remove/reorder any variables here.

    // Move
    std::int16_t m_move;
    // UCT
    std::atomic<std::int16_t> m_virtual_loss{0};
    std::atomic<int> m_visits{0};
    // UCT eval
    float m_policy;
    // Original net eval for this node (not children).
    float m_net_eval{0.5f};
    //    float m_net_value{0.5f};
    float m_net_alpkt{0.0f}; // alpha + \tilde k
    float m_net_beta{1.0f};
    float m_eval_bonus{0.0f}; // x bar
    float m_eval_base{0.0f}; // x base
    float m_eval_base_father{0.0f}; // x base of father node
    float m_eval_bonus_father{0.0f}; // x bar of father node
#ifdef USE_EVALCMD
    int m_progid{-1}; // progressive unique identifier
#endif
#ifndef NDEBUG
    std::array<float, 5> m_last_urgency;
#endif

    // the following is used only in fpu, with reduction
    float m_agent_eval{0.5f}; // eval_with_bonus(eval_bonus()) no father
    // Variable used for calculating variance of evaluations.
    // Initialized to small non-zero value to avoid accidental zero variances
    // at low visits.
    std::atomic<float> m_squared_eval_diff{1e-4f};
    std::atomic<double> m_blackevals{0.0};
    std::atomic<Status> m_status{ACTIVE};

    std::atomic<float> m_alpkt_median{0.0f};

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

    //  m_expand_state manipulation methods
    // INITIAL -> EXPANDING
    // Return false if current state is not INITIAL
    bool acquire_expanding();

    // EXPANDING -> DONE
    void expand_done();

    // EXPANDING -> INITIAL
    void expand_cancel();

    // wait until we are on EXPANDED state
    void wait_expanded();
};

#endif
