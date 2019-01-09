/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2018 Gian-Carlo Pascutto and contributors
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

#ifndef NETWORK_H_INCLUDED
#define NETWORK_H_INCLUDED

#include "config.h"

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <fstream>

#include "FastState.h"
#include "GameState.h"



std::pair<float,float> sigmoid(float alpha, float beta, float bonus);

extern bool is_mult_komi_net;
extern std::array<std::array<int, BOARD_SQUARES>, 8>
    symmetry_nn_idx_table;   
    


class Network {
public:
    enum Ensemble {
        DIRECT, RANDOM_SYMMETRY, AVERAGE
    };
    using ScoreVertexPair = std::pair<float,int>;

    struct Netresult {
        // 19x19 board positions
        std::vector<float> policy;

        // pass
        float policy_pass;

        // winrate
        float value;

        // sigmoid alpha
        float alpha;

        // sigmoid beta
        float beta;

        Netresult() : policy(BOARD_SQUARES), policy_pass(0.0f), alpha(0.0f), beta(0.0f) {}
    };

    // Results which may obtained by a Netresult together with a FastState
    struct Netresult_extended {
        float winrate;
        float alpkt;
        float pi;
        float eval_bonus;
        float eval_base;
        float agent_eval;
    };

    static Netresult get_scored_moves(const GameState* const state,
                                      const Ensemble ensemble,
                                      const int symmetry = -1,
                                      const bool skip_cache = false);

    static constexpr unsigned short int SINGLE = 1;
    static constexpr unsigned short int DOUBLE_V = 2;
    static constexpr unsigned short int DOUBLE_Y = 3;
    static constexpr unsigned short int DOUBLE_T = 4;
    static constexpr unsigned short int DOUBLE_I = 5;
    static constexpr unsigned int DEFAULT_INPUT_MOVES = 8;
    static constexpr unsigned int REDUCED_INPUT_MOVES = 4;
    static constexpr unsigned int DEFAULT_ADV_FEATURES = 0;
    static constexpr auto DEFAULT_COLOR_INPUT_PLANES = (2 + DEFAULT_ADV_FEATURES) * DEFAULT_INPUT_MOVES + 2;
    //    static constexpr auto DEFAULT_NOCOL_INPUT_PLANES = (2 + DEFAULT_ADV_FEATURES) * INPUT_MOVES + 1;

    // Winograd filter transformation changes 3x3 filters to 4x4
    static constexpr auto WINOGRAD_ALPHA = 4;
    static constexpr auto WINOGRAD_TILE = WINOGRAD_ALPHA * WINOGRAD_ALPHA;

    static void initialize();
    static void benchmark(const GameState * const state,
                          const int iterations = 1600);
    static void show_heatmap(const FastState * const state,
                             const Netresult & netres, const bool topmoves);

    static Netresult_extended get_extended(const FastState&, const Netresult& result);

    static std::vector<net_t> gather_features(const GameState* const state,
                                              const int symmetry,
					      const int input_moves = DEFAULT_INPUT_MOVES,
					      const bool adv_features = false,
					      const bool include_color = false);
private:
    static int load_v1_network(std::istream& wtfile, int format_version);
    static int load_network_file(const std::string& filename);
    static void process_bn_var(std::vector<float>& weights,
                               const float epsilon = 1e-5f);

    static std::vector<float> winograd_transform_f(const std::vector<float>& f,
        const int outputs, const int channels);
    static std::vector<float> zeropad_U(const std::vector<float>& U,
        const int outputs, const int channels,
        const int outputs_pad, const int channels_pad);
    static void winograd_transform_in(const std::vector<float>& in,
                                      std::vector<float>& V,
                                      const int C);
    static void winograd_transform_out(const std::vector<float>& M,
                                       std::vector<float>& Y,
                                       const int K);
    static void winograd_convolve3(const int outputs,
                                   const std::vector<float>& input,
                                   const std::vector<float>& U,
                                   std::vector<float>& V,
                                   std::vector<float>& M,
                                   std::vector<float>& output);
    static void winograd_sgemm(const std::vector<float>& U,
                               const std::vector<float>& V,
                               std::vector<float>& M, const int C, const int K);
    static int get_nn_idx_symmetry(const int vertex, int symmetry);
    static void fill_input_plane_pair(const FullBoard& board,
                                      std::vector<net_t>::iterator black,
                                      std::vector<net_t>::iterator white,
                                      const int symmetry);
    static void fill_input_plane_advfeat(std::shared_ptr<const KoState> const state,
					 std::vector<net_t>::iterator legal,
					 std::vector<net_t>::iterator atari,
					 const int symmetry);
    static Netresult get_scored_moves_internal(const GameState* const state,
                                               const int symmetry);
#if defined(USE_BLAS)
    static void forward_cpu(const std::vector<float>& input,
                            std::vector<float>& output_pol,
                            std::vector<float>& output_val,
                            std::vector<float>& output_vbe);

#endif
};


struct netarch {
    int value_head_type = Network::SINGLE;
    size_t residual_blocks = size_t{3};
    size_t channels = size_t{128};
    size_t input_moves = size_t{Network::DEFAULT_INPUT_MOVES};
    size_t input_planes = size_t{Network::DEFAULT_COLOR_INPUT_PLANES};
    bool adv_features = false;
    bool komi_policy = false;
    bool include_color = true;
    size_t policy_outputs = size_t{2};
    size_t komipolicy_chans = size_t{0};
    size_t val_outputs = size_t{1};
    size_t vbe_outputs = size_t{0};
    size_t val_chans = size_t{256};
    size_t vbe_chans = size_t{0};
    size_t value_head_rets = size_t{1};
};

extern netarch arch;


#endif
