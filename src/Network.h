/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2018 Gian-Carlo Pascutto and contributors

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

#include <deque>
#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <fstream>
#include <tuple>

#include "NNCache.h"
#include "FastState.h"
#ifdef USE_OPENCL
#include "OpenCLScheduler.h"
#endif
#include "GameState.h"
#include "ForwardPipe.h"
#ifdef USE_OPENCL
#include "OpenCLScheduler.h"
#endif
#ifdef USE_OPENCL_SELFCHECK
#include "SMP.h"
#endif

// Winograd filter transformation changes 3x3 filters to M + 3 - 1
constexpr auto WINOGRAD_M = 4;
constexpr auto WINOGRAD_ALPHA = WINOGRAD_M + 3 - 1;
constexpr auto WINOGRAD_WTILES = BOARD_SIZE / WINOGRAD_M + (BOARD_SIZE % WINOGRAD_M != 0);
constexpr auto WINOGRAD_TILE = WINOGRAD_ALPHA * WINOGRAD_ALPHA;
constexpr auto WINOGRAD_P = WINOGRAD_WTILES * WINOGRAD_WTILES;
constexpr auto SQ2 = 1.4142135623730951f; // Square root of 2

std::pair<float, float> sigmoid(float alpha, float beta, float bonus);

extern std::array<std::array<int, NUM_INTERSECTIONS>, 8>
    symmetry_nn_idx_table;

class Network
{
    using ForwardPipeWeights = ForwardPipe::ForwardPipeWeights;

  public:
    static constexpr auto NUM_SYMMETRIES = 8;
    static constexpr auto IDENTITY_SYMMETRY = 0;
    enum Ensemble
    {
        DIRECT,
        RANDOM_SYMMETRY,
        AVERAGE
    };
    using PolicyVertexPair = std::pair<float, int>;
    using Netresult = NNCache::Netresult;

    // Results which may obtained by a Netresult together with a FastState
    struct Netresult_extended {
        float winrate;
        float alpkt;
        float pi;
        float eval_bonus;
        float eval_base;
        float agent_eval;
    };

    Netresult get_output(const GameState *const state,
                         const Ensemble ensemble,
                         const int symmetry = -1,
                         const bool skip_cache = false,
                         const bool force_selfcheck = false);

    static constexpr unsigned short int SINGLE = 1;
    static constexpr unsigned short int DOUBLE_V = 2;
    static constexpr unsigned short int DOUBLE_Y = 3;
    static constexpr unsigned short int DOUBLE_T = 4;
    static constexpr unsigned short int DOUBLE_I = 5;
    static constexpr unsigned int DEFAULT_INPUT_MOVES = 8;
    static constexpr unsigned int REDUCED_INPUT_MOVES = 4;
    static constexpr unsigned int DEFAULT_ADV_FEATURES = 0;
    static constexpr auto DEFAULT_COLOR_INPUT_PLANES = (2 + DEFAULT_ADV_FEATURES) * DEFAULT_INPUT_MOVES + 2;

    void initialize(int playouts, const std::string &weightsfile);

    float benchmark_time(int centiseconds);
    void benchmark(const GameState *const state,
                   const int iterations = 1600);
    static void show_heatmap(const FastState *const state,
                             const Netresult &netres, const bool topmoves);
    static Netresult_extended get_extended(const FastState &, const Netresult &result);
    static std::vector<float> gather_features(const GameState *const state,
                                              const int symmetry,
                                              const int input_moves = DEFAULT_INPUT_MOVES,
                                              const bool adv_features = false,
                                              const bool include_color = false);
    static std::pair<int, int> get_symmetry(const std::pair<int, int> &vertex,
                                            const int symmetry,
                                            const int board_size = BOARD_SIZE);

    size_t get_estimated_size();
    size_t get_estimated_cache_size();
    void nncache_resize(int max_count);

    int m_value_head_type = SINGLE;
    bool m_value_head_sai; // was is_multi_komi_net
    size_t m_residual_blocks = size_t{3};
    size_t m_channels = size_t{128};
    size_t m_input_moves = size_t{DEFAULT_INPUT_MOVES};
    size_t m_input_planes = size_t{DEFAULT_COLOR_INPUT_PLANES};
    bool m_adv_features = false;
    bool m_komi_policy = false;
    bool m_include_color = true;
    size_t m_policy_outputs = size_t{2};
    size_t m_komipolicy_chans = size_t{0};
    size_t m_val_outputs = size_t{1};
    size_t m_vbe_outputs = size_t{0};
    size_t m_val_chans = size_t{256};
    size_t m_vbe_chans = size_t{0};
    size_t m_value_head_rets = size_t{1};

  private:
    int load_v1_network(std::istream &wtfile, int format_version);
    int load_network_file(const std::string &filename);

    static std::vector<float> winograd_transform_f(const std::vector<float> &f,
                                                   const int outputs, const int channels);
    static std::vector<float> zeropad_U(const std::vector<float> &U,
                                        const int outputs, const int channels,
                                        const int outputs_pad, const int channels_pad);
    static void winograd_transform_in(const std::vector<float> &in,
                                      std::vector<float> &V,
                                      const int C);
    static void winograd_transform_out(const std::vector<float> &M,
                                       std::vector<float> &Y,
                                       const int K);
    static void winograd_convolve3(const int outputs,
                                   const std::vector<float> &input,
                                   const std::vector<float> &U,
                                   std::vector<float> &V,
                                   std::vector<float> &M,
                                   std::vector<float> &output);
    static void winograd_sgemm(const std::vector<float> &U,
                               const std::vector<float> &V,
                               std::vector<float> &M, const int C, const int K);
    Netresult get_output_internal(const GameState *const state,
                                  const int symmetry, bool selfcheck = false);
    static void fill_input_plane_pair(const FullBoard &board,
                                      std::vector<float>::iterator black,
                                      std::vector<float>::iterator white,
                                      const int symmetry);
    static void fill_input_plane_advfeat(std::shared_ptr<const KoState> const state,
                                         std::vector<float>::iterator legal,
                                         std::vector<float>::iterator atari,
                                         const int symmetry);

    bool probe_cache(const GameState *const state, Network::Netresult &result);
    std::unique_ptr<ForwardPipe> &&init_net(int channels,
                                            std::unique_ptr<ForwardPipe> &&pipe);
#ifdef USE_HALF
    void select_precision(int channels);
#endif
    std::unique_ptr<ForwardPipe> m_forward;
#ifdef USE_OPENCL_SELFCHECK
    void compare_net_outputs(const Netresult &data, const Netresult &ref);
    std::unique_ptr<ForwardPipe> m_forward_cpu;
#endif

    NNCache m_nncache;

    size_t estimated_size{0};

    // Residual tower
    std::shared_ptr<ForwardPipeWeights> m_fwd_weights;

    // Policy head
    std::vector<float> m_bn_pol_w1; // policy_outputs
    std::vector<float> m_bn_pol_w2; // policy_outputs

    std::vector<float> m_kp1_pol_w; // (board_sq*policy_outputs+1)*komipolicy_chans
    std::vector<float> m_kp1_pol_b; // komipolicy_chans

    std::vector<float> m_kp2_pol_w; // komipolicy_chans*komipolicy_chans
    std::vector<float> m_kp2_pol_b; // board_sq*policy_outputs*(board_sq+1)

    std::vector<float> m_ip_pol_w;  // (board_sq*policy_outputs + komipolicy_chans)*(board_sq+1)
    std::vector<float> m_ip_pol_b;  // board_sq+1

    // Value head alpha (val=Value ALpha)
    std::vector<float> m_bn_val_w1; // val_outputs
    std::vector<float> m_bn_val_w2; // val_outputs

    std::vector<float> m_ip1_val_w; // board_sq*val_outputs*val_chans
    std::vector<float> m_ip1_val_b; // val_chans

    std::vector<float> m_ip2_val_w; // val_chans (*2 in SINGLE head type)
    std::vector<float> m_ip2_val_b; // 1 (2 in SINGLE head type)

    bool m_value_head_not_stm;

    // Value head beta (vbe=Value BEta)
    std::vector<float> m_bn_vbe_w1; // vbe_outputs
    std::vector<float> m_bn_vbe_w2; // vbe_outputs

    std::vector<float> m_ip1_vbe_w; // board_sq*vbe_outputs*vbe_chans
    std::vector<float> m_ip1_vbe_b; // vbe_chans

    std::vector<float> m_ip2_vbe_w; // vbe_chans
    std::vector<float> m_ip2_vbe_b; // 1
};
#endif
