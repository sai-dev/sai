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


#include "config.h"
#include "Network.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <boost/utility.hpp>
#include <boost/format.hpp>
#include <boost/spirit/home/x3.hpp>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif
#ifdef USE_MKL
#include <mkl.h>
#endif
#ifdef USE_OPENBLAS
#include <cblas.h>
#endif
#include "zlib.h"
#ifdef USE_OPENCL
#include "OpenCLScheduler.h"
#include "UCTNode.h"
#endif

#include "FastBoard.h"
#include "FastState.h"
#include "FullBoard.h"
#include "GameState.h"
#include "GTP.h"
#include "Im2Col.h"
#include "NNCache.h"
#include "Random.h"
#include "ThreadPool.h"
#include "Timing.h"
#include "Utils.h"

namespace x3 = boost::spirit::x3;
using namespace Utils;

netarch arch;
bool is_mult_komi_net = false;

// Input + residual block tower
static std::vector<std::vector<float>> conv_weights;
static std::vector<std::vector<float>> conv_biases;
static std::vector<std::vector<float>> batchnorm_means;
static std::vector<std::vector<float>> batchnorm_stddivs;

// Policy head
static std::vector<float> conv_pol_w;    // channels*policy_outputs
static std::vector<float> conv_pol_b;    // policy_outputs
static std::vector<float> bn_pol_w1;     // policy_outputs
static std::vector<float> bn_pol_w2;     // policy_outputs

static std::vector<float> kp1_pol_w;      // (board_sq*policy_outputs+1)*komipolicy_chans
static std::vector<float> kp1_pol_b;      // komipolicy_chans

static std::vector<float> kp2_pol_w;      // komipolicy_chans*komipolicy_chans
static std::vector<float> kp2_pol_b;      // board_sq*policy_outputs*(board_sq+1)

static std::vector<float> ip_pol_w;      // (board_sq*policy_outputs + komipolicy_chans)*(board_sq+1)
static std::vector<float> ip_pol_b;      // board_sq+1

// Value head alpha (val=Value ALpha)
static std::vector<float> conv_val_w;    // channels*val_outputs
static std::vector<float> conv_val_b;    // val_outputs
static std::vector<float> bn_val_w1;     // val_outputs
static std::vector<float> bn_val_w2;     // val_outputs

static std::vector<float> ip1_val_w;     // board_sq*val_outputs*val_chans
static std::vector<float> ip1_val_b;     // val_chans

static std::vector<float> ip2_val_w;     // val_chans (*2 in SINGLE head type)
static std::vector<float> ip2_val_b;     // 1 (2 in SINGLE head type)
static bool value_head_not_stm;

// Symmetry helper
std::array<std::array<int, BOARD_SQUARES>, 8> symmetry_nn_idx_table;

// Value head beta (vbe=Value BEta)
static std::vector<float> conv_vbe_w;    // channels*vbe_outputs
static std::vector<float> conv_vbe_b;    // vbe_outputs
static std::vector<float> bn_vbe_w1;     // vbe_outputs
static std::vector<float> bn_vbe_w2;     // vbe_outputs

static std::vector<float> ip1_vbe_w;     // board_sq*vbe_outputs*vbe_chans
static std::vector<float> ip1_vbe_b;     // vbe_chans

static std::vector<float> ip2_vbe_w;     // vbe_chans
static std::vector<float> ip2_vbe_b;     // 1

void Network::benchmark(const GameState* const state, const int iterations) {
    const auto cpus = cfg_num_threads;
    const Time start;

    ThreadGroup tg(thread_pool);
    std::atomic<int> runcount{0};

    for (auto i = 0; i < cpus; i++) {
        tg.add_task([&runcount, iterations, state]() {
            while (runcount < iterations) {
                runcount++;
                get_scored_moves(state, Ensemble::RANDOM_SYMMETRY, -1, true);
            }
        });
    }
    tg.wait_all();

    const Time end;
    const auto elapsed = Time::timediff_seconds(start, end);
    myprintf("%5d evaluations in %5.2f seconds -> %d n/s\n",
             runcount.load(), elapsed, int(runcount.load() / elapsed));
}

void Network::process_bn_var(std::vector<float>& weights, const float epsilon) {
    for (auto&& w : weights) {
        w = 1.0f / std::sqrt(w + epsilon);
    }
}

std::vector<float> Network::winograd_transform_f(const std::vector<float>& f,
                                                 const int outputs,
                                                 const int channels) {
    // F(2x2, 3x3) Winograd filter transformation
    // transpose(G.dot(f).dot(G.transpose()))
    // U matrix is transposed for better memory layout in SGEMM
    auto U = std::vector<float>(WINOGRAD_TILE * outputs * channels);
    // [fm] the following array has 16 components, but only 12 are set
    const auto G = std::array<float, WINOGRAD_TILE>{ 1.0,  0.0,  0.0,
                                                     0.5,  0.5,  0.5,
                                                     0.5, -0.5,  0.5,
                                                     0.0,  0.0,  1.0};
    auto temp = std::array<float, 12>{};

    for (auto o = 0; o < outputs; o++) {
        for (auto c = 0; c < channels; c++) {
            for (auto i = 0; i < 4; i++){
                for (auto j = 0; j < 3; j++) {
                    auto acc = 0.0f;
                    for (auto k = 0; k < 3; k++) {
                        acc += G[i*3 + k] * f[o*channels*9 + c*9 + k*3 + j];
                    }
                    temp[i*3 + j] = acc;
                }
            }

            for (auto xi = 0; xi < 4; xi++) {
                for (auto nu = 0; nu < 4; nu++) {
                    auto acc = 0.0f;
                    for (auto k = 0; k < 3; k++) {
                        acc += temp[xi*3 + k] * G[nu*3 + k];
                    }
                    U[xi * (4 * outputs * channels)
                      + nu * (outputs * channels)
                      + c * outputs
                      + o] = acc;
                }
            }
        }
    }

    return U;
}

std::vector<float> Network::zeropad_U(const std::vector<float>& U,
                                      const int outputs, const int channels,
                                      const int outputs_pad,
                                      const int channels_pad) {
    // Fill with zeroes
    auto Upad = std::vector<float>(WINOGRAD_TILE * outputs_pad * channels_pad);

    for (auto o = 0; o < outputs; o++) {
        for (auto c = 0; c < channels; c++) {
            for (auto xi = 0; xi < WINOGRAD_ALPHA; xi++){
                for (auto nu = 0; nu < WINOGRAD_ALPHA; nu++) {
                    Upad[xi * (WINOGRAD_ALPHA * outputs_pad * channels_pad)
                         + nu * (outputs_pad * channels_pad)
                         + c * outputs_pad +
                          o] =
                    U[xi * (WINOGRAD_ALPHA * outputs * channels)
                      + nu * (outputs * channels)
                      + c * outputs
                      + o];
                }
            }
        }
    }

    return Upad;
}

//v1 refers to the actual weight file format, to be changed when/if the weight file format changes
int Network::load_v1_network(std::istream& wtfile, int format_version) {
    // Count size of the network
    myprintf("Detecting residual layers... v%d\n", format_version);

    auto komipolicy_lines = 0;
    arch.komipolicy_chans = 0; 
    if (format_version == 49) {
        // in this format there are 4 additional lines between policy
        // 1x1 convolution and policy dense layer
        
        komipolicy_lines = 4;
    }
    
    // First line was the version number
    auto linecount = size_t{1};
    int lastlines = 0;
    auto line = std::string{};
    size_t plain_conv_layers = 0;
    size_t plain_conv_wts = 0;
    std::array<std::vector<float>, 8> wts_2nd_val_head;
    std::array<std::vector<float>::size_type, 8> n_wts_2nd_val_head;

    bool is_head_line = false;
    linecount = 0;
    auto n_wts_1st_layer = size_t{0};

    while (std::getline(wtfile, line)) {
        std::vector<float> weights;
        auto it_line = line.cbegin();
        const auto ok = phrase_parse(it_line, line.cend(),
                                     *x3::float_, x3::space, weights);
        if (!ok || it_line != line.cend()) {
            myprintf("\nFailed to parse weight file. Error on line %d.\n",
                    linecount + 2); //+1 from version line, +1 from 0-indexing
            return 1;
        }
	auto n_wts = weights.size();
        if (!is_head_line) {
            // we should be still in the convolutional tower
            // (or we just exit)

            if (linecount % 4 == 0) {
                // first line of 4: holds convolutional weights

                if (linecount == 0)
		n_wts_1st_layer = n_wts;

                // check if we are still in the resconv tower
	      if (linecount==0 || n_wts==arch.channels*9*arch.channels)

                  // yes: these are convolutional weights
                conv_weights.emplace_back(weights);
	      else {

                  // no: first line of policy head [1x1 conv weights]
		is_head_line = true;
		arch.policy_outputs = n_wts/arch.channels;
		assert (n_wts == arch.channels*arch.policy_outputs);
		conv_pol_w = std::move(weights);
		arch.residual_blocks = (linecount-4)/8;
		plain_conv_layers = 1 + (arch.residual_blocks * 2);
		plain_conv_wts = plain_conv_layers * 4;
		assert(plain_conv_wts == linecount);
		myprintf(" %d blocks\n%d policy outputs", arch.residual_blocks, arch.policy_outputs);
	      }
            } else if (linecount % 4 == 1) {
                // second line of 4: holds convolutional biases

                if (linecount == 1) {
		  // second line of weights, holds the biases for the
		  // input convolutional layer, hence its size gives
		  // the number of channels of subsequent resconv
		  // layers
		  arch.channels = n_wts;

		  // we recover the number of input planes
		  arch.input_planes = n_wts_1st_layer/9/arch.channels;

		  // if it is even, color of the current player is
		  // used, if it is odd, only komi is used
		  arch.include_color = (0 == arch.input_planes % 2);

		  // we recover the number of input moves, knowing
		  // that for each move there are 2 bitplanes with
		  // stones positions and possibly 2 more bitplanes
		  // with some advanced features (legal and atari)
		  arch.input_moves = (arch.input_planes - (arch.include_color ? 2 : 1)) /
		      (arch.adv_features ? 4 : 2);

		  assert (n_wts_1st_layer == arch.input_planes*9*arch.channels);
		  myprintf("%d input planes, %d input moves\n%d channels...",
			   arch.input_planes,
			   arch.input_moves,
			   arch.channels);
	      }
	      else
		assert (n_wts == arch.channels);

	      conv_biases.emplace_back(weights);
            } else if (linecount % 4 == 2) {
		assert (n_wts == arch.channels);
                batchnorm_means.emplace_back(weights);
            } else if (linecount % 4 == 3) {
	        assert (n_wts == arch.channels);
                process_bn_var(weights);
                batchnorm_stddivs.emplace_back(weights);
            }
        } else if (linecount == plain_conv_wts + 1) {
            // line 2 of policy head [1x1 convolutional biases]

            assert (n_wts == arch.policy_outputs);
	    conv_pol_b = std::move(weights);
        } else if (linecount == plain_conv_wts + 2) {
            // line 3 of policy head [1x1 convolutional bn 1]

	    assert (n_wts == arch.policy_outputs);
	    bn_pol_w1 = std::move(weights);
        } else if (linecount == plain_conv_wts + 3) {
            // line 4 of policy head [1x1 convolutional bn 2]

	    process_bn_var(weights);
	    assert (n_wts == arch.policy_outputs);
	    bn_pol_w2 = std::move(weights);

            // if the net has 'komi policy' layers, then their weigths
            // go here
        } else if (komipolicy_lines && linecount == plain_conv_wts + 4) {
            // line 1 of komi policy layers

            arch.komipolicy_chans = n_wts /
                (BOARD_SQUARES * arch.policy_outputs + 1);
            assert (n_wts == (BOARD_SQUARES * arch.policy_outputs + 1)
                    * arch.komipolicy_chans);
            kp1_pol_w = std::move(weights);
        } else if (komipolicy_lines && linecount == plain_conv_wts + 5) {
            // line 2 of komi policy layers

            assert (n_wts == arch.komipolicy_chans);
            kp1_pol_b = std::move(weights);
        } else if (komipolicy_lines && linecount == plain_conv_wts + 6) {
            // line 3 of komi policy layers

            assert (n_wts == arch.komipolicy_chans * arch.komipolicy_chans);
            kp2_pol_w = std::move(weights);
        } else if (komipolicy_lines && linecount == plain_conv_wts + 7) {
            // line 4 of komi policy layers

            assert (n_wts == arch.komipolicy_chans);
            kp2_pol_b = std::move(weights);
        } else if (linecount == plain_conv_wts + komipolicy_lines + 4) {
            // line [-2] of policy head [dense layer weights]

            // we do not check the board size here as it would be
            // uselessly complicate

            assert (n_wts == (arch.policy_outputs * BOARD_SQUARES
                              + arch.komipolicy_chans ) // assumes this is 0 when
                                                        // komi policy is not used
                              * (BOARD_SQUARES + 1) );

	    ip_pol_w = std::move(weights);

        } else if (linecount == plain_conv_wts + komipolicy_lines + 5) {
            // line [-1] of policy head [dense layer biases]
            // check if the board size is correct

	    if (n_wts != BOARD_SQUARES+1) {
                const auto netboardsize = std::sqrt(n_wts-1);
                myprintf("\nGiven network is for %.0fx%.0f, but this version "
                         "of SAI was compiled for %dx%d board!\n",
                         netboardsize, netboardsize, BOARD_SIZE, BOARD_SIZE);
                return 1;
            }

            myprintf("\n%dx%d board.\n", BOARD_SIZE, BOARD_SIZE);

            ip_pol_b = std::move(weights);

        } else if (linecount == plain_conv_wts + komipolicy_lines + 6) {
            // line 1 of value head [1x1 convolutional weights]

	    arch.val_outputs = n_wts/arch.channels;
	    assert (n_wts == arch.channels*arch.val_outputs);
	    conv_val_w = std::move(weights);

        } else if (linecount == plain_conv_wts + komipolicy_lines + 7) {
            // line 2 of value head [1x1 convolutional biases]

	    assert (n_wts == arch.val_outputs);
            conv_val_b = std::move(weights);

        } else if (linecount == plain_conv_wts + komipolicy_lines + 8) {
            // line 3 of value head [1x1 convolutional bn 1]

	    assert (n_wts == arch.val_outputs);
            bn_val_w1 = std::move(weights);

        } else if (linecount == plain_conv_wts + komipolicy_lines + 9) {
            // line 4 of value head [1x1 convolutional bn 2]

	    assert (n_wts == arch.val_outputs);
            process_bn_var(weights);
            bn_val_w2 = std::move(weights);

        } else if (linecount == plain_conv_wts + komipolicy_lines + 10) {
            // line 5 of value head [dense layer weights]

	    arch.val_chans = n_wts/arch.val_outputs/(BOARD_SQUARES);
	    assert (n_wts == arch.val_chans*arch.val_outputs*BOARD_SQUARES);
	    ip1_val_w = std::move(weights);

        } else if (linecount == plain_conv_wts + komipolicy_lines + 11) {
            // line 6 of value head [dense layer biases]

	    assert (n_wts == arch.val_chans);
            ip1_val_b = std::move(weights);
        } else if (linecount == plain_conv_wts + komipolicy_lines + 12) {
            // line 7 of value head [last unit(s) weights]

	    arch.value_head_rets = n_wts/arch.val_chans;
	    assert (n_wts == arch.val_chans*arch.value_head_rets);
	    assert (arch.value_head_rets == 1 || arch.value_head_rets == 2);
	    ip2_val_w = std::move(weights);

        } else if (linecount == plain_conv_wts + komipolicy_lines + 13) {
            // line 8 of value head [last unit(s) biases]

	    assert (n_wts == arch.value_head_rets);
            ip2_val_b = std::move(weights);

        } else if (linecount >= plain_conv_wts + komipolicy_lines + 14) {
            // read the second value head, if present, store it
            // temporarily and count how many lines long it is

	    auto i = lastlines;
	    assert (i>=0 && i<8);
            wts_2nd_val_head[i] = std::move(weights);
	    n_wts_2nd_val_head[i] = n_wts;
            lastlines++;
        }
        linecount++;
    }

    if (lastlines == 8) {
        arch.value_head_type = DOUBLE_V;
        arch.value_head_rets = 2;

	arch.vbe_outputs = n_wts_2nd_val_head[0]/arch.channels;
	assert (n_wts_2nd_val_head[0] == arch.channels*arch.vbe_outputs);
	conv_vbe_w = std::move(wts_2nd_val_head[0]);

	assert (n_wts_2nd_val_head[1] == arch.vbe_outputs);
	conv_vbe_b = std::move(wts_2nd_val_head[1]);

	assert (n_wts_2nd_val_head[2] == arch.vbe_outputs);
	bn_vbe_w1 = std::move(wts_2nd_val_head[2]);

	assert (n_wts_2nd_val_head[3] == arch.vbe_outputs);
	process_bn_var(wts_2nd_val_head[3]);
	bn_vbe_w2 = std::move(wts_2nd_val_head[3]);

	arch.vbe_chans = n_wts_2nd_val_head[4]/arch.vbe_outputs/(BOARD_SQUARES);
	assert (n_wts_2nd_val_head[4] == arch.vbe_chans*arch.vbe_outputs*BOARD_SQUARES);
	ip1_vbe_w = std::move(wts_2nd_val_head[4]);

	assert (n_wts_2nd_val_head[5] == arch.vbe_chans);
	ip1_vbe_b = std::move(wts_2nd_val_head[5]);

	int ret2 = n_wts_2nd_val_head[6]/arch.vbe_chans;
	assert (n_wts_2nd_val_head[6] == arch.vbe_chans*ret2);
	if (ret2 != 1) {
	  myprintf ("Unexpected in weights file: ret2=%d. %d -- %d -- %d.\n",
		    ret2,
		    n_wts_2nd_val_head[6],
		    arch.vbe_chans,
		    n_wts_2nd_val_head[6]/arch.vbe_chans);
	  return 1;
	}
	ip2_vbe_w = std::move(wts_2nd_val_head[6]);

	assert (n_wts_2nd_val_head[7] == 1);
	ip2_vbe_b = std::move(wts_2nd_val_head[7]);

	myprintf("Double value head. Type V.\n");
	myprintf("Alpha head: %d outputs, %d channels.\n", arch.val_outputs, arch.val_chans);
	myprintf("Beta head: %d outputs, %d channels.\n", arch.vbe_outputs, arch.vbe_chans);
    } else if (lastlines == 4) {
        arch.value_head_type = DOUBLE_Y;
        arch.value_head_rets = 2;

	arch.vbe_chans = n_wts_2nd_val_head[0]/arch.val_outputs/(BOARD_SQUARES);
	assert (n_wts_2nd_val_head[0] == arch.vbe_chans*arch.val_outputs*BOARD_SQUARES);
	ip1_vbe_w = std::move(wts_2nd_val_head[0]);

	assert (n_wts_2nd_val_head[1] == arch.vbe_chans);
	ip1_vbe_b = std::move(wts_2nd_val_head[1]);

	int ret2 = n_wts_2nd_val_head[2]/arch.vbe_chans;
	assert (n_wts_2nd_val_head[2] == arch.vbe_chans*ret2);
	if (ret2 != 1)
	  return 1;
	ip2_vbe_w = std::move(wts_2nd_val_head[2]);

	assert (n_wts_2nd_val_head[3] == 1);
	ip2_vbe_b = std::move(wts_2nd_val_head[3]);

	myprintf("Double value head. Type Y.\n");
	myprintf("Common convolution: %d outputs.\n", arch.val_outputs);
	myprintf("Alpha head: %d channels. Beta head: %d channels.\n", arch.val_chans, arch.vbe_chans);
    } else if (lastlines == 2) {
        arch.value_head_type = DOUBLE_T;
	arch.value_head_rets = 2;

	int ret2 = n_wts_2nd_val_head[0]/arch.val_chans;
	assert (n_wts_2nd_val_head[0] == arch.val_chans*ret2);
	if (ret2 != 1)
	  return 1;
	ip2_vbe_w = std::move(wts_2nd_val_head[0]);

	assert (n_wts_2nd_val_head[1] == 1);
	ip2_vbe_b = std::move(wts_2nd_val_head[1]);

	myprintf("Double value head. Type T: %d outputs, %d channels.\n",
		 arch.val_outputs, arch.val_chans);
    } else if (lastlines == 0) {
        if (arch.value_head_rets == 2) {
	  arch.value_head_type = DOUBLE_I;

	  myprintf("Double value head. Type I: %d outputs, %d channels.\n",
		   arch.val_outputs, arch.val_chans);
	}
	else if (arch.value_head_rets == 1) {
          arch.value_head_type = SINGLE;

	  myprintf("Single value head: %d outputs, %d channels.\n",
		   arch.val_outputs, arch.val_chans);
	}
    } else {
      myprintf ("\nFailed to parse weight file.\n");
      return 1;
    }

    return 0;
}

int Network::load_network_file(const std::string& filename) {
    // gzopen supports both gz and non-gz files, will decompress
    // or just read directly as needed.
    auto gzhandle = gzopen(filename.c_str(), "rb");
    if (gzhandle == nullptr) {
        myprintf("Could not open weights file: %s\n", filename.c_str());
        return 1;
    }
    // Stream the gz file in to a memory buffer stream.
    auto buffer = std::stringstream{};
    constexpr auto chunkBufferSize = 64 * 1024;
    std::vector<char> chunkBuffer(chunkBufferSize);
    while (true) {
        auto bytesRead = gzread(gzhandle, chunkBuffer.data(), chunkBufferSize);
        if (bytesRead == 0) break;
        if (bytesRead < 0) {
            myprintf("Failed to decompress or read: %s\n", filename.c_str());
            gzclose(gzhandle);
            return 1;
        }
        assert(bytesRead <= chunkBufferSize);
        buffer.write(chunkBuffer.data(), bytesRead);
    }
    gzclose(gzhandle);

    // Read format version
    auto line = std::string{};
    auto format_version = -1;
    if (std::getline(buffer, line)) {
        auto iss = std::stringstream{line};
        // First line is the file format version id
        iss >> format_version;
        if (iss.fail() || (format_version != 1 &&
			   format_version != 2 &&
			   format_version != 17 &&
			   format_version != 49)) {
            myprintf("Weights file is the wrong version.\n");
            return 1;
        } else {
            // Version 2 networks are identical to v1, except
            // that they return the score for black instead of
            // the player to move. This is used by ELF Open Go.
            if (format_version == 2) {
		myprintf("Version 2 weights file (ELF).\n");
                value_head_not_stm = true;
            } else {
		if (format_version == 1) {
		    myprintf("Version 1 weights file (LZ).\n");
		}
                value_head_not_stm = false;
            }
	    if (format_version == 17) {
		myprintf("Version 17 weights file (advanced board features).\n");
		arch.adv_features = true;
                arch.komi_policy = false;
	    } else if (format_version == 49) {
		myprintf("Version 49 weights file (komi policy + advanced board features).\n");
		arch.adv_features = true;
                arch.komi_policy = true;
            } else {
		arch.adv_features = false;
                arch.komi_policy = false;
	    }
            return load_v1_network(buffer, format_version);
        }
    }
    return 1;
}

void Network::initialize() {
    // Prepare symmetry table
    for (auto s = 0; s < 8; s++) {
        for (auto v = 0; v < BOARD_SQUARES; v++) {
            symmetry_nn_idx_table[s][v] = get_nn_idx_symmetry(v, s);
        }
    }

    // Load network from file
    if(load_network_file(cfg_weightsfile)) {
        exit(EXIT_FAILURE);
    }

    is_mult_komi_net = (arch.value_head_type != SINGLE);

    auto weight_index = size_t{0};
    // Input convolution
    // Winograd transform convolution weights
    conv_weights[weight_index] =
        winograd_transform_f(conv_weights[weight_index],
                             arch.channels, arch.input_planes);
    weight_index++;

    // Residual block convolutions
    for (auto i = size_t{0}; i < arch.residual_blocks * 2; i++) {
        conv_weights[weight_index] =
            winograd_transform_f(conv_weights[weight_index],
                                 arch.channels, arch.channels);
        weight_index++;
    }

    // Biases are not calculated and are typically zero but some networks might
    // still have non-zero biases.
    // Move biases to batchnorm means to make the output match without having
    // to separately add the biases.
    for (auto i = size_t{0}; i < conv_biases.size(); i++) {
        for (auto j = size_t{0}; j < batchnorm_means[i].size(); j++) {
            batchnorm_means[i][j] -= conv_biases[i][j];
            conv_biases[i][j] = 0.0f;
        }
    }

    for (auto i = size_t{0}; i < bn_val_w1.size(); i++) {
        bn_val_w1[i] -= conv_val_b[i];
        conv_val_b[i] = 0.0f;
    }

    for (auto i = size_t{0}; i < bn_vbe_w1.size(); i++) {
        bn_vbe_w1[i] -= conv_vbe_b[i];
        conv_vbe_b[i] = 0.0f;
    }

    for (auto i = size_t{0}; i < bn_pol_w1.size(); i++) {
        bn_pol_w1[i] -= conv_pol_b[i];
        conv_pol_b[i] = 0.0f;
    }

#ifdef USE_OPENCL
    myprintf("Initializing OpenCL.\n");
    opencl.initialize(arch.channels);

    for (const auto & opencl_net : opencl.get_networks()) {
        const auto tuners = opencl_net->getOpenCL().get_sgemm_tuners();

        const auto mwg = tuners[0];
        const auto kwg = tuners[2];
        const auto vwm = tuners[3];

        weight_index = 0;

        const auto m_ceil = ceilMultiple(ceilMultiple(arch.channels, mwg), vwm);
        const auto k_ceil = ceilMultiple(ceilMultiple(arch.input_planes, kwg), vwm);

        const auto Upad = zeropad_U(conv_weights[weight_index],
                                    arch.channels, arch.input_planes,
                                    m_ceil, k_ceil);

        // Winograd filter transformation changes filter size to 4x4
        opencl_net->push_input_convolution(WINOGRAD_ALPHA, arch.input_planes,
            arch.channels, Upad,
            batchnorm_means[weight_index], batchnorm_stddivs[weight_index]);
        weight_index++;

        // residual blocks
        for (auto i = size_t{0}; i < arch.residual_blocks; i++) {
            const auto Upad1 = zeropad_U(conv_weights[weight_index],
                                         arch.channels, arch.channels,
                                         m_ceil, m_ceil);
            const auto Upad2 = zeropad_U(conv_weights[weight_index + 1],
                                         arch.channels, arch.channels,
                                         m_ceil, m_ceil);
            opencl_net->push_residual(WINOGRAD_ALPHA, arch.channels, arch.channels,
                                      Upad1,
                                      batchnorm_means[weight_index],
                                      batchnorm_stddivs[weight_index],
                                      Upad2,
                                      batchnorm_means[weight_index + 1],
                                      batchnorm_stddivs[weight_index + 1]);
            weight_index += 2;
        }

        // Output head convolutions
        opencl_net->push_convolve1(arch.channels, arch.policy_outputs, conv_pol_w);
        opencl_net->push_convolve1(arch.channels, arch.val_outputs, conv_val_w);
	if (arch.value_head_type == DOUBLE_V) {
	  opencl_net->push_convolve1(arch.channels, arch.vbe_outputs, conv_vbe_w);
	}
    }
#endif
#ifdef USE_BLAS
#ifndef __APPLE__
#ifdef USE_OPENBLAS
    openblas_set_num_threads(1);
    myprintf("BLAS Core: %s\n", openblas_get_corename());
#endif
#ifdef USE_MKL
    //mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL);
    mkl_set_num_threads(1);
    MKLVersion Version;
    mkl_get_version(&Version);
    myprintf("BLAS core: MKL %s\n", Version.Processor);
#endif
#endif
#endif
}

#ifdef USE_BLAS
void Network::winograd_transform_in(const std::vector<float>& in,
                                    std::vector<float>& V,
                                    const int C) {
    constexpr auto W = BOARD_SIZE;
    constexpr auto H = BOARD_SIZE;
    constexpr auto WTILES = (W + 1) / 2;
    constexpr auto P = WTILES * WTILES;

    std::array<std::array<float, WTILES * 2 + 2>, WTILES * 2 + 2> in_pad;
    for (auto xin = size_t{0}; xin < in_pad.size(); xin++) {
        in_pad[0][xin]     = 0.0f;
        in_pad[H + 1][xin] = 0.0f;
        in_pad[H + 2][xin] = 0.0f;
    }
    for (auto yin = size_t{1}; yin < in_pad[0].size() - 2; yin++) {
        in_pad[yin][0]     = 0.0f;
        in_pad[yin][W + 1] = 0.0f;
        in_pad[yin][W + 2] = 0.0f;
    }

    for (auto ch = 0; ch < C; ch++) {
        for (auto yin = 0; yin < H; yin++) {
            for (auto xin = 0; xin < W; xin++) {
                in_pad[yin + 1][xin + 1] = in[ch*(W*H) + yin*W + xin];
            }
        }
        for (auto block_y = 0; block_y < WTILES; block_y++) {
            // Tiles overlap by 2
            const auto yin = 2 * block_y;
            for (auto block_x = 0; block_x < WTILES; block_x++) {
                const auto xin = 2 * block_x;

                // Calculates transpose(B).x.B
                // B = [[ 1.0,  0.0,  0.0,  0.0],
                //      [ 0.0,  1.0, -1.0,  1.0],
                //      [-1.0,  1.0,  1.0,  0.0],
                //      [ 0.0,  0.0,  0.0, -1.0]]

                using WinogradTile =
                    std::array<std::array<float, WINOGRAD_ALPHA>, WINOGRAD_ALPHA>;
                WinogradTile T1, T2;

                T1[0][0] = in_pad[yin + 0][xin + 0] - in_pad[yin + 2][xin + 0];
                T1[0][1] = in_pad[yin + 0][xin + 1] - in_pad[yin + 2][xin + 1];
                T1[0][2] = in_pad[yin + 0][xin + 2] - in_pad[yin + 2][xin + 2];
                T1[0][3] = in_pad[yin + 0][xin + 3] - in_pad[yin + 2][xin + 3];
                T1[1][0] = in_pad[yin + 1][xin + 0] + in_pad[yin + 2][xin + 0];
                T1[1][1] = in_pad[yin + 1][xin + 1] + in_pad[yin + 2][xin + 1];
                T1[1][2] = in_pad[yin + 1][xin + 2] + in_pad[yin + 2][xin + 2];
                T1[1][3] = in_pad[yin + 1][xin + 3] + in_pad[yin + 2][xin + 3];
                T1[2][0] = in_pad[yin + 2][xin + 0] - in_pad[yin + 1][xin + 0];
                T1[2][1] = in_pad[yin + 2][xin + 1] - in_pad[yin + 1][xin + 1];
                T1[2][2] = in_pad[yin + 2][xin + 2] - in_pad[yin + 1][xin + 2];
                T1[2][3] = in_pad[yin + 2][xin + 3] - in_pad[yin + 1][xin + 3];
                T1[3][0] = in_pad[yin + 1][xin + 0] - in_pad[yin + 3][xin + 0];
                T1[3][1] = in_pad[yin + 1][xin + 1] - in_pad[yin + 3][xin + 1];
                T1[3][2] = in_pad[yin + 1][xin + 2] - in_pad[yin + 3][xin + 2];
                T1[3][3] = in_pad[yin + 1][xin + 3] - in_pad[yin + 3][xin + 3];

                T2[0][0] = T1[0][0] - T1[0][2];
                T2[0][1] = T1[0][1] + T1[0][2];
                T2[0][2] = T1[0][2] - T1[0][1];
                T2[0][3] = T1[0][1] - T1[0][3];
                T2[1][0] = T1[1][0] - T1[1][2];
                T2[1][1] = T1[1][1] + T1[1][2];
                T2[1][2] = T1[1][2] - T1[1][1];
                T2[1][3] = T1[1][1] - T1[1][3];
                T2[2][0] = T1[2][0] - T1[2][2];
                T2[2][1] = T1[2][1] + T1[2][2];
                T2[2][2] = T1[2][2] - T1[2][1];
                T2[2][3] = T1[2][1] - T1[2][3];
                T2[3][0] = T1[3][0] - T1[3][2];
                T2[3][1] = T1[3][1] + T1[3][2];
                T2[3][2] = T1[3][2] - T1[3][1];
                T2[3][3] = T1[3][1] - T1[3][3];

                const auto offset = ch * P + block_y * WTILES + block_x;
                for (auto i = 0; i < WINOGRAD_ALPHA; i++) {
                    for (auto j = 0; j < WINOGRAD_ALPHA; j++) {
                        V[(i*WINOGRAD_ALPHA + j)*C*P + offset] = T2[i][j];
                    }
                }
            }
        }
    }
}

void Network::winograd_sgemm(const std::vector<float>& U,
                             const std::vector<float>& V,
                             std::vector<float>& M,
                             const int C, const int K) {
    constexpr auto P = (BOARD_SIZE + 1) * (BOARD_SIZE + 1) / WINOGRAD_ALPHA;

    for (auto b = 0; b < WINOGRAD_TILE; b++) {
        const auto offset_u = b * K * C;
        const auto offset_v = b * C * P;
        const auto offset_m = b * K * P;

        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    K, P, C,
                    1.0f,
                    &U[offset_u], K,
                    &V[offset_v], P,
                    0.0f,
                    &M[offset_m], P);
    }
}

void Network::winograd_transform_out(const std::vector<float>& M,
                                     std::vector<float>& Y,
                                     const int K) {
    constexpr auto W = BOARD_SIZE;
    constexpr auto H = BOARD_SIZE;
    constexpr auto WTILES = (W + 1) / 2;
    constexpr auto P = WTILES * WTILES;

    for (auto k = 0; k < K; k++) {
        const auto kHW = k * W * H;
        for (auto block_x = 0; block_x < WTILES; block_x++) {
            const auto x = 2 * block_x;
            for (auto block_y = 0; block_y < WTILES; block_y++) {
                const auto y = 2 * block_y;

                const auto b = block_y * WTILES + block_x;
                using WinogradTile =
                    std::array<std::array<float, WINOGRAD_ALPHA>, WINOGRAD_ALPHA>;
                WinogradTile temp_m;
                for (auto xi = 0; xi < WINOGRAD_ALPHA; xi++) {
                    for (auto nu = 0; nu < WINOGRAD_ALPHA; nu++) {
                        temp_m[xi][nu] =
                            M[xi*(WINOGRAD_ALPHA*K*P) + nu*(K*P)+ k*P + b];
                    }
                }

                // Calculates transpose(A).temp_m.A
                //    A = [1.0,  0.0],
                //        [1.0,  1.0],
                //        [1.0, -1.0],
                //        [0.0, -1.0]]

                const std::array<std::array<float, 2>, 2> o = {
                    temp_m[0][0] + temp_m[0][1] + temp_m[0][2] +
                    temp_m[1][0] + temp_m[1][1] + temp_m[1][2] +
                    temp_m[2][0] + temp_m[2][1] + temp_m[2][2],
                    temp_m[0][1] - temp_m[0][2] - temp_m[0][3] +
                    temp_m[1][1] - temp_m[1][2] - temp_m[1][3] +
                    temp_m[2][1] - temp_m[2][2] - temp_m[2][3],
                    temp_m[1][0] + temp_m[1][1] + temp_m[1][2] -
                    temp_m[2][0] - temp_m[2][1] - temp_m[2][2] -
                    temp_m[3][0] - temp_m[3][1] - temp_m[3][2],
                    temp_m[1][1] - temp_m[1][2] - temp_m[1][3] -
                    temp_m[2][1] + temp_m[2][2] + temp_m[2][3] -
                    temp_m[3][1] + temp_m[3][2] + temp_m[3][3]
                };

                const auto y_ind = kHW + (y)*W + (x);
                Y[y_ind] = o[0][0];
                if (x + 1 < W) {
                    Y[y_ind + 1] = o[0][1];
                }
                if (y + 1 < H) {
                    Y[y_ind + W] = o[1][0];
                    if (x + 1 < W) {
                        Y[y_ind + W + 1] = o[1][1];
                    }
                }
            }
        }
    }
}

void Network::winograd_convolve3(const int outputs,
                                 const std::vector<float>& input,
                                 const std::vector<float>& U,
                                 std::vector<float>& V,
                                 std::vector<float>& M,
                                 std::vector<float>& output) {

    constexpr unsigned int filter_len = WINOGRAD_ALPHA * WINOGRAD_ALPHA;
    const auto input_channels = U.size() / (outputs * filter_len);

    winograd_transform_in(input, V, input_channels);
    winograd_sgemm(U, V, M, input_channels, outputs);
    winograd_transform_out(M, output, outputs);
}

template<unsigned int filter_size>
void convolve(const size_t outputs,
              const std::vector<float>& input,
              const std::vector<float>& weights,
              const std::vector<float>& biases,
              std::vector<float>& output) {
    // The size of the board is defined at compile time
    constexpr unsigned int width = BOARD_SIZE;
    constexpr unsigned int height = BOARD_SIZE;
    constexpr auto board_squares = width * height;
    constexpr auto filter_len = filter_size * filter_size;
    const auto input_channels = weights.size() / (biases.size() * filter_len);
    const auto filter_dim = filter_len * input_channels;
    assert(outputs * board_squares == output.size());

    std::vector<float> col(filter_dim * width * height);
    im2col<filter_size>(input_channels, input, col);

    // Weight shape (output, input, filter_size, filter_size)
    // 96 18 3 3
    // C←αAB + βC
    // outputs[96,19x19] = weights[96,18x3x3] x col[18x3x3,19x19]
    // M Number of rows in matrices A and C.
    // N Number of columns in matrices B and C.
    // K Number of columns in matrix A; number of rows in matrix B.
    // lda The size of the first dimention of matrix A; if you are
    // passing a matrix A[m][n], the value should be m.
    //    cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
    //                ldb, beta, C, N);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                // M        N            K
                outputs, board_squares, filter_dim,
                1.0f, &weights[0], filter_dim,
                &col[0], board_squares,
                0.0f, &output[0], board_squares);

    for (unsigned int o = 0; o < outputs; o++) {
        for (unsigned int b = 0; b < board_squares; b++) {
            output[(o * board_squares) + b] += biases[o];
        }
    }
}

template<bool ReLU>
std::vector<float> innerproduct(const std::vector<float>& input,
                                const std::vector<float>& weights,
                                const std::vector<float>& biases) {
    const auto inputs = input.size();
    const auto outputs = biases.size();
    std::vector<float> output(outputs);

    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                // M     K
                outputs, inputs,
                1.0f, &weights[0], inputs,
                &input[0], 1,
                0.0f, &output[0], 1);

    const auto lambda_ReLU = [](const auto val) { return (val > 0.0f) ?
                                                          val : 0.0f; };
    for (unsigned int o = 0; o < outputs; o++) {
        auto val = biases[o] + output[o];
        if (ReLU) {
            val = lambda_ReLU(val);
        }
        output[o] = val;
    }

    return output;
}

template <size_t spatial_size>
void batchnorm(const size_t channels,
               std::vector<float>& data,
               const float* const means,
               const float* const stddivs,
               const float* const eltwise = nullptr)
{
    const auto lambda_ReLU = [](const auto val) { return (val > 0.0f) ?
                                                          val : 0.0f; };
    for (auto c = size_t{0}; c < channels; ++c) {
        const auto mean = means[c];
        const auto scale_stddiv = stddivs[c];

        if (eltwise == nullptr) {
            // Classical BN
            const auto arr = &data[c * spatial_size];
            for (auto b = size_t{0}; b < spatial_size; b++) {
                arr[b] = lambda_ReLU(scale_stddiv * (arr[b] - mean));
            }
        } else {
            // BN + residual add
            const auto arr = &data[c * spatial_size];
            const auto res = &eltwise[c * spatial_size];
            for (auto b = size_t{0}; b < spatial_size; b++) {
                arr[b] = lambda_ReLU((scale_stddiv * (arr[b] - mean)) + res[b]);
            }
        }
    }
}

// output_val, output_vbe are the features before the fully connected step
void Network::forward_cpu(const std::vector<float>& input,
                          std::vector<float>& output_pol,
                          std::vector<float>& output_val,
                          std::vector<float>& output_vbe) {
    // Input convolution
    constexpr auto width = BOARD_SIZE;
    constexpr auto height = BOARD_SIZE;
    constexpr auto tiles = (width + 1) * (height + 1) / 4;
    // input_channels is the maximum number of input channels of any
    // convolution. Residual blocks are identical, but the first convolution
    // might be bigger when the network has very few filters
    const auto input_channels = std::max(static_cast<size_t>(arch.channels),
                                         static_cast<size_t>(arch.input_planes));
    auto conv_out = std::vector<float>(arch.channels * width * height);

    auto V = std::vector<float>(WINOGRAD_TILE * input_channels * tiles);
    auto M = std::vector<float>(WINOGRAD_TILE * arch.channels * tiles);

    winograd_convolve3(arch.channels, input, conv_weights[0], V, M, conv_out);
    batchnorm<BOARD_SQUARES>(arch.channels, conv_out,
                             batchnorm_means[0].data(),
                             batchnorm_stddivs[0].data());

    // Residual tower
    auto conv_in = std::vector<float>(arch.channels * width * height);
    auto res = std::vector<float>(arch.channels * width * height);
    for (auto i = size_t{1}; i < conv_weights.size(); i += 2) {
        auto output_channels = conv_biases[i].size();
        std::swap(conv_out, conv_in);
        winograd_convolve3(output_channels, conv_in,
                           conv_weights[i], V, M, conv_out);
        batchnorm<BOARD_SQUARES>(output_channels, conv_out,
                                 batchnorm_means[i].data(),
                                 batchnorm_stddivs[i].data());

        output_channels = conv_biases[i + 1].size();
        std::swap(conv_in, res);
        std::swap(conv_out, conv_in);
        winograd_convolve3(output_channels, conv_in,
                           conv_weights[i + 1], V, M, conv_out);
        batchnorm<BOARD_SQUARES>(output_channels, conv_out,
                                 batchnorm_means[i + 1].data(),
                                 batchnorm_stddivs[i + 1].data(),
                                 res.data());
    }
    convolve<1>(arch.policy_outputs, conv_out, conv_pol_w, conv_pol_b, output_pol);
    convolve<1>(arch.val_outputs, conv_out, conv_val_w, conv_val_b, output_val);
    if (arch.value_head_type == DOUBLE_V) {
      convolve<1>(arch.vbe_outputs, conv_out, conv_vbe_w, conv_vbe_b, output_vbe);
    }
}

template<typename T>
T relative_difference(const T a, const T b) {
    // Handle NaN
    if (std::isnan(a) || std::isnan(b)) {
        return std::numeric_limits<T>::max();
    }

    constexpr auto small_number = 1e-3f;
    auto fa = std::fabs(a);
    auto fb = std::fabs(b);

    if (fa > small_number && fb > small_number) {
        // Handle sign difference
        if ((a < 0) != (b < 0)) {
            return std::numeric_limits<T>::max();
        }
    } else {
        // Handle underflow
        fa = std::max(fa, small_number);
        fb = std::max(fb, small_number);
    }

    return fabs(fa - fb) / std::min(fa, fb);
}

void compare_net_outputs(std::vector<float>& data,
                         std::vector<float>& ref) {
    // We accept an error up to 5%, but output values
    // smaller than 1/1000th are "rounded up" for the comparison.
    constexpr auto relative_error = 5e-2f;
    for (auto idx = size_t{0}; idx < data.size(); ++idx) {
        const auto err = relative_difference(data[idx], ref[idx]);
        if (err > relative_error) {
            printf("Error in OpenCL calculation: expected %f got %f "
                   "(error=%f%%)\n", ref[idx], data[idx], err * 100.0);
            printf("Update your GPU drivers or reduce the amount of games "
                   "played simultaneously.\n");
            throw std::runtime_error("OpenCL self-check mismatch.");
        }
    }
}
#endif

std::vector<float> softmax(const std::vector<float>& input,
                           const float temperature = 1.0f) {
    auto output = std::vector<float>{};
    output.reserve(input.size());

    const auto alpha = *std::max_element(cbegin(input), cend(input));
    auto denom = 0.0f;

    for (const auto in_val : input) {
        auto val = std::exp((in_val - alpha) / temperature);
        denom += val;
        output.push_back(val);
    }

    for (auto& out : output) {
        out /= denom;
    }

    return output;
}

std::pair<float,float> sigmoid(float alpha, float beta, float bonus) {
    const auto arg = beta*(alpha+bonus);
    const auto absarg = std::abs(arg);
    float ret;
    
    if (absarg > 30.0f) {
        ret = std::exp(-absarg);
    } else {
        ret = 1.0f/(1.0f+std::exp(absarg));
    }
    return arg<0 ? std::make_pair(ret, 1.0f-ret)
               : std::make_pair(1.0f-ret, ret);
}

Network::Netresult Network::get_scored_moves(
    const GameState* const state, const Ensemble ensemble,
    const int symmetry, const bool skip_cache) {
    Netresult result;
    if (state->board.get_boardsize() != BOARD_SIZE) {
        return result;
    }

    if (!skip_cache) {
        // See if we already have this in the cache.
        if (NNCache::get_NNCache().lookup(state->board.get_hash(), result)) {
            return result;
        }
    }

    if (ensemble == DIRECT) {
        assert(symmetry >= 0 && symmetry <= 7);
        result = get_scored_moves_internal(state, symmetry);
    } else if (ensemble == AVERAGE) {
        for (auto sym = 0; sym < 8; ++sym) {
            auto tmpresult = get_scored_moves_internal(state, sym);
            result.policy_pass += tmpresult.policy_pass / 8.0f;
            result.value += tmpresult.value / 8.0f;
            result.alpha += tmpresult.alpha / 8.0f;
            result.beta += tmpresult.beta / 8.0f;

            for (auto idx = size_t{0}; idx < BOARD_SQUARES; idx++) {
                result.policy[idx] += tmpresult.policy[idx] / 8.0f;
            }
        }
    } else {
        assert(ensemble == RANDOM_SYMMETRY);
        assert(symmetry == -1);
        const auto rand_sym = Random::get_Rng().randfix<8>();
        result = get_scored_moves_internal(state, rand_sym);
    }

    if (!cfg_symm_nonrandom) {
        // Insert result into cache.
        NNCache::get_NNCache().insert(state->board.get_hash(), result);
    }

    return result;
}

Network::Netresult Network::get_scored_moves_internal(
    const GameState* const state, const int symmetry) {
    assert(symmetry >= 0 && symmetry <= 7);
    constexpr auto width = BOARD_SIZE;
    constexpr auto height = BOARD_SIZE;

    // if the input planes of the loaded network are even, then the
    // color of the current player is encoded in the last two planes
    const auto include_color = (0 == arch.input_planes % 2);

    const auto input_data = gather_features(state, symmetry,
					    arch.input_moves,
					    arch.adv_features,
					    include_color);
    std::vector<float> policy_data(arch.policy_outputs * width * height);
    std::vector<float> val_data(arch.val_outputs * width * height);
    std::vector<float> vbe_data(arch.vbe_outputs * width * height);

#ifdef USE_HALF
    std::vector<net_t> policy_data_n(arch.policy_outputs * width * height);
    std::vector<net_t> val_data_n(arch.val_outputs * width * height);
    std::vector<net_t> vbe_data_n(arch.vbe_outputs * width * height);
#endif


#ifdef USE_OPENCL
#ifdef USE_HALF
    opencl.forward(input_data, policy_data_n, val_data_n, vbe_data_n);
    std::copy(begin(policy_data_n), end(policy_data_n), begin(policy_data));
    std::copy(begin(val_data_n), end(val_data_n), begin(val_data));
    std::copy(begin(vbe_data_n), end(vbe_data_n), begin(vbe_data));
#else
    opencl.forward(input_data, policy_data, val_data, vbe_data);
#endif

#elif defined(USE_BLAS) && !defined(USE_OPENCL)
    forward_cpu(input_data, policy_data, val_data, vbe_data);
#endif
#ifdef USE_OPENCL_SELFCHECK
    // Both implementations are available, self-check the OpenCL driver by
    // running both with a probability of 1/2000.
    if (Random::get_Rng().randfix<SELFCHECK_PROBABILITY>() == 0) {
        auto cpu_policy_data = std::vector<float>(policy_data.size());
        auto cpu_val_data = std::vector<float>(val_data.size());
        auto cpu_vbe_data = std::vector<float>(vbe_data.size());
        forward_cpu(input_data, cpu_policy_data, cpu_val_data, cpu_vbe_data);
        compare_net_outputs(policy_data, cpu_policy_data);
        compare_net_outputs(val_data, cpu_val_data);
        compare_net_outputs(vbe_data, cpu_vbe_data);
    }
#endif
    // Get the moves
    batchnorm<BOARD_SQUARES>(arch.policy_outputs, policy_data,
        bn_pol_w1.data(), bn_pol_w2.data());

    if (arch.komi_policy) {
        float komi = state->get_komi();
        komi *= ( state->get_to_move() == FastBoard::BLACK ? -1.0 : 1.0 );
        policy_data.push_back(komi);
        const auto kp1 =
            innerproduct<true>(policy_data, kp1_pol_w, kp1_pol_b);
        const auto kp2 =
            innerproduct<true>(kp1, kp2_pol_w, kp2_pol_b);
        policy_data.pop_back();
        for (auto & i : kp2) {
            policy_data.push_back(i);
        }
    }
    
    const auto policy_out =
        innerproduct<false>(
            policy_data, ip_pol_w, ip_pol_b);

    const auto outputs = softmax(policy_out, cfg_softmax_temp);

    // Get alpha or value
    batchnorm<BOARD_SQUARES>(arch.val_outputs, val_data,
        bn_val_w1.data(), bn_val_w2.data());
    const auto val_channels =
        innerproduct<true>(val_data, ip1_val_w, ip1_val_b);
    const auto val_output =
        innerproduct<false>(val_channels, ip2_val_w, ip2_val_b);

    Netresult result;

    if (arch.value_head_type==DOUBLE_V) {
	// If double head value, also get beta
	batchnorm<BOARD_SQUARES>(arch.vbe_outputs, vbe_data,
				 bn_vbe_w1.data(), bn_vbe_w2.data());
	const auto vbe_channels =
	    innerproduct<true>(vbe_data, ip1_vbe_w, ip1_vbe_b);
	const auto vbe_output =
	    innerproduct<false>(vbe_channels, ip2_vbe_w, ip2_vbe_b);

	result.value = 0.5f;
	result.alpha = val_output[0];
	result.beta = std::exp(vbe_output[0]) * 10.0f / BOARD_SQUARES;

    } else if (arch.value_head_type==DOUBLE_Y) {
	const auto vbe_channels =
	    innerproduct<true>(val_data, ip1_vbe_w, ip1_vbe_b);
	const auto vbe_output =
	    innerproduct<false>(vbe_channels, ip2_vbe_w, ip2_vbe_b);

	result.value = 0.5f;
	result.alpha = val_output[0];
	result.beta = std::exp(vbe_output[0]) * 10.0f / BOARD_SQUARES;

    } else if (arch.value_head_type==DOUBLE_T) {
	const auto vbe_output =
	    innerproduct<false>(val_channels, ip2_vbe_w, ip2_vbe_b);

	result.value = 0.5f;
	result.alpha = val_output[0];
	result.beta = std::exp(vbe_output[0]) * 10.0f / BOARD_SQUARES;

    } else if (arch.value_head_type==DOUBLE_I) {
	result.value = 0.5f;
	result.alpha = val_output[0];
	result.beta = std::exp(val_output[1]) * 10.0f / BOARD_SQUARES;

    } else if (arch.value_head_type==SINGLE) {
	result.value = (1.0f + std::tanh(val_output[0])) / 2.0f;
	result.alpha = 0.0f;
	result.beta = 1.0f;
    }


    for (auto idx = size_t{0}; idx < BOARD_SQUARES; idx++) {
        const auto sym_idx = symmetry_nn_idx_table[symmetry][idx];
        result.policy[sym_idx] = outputs[idx];
    }

    result.policy_pass = outputs[BOARD_SQUARES];

    return result;
}

Network::Netresult_extended Network::get_extended(const FastState& state, const Netresult& result) {
    const auto komi = state.get_komi();
    const auto alpha = result.alpha;
    const auto beta = result.beta;

    const auto winrate = sigmoid(alpha,  beta, state.board.black_to_move() ? -komi : komi);
    const auto alpkt = (state.board.black_to_move() ? alpha : -alpha) - komi;
    
    const auto pi = sigmoid(alpkt, beta, 0.0f);
    // if pi is near to 1, this is much more precise than 1-pi
    //    const auto one_m_pi = sigmoid(-alpkt, beta, 0.0f);
    
    const auto pi_lambda = std::make_pair((1-cfg_lambda)*pi.first + cfg_lambda*0.5f,
                                          (1-cfg_lambda)*pi.second + cfg_lambda*0.5f);
    const auto pi_mu = std::make_pair((1-cfg_mu)*pi.first + cfg_mu*0.5f,
                                      (1-cfg_mu)*pi.second + cfg_mu*0.5f);
    
    // this is useful when lambda is near to 0 and pi near 1
    //    const auto one_m_pi_lambda = (1-cfg_lambda)*one_m_pi + cfg_lambda*0.5f;
    const auto sigma_inv_pi_lambda = std::log(pi_lambda.first) - std::log(pi_lambda.second);
    const auto eval_bonus = (cfg_lambda == 0) ? 0.0f : sigma_inv_pi_lambda / beta - alpkt;
    
    //    const auto one_m_pi_mu = (1-cfg_mu)*one_m_pi + cfg_mu*0.5f;
    const auto sigma_inv_pi_mu = std::log(pi_mu.first) - std::log(pi_mu.second);
    const auto eval_base = (cfg_mu == 0) ? 0.0f : sigma_inv_pi_mu / beta - alpkt;
    
    const auto agent_eval = Utils::sigmoid_interval_avg(alpkt, beta, eval_base, eval_bonus);

    return { winrate.first, alpkt, pi.first, eval_bonus, eval_base, agent_eval };
}

void Network::show_heatmap(const FastState* const state,
                           const Netresult& result,
                           const bool topmoves) {
    std::vector<std::string> display_map;
    std::string line;

    float legal_score = 0.0f;
    float illegal_score = 0.0f;

    std::array<float, BOARD_SQUARES> scores;
    
    const auto color = state->get_to_move();

    for (unsigned int y = 0; y < BOARD_SIZE; y++) {
        for (unsigned int x = 0; x < BOARD_SIZE; x++) {
            const auto vertex = state->board.get_vertex(x, y);
            const auto score = result.policy[y * BOARD_SIZE + x];
            if (state->is_move_legal(color, vertex)) {
                legal_score += score;
                scores[y * BOARD_SIZE + x] = score;
            } else {
                illegal_score += score;
                scores[y * BOARD_SIZE + x] = 0.0f;
            }
        }
    }

    for (unsigned int y = 0; y < BOARD_SIZE; y++) {
        for (unsigned int x = 0; x < BOARD_SIZE; x++) {
            const auto clean_score = int(scores[y * BOARD_SIZE + x] * 1000.0f / legal_score);
            line += boost::str(boost::format("%3d ") % clean_score);
        }

        display_map.push_back(line);
        line.clear();
    }

    for (int i = display_map.size() - 1; i >= 0; --i) {
        myprintf("%s\n", display_map[i].c_str());
    }
    const auto pass_score = int(result.policy_pass * 1000);
    const auto illegal_millis = int(illegal_score * 1000);

    myprintf("pass: %d, illegal: %d\n", pass_score, illegal_millis);
    if (is_mult_komi_net) {
        const auto result_extended = get_extended(*state, result);
        myprintf("alpha: %.2f, ", result.alpha);
        myprintf("beta: %.2f, ", result.beta);
        myprintf("winrate: %.1f%%\n", result_extended.winrate*100);
        myprintf("black alpkt: %.2f,", result_extended.alpkt);
        myprintf(" x_bar: %.2f,", result_extended.eval_bonus);
        myprintf(" x_base: %.2f\n", result_extended.eval_base);
    } else {
        myprintf("value: %.1f%%\n", result.value*100);
    }

    if (topmoves) {
        std::vector<Network::ScoreVertexPair> moves;
        for (auto i=0; i < BOARD_SQUARES; i++) {
            const auto x = i % BOARD_SIZE;
            const auto y = i / BOARD_SIZE;
            const auto vertex = state->board.get_vertex(x, y);
            if (state->board.get_square(vertex) == FastBoard::EMPTY) {
                moves.emplace_back(result.policy[i], vertex);
            }
        }
        moves.emplace_back(result.policy_pass, FastBoard::PASS);

        std::stable_sort(rbegin(moves), rend(moves));

        auto cum = 0.0f;
        size_t tried = 0;
        while (cum < 0.85f && tried < moves.size()) {
            if (moves[tried].first < 0.01f) break;
            myprintf("%1.3f (%s)\n",
                    moves[tried].first,
                    state->board.move_to_text(moves[tried].second).c_str());
            cum += moves[tried].first;
            tried++;
        }
    }
}

void Network::fill_input_plane_pair(const FullBoard& board,
                                    std::vector<net_t>::iterator black,
                                    std::vector<net_t>::iterator white,
                                    const int symmetry) {
    for (auto idx = 0; idx < BOARD_SQUARES; idx++) {
        const auto sym_idx = symmetry_nn_idx_table[symmetry][idx];
        const auto x = sym_idx % BOARD_SIZE;
        const auto y = sym_idx / BOARD_SIZE;
        const auto color = board.get_square(x, y);
        if (color == FastBoard::BLACK) {
            black[idx] = net_t(true);
        } else if (color == FastBoard::WHITE) {
            white[idx] = net_t(true);
        }
    }
}

void Network::fill_input_plane_advfeat(std::shared_ptr<const KoState> const state,
                                    std::vector<net_t>::iterator legal,
                                    std::vector<net_t>::iterator atari,
                                    const int symmetry) {
    for (auto idx = 0; idx < BOARD_SQUARES; idx++) {
        const auto sym_idx = symmetry_nn_idx_table[symmetry][idx];
        const auto x = sym_idx % BOARD_SIZE;
        const auto y = sym_idx / BOARD_SIZE;
	const auto vertex = state->board.get_vertex(x,y);
	const auto tomove = state->get_to_move();
	const auto is_legal = state->is_move_legal(tomove, vertex);
	legal[idx] = !is_legal;
	atari[idx] = is_legal && (1 == state->board.liberties_to_capture(vertex));
    }
}

std::vector<net_t> Network::gather_features(const GameState* const state,
					    const int symmetry,
					    const int input_moves,
					    const bool adv_features,
					    const bool include_color) {
    assert(symmetry >= 0 && symmetry <= 7);

    // if advanced board features are included, for every input move
    // in addition to 2 planes with the stones there are 2 planes with
    // legal moves for current player and "atari" intersections for
    // either player
    auto moves_planes = input_moves * (2 + (adv_features ? 2 : 0));

    // if the color of the current player is included, two more input
    // planes are needed, otherwise one input plane filled with ones
    // will provide information on the border of the board for the CNN
    auto input_planes = moves_planes + (include_color ? 2 : 1);

    auto input_data = std::vector<net_t>(input_planes * BOARD_SQUARES);

    const auto current_it = begin(input_data);
    const auto opponent_it = begin(input_data) + input_moves * BOARD_SQUARES;
    auto legal_it = current_it;
    auto atari_it = current_it;

    if (adv_features) {
	legal_it += 2 * input_moves * BOARD_SQUARES;
	atari_it += 3 * input_moves * BOARD_SQUARES;
    }

    const auto to_move = state->get_to_move();
    const auto blacks_move = to_move == FastBoard::BLACK;
    const auto black_it = blacks_move ? current_it : opponent_it;
    const auto white_it = blacks_move ? opponent_it : current_it;
    // myprintf("input moves: %d, advanced features: %d, include color: %d\n"
    // 	     "moves planes: %d, input planes: %d, to move: %d, blacks_move: %d\n",
    // 	     input_moves, adv_features, include_color,
    // 	     moves_planes, input_planes, to_move, blacks_move);

    // we fill one plane with ones: this is the only one remaining
    // when the color of current player is not included, otherwise it
    // is one of the two last plane, depending on current player
    const auto onesfilled_it = 	blacks_move || !include_color ?
	begin(input_data) + moves_planes * BOARD_SQUARES :
	begin(input_data) + (moves_planes + 1) * BOARD_SQUARES;
    std::fill(onesfilled_it, onesfilled_it + BOARD_SQUARES, net_t(true));

    const auto moves = std::min<size_t>(state->get_movenum() + 1, input_moves);
    // Go back in time, fill history boards
    for (auto h = size_t{0}; h < moves; h++) {
        // collect white, black occupation planes
        fill_input_plane_pair(state->get_past_state(h)->board,
                              black_it + h * BOARD_SQUARES,
                              white_it + h * BOARD_SQUARES,
                              symmetry);
	if (adv_features) {
	    fill_input_plane_advfeat(state->get_past_state(h),
				     legal_it + h * BOARD_SQUARES,
				     atari_it + h * BOARD_SQUARES,
				     symmetry);

	}
    }


    return input_data;
}

int Network::get_nn_idx_symmetry(const int vertex, int symmetry) {
    assert(vertex >= 0 && vertex < BOARD_SQUARES);
    assert(symmetry >= 0 && symmetry < 8);
    auto x = vertex % BOARD_SIZE;
    auto y = vertex / BOARD_SIZE;
    int newx;
    int newy;

    if (symmetry >= 4) {
        std::swap(x, y);
        symmetry -= 4;
    }

    if (symmetry == 0) {
        newx = x;
        newy = y;
    } else if (symmetry == 1) {
        newx = x;
        newy = BOARD_SIZE - y - 1;
    } else if (symmetry == 2) {
        newx = BOARD_SIZE - x - 1;
        newy = y;
    } else {
        assert(symmetry == 3);
        newx = BOARD_SIZE - x - 1;
        newy = BOARD_SIZE - y - 1;
    }

    const auto newvtx = (newy * BOARD_SIZE) + newx;
    assert(newvtx >= 0 && newvtx < BOARD_SQUARES);
    return newvtx;
}
