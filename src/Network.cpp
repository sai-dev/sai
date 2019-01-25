/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2018 Gian-Carlo Pascutto and contributors
    Copyright (C) 2018, 2019 SAI Team

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

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <boost/utility.hpp>
#include <boost/format.hpp>
#include <boost/spirit/home/x3.hpp>
#ifndef USE_BLAS
#include <Eigen/Dense>
#endif

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

#include "Network.h"
#include "CPUPipe.h"
#ifdef USE_OPENCL
#include "OpenCLScheduler.h"
#include "UCTNode.h"
#endif
#include "FastBoard.h"
#include "FastState.h"
#include "FullBoard.h"
#include "GameState.h"
#include "GTP.h"
#include "NNCache.h"
#include "Random.h"
#include "ThreadPool.h"
#include "Timing.h"
#include "Utils.h"

namespace x3 = boost::spirit::x3;
using namespace Utils;

#ifndef USE_BLAS
// Eigen helpers
template <typename T>
using EigenVectorMap =
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>;
template <typename T>
using ConstEigenVectorMap =
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>;
template <typename T>
using ConstEigenMatrixMap =
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;
#endif

// Symmetry helper
std::array<std::array<int, NUM_INTERSECTIONS>,
                  Network::NUM_SYMMETRIES> symmetry_nn_idx_table;

float Network::benchmark_time(int centiseconds) {
    const auto cpus = cfg_num_threads;

    ThreadGroup tg(thread_pool);
    std::atomic<int> runcount{0};

    GameState state;
    state.init_game(BOARD_SIZE, 7.5);

    // As a sanity run, try one run with self check.
    // Isn't enough to guarantee correctness but better than nothing,
    // plus for large nets self-check takes a while (1~3 eval per second)
    get_output(&state, Ensemble::RANDOM_SYMMETRY, -1, true, true);

    const Time start;
    for (auto i = 0; i < cpus; i++) {
        tg.add_task([this, &runcount, start, centiseconds, state]() {
            while (true) {
                runcount++;
                get_output(&state, Ensemble::RANDOM_SYMMETRY, -1, true);
                const Time end;
                const auto elapsed = Time::timediff_centis(start, end);
                if (elapsed >= centiseconds) {
                    break;
                }
            }
        });
    }
    tg.wait_all();

    const Time end;
    const auto elapsed = Time::timediff_centis(start, end);
    return 100.0f * runcount.load() / elapsed;
}

void Network::benchmark(const GameState* const state, const int iterations) {
    const auto cpus = cfg_num_threads;
    const Time start;

    ThreadGroup tg(thread_pool);
    std::atomic<int> runcount{0};

    for (auto i = 0; i < cpus; i++) {
        tg.add_task([this, &runcount, iterations, state]() {
            while (runcount < iterations) {
                runcount++;
                get_output(state, Ensemble::RANDOM_SYMMETRY, -1, true);
            }
        });
    }
    tg.wait_all();

    const Time end;
    const auto elapsed = Time::timediff_seconds(start, end);
    myprintf("%5d evaluations in %5.2f seconds -> %d n/s\n",
             runcount.load(), elapsed, int(runcount.load() / elapsed));
}

template<class container>
void process_bn_var(container& weights) {
    constexpr float epsilon = 1e-5f;
    for (auto&& w : weights) {
        w = 1.0f / std::sqrt(w + epsilon);
    }
}

std::vector<float> Network::winograd_transform_f(const std::vector<float>& f,
                                                 const int outputs,
                                                 const int channels) {
    // F(4x4, 3x3) Winograd filter transformation
    // transpose(G.dot(f).dot(G.transpose()))
    // U matrix is transposed for better memory layout in SGEMM
    auto U = std::vector<float>(WINOGRAD_TILE * outputs * channels);
    const auto G = std::array<float, 3 * WINOGRAD_ALPHA>
                    { 1.0f,        0.0f,      0.0f,
                      -2.0f/3.0f, -SQ2/3.0f, -1.0f/3.0f,
                      -2.0f/3.0f,  SQ2/3.0f, -1.0f/3.0f,
                      1.0f/6.0f,   SQ2/6.0f,  1.0f/3.0f,
                      1.0f/6.0f,  -SQ2/6.0f,  1.0f/3.0f,
                      0.0f,        0.0f,      1.0f};

    auto temp = std::array<float, 3 * WINOGRAD_ALPHA>{};

    constexpr auto max_buffersize = 8;
    auto buffersize = max_buffersize;

    if (outputs % buffersize != 0) {
        buffersize = 1;
    }

    std::array<float, max_buffersize * WINOGRAD_ALPHA * WINOGRAD_ALPHA> buffer;

    for (auto c = 0; c < channels; c++) {
        for (auto o_b = 0; o_b < outputs/buffersize; o_b++) {
            for (auto bufferline = 0; bufferline < buffersize; bufferline++) {
                const auto o = o_b * buffersize + bufferline;

                for (auto i = 0; i < WINOGRAD_ALPHA; i++) {
                    for (auto j = 0; j < 3; j++) {
                        auto acc = 0.0f;
                        for (auto k = 0; k < 3; k++) {
                            acc += G[i*3 + k] * f[o*channels*9 + c*9 + k*3 + j];
                        }
                        temp[i*3 + j] = acc;
                    }
                }

                for (auto xi = 0; xi < WINOGRAD_ALPHA; xi++) {
                    for (auto nu = 0; nu < WINOGRAD_ALPHA; nu++) {
                        auto acc = 0.0f;
                        for (auto k = 0; k < 3; k++) {
                            acc += temp[xi*3 + k] * G[nu*3 + k];
                        }
                        buffer[(xi * WINOGRAD_ALPHA + nu) * buffersize + bufferline] = acc;
                    }
                }
            }
            for (auto i = 0; i < WINOGRAD_ALPHA * WINOGRAD_ALPHA; i++) {
                for (auto entry = 0; entry < buffersize; entry++) {
                    const auto o = o_b * buffersize + entry;
                    U[i * outputs * channels
                      + c * outputs
                      + o] =
                    buffer[buffersize * i + entry];
                }
            }
        }
    }

    return U;
}

//v1 refers to the actual weight file format, to be changed when/if the weight file format changes
int Network::load_v1_network(std::istream& wtfile, int format_version) {
    // Count size of the network
    myprintf("Detecting residual layers... v%d\n", format_version);

    auto komipolicy_lines = 0;
    m_komipolicy_chans = 0;
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
          if (linecount==0 || n_wts==m_channels*9*m_channels)
            // yes: these are convolutional weights
            m_fwd_weights->m_conv_weights.emplace_back(weights);
      else {
        // no: first line of policy head [1x1 conv weights]
		is_head_line = true;
		m_policy_outputs = n_wts/m_channels;
		assert (n_wts == m_channels*m_policy_outputs);
		m_fwd_weights->m_conv_pol_w = std::move(weights);
		m_residual_blocks = (linecount-4)/8;
		plain_conv_layers = 1 + (m_residual_blocks * 2);
		plain_conv_wts = plain_conv_layers * 4;
		assert(plain_conv_wts == linecount);
		myprintf(" %d blocks\n%d policy outputs", m_residual_blocks, m_policy_outputs);
	      }
            } else if (linecount % 4 == 1) {
                // second line of 4: holds convolutional biases

                if (linecount == 1) {
		  // second line of weights, holds the biases for the
		  // input convolutional layer, hence its size gives
		  // the number of channels of subsequent resconv
		  // layers
		  m_channels = n_wts;

		  // we recover the number of input planes
		  m_input_planes = n_wts_1st_layer/9/m_channels;

		  // if it is even, color of the current player is
		  // used, if it is odd, only komi is used
		  m_include_color = (0 == m_input_planes % 2);

		  // we recover the number of input moves, knowing
		  // that for each move there are 2 bitplanes with
		  // stones positions and possibly 2 more bitplanes
		  // with some advanced features (legal and atari)
		  m_input_moves = (m_input_planes - (m_include_color ? 2 : 1)) /
		      (m_adv_features ? 4 : 2);

		  assert (n_wts_1st_layer == m_input_planes*9*m_channels);
		  myprintf("%d input planes, %d input moves\n%d channels...",
			   m_input_planes,
			   m_input_moves,
			   m_channels);
	      }
	      else
		assert (n_wts == m_channels);

	      m_fwd_weights->m_conv_biases.emplace_back(weights);
            } else if (linecount % 4 == 2) {
		assert (n_wts == m_channels);
                m_fwd_weights->m_batchnorm_means.emplace_back(weights);
            } else if (linecount % 4 == 3) {
	        assert (n_wts == m_channels);
                process_bn_var(weights);
                m_fwd_weights->m_batchnorm_stddevs.emplace_back(weights);
            }
        } else if (linecount == plain_conv_wts + 1) {
        // line 2 of policy head [1x1 convolutional biases]
        assert (n_wts == m_policy_outputs);
	    m_fwd_weights->m_conv_pol_b = std::move(weights);
        } else if (linecount == plain_conv_wts + 2) {
            // line 3 of policy head [1x1 convolutional bn 1]
	    assert (n_wts == m_policy_outputs);
	    m_bn_pol_w1 = std::move(weights);
        } else if (linecount == plain_conv_wts + 3) {
            // line 4 of policy head [1x1 convolutional bn 2]

	    process_bn_var(weights);
	    assert (n_wts == m_policy_outputs);
	    m_bn_pol_w2 = std::move(weights);

            // if the net has 'komi policy' layers, then their weigths
            // go here
        } else if (komipolicy_lines && linecount == plain_conv_wts + 4) {
            // line 1 of komi policy layers

            m_komipolicy_chans = n_wts /
                (NUM_INTERSECTIONS * m_policy_outputs + 1);
            assert (n_wts == (NUM_INTERSECTIONS * m_policy_outputs + 1)
                    * m_komipolicy_chans);
            m_kp1_pol_w = std::move(weights);
        } else if (komipolicy_lines && linecount == plain_conv_wts + 5) {
            // line 2 of komi policy layers

            assert (n_wts == m_komipolicy_chans);
            m_kp1_pol_b = std::move(weights);
        } else if (komipolicy_lines && linecount == plain_conv_wts + 6) {
            // line 3 of komi policy layers

            assert (n_wts == m_komipolicy_chans * m_komipolicy_chans);
            m_kp2_pol_w = std::move(weights);
        } else if (komipolicy_lines && linecount == plain_conv_wts + 7) {
            // line 4 of komi policy layers

            assert (n_wts == m_komipolicy_chans);
            m_kp2_pol_b = std::move(weights);
        } else if (linecount == plain_conv_wts + komipolicy_lines + 4) {
            // line [-2] of policy head [dense layer weights]

            // we do not check the board size here as it would be
            // uselessly complicate

            assert (n_wts == (m_policy_outputs * NUM_INTERSECTIONS
                              + (m_komi_policy ? 1 : 0)
                              + m_komipolicy_chans ) // assumes this is 0 when
                                                     // komi policy is not used)
                              * POTENTIAL_MOVES );

	        m_ip_pol_w = std::move(weights);

        } else if (linecount == plain_conv_wts + komipolicy_lines + 5) {
            // line [-1] of policy head [dense layer biases]
            // check if the board size is correct

	    if (n_wts != POTENTIAL_MOVES) {
                const auto netboardsize = std::sqrt(n_wts-1);
                myprintf("\nGiven network is for %.0fx%.0f, but this version "
                         "of SAI was compiled for %dx%d board!\n",
                         netboardsize, netboardsize, BOARD_SIZE, BOARD_SIZE);
                return 1;
            }

            m_ip_pol_b = std::move(weights);

        } else if (linecount == plain_conv_wts + komipolicy_lines + 6) {
            // line 1 of value head [1x1 convolutional weights]
	    m_val_outputs = n_wts/m_channels;
	    assert (n_wts == m_channels*m_val_outputs);
	    m_fwd_weights->m_conv_val_w = std::move(weights);
        } else if (linecount == plain_conv_wts + komipolicy_lines + 7) {
            // line 2 of value head [1x1 convolutional biases]
	    assert (n_wts == m_val_outputs);
            m_fwd_weights->m_conv_val_b = std::move(weights);
        } else if (linecount == plain_conv_wts + komipolicy_lines + 8) {
            // line 3 of value head [1x1 convolutional bn 1]
	    assert (n_wts == m_val_outputs);
            m_bn_val_w1 = std::move(weights);
        } else if (linecount == plain_conv_wts + komipolicy_lines + 9) {
            // line 4 of value head [1x1 convolutional bn 2]
	    assert (n_wts == m_val_outputs);
            process_bn_var(weights);
            m_bn_val_w2 = std::move(weights);
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
        m_value_head_type = DOUBLE_V;
        m_value_head_rets = 2;

	m_vbe_outputs = n_wts_2nd_val_head[0]/m_channels;
	assert (n_wts_2nd_val_head[0] == m_channels*m_vbe_outputs);
	m_fwd_weights->m_conv_vbe_w = std::move(wts_2nd_val_head[0]);

	assert (n_wts_2nd_val_head[1] == m_vbe_outputs);
	m_fwd_weights->m_conv_vbe_b = std::move(wts_2nd_val_head[1]);

	assert (n_wts_2nd_val_head[2] == m_vbe_outputs);
	m_bn_vbe_w1 = std::move(wts_2nd_val_head[2]);

	assert (n_wts_2nd_val_head[3] == m_vbe_outputs);
	process_bn_var(wts_2nd_val_head[3]);
	m_bn_vbe_w2 = std::move(wts_2nd_val_head[3]);

	m_vbe_chans = n_wts_2nd_val_head[4]/m_vbe_outputs/(NUM_INTERSECTIONS);
	assert (n_wts_2nd_val_head[4] == m_vbe_chans*m_vbe_outputs*NUM_INTERSECTIONS);
	m_ip1_vbe_w = std::move(wts_2nd_val_head[4]);

	assert (n_wts_2nd_val_head[5] == m_vbe_chans);
	m_ip1_vbe_b = std::move(wts_2nd_val_head[5]);

	int ret2 = n_wts_2nd_val_head[6]/m_vbe_chans;
	assert (n_wts_2nd_val_head[6] == m_vbe_chans*ret2);
	if (ret2 != 1) {
	  myprintf ("Unexpected in weights file: ret2=%d. %d -- %d -- %d.\n",
		    ret2,
		    n_wts_2nd_val_head[6],
		    m_vbe_chans,
		    n_wts_2nd_val_head[6]/m_vbe_chans);
	  return 1;
	}
	m_ip2_vbe_w = std::move(wts_2nd_val_head[6]);

	assert (n_wts_2nd_val_head[7] == 1);
	m_ip2_vbe_b = std::move(wts_2nd_val_head[7]);

	myprintf("Double value head. Type V.\n");
	myprintf("Alpha head: %d outputs, %d channels.\n", m_val_outputs, m_val_chans);
	myprintf("Beta head: %d outputs, %d channels.\n", m_vbe_outputs, m_vbe_chans);
    } else if (lastlines == 4) {
        m_value_head_type = DOUBLE_Y;
        m_value_head_rets = 2;

	m_vbe_chans = n_wts_2nd_val_head[0]/m_val_outputs/(NUM_INTERSECTIONS);
	assert (n_wts_2nd_val_head[0] == m_vbe_chans*m_val_outputs*NUM_INTERSECTIONS);
	m_ip1_vbe_w = std::move(wts_2nd_val_head[0]);

	assert (n_wts_2nd_val_head[1] == m_vbe_chans);
	m_ip1_vbe_b = std::move(wts_2nd_val_head[1]);

	int ret2 = n_wts_2nd_val_head[2]/m_vbe_chans;
	assert (n_wts_2nd_val_head[2] == m_vbe_chans*ret2);
	if (ret2 != 1)
	  return 1;
	m_ip2_vbe_w = std::move(wts_2nd_val_head[2]);

	assert (n_wts_2nd_val_head[3] == 1);
	m_ip2_vbe_b = std::move(wts_2nd_val_head[3]);

	myprintf("Double value head. Type Y.\n");
	myprintf("Common convolution: %d outputs.\n", m_val_outputs);
	myprintf("Alpha head: %d channels. Beta head: %d channels.\n", m_val_chans, m_vbe_chans);
    } else if (lastlines == 2) {
        m_value_head_type = DOUBLE_T;
	m_value_head_rets = 2;

	int ret2 = n_wts_2nd_val_head[0]/m_val_chans;
	assert (n_wts_2nd_val_head[0] == m_val_chans*ret2);
	if (ret2 != 1)
	  return 1;
	m_ip2_vbe_w = std::move(wts_2nd_val_head[0]);

	assert (n_wts_2nd_val_head[1] == 1);
	m_ip2_vbe_b = std::move(wts_2nd_val_head[1]);

	myprintf("Double value head. Type T: %d outputs, %d channels.\n",
		 m_val_outputs, m_val_chans);
    } else if (lastlines == 0) {
        if (m_value_head_rets == 2) {
	  m_value_head_type = DOUBLE_I;

	  myprintf("Double value head. Type I: %d outputs, %d channels.\n",
		   m_val_outputs, m_val_chans);
	}
	else if (m_value_head_rets == 1) {
          m_value_head_type = SINGLE;

	  myprintf("Single value head: %d outputs, %d channels.\n",
		   m_val_outputs, m_val_chans);
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
            // that they return the value for black instead of
            // the player to move. This is used by ELF Open Go.
            if (format_version == 2) {
                myprintf("Version 2 weights file (ELF).\n");
                m_value_head_not_stm = true;
            } else {
                if (format_version == 1) {
		            myprintf("Version 1 weights file (LZ).\n");
		        }
                m_value_head_not_stm = false;
            }
            if (format_version == 17) {
		        myprintf("Version 17 weights file (advanced board features).\n");
                m_adv_features = true;
                m_komi_policy = false;
            } else if (format_version == 49) {
		        myprintf("Version 49 weights file (komi policy + advanced board features).\n");
		        m_adv_features = true;
                m_komi_policy = true;
            } else {
		        m_adv_features = false;
                m_komi_policy = false;
	        }
            return load_v1_network(buffer, format_version);
        }
    }
    return 1;
}

std::unique_ptr<ForwardPipe>&& Network::init_net(int channels,
    std::unique_ptr<ForwardPipe>&& pipe) {

    pipe->initialize(channels);
    pipe->push_weights(WINOGRAD_ALPHA, m_input_planes, channels, m_fwd_weights);

    return std::move(pipe);
}

#ifdef USE_HALF
void Network::select_precision(int channels) {
    if (cfg_precision == precision_t::AUTO) {
        auto score_fp16 = float{-1.0};
        auto score_fp32 = float{-1.0};

        myprintf("Initializing OpenCL (autodetecting precision).\n");

        // Setup fp16 here so that we can see if we can skip autodetect.
        // However, if fp16 sanity check fails we will return a fp32 and pray it works.
        auto fp16_net = std::make_unique<OpenCLScheduler<half_float::half>>();
        if (!fp16_net->needs_autodetect()) {
            try {
                myprintf("OpenCL: using fp16/half compute support.\n");
                m_forward = init_net(channels, std::move(fp16_net));
                benchmark_time(1); // a sanity check run
            } catch (...) {
                myprintf("OpenCL: fp16/half failed despite driver claiming support.\n");
                myprintf("Falling back to single precision\n");
                m_forward.reset();
                m_forward = init_net(channels,
                    std::make_unique<OpenCLScheduler<float>>());
            }
            return;
        }

        // Start by setting up fp32.
        try {
            m_forward.reset();
            m_forward = init_net(channels,
                std::make_unique<OpenCLScheduler<float>>());
            score_fp32 = benchmark_time(100);
        } catch (...) {
            // empty - if exception thrown just throw away fp32 net
        }

        // Now benchmark fp16.
        try {
            m_forward.reset();
            m_forward = init_net(channels, std::move(fp16_net));
            score_fp16 = benchmark_time(100);
        } catch (...) {
            // empty - if exception thrown just throw away fp16 net
        }

        if (score_fp16 < 0.0f && score_fp32 < 0.0f) {
            myprintf("Both single precision and half precision failed to run.\n");
            throw std::runtime_error("Failed to initialize net.");
        } else if (score_fp16 < 0.0f) {
            myprintf("Using OpenCL single precision (half precision failed to run).\n");
            m_forward.reset();
            m_forward = init_net(channels,
                std::make_unique<OpenCLScheduler<float>>());
        } else if (score_fp32 < 0.0f) {
            myprintf("Using OpenCL half precision (single precision failed to run).\n");
        } else if (score_fp32 * 1.05f > score_fp16) {
            myprintf("Using OpenCL single precision (less than 5%% slower than half).\n");
            m_forward.reset();
            m_forward = init_net(channels,
                std::make_unique<OpenCLScheduler<float>>());
        } else {
            myprintf("Using OpenCL half precision (at least 5%% faster than single).\n");
        }
        return;
    } else if (cfg_precision == precision_t::SINGLE) {
        myprintf("Initializing OpenCL (single precision).\n");
        m_forward = init_net(channels,
            std::make_unique<OpenCLScheduler<float>>());
        return;
    } else if (cfg_precision == precision_t::HALF) {
        myprintf("Initializing OpenCL (half precision).\n");
        m_forward = init_net(channels,
            std::make_unique<OpenCLScheduler<half_float::half>>());
        return;
    }
}
#endif

void Network::initialize(int playouts, const std::string & weightsfile) {
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
#else
    myprintf("BLAS Core: built-in Eigen %d.%d.%d library.\n",
             EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION, EIGEN_MINOR_VERSION);
#endif

    m_fwd_weights = std::make_shared<ForwardPipeWeights>();

    // Make a guess at a good size as long as the user doesn't
    // explicitly set a maximum memory usage.
    m_nncache.set_size_from_playouts(playouts);

    // Prepare symmetry table
    for (auto s = 0; s < NUM_SYMMETRIES; ++s) {
        for (auto v = 0; v < NUM_INTERSECTIONS; ++v) {
            const auto newvtx =
                get_symmetry({v % BOARD_SIZE, v / BOARD_SIZE}, s);
            symmetry_nn_idx_table[s][v] =
                (newvtx.second * BOARD_SIZE) + newvtx.first;
            assert(symmetry_nn_idx_table[s][v] >= 0
                   && symmetry_nn_idx_table[s][v] < NUM_INTERSECTIONS);
        }
    }

    // Load network from file
    if (load_network_file(cfg_weightsfile)) {
        exit(EXIT_FAILURE);
    }
    m_value_head_sai = (m_value_head_type != SINGLE);

    auto weight_index = size_t{0};
    // Input convolution
    // Winograd transform convolution weights
    m_fwd_weights->m_conv_weights[weight_index] =
        winograd_transform_f(m_fwd_weights->m_conv_weights[weight_index],
                             m_channels, m_input_planes);
    weight_index++;

    // Residual block convolutions
    for (auto i = size_t{0}; i < m_residual_blocks * 2; i++) {
        m_fwd_weights->m_conv_weights[weight_index] =
            winograd_transform_f(m_fwd_weights->m_conv_weights[weight_index],
                                 m_channels, m_channels);
        weight_index++;
    }

    // Biases are not calculated and are typically zero but some networks might
    // still have non-zero biases.
    // Move biases to batchnorm means to make the output match without having
    // to separately add the biases.
    auto bias_size = m_fwd_weights->m_conv_biases.size();
    for (auto i = size_t{0}; i < bias_size; i++) {
        auto means_size = m_fwd_weights->m_batchnorm_means[i].size();
        for (auto j = size_t{0}; j < means_size; j++) {
            m_fwd_weights->m_batchnorm_means[i][j] -= m_fwd_weights->m_conv_biases[i][j];
            m_fwd_weights->m_conv_biases[i][j] = 0.0f;
        }
    }

    for (auto i = size_t{0}; i < m_bn_val_w1.size(); i++) {
        m_bn_val_w1[i] -= m_fwd_weights->m_conv_val_b[i];
        m_fwd_weights->m_conv_val_b[i] = 0.0f;
    }

    for (auto i = size_t{0}; i < m_bn_vbe_w1.size(); i++) {
        m_bn_vbe_w1[i] -= m_fwd_weights->m_conv_vbe_b[i];
        m_fwd_weights->m_conv_vbe_b[i] = 0.0f;
    }

    for (auto i = size_t{0}; i < m_bn_pol_w1.size(); i++) {
        m_bn_pol_w1[i] -= m_fwd_weights->m_conv_pol_b[i];
        m_fwd_weights->m_conv_pol_b[i] = 0.0f;
    }

#ifdef USE_OPENCL
    if (cfg_cpu_only) {
        myprintf("Initializing CPU-only evaluation.\n");
        m_forward = init_net(m_channels, std::make_unique<CPUPipe>());
    } else {
#ifdef USE_OPENCL_SELFCHECK
        // initialize CPU reference first, so that we can self-check
        // when doing fp16 vs. fp32 detections
        m_forward_cpu = init_net(m_channels, std::make_unique<CPUPipe>());
#endif
#ifdef USE_HALF
        // HALF support is enabled, and we are using the GPU.
        // Select the precision to use at runtime.
        select_precision(m_channels);
#else
        myprintf("Initializing OpenCL (single precision).\n");
        m_forward = init_net(m_channels,
                             std::make_unique<OpenCLScheduler<float>>());
#endif
    }

#else //!USE_OPENCL
    myprintf("Initializing CPU-only evaluation.\n");
    m_forward = init_net(m_channels, std::make_unique<CPUPipe>());
#endif

    // Need to estimate size before clearing up the pipe.
    get_estimated_size();
    m_fwd_weights.reset();
}

template<bool ReLU>
std::vector<float> innerproduct(const std::vector<float>& input,
                                const std::vector<float>& weights,
                                const std::vector<float>& biases) {
    const auto inputs = input.size();
    const auto outputs = biases.size();
    std::vector<float> output(outputs);

#ifdef USE_BLAS
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                // M     K
                outputs, inputs,
                1.0f, &weights[0], inputs,
                &input[0], 1,
                0.0f, &output[0], 1);
#else
    EigenVectorMap<float> y(output.data(), outputs);
    y.noalias() =
        ConstEigenMatrixMap<float>(weights.data(),
                                   inputs,
                                   outputs).transpose()
        * ConstEigenVectorMap<float>(input.data(), inputs);
#endif
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
               const float* const eltwise = nullptr) {
    const auto lambda_ReLU = [](const auto val) { return (val > 0.0f) ?
                                                          val : 0.0f; };
    for (auto c = size_t{0}; c < channels; ++c) {
        const auto mean = means[c];
        const auto scale_stddiv = stddivs[c];
        const auto arr = &data[c * spatial_size];

        if (eltwise == nullptr) {
            // Classical BN
            for (auto b = size_t{0}; b < spatial_size; b++) {
                arr[b] = lambda_ReLU(scale_stddiv * (arr[b] - mean));
            }
        } else {
            // BN + residual add
            const auto res = &eltwise[c * spatial_size];
            for (auto b = size_t{0}; b < spatial_size; b++) {
                arr[b] = lambda_ReLU((scale_stddiv * (arr[b] - mean)) + res[b]);
            }
        }
    }
}

#ifdef USE_OPENCL_SELFCHECK
void Network::compare_net_outputs(const Netresult& data,
                                  const Netresult& ref) {
    // Calculates L2-norm between data and ref.
    constexpr auto max_error = 0.2f;

    auto error = 0.0f;

    for (auto idx = size_t{0}; idx < data.policy.size(); ++idx) {
        const auto diff = data.policy[idx] - ref.policy[idx];
        error += diff * diff;
    }
    const auto diff_pass = data.policy_pass - ref.policy_pass;
    const auto diff_winrate = data.value - ref.value;
    error += diff_pass * diff_pass;
    error += diff_winrate * diff_winrate;

    error = std::sqrt(error);

    if (error > max_error || std::isnan(error)) {
        printf("Error in OpenCL calculation: Update your GPU drivers "
               "or reduce the amount of games played simultaneously.\n");
        throw std::runtime_error("OpenCL self-check mismatch.");
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

bool Network::probe_cache(const GameState* const state,
                          Network::Netresult& result) {
    if (m_nncache.lookup(state->board.get_hash(), result)) {
        return true;
    }
    // If we are not generating a self-play game, try to find
    // symmetries if we are in the early opening.
    if (!cfg_noise && !cfg_random_cnt
        && state->get_movenum()
           < (state->get_timecontrol().opening_moves(BOARD_SIZE) / 2)) {
        for (auto sym = 0; sym < Network::NUM_SYMMETRIES; ++sym) {
            if (sym == Network::IDENTITY_SYMMETRY) {
                continue;
            }
            const auto hash = state->get_symmetry_hash(sym);
            if (m_nncache.lookup(hash, result)) {
                decltype(result.policy) corrected_policy;
                for (auto idx = size_t{0}; idx < NUM_INTERSECTIONS; ++idx) {
                    const auto sym_idx = symmetry_nn_idx_table[sym][idx];
                    corrected_policy[idx] = result.policy[sym_idx];
                }
                result.policy = std::move(corrected_policy);
                return true;
            }
        }
    }

    return false;
}

Network::Netresult Network::get_output(
    const GameState* const state, const Ensemble ensemble,
    const int symmetry, const bool skip_cache, const bool force_selfcheck) {
    Netresult result;
    if (state->board.get_boardsize() != BOARD_SIZE) {
        return result;
    }

    if (!skip_cache) {
        // See if we already have this in the cache.
        if (probe_cache(state, result)) {
            return result;
        }
    }

    if (ensemble == DIRECT) {
        assert(symmetry >= 0 && symmetry < NUM_SYMMETRIES);
        result = get_output_internal(state, symmetry);
    } else if (ensemble == AVERAGE) {
        for (auto sym = 0; sym < NUM_SYMMETRIES; ++sym) {
            auto tmpresult = get_output_internal(state, sym);
            result.policy_pass +=
                tmpresult.policy_pass / static_cast<float>(NUM_SYMMETRIES);
            result.value += tmpresult.value / static_cast<float>(NUM_SYMMETRIES);;
            result.alpha += tmpresult.alpha / static_cast<float>(NUM_SYMMETRIES);;
            result.beta += tmpresult.beta / static_cast<float>(NUM_SYMMETRIES);;

            for (auto idx = size_t{0}; idx < NUM_INTERSECTIONS; idx++) {
                result.policy[idx] +=
                    tmpresult.policy[idx] / static_cast<float>(NUM_SYMMETRIES);
            }
        }
    } else {
        assert(ensemble == RANDOM_SYMMETRY);
        assert(symmetry == -1);
        const auto rand_sym = Random::get_Rng().randfix<NUM_SYMMETRIES>();
        result = get_output_internal(state, rand_sym);
#ifdef USE_OPENCL_SELFCHECK
        // Both implementations are available, self-check the OpenCL driver by
        // running both with a probability of 1/2000.
        // selfcheck is done here because this is the only place NN
        // evaluation is done on actual gameplay.
        if (m_forward_cpu != nullptr
            && (force_selfcheck || Random::get_Rng().randfix<SELFCHECK_PROBABILITY>() == 0)
        ) {
            auto result_ref = get_output_internal(state, rand_sym, true);
            compare_net_outputs(result, result_ref);
        }
#else
        (void)force_selfcheck;
#endif
    }

    // v2 format (ELF Open Go) returns black value, not stm
    if (m_value_head_not_stm) {
        if (state->board.get_to_move() == FastBoard::WHITE) {
            result.value = 1.0f - result.value;
        }
    }

    if (!cfg_symm_nonrandom) {
        // Insert result into cache.
        m_nncache.insert(state->board.get_hash(), result);
    }

    return result;
}

Network::Netresult Network::get_output_internal(
    const GameState* const state, const int symmetry, bool selfcheck) {
    assert(symmetry >= 0 && symmetry < NUM_SYMMETRIES);
    constexpr auto width = BOARD_SIZE;
    constexpr auto height = BOARD_SIZE;

    // if the input planes of the loaded network are even, then the
    // color of the current player is encoded in the last two planes
    const auto include_color = (0 == m_input_planes % 2);

    const auto input_data = gather_features(state, symmetry, m_input_moves, m_adv_features, include_color);
    std::vector<float> policy_data(m_policy_outputs * width * height);
    std::vector<float> val_data(m_val_outputs * width * height);
    std::vector<float> vbe_data(m_vbe_outputs * width * height);
#ifdef USE_OPENCL_SELFCHECK
    if (selfcheck) {
        m_forward_cpu->forward(input_data, policy_data, val_data, vbe_data);
    } else {
        m_forward->forward(input_data, policy_data, val_data, vbe_data);
    }
#else
    m_forward->forward(input_data, policy_data, val_data, vbe_data);
    (void) selfcheck;
#endif

    // Get the moves
    batchnorm<NUM_INTERSECTIONS>(m_policy_outputs, policy_data,
        m_bn_pol_w1.data(), m_bn_pol_w2.data());

    if (m_komi_policy) {
        float komi = state->get_komi();
        komi *= ( state->get_to_move() == FastBoard::BLACK ? -1.0 : 1.0 );
        policy_data.push_back(komi);
        const auto kp1 =
            innerproduct<true>(policy_data, m_kp1_pol_w, m_kp1_pol_b);
        const auto kp2 =
            innerproduct<true>(kp1, m_kp2_pol_w, m_kp2_pol_b);
        policy_data.pop_back();
        for (auto & i : kp2) {
            policy_data.push_back(i);
        }
    }

    const auto policy_out =
        innerproduct<false>(
            policy_data, m_ip_pol_w, m_ip_pol_b);
    const auto outputs = softmax(policy_out, cfg_softmax_temp);

    // Now get the value
    batchnorm<NUM_INTERSECTIONS>(m_val_outputs, val_data,
        m_bn_val_w1.data(), m_bn_val_w2.data());
    const auto val_channels =
        innerproduct<true>(
            val_data, m_ip1_val_w, m_ip1_val_b);
    const auto val_output =
        innerproduct<false>(val_data, m_ip2_val_w, m_ip2_val_b);

    Netresult result;

    if (m_value_head_type==DOUBLE_V) {
        // If double head value, also get beta
        batchnorm<NUM_INTERSECTIONS>(m_vbe_outputs, vbe_data,
                    m_bn_vbe_w1.data(), m_bn_vbe_w2.data());
        const auto vbe_channels =
            innerproduct<true>(vbe_data, m_ip1_vbe_w, m_ip1_vbe_b);
        const auto vbe_output =
            innerproduct<false>(vbe_channels, m_ip2_vbe_w, m_ip2_vbe_b);

        result.value = 0.5f;
        result.alpha = val_output[0];
        result.beta = std::exp(vbe_output[0]) * 10.0f / NUM_INTERSECTIONS;
        result.is_sai = true;
    } else if (m_value_head_type==DOUBLE_Y) {
        const auto vbe_channels =
            innerproduct<true>(val_data, m_ip1_vbe_w, m_ip1_vbe_b);
        const auto vbe_output =
            innerproduct<false>(vbe_channels, m_ip2_vbe_w, m_ip2_vbe_b);

        result.value = 0.5f;
        result.alpha = val_output[0];
        result.beta = std::exp(vbe_output[0]) * 10.0f / NUM_INTERSECTIONS;
        result.is_sai = true;
    } else if (m_value_head_type==DOUBLE_T) {
        const auto vbe_output =
            innerproduct<false>(val_channels, m_ip2_vbe_w, m_ip2_vbe_b);
        result.value = 0.5f;
        result.alpha = val_output[0];
        result.beta = std::exp(vbe_output[0]) * 10.0f / NUM_INTERSECTIONS;
        result.is_sai = true;
    } else if (m_value_head_type==DOUBLE_I) {
        result.value = 0.5f;
        result.alpha = val_output[0];
        result.beta = std::exp(val_output[1]) * 10.0f / NUM_INTERSECTIONS;
        result.is_sai = true;
    } else if (m_value_head_type==SINGLE) {
        result.value = (1.0f + std::tanh(val_output[0])) / 2.0f;
        result.alpha = 0.0f;
        result.beta = 1.0f;
        result.is_sai = false;
    }

    for (auto idx = size_t{0}; idx < NUM_INTERSECTIONS; idx++) {
        const auto sym_idx = symmetry_nn_idx_table[symmetry][idx];
        result.policy[sym_idx] = outputs[idx];
    }

    result.policy_pass = outputs[NUM_INTERSECTIONS];

    return result;
}

Network::Netresult_extended Network::get_extended(const FastState& state, const Network::Netresult& result) {
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

    float legal_policy = result.policy_pass;
    float illegal_policy = 0.0f;

    std::array<float, NUM_INTERSECTIONS> policies;

    const auto color = state->get_to_move();
    for (unsigned int y = 0; y < NUM_INTERSECTIONS; y++) {
        for (unsigned int x = 0; x < NUM_INTERSECTIONS; x++) {
            const auto vertex = state->board.get_vertex(x, y);
            const auto policy = result.policy[y * NUM_INTERSECTIONS + x];
            if (state->is_move_legal(color, vertex)) {
                legal_policy += policy;
                policies[y * BOARD_SIZE + x] = policy;
            } else {
                illegal_policy += policy;
                policies[y * BOARD_SIZE + x] = 0.0f;
            }
        }
    }

    for (unsigned int y = 0; y < NUM_INTERSECTIONS; y++) {
        for (unsigned int x = 0; x < NUM_INTERSECTIONS; x++) {
            const auto clean_policy = int(policies[y * NUM_INTERSECTIONS + x] * 1000.0f / legal_policy);
            line += boost::str(boost::format("%3d ") % clean_policy);
        }

        display_map.push_back(line);
        line.clear();
    }

    for (int i = display_map.size() - 1; i >= 0; --i) {
        myprintf("%s\n", display_map[i].c_str());
    }
    const auto pass_policy = int(result.policy_pass * 1000 / legal_policy);
    const auto illegal_millis = int(illegal_policy * 1000);

    myprintf("pass: %d, illegal: %d\n", pass_policy, illegal_millis);
    if (result.is_sai) {
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
        std::vector<Network::PolicyVertexPair> moves;
        for (auto i=0; i < NUM_INTERSECTIONS; i++) {
            const auto x = i % BOARD_SIZE;
            const auto y = i / BOARD_SIZE;
            const auto vertex = state->board.get_vertex(x, y);
            if (state->board.get_state(vertex) == FastBoard::EMPTY) {
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
                                    std::vector<float>::iterator black,
                                    std::vector<float>::iterator white,
                                    const int symmetry) {
    for (auto idx = 0; idx < NUM_INTERSECTIONS; idx++) {
        const auto sym_idx = symmetry_nn_idx_table[symmetry][idx];
        const auto x = sym_idx % BOARD_SIZE;
        const auto y = sym_idx / BOARD_SIZE;
        const auto color = board.get_state(x, y);
        if (color == FastBoard::BLACK) {
            black[idx] = float(true);
        } else if (color == FastBoard::WHITE) {
            white[idx] = float(true);
        }
    }
}

void Network::fill_input_plane_advfeat(std::shared_ptr<const KoState> const state,
                                    std::vector<float>::iterator legal,
                                    std::vector<float>::iterator atari,
                                    const int symmetry) {
    for (auto idx = 0; idx < NUM_INTERSECTIONS; idx++) {
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

std::vector<float> Network::gather_features(const GameState* const state,
					    const int symmetry,
					    const int input_moves,
					    const bool adv_features,
					    const bool include_color) {
    assert(symmetry >= 0 && symmetry < NUM_SYMMETRIES);

    // if advanced board features are included, for every input move
    // in addition to 2 planes with the stones there are 2 planes with
    // legal moves for current player and "atari" intersections for
    // either player
    auto moves_planes = input_moves * (2 + (adv_features ? 2 : 0));

    // if the color of the current player is included, two more input
    // planes are needed, otherwise one input plane filled with ones
    // will provide information on the border of the board for the CNN
    auto input_planes = moves_planes + (include_color ? 2 : 1);

    auto input_data = std::vector<float>(input_planes * NUM_INTERSECTIONS);

    const auto current_it = begin(input_data);
    const auto opponent_it = begin(input_data) + input_moves * NUM_INTERSECTIONS;
    auto legal_it = current_it;
    auto atari_it = current_it;

    if (adv_features) {
	legal_it += 2 * input_moves * NUM_INTERSECTIONS;
	atari_it += 3 * input_moves * NUM_INTERSECTIONS;
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
	begin(input_data) + moves_planes * NUM_INTERSECTIONS :
	begin(input_data) + (moves_planes + 1) * NUM_INTERSECTIONS;
    std::fill(onesfilled_it, onesfilled_it + NUM_INTERSECTIONS, float(true));

    const auto moves = std::min<size_t>(state->get_movenum() + 1, input_moves);
    // Go back in time, fill history boards
    for (auto h = size_t{0}; h < moves; h++) {
        // collect white, black occupation planes
        fill_input_plane_pair(state->get_past_state(h)->board,
                              black_it + h * NUM_INTERSECTIONS,
                              white_it + h * NUM_INTERSECTIONS,
                              symmetry);
	if (adv_features) {
	    fill_input_plane_advfeat(state->get_past_state(h),
				     legal_it + h * NUM_INTERSECTIONS,
				     atari_it + h * NUM_INTERSECTIONS,
				     symmetry);

	}
    }

    return input_data;
}

std::pair<int, int> Network::get_symmetry(const std::pair<int, int>& vertex,
                                          const int symmetry,
                                          const int board_size) {
    auto x = vertex.first;
    auto y = vertex.second;
    assert(x >= 0 && x < board_size);
    assert(y >= 0 && y < board_size);
    assert(symmetry >= 0 && symmetry < NUM_SYMMETRIES);

    if ((symmetry & 4) != 0) {
        std::swap(x, y);
    }

    if ((symmetry & 2) != 0) {
        x = board_size - x - 1;
    }

    if ((symmetry & 1) != 0) {
        y = board_size - y - 1;
    }

    assert(x >= 0 && x < board_size);
    assert(y >= 0 && y < board_size);
    assert(symmetry != IDENTITY_SYMMETRY || vertex == std::make_pair(x, y));
    return {x, y};
}

size_t Network::get_estimated_size() {
    if (estimated_size != 0) {
        return estimated_size;
    }
    auto result = size_t{0};

    const auto lambda_vector_size =  [](const std::vector<std::vector<float>> &v) {
        auto result = size_t{0};
        for (auto it = begin(v); it != end(v); ++it) {
            result += it->size() * sizeof(float);
        }
        return result;
    };

    result += lambda_vector_size(m_fwd_weights->m_conv_weights);
    result += lambda_vector_size(m_fwd_weights->m_conv_biases);
    result += lambda_vector_size(m_fwd_weights->m_batchnorm_means);
    result += lambda_vector_size(m_fwd_weights->m_batchnorm_stddevs);

    result += m_fwd_weights->m_conv_pol_w.size() * sizeof(float);
    result += m_fwd_weights->m_conv_pol_b.size() * sizeof(float);

    // Policy head
    result += m_policy_outputs * sizeof(float); // m_bn_pol_w1
    result += m_policy_outputs * sizeof(float); // m_bn_pol_w2
    result += m_policy_outputs * NUM_INTERSECTIONS
                             * POTENTIAL_MOVES * sizeof(float); //m_ip_pol_w
    result += POTENTIAL_MOVES * sizeof(float); // m_ip_pol_b

    // Value head
    result += m_fwd_weights->m_conv_val_w.size() * sizeof(float);
    result += m_fwd_weights->m_conv_val_b.size() * sizeof(float);
    result += m_val_outputs * sizeof(float); // m_bn_val_w1
    result += m_val_outputs * sizeof(float); // m_bn_val_w2

    result += m_val_outputs * NUM_INTERSECTIONS
                            * m_val_chans * sizeof(float); // m_ip1_val_w
    result += m_val_chans * sizeof(float);  // m_ip1_val_b

    result += m_val_chans * sizeof(float); // m_ip2_val_w
    result += sizeof(float); // m_ip2_val_b
    return estimated_size = result;
}

size_t Network::get_estimated_cache_size() {
    return m_nncache.get_estimated_size();
}

void Network::nncache_resize(int max_count) {
    return m_nncache.resize(max_count);
}
