// Copyright 2020 shiinamiyuki
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <random>
#include <spdlog/spdlog.h>
#include <akari/render.h>
#include <akari/render_mlt.h>
#include <numeric>
namespace akari::render {
    namespace smcmc {
        using namespace mlt;
        struct State {
            ivec2 p_center;
            std::optional<MarkovChain> chain;
            // std::array<RadianceRecord, 5> Xs;
            struct Current {
                std::pair<Spectrum, Spectrum> radiance;
                Float Ts = 0.0;
            };
            Current current;
        };
    } // namespace smcmc
    Image render_smcmc(MLTConfig config, const Scene &scene) {
        using namespace mlt;
        using namespace smcmc;
        PTConfig pt_config;
        pt_config.max_depth = config.max_depth;
        pt_config.min_depth = config.min_depth;
        auto T = [](const std::pair<Spectrum, Spectrum> &s) -> Float { return hmax(s.second - s.first * 0.95); };
        const std::array<ivec2, 5> offsets = {
            ivec2(0, 0), ivec2(0, 1), ivec2(0, -1), ivec2(1, 0), ivec2(-1, 0),
        };
        auto run_uniform_global_mcmc = [&](PTConfig config, Allocator<> allocator, Sampler &sampler) {
            sampler.start_next_sample();
            ivec2 p_center =
                glm::min(scene.camera->resolution() - 1, ivec2(sampler.next2d() * vec2(scene.camera->resolution())));
            auto idx = std::min<int>(sampler.next1d() * 5, 4);
            auto p_film = p_center + offsets[idx];
            auto L = render_pt_pixel_separete_emitter_direct(pt_config, allocator, scene, sampler, p_film);
            return std::make_pair(p_center, L);
        };
        auto run_mcmc = [&](PTConfig config, Allocator<> allocator, const ivec2 &p_center, Sampler &sampler) {
            sampler.start_next_sample();
            (void)sampler.next2d();
            auto idx = std::min<int>(sampler.next1d() * 5, 4);
            auto p_film = p_center + offsets[idx];
            auto L = render_pt_pixel_separete_emitter_direct(pt_config, allocator, scene, sampler, p_film);
            return std::make_pair(idx, L);
        };
        auto run_mcmc_coherent = [&](PTConfig config, Allocator<> allocator, const ivec2 &p_center,
                                     const MLTSampler &base, int idx) {
            astd::pmr::vector<Float> Xs(allocator);
            for (auto &X : base.X) {
                Xs.push_back(X.value);
            }
            Sampler sampler = ReplaySampler(std::move(Xs), base.rng);
            sampler.start_next_sample();
            (void)sampler.next2d();
            (void)sampler.next1d();
            auto p_film = p_center + offsets[idx];
            auto L = render_pt_pixel_separete_emitter_direct(pt_config, allocator, scene, sampler, p_film);
            return L;
        };
        std::optional<MarkovChain> global_chain;
        std::random_device rd;
        {
            std::vector<uint64_t> seeds;
            seeds.reserve(config.num_bootstrap);
            {
                std::uniform_int_distribution<uint64_t> dist;
                for (int i = 0; i < config.num_bootstrap; i++) {
                    seeds.push_back(dist(rd));
                }
            }
            std::vector<Float> Ts;
            {
                astd::pmr::monotonic_buffer_resource resource;
                for (auto seed : seeds) {
                    Sampler sampler = MLTSampler(seed);
                    auto [p_film, L] = run_uniform_global_mcmc(pt_config, Allocator<>(&resource), sampler);
                    Ts.push_back(T(L));
                }
            }
            Distribution1D distribution(Ts.data(), Ts.size(), Allocator<>());
            std::uniform_real_distribution<> dist;
            global_chain = MarkovChain(MLTSampler(seeds[distribution.sample_discrete(dist(rd)).first]));
        }

        Array2D<State> states(scene.camera->resolution());
        Film film(scene.camera->resolution());
        {
            std::uniform_int_distribution<uint32_t> dist;
            astd::pmr::monotonic_buffer_resource resource;
            for (int i = 0; i < hprod(scene.camera->resolution()); i++) {
                auto [p_film, L] = run_uniform_global_mcmc(pt_config, Allocator<>(&resource), global_chain->sampler);
                if (!states(p_film).chain.has_value()) {
                    states(p_film).chain = MarkovChain(*global_chain);
                    states(p_film).chain->sampler.get<MLTSampler>()->rng = Rng(dist(rd));
                    states(p_film).p_center = p_film;
                }
            }
        }
        auto estimator = [](const State &state,
                            const std::array<RadianceRecord, 5> &Xs) -> std::pair<Spectrum, Spectrum> {
            for (auto &X : Xs) {
                if (glm::all(glm::equal(state.p_center, X.p_film))) {
                    return X.radiance;
                }
            }
            AKR_ASSERT(false);
        };
        auto run_mcmc2 = [&](Allocator<> alloc, State &state) -> std::array<RadianceRecord, 5> {
            std::array<RadianceRecord, 5> Xs;
            auto [X_idx, X] = run_mcmc(pt_config, alloc, state.p_center, state.chain->sampler);
            Xs[0] = RadianceRecord{state.p_center + offsets[X_idx], X};
            int cnt = 1;
            for (int i = 0; i < 5; i++) {
                if (i != X_idx) {
                    Xs[cnt] = RadianceRecord{state.p_center + offsets[i],
                                             run_mcmc_coherent(pt_config, alloc, state.p_center,
                                                               *state.chain->sampler.get<MLTSampler>(), i)};
                    cnt++;
                }
            }
            return Xs;
        };
        auto Ts = [&](const std::array<RadianceRecord, 5> &Xs) {
            auto value = T(Xs[0].radiance);
            for (int i = 1; i < 5; i++) {
                value = std::max(value, T(Xs[i].radiance));
            }
            return value;
        };
        // ensures every tile is initialized
        thread::parallel_for(thread::blocked_range<2>(states.dimension(), ivec2(16, 16)), [&](ivec2 id, uint32_t tid) {
            std::uniform_int_distribution<uint32_t> dist;
            astd::pmr::monotonic_buffer_resource resource;
            if (!states(id).chain.has_value()) {
                const int num_tries = 16;
                for (int i = 0; i < num_tries; i++) {
                    Sampler sampler = MLTSampler(dist(rd));
                    auto [idx, L] = run_mcmc(pt_config, Allocator<>(&resource), id, sampler);
                    if (T(L) > 0.0 || i == num_tries - 1) {
                        states(id).chain = MarkovChain(*sampler.get<MLTSampler>());
                        states(id).chain->sampler.get<MLTSampler>()->rng = Rng(dist(rd));
                        states(id).p_center = id;
                        break;
                    }
                }
            }
            AKR_ASSERT(states(id).chain.has_value());
            auto L = run_mcmc2(Allocator<>(&resource), states(id));
            states(id).current.radiance = estimator(states(id), L);
            states(id).current.Ts = Ts(L);
        });
        AtomicDouble acc_b(0.0);
        std::atomic_uint64_t n_large(0);
        auto replica_exchange = [&](Allocator<> alloc, State &a, State &b) {
            std::uniform_real_distribution<Float> dist;
            State ab;
            ab.p_center = a.p_center;
            ab.chain = b.chain;
            State ba;
            ba.p_center = b.p_center;
            ba.chain = a.chain;
            const auto Lab = run_mcmc2(alloc, ab);
            const auto Lba = run_mcmc2(alloc, ba);
            const auto Tab = Ts(Lab);
            const auto Tba = Ts(Lba);
            const auto Ta = a.current.Ts;
            const auto Tb = b.current.Ts;
            const auto accept = std::clamp<Float>(Tab * Tba / (Ta * Tb), 0.0, 1.0);

            if (accept > 0) {
                film.splat(ab.p_center, estimator(ab, Lab).second * accept / Tab);
                film.splat(ba.p_center, estimator(ba, Lba).second * accept / Tba);
            }
            if (1.0 - accept > 0) {
                film.splat(a.p_center, a.current.radiance.second * (1.0 - accept) / Ta);
                film.splat(b.p_center, b.current.radiance.second * (1.0 - accept) / Tb);
            }

            if (dist(rd) < accept) {
                a = ab;
                b = ba;

                a.current.radiance = estimator(ab, Lab);
                a.current.Ts = Tab;

                b.current.radiance = estimator(ba, Lba);
                b.current.Ts = Tba;
            }
        };
        auto independent_mcmc = [&](Allocator<> alloc, State &s) {
            std::uniform_real_distribution<Float> dist;
            const auto L = run_mcmc2(alloc, s);
            const auto Tnew = Ts(L);
            const auto accept = std::clamp<Float>(Tnew / s.current.Ts, 0.0, 1.0);
            auto mlt_sampler = s.chain->sampler.get<MLTSampler>();
            if (mlt_sampler->large_step) {
                acc_b.add(T(estimator(s, L)));
                n_large++;
            }
            if (accept > 0) {
                film.splat(s.p_center, estimator(s, L).second * accept / Tnew);
            }
            if (1.0 - accept > 0) {
                film.splat(s.p_center, s.current.radiance.second * (1.0 - accept) / s.current.Ts);
            }
            if (dist(rd) < accept) {
                mlt_sampler->accept();
                s.current.radiance = estimator(s, L);
                s.current.Ts = Tnew;
            } else {
                mlt_sampler->reject();
            }
        };
        // run mcmc

        std::vector<astd::pmr::monotonic_buffer_resource *> buffers;
        for (size_t i = 0; i < thread::num_work_threads(); i++) {
            buffers.emplace_back(new astd::pmr::monotonic_buffer_resource(astd::pmr::new_delete_resource()));
        }

        for (int m = 0; m < config.spp; m++) {
            // if (m % 2 == 0) {
            thread::parallel_for(
                thread::blocked_range<2>(states.dimension(), ivec2(16, 16)), [&](ivec2 id, uint32_t tid) {
                    independent_mcmc(Allocator<>(buffers[tid]), states(id)); //                           .
                    buffers[tid]->release();
                });
            // } else {
            //     int n = m / 2;
            //     switch (n % 4) {
            //     case 0:
            //         break;
            //     case 1:
            //         break;
            //     case 2:
            //         break;
            //     case 3:
            //         break;
            //     }
            // }
        }
        for (auto buf : buffers) {
            delete buf;
        }
        spdlog::info("render smcmc done");
        return rgba_image(ivec2(1));
    }
} // namespace akari::render