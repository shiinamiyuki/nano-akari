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

#include <akari/util.h>
#include <akari/render_ppg.h>
#include <spdlog/spdlog.h>

namespace akari::render {
    namespace ppg {
        struct DirectLighting {
            Ray shadow_ray;
            Spectrum color;
            Float pdf;
        };

        struct SurfaceVertex {
            SurfaceInteraction si;
            Vec3 wo;
            Ray ray;
            Spectrum beta;
            std::optional<BSDF> bsdf;
            Float pdf = 0.0;
            BSDFType sampled_lobe = BSDFType::Unset;
            // SurfaceVertex() = default;
            SurfaceVertex(const Vec3 &wo, const SurfaceInteraction si) : wo(wo), si(si) {}
            Vec3 p() const { return si.p; }
            Vec3 ng() const { return si.ng; }
        };
        struct PathVertex : Variant<SurfaceVertex> {
            using Variant::Variant;
            Vec3 p() const {
                return dispatch([](auto &&arg) { return arg.p(); });
            }
            Vec3 ng() const {
                return dispatch([](auto &&arg) { return arg.ng(); });
            }
            Float pdf() const {
                return dispatch([](auto &&arg) { return arg.pdf; });
            }
            BSDFType sampled_lobe() const {
                return dispatch([=](auto &&arg) -> BSDFType {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_same_v<T, SurfaceVertex>) {
                        return arg.sampled_lobe;
                    }
                    return BSDFType::Unset;
                });
            }
            const Light *light(const Scene *scene) const {
                return dispatch([=](auto &&arg) -> const Light * {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_same_v<T, SurfaceVertex>) {
                        return arg.si.light();
                    }
                    return nullptr;
                });
            }
        };

        class GuidedPathTracer {
          public:
            const Scene *scene = nullptr;
            Sampler *sampler = nullptr;
            Spectrum L;
            Spectrum beta = Spectrum(1.0f);
            Allocator<> allocator;
            int depth = 0;
            int min_depth = 5;
            int max_depth = 5;
            bool useNEE = true;
            std::shared_ptr<STree> sTree;
            struct PPGVertex {
                vec3 wi;
                vec3 p;
                Spectrum L;
                Spectrum beta;
            };
            BufferView<PPGVertex> vertices;
            int n_vertices = 0;
            bool training = false;
            DTreeWrapper *dTree = nullptr;
            Float bsdfSamplingFraction = 0.5;
            static Float mis_weight(Float pdf_A, Float pdf_B) {
                pdf_A *= pdf_A;
                pdf_B *= pdf_B;
                return pdf_A / (pdf_A + pdf_B);
            }
            CameraSample camera_ray(const Camera *camera, const ivec2 &p) noexcept {
                CameraSample sample = camera->generate_ray(sampler->next2d(), sampler->next2d(), p);
                return sample;
            }
            std::pair<const Light *, Float> select_light() noexcept {
                return scene->light_sampler->sample(sampler->next2d());
            }

            std::optional<DirectLighting>
            compute_direct_lighting(SurfaceVertex &vertex, const std::pair<const Light *, Float> &selected) noexcept {
                auto [light, light_pdf] = selected;
                if (light) {
                    auto &si = vertex.si;
                    DirectLighting lighting;
                    LightSampleContext light_ctx;
                    light_ctx.u = sampler->next2d();
                    light_ctx.p = si.p;
                    LightSample light_sample = light->sample_incidence(light_ctx);
                    if (light_sample.pdf <= 0.0)
                        return std::nullopt;
                    light_pdf *= light_sample.pdf;
                    auto f = light_sample.I * vertex.bsdf->evaluate(vertex.wo, light_sample.wi) *
                             std::abs(dot(si.ns, light_sample.wi));
                    Float bsdf_pdf = vertex.bsdf->evaluate_pdf(vertex.wo, light_sample.wi);
                    lighting.color = f / light_pdf * mis_weight(light_pdf, bsdf_pdf);
                    lighting.shadow_ray = light_sample.shadow_ray;
                    lighting.pdf = light_pdf;
                    return lighting;
                } else {
                    return std::nullopt;
                }
            }

            void on_miss(const Ray &ray, const std::optional<PathVertex> &prev_vertex) noexcept {
                // if (scene->envmap) {
                //     on_hit_light(scene->envmap.get(), -ray.d, ShadingPoint(), prev_vertex);
                // }
            }

            void accumulate_radiance_wo_beta(const Spectrum &r) {
                Float irradiance = luminance(r);
                AKR_ASSERT(irradiance >= 0 && !std::isnan(irradiance) && !std::isinf(irradiance));
                L += beta * r;
                for (int i = 0; i < n_vertices && training; i++) {
                    vertices[i].L += vertices[i].beta * r;
                }
            }

            void accumulate_beta(const Spectrum &k) {
                beta *= k;
                for (int i = 0; i < n_vertices && training; i++) {
                    vertices[i].beta *= k;
                }
            }

            void on_hit_light(const Light *light, const Vec3 &wo, const ShadingPoint &sp,
                              const std::optional<PathVertex> &prev_vertex) {
                Spectrum I = light->Le(wo, sp);
                if (!useNEE || depth == 0 || BSDFType::Unset != (prev_vertex->sampled_lobe() & BSDFType::Specular)) {
                    accumulate_radiance_wo_beta(I);
                } else {
                    PointGeometry ref;
                    ref.n = prev_vertex->ng();
                    ref.p = prev_vertex->p();
                    auto light_pdf = light->pdf_incidence(ref, -wo) * scene->light_sampler->pdf(light);
                    if ((prev_vertex->sampled_lobe() & BSDFType::Specular) != BSDFType::Unset) {
                        accumulate_radiance_wo_beta(I);
                    } else {
                        Float weight_bsdf = mis_weight(prev_vertex->pdf(), light_pdf);
                        accumulate_radiance_wo_beta(weight_bsdf * I);
                    }
                }
            }

            // @param mat_pdf: supplied if material is already chosen
            std::optional<SurfaceVertex> on_surface_scatter(const Vec3 &wo, SurfaceInteraction &si,
                                                            const std::optional<PathVertex> &prev_vertex) noexcept {
                auto *material = si.material();
                if (si.triangle.light) {
                    on_hit_light(si.triangle.light, wo, si.sp(), prev_vertex);
                    return std::nullopt;
                } else if (depth < max_depth) {
                    auto u0 = sampler->next1d();
                    auto u1 = sampler->next2d();
                    auto u2 = sampler->next2d();
                    SurfaceVertex vertex(wo, si);
                    auto bsdf = material->evaluate(*sampler, allocator, si);
                    BSDFSampleContext sample_ctx{sampler->next1d(), sampler->next2d(), wo};
                    auto sample = bsdf.sample(sample_ctx);
                    if (u0 < bsdfSamplingFraction) {
                        sample = bsdf.sample(sample_ctx);
                        if (!sample)
                            return std::nullopt;
                        sample->pdf *= bsdfSamplingFraction;
                        if (BSDFType::Unset == (sample->type & BSDFType::Specular)) {
                            sample->pdf = sample->pdf + (1.0f - bsdfSamplingFraction) * dTree->pdf(sample->wi);
                        }
                    } else {
                        sample = BSDFSample{};
                        auto w = dTree->sample(u1, u2);
                        sample->wi = w;
                        sample->pdf = dTree->pdf(w);
                        AKR_CHECK(sample->pdf >= 0);
                        sample->f = bsdf.evaluate(wo, sample->wi);
                        sample->type = (BSDFType::All & ~BSDFType::Specular);
                        sample->pdf *= 1.0f - bsdfSamplingFraction;
                        sample->pdf = sample->pdf + bsdfSamplingFraction * bsdf.evaluate_pdf(wo, sample->wi);
                    }
                    AKR_CHECK(!std::isnan(sample->pdf));
                    AKR_CHECK(sample->pdf >= 0.0);
                    AKR_CHECK(hmin(sample->f) >= 0.0f);
                    if (std::isnan(sample->pdf) || sample->pdf == 0.0f) {
                        return std::nullopt;
                    }
                    vertex.bsdf = bsdf;
                    vertex.ray = Ray(si.p, sample->wi, Eps / std::abs(glm::dot(si.ng, sample->wi)));
                    vertex.beta = sample->f * std::abs(glm::dot(si.ns, sample->wi)) / sample->pdf;
                    vertex.pdf = sample->pdf;
                    vertex.sampled_lobe = sample->type;
                    return vertex;
                }
                return std::nullopt;
            }
            void run_megakernel(const Camera *camera, const ivec2 &raster) noexcept {
                auto camera_sample = camera_ray(camera, raster);
                Ray ray = camera_sample.ray;
                std::optional<PathVertex> prev_vertex;
                while (true) {
                    auto si = scene->intersect(ray);
                    if (!si) {
                        on_miss(ray, prev_vertex);
                        break;
                    }
                    auto [_dTree, tree_depth] = sTree->dTree(si->p);
                    dTree = _dTree;
                    auto wo = -ray.d;
                    auto vertex = on_surface_scatter(wo, *si, prev_vertex);
                    if (!vertex) {
                        break;
                    }
                    if ((vertex->sampled_lobe & BSDFType::Specular) == BSDFType::Unset && useNEE) {
                        std::optional<DirectLighting> has_direct = compute_direct_lighting(*vertex, select_light());
                        if (has_direct) {
                            auto &direct = *has_direct;
                            if (!is_black(direct.color) && !scene->occlude(direct.shadow_ray)) {
                                accumulate_radiance_wo_beta(direct.color);
                            }
                        }
                    }
                    vertices[n_vertices].L = Spectrum(0);
                    {
                        // vec3 u = vec3(sampler->next1d(), sampler->next1d(), sampler->next1d());
                        auto p = si->p;
                        // p = sTree->box.clip(p + dtree_voxel_size * (u - vec3(0.5)));
                        vertices[n_vertices].p = p;
                    }
                    vertices[n_vertices].wi = vertex->ray.d;
                    vertices[n_vertices].beta = Spectrum(1.0 / vertex->pdf);
                    accumulate_beta(vertex->beta);
                    n_vertices++;
                    depth++;
                    // if (depth > min_depth) {
                    //     Float continue_prob = std::min<Float>(1.0, hmax(beta)) * 0.95;
                    //     if (continue_prob > 0.0 && continue_prob < sampler->next1d()) {
                    //         accumulate_beta(Spectrum(1.0 / continue_prob));
                    //     } else {
                    //         break;
                    //     }
                    // }
                    ray = vertex->ray;
                    prev_vertex = PathVertex(*vertex);
                }
                if (training) {
                    for (int i = 0; i < n_vertices; i++) {
                        auto irradiance = luminance(clamp_zero(vertices[i].L));
                        AKR_CHECK(irradiance >= 0);
                        sTree->deposit(vertices[i].p, vertices[i].wi, irradiance);
                    }
                }
            }
        };
    } // namespace ppg
    struct RatioStat {
        std::atomic_uint64_t good, total;
        void accumluate(bool g) {
            if (g) {
                good++;
            }
            total++;
        }
        void clear() {
            good = 0;
            total = 0;
        }
        double ratio() const { return double(good.load()) / double(total.load()); }
    };
    Image render_ppg(PPGConfig config, const Scene &scene) {
        std::shared_ptr<STree> sTree(new STree(scene.accel->world_bounds()));
        RatioStat non_zero_path;
        Array2D<Spectrum> sum(scene.camera->resolution());
        Spectrum sum_weights(0.0);
        std::vector<astd::pmr::monotonic_buffer_resource *> buffers;
        for (size_t i = 0; i < thread::num_work_threads(); i++) {
            buffers.emplace_back(new astd::pmr::monotonic_buffer_resource(astd::pmr::new_delete_resource()));
        }
        std::vector<Sampler> samplers(hprod(sum.dimension()));
        for (size_t i = 0; i < samplers.size(); i++) {
            samplers[i] = config.sampler;
            samplers[i].set_sample_index(i);
        }
        uint32_t pass = 0;
        uint32_t accumulatedSamples = 0;
        uint32_t trainingSamples = config.spp;
        for (pass = 0; accumulatedSamples < trainingSamples; pass++) {
            non_zero_path.clear();
            size_t samples;
            samples = 2ull << pass;
            auto nextPassSamples = 4u << pass;
            if (accumulatedSamples + samples + nextPassSamples > (uint32_t)trainingSamples) {
                samples = (uint32_t)trainingSamples - accumulatedSamples;
            }
            spdlog::info("Learning pass {}, spp:{}", pass + 1, samples);
            accumulatedSamples += samples;
            Film film(scene.camera->resolution());
            Array2D<Spectrum> variance(scene.camera->resolution());
            thread::parallel_for(
                thread::blocked_range<2>(film.resolution(), ivec2(16, 16)), [&](ivec2 id, uint32_t tid) {
                    auto Li = [&](const ivec2 p, Sampler &sampler) -> Spectrum {
                        ppg::GuidedPathTracer pt;
                        pt.min_depth = config.min_depth;
                        pt.max_depth = config.max_depth;
                        pt.n_vertices = 0;
                        pt.vertices =
                            BufferView(Allocator<>(buffers[tid])
                                           .allocate_object<ppg::GuidedPathTracer::PPGVertex>(config.max_depth + 1),
                                       config.max_depth + 1);
                        pt.L = Spectrum(0.0);
                        pt.beta = Spectrum(1.0);
                        pt.sampler = &sampler;
                        pt.scene = &scene;
                        pt.useNEE = false;
                        pt.training = true;
                        pt.sTree = sTree;
                        pt.allocator = Allocator<>(buffers[tid]);
                        pt.run_megakernel(&scene.camera.value(), p);
                        non_zero_path.accumluate(!is_black(pt.L));
                        buffers[tid]->release();
                        return pt.L;
                    };
                    Sampler &sampler = samplers[id.x + id.y * film.resolution().x];
                    VarianceTracker<Spectrum> var;
                    for (int s = 0; s < config.spp; s++) {
                        sampler.start_next_sample();
                        auto L = Li(id, sampler);
                        var.update(L);
                        film.add_sample(id, L, 1.0);
                    }
                    variance(id) = var.variance().value();
                });
            Array2D<Spectrum> pixel_weight = Array2D<Spectrum>::ones(variance.dimension()).safe_div(variance);
            Spectrum weight = pixel_weight.sum();
            sum += Spectrum(weight) * film.to_array2d();
            sum_weights += weight;
            spdlog::info("Refining SDTree; pass: {}", pass + 1);
            spdlog::info("nodes: {}", sTree->nodes.size());
            spdlog::info("non zero path:{}%", non_zero_path.ratio() * 100);
            sTree->refine(12000 * std::sqrt(samples));
        }
        Film film(scene.camera->resolution());
        Array2D<Spectrum> variance(scene.camera->resolution());
        thread::parallel_for(thread::blocked_range<2>(film.resolution(), ivec2(16, 16)), [&](ivec2 id, uint32_t tid) {
            auto Li = [&](const ivec2 p, Sampler &sampler) -> Spectrum {
                ppg::GuidedPathTracer pt;
                pt.min_depth = config.min_depth;
                pt.max_depth = config.max_depth;
                pt.n_vertices = 0;
                pt.vertices = BufferView(
                    Allocator<>(buffers[tid]).allocate_object<ppg::GuidedPathTracer::PPGVertex>(config.max_depth + 1),
                    config.max_depth + 1);
                pt.L = Spectrum(0.0);
                pt.beta = Spectrum(1.0);
                pt.sampler = &sampler;
                pt.scene = &scene;
                pt.training = false;
                pt.useNEE = false;
                pt.sTree = sTree;
                pt.allocator = Allocator<>(buffers[tid]);
                pt.run_megakernel(&scene.camera.value(), p);
                non_zero_path.accumluate(!is_black(pt.L));
                buffers[tid]->release();
                return pt.L;
            };
            Sampler &sampler = samplers[id.x + id.y * film.resolution().x];
            VarianceTracker<Spectrum> var;
            for (int s = 0; s < config.spp; s++) {
                sampler.start_next_sample();
                auto L = Li(id, sampler);
                var.update(L);
                film.add_sample(id, L, 1.0);
            }
            variance(id) = var.variance().value();
        });
        {
            Array2D<Spectrum> pixel_weight = Array2D<Spectrum>::ones(variance.dimension()).safe_div(variance);
            Spectrum weight = pixel_weight.sum();
            sum += Spectrum(weight) * film.to_array2d();
            sum_weights += weight;
        }
        for (auto buf : buffers) {
            delete buf;
        }
        spdlog::info("render ppg done");
        return array2d_to_rgb(sum / Spectrum(sum_weights));
    }
} // namespace akari::render