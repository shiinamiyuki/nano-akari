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
#include <akari/render.h>
#include <spdlog/spdlog.h>

namespace akari::render {
    struct BasicFloatEvaluator : TextureEvaluator<Float, BasicFloatEvaluator> {
        result_t do_evaluate(const Texture &tex, const ShadingPoint &sp) const {
            return tex.dispatch(overloaded{[&](const ConstantTexture &tex) -> result_t { return tex.value[0]; },
                                           [&](const ImageTexture &tex) -> result_t { return 0.0; }});
        }
    };

    struct BasicSpectrumEvaluator : TextureEvaluator<Spectrum, BasicSpectrumEvaluator> {
        result_t do_evaluate(const Texture &tex, const ShadingPoint &sp) const {
            return tex.dispatch(overloaded{[&](const ConstantTexture &tex) -> result_t { return tex.value; },
                                           [&](const ImageTexture &tex) -> result_t { return Spectrum(0.0); }});
        }
    };

    Film render_pt(PTConfig config, const Scene &scene) {
        Film film(scene.camera.resolution());
        auto Li = [&](const Ray &primary, Sampler &sampler) -> Spectrum {
            Spectrum L(0.0);
            Spectrum beta(1.0);
            Ray ray = primary;
            int depth = 0;
            while (true) {
                auto si = scene.intersect(ray);
                if (!si) {
                    L += Spectrum(1.0) * beta;
                    break;
                }
                if (depth > config.max_depth) {
                    break;
                }
                auto wo = -ray.d;
                BSDF bsdf = MaterialEvaluator<BasicFloatEvaluator, BasicSpectrumEvaluator>().evaluate(*si);
                auto sample = bsdf.sample(sampler.next1d(), sampler.next2d(), wo, si->sp());
                if (!sample || is_black(sample->f) || sample->pdf == 0.0) {
                    break;
                }

                beta *= sample->f * std::abs(dot(sample->wi, si->ns)) / sample->pdf;
                ray = Ray(si->p, sample->wi);
                depth++;
            }
            return L;
        };
        thread::parallel_for(thread::blocked_range<2>(film.resolution(), ivec2(16, 16)), [&](ivec2 id, uint32_t tid) {
            //  film.add_sample(id, Spectrum(1.0), 1.0);
            Sampler sampler = config.sampler;
            sampler.set_sample_index(id.y * film.resolution().x + id.x);
            for (int s = 0; s < config.spp; s++) {
                auto camera_sample = scene.camera.generate_ray(sampler.next2d(), sampler.next2d(), id);
                auto L = Li(camera_sample.ray, sampler);
                film.add_sample(id, L, 1.0);
            }
        });
        return film;
    }
} // namespace akari::render