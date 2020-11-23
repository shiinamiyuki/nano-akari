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

#include <akari/render_ppg.h>
using namespace akari;
using render::DTreeWrapper;
using render::QTreeNode;
using render::Rng;
using render::STree;

int main() {
    auto f = [](double x, double y, double z) {
        return std::exp(-std::pow(x - 0.5, 2)) * std::exp(-std::pow(y - 0.5, 2));
    };
    DTreeWrapper dtree;
    Rng rng;
    double v1, v2;
    {
        VarianceTracker<double> var;
        for (int i = 0; i < 100; i++) {
            auto w = dtree.sample(vec2(rng.uniform_float(), rng.uniform_float()),
                                  vec2(rng.uniform_float(), rng.uniform_float()));
            auto pdf = dtree.pdf(w);
            // printf("%lf\n", f(w.x, w.y, w.z) / pdf);
            var.update(f(w.x, w.y, w.z) / pdf);
        }
        v1 = *var.variance();
    }

    for (int iter = 0; iter < 16; iter++) {
        for (int i = 0; i < 1024 * 1024; i++) {
            auto w = vec3(rng.uniform_float(), rng.uniform_float(), rng.uniform_float());
            dtree.deposit(w, f(w.x, w.y, w.z));
        }
        dtree.refine();
    }
   {
        VarianceTracker<double> var;
        for (int i = 0; i < 100; i++) {
            auto w = dtree.sample(vec2(rng.uniform_float(), rng.uniform_float()),
                                  vec2(rng.uniform_float(), rng.uniform_float()));
            auto pdf = dtree.pdf(w);
            // printf("%lf\n", f(w.x, w.y, w.z) / pdf);
            var.update(f(w.x, w.y, w.z) / pdf);
        }
        v2 = *var.variance();
    }
    printf("%lf %lf\n",v1,v2);
}