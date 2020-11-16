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
    std::shared_ptr<const Scene> create_scene(const std::shared_ptr<scene::SceneGraph> &scene_graph) {
        scene_graph->commit();
        std::shared_ptr<Scene> scene(new Scene());
        scene->camera = [&] {
            Camera camera;
            if (auto perspective = scene_graph->camera->as<scene::PerspectiveCamera>()) {
                TRSTransform TRS{perspective->transform.translation, perspective->transform.rotation, Vec3(1.0)};
                auto c2w = TRS();
                camera = PerspectiveCamera(perspective->resolution, c2w, perspective->fov);
            }
            return camera;
        }();
        scene->accel = create_embree_accel();
        scene->accel->build(scene_graph);
        return scene;
    }
} // namespace akari::render