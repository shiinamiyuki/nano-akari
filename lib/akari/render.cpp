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
    bool Scene::occlude(const Ray &ray) const { return accel->occlude1(ray); }
    std::optional<SurfaceInteraction> Scene::intersect(const Ray &ray) const {
        std::optional<Intersection> isct = accel->intersect1(ray);
        if (!isct) {
            return std::nullopt;
        }
        Triangle triangle = instances[isct->geom_id].get_triangle(isct->prim_id);
        SurfaceInteraction si(isct->uv, triangle);
        si.shape = &instances[isct->geom_id];
        return si;
    }
    Scene::~Scene() {
        camera.reset();
        light_sampler.reset();
        materials.clear();
        lights.clear();
        delete rsrc;
    }
    std::shared_ptr<const Scene> create_scene(Allocator<> alloc,
                                              const std::shared_ptr<scene::SceneGraph> &scene_graph) {
        scene_graph->commit();
        auto scene = make_pmr_shared<Scene>(alloc);
        {
            auto rsrc = alloc.resource();
            scene->rsrc = new astd::pmr::monotonic_buffer_resource(rsrc);
            scene->allocator = Allocator<>(scene->rsrc);
        }
        scene->camera = [&] {
            std::optional<Camera> camera;
            if (auto perspective = scene_graph->camera->as<scene::PerspectiveCamera>()) {
                TRSTransform TRS{perspective->transform.translation, perspective->transform.rotation, Vec3(1.0)};
                auto c2w = TRS();
                camera.emplace(PerspectiveCamera(perspective->resolution, c2w, perspective->fov));
            }
            return camera;
        }();
        std::unordered_map<const scene::Material *, const Material *> mat_map;
        auto create_tex = [&](const scene::P<scene::Texture> &tex_node) -> Texture {
            if (!tex_node) {
                return Texture(ConstantTexture(0.0));
            }
            std::optional<Texture> tex;
            std::visit(
                [&](auto &&arg) {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_same_v<T, Float>) {
                        tex.emplace(ConstantTexture(arg));
                    } else if constexpr (std::is_same_v<T, Spectrum>) {
                        tex.emplace(ConstantTexture(arg));
                    }
                },
                tex_node->value);
            return tex.value();
        };
        auto create_mat = [&](const scene::P<scene::Material> &mat_node) -> const Material * {
            auto it = mat_map.find(mat_node.get());
            if (it != mat_map.end())
                return it->second;
            auto mat = scene->allocator.new_object<Material>();
            mat->color = create_tex(mat_node->color);
            mat->emission = create_tex(mat_node->emission);
            mat_map.emplace(mat_node.get(), mat);
            scene->materials.emplace_back(mat);
            return mat;
        };
        auto create_instance = [&](Transform parent_transform, scene::P<scene::Node> node, auto &&self) -> void {
            Transform node_T = parent_transform * node->transform();
            for (auto &instance : node->instances) {
                Transform T = node_T * instance->transform();
                MeshInstance inst;
                inst.transform = T;
                inst.material = create_mat(instance->material);
                inst.indices = instance->mesh->indices;
                inst.normals = instance->mesh->normals;
                inst.texcoords = instance->mesh->texcoords;
                inst.vertices = instance->mesh->vertices;
                inst.mesh = instance->mesh.get();
                if (inst.material->emission.isa<ConstantTexture>() &&
                    luminance(inst.material->emission.get<ConstantTexture>()->evaluate_s(ShadingPoint())) <= 0.0) {
                    // not emissive
                } else {
                    // emissived
                    for (int i = 0; i < (int)inst.indices.size(); i++) {
                        AreaLight area_light(inst.get_triangle(i), inst.material->emission, false);
                        auto light = alloc.new_object<Light>(area_light);
                        scene->lights.emplace_back(light);
                        inst.lights.emplace_back(light);
                    }
                }
                scene->instances.emplace_back(std::move(inst));
            }
            for (auto &child : node->children) {
                self(node_T, child, self);
            }
        };
        create_instance(Transform(), scene_graph->root, create_instance);
        {
            BufferView<const Light *> lights(scene->lights.data(), scene->lights.size());
            std::vector<Float> power;
            for (auto light : lights) {
                power.emplace_back(1.0);
            }
            scene->light_sampler = std::make_shared<PowerLightSampler>(alloc, lights, power);
        }
        scene->accel = create_embree_accel();
        scene->accel->build(*scene, scene_graph);
        return scene;
    }
} // namespace akari::render