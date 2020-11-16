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

#include <unordered_map>
#include <akari/scenegraph.h>
#include <akari/render.h>
#include <spdlog/spdlog.h>
#include <embree3/rtcore.h>
namespace akari::render {
    static inline RTCRay toRTCRay(const Ray &_ray) {
        RTCRay ray;
        auto _o = _ray.o;
        ray.dir_x = _ray.d.x;
        ray.dir_y = _ray.d.y;
        ray.dir_z = _ray.d.z;
        ray.org_x = _o.x;
        ray.org_y = _o.y;
        ray.org_z = _o.z;
        ray.tnear = _ray.tmin;
        ray.tfar = _ray.tmax;
        ray.flags = 0;
        return ray;
    }
    using scene::Mesh;
    using scene::P;
    class EmbreeAccelImpl : public EmbreeAccel {
        RTCScene rtcScene = nullptr;
        RTCDevice device = nullptr;
        std::unordered_map<const Mesh *, RTCScene> per_mesh_scene;

      public:
        EmbreeAccelImpl() { device = rtcNewDevice(nullptr); }
        void build(const std::shared_ptr<scene::SceneGraph> &scene) override {
            spdlog::info("building acceleration structure for {} meshes", scene->meshes.size());
            if (rtcScene) {
                rtcReleaseScene(rtcScene);
            }
            rtcScene = rtcNewScene(device);
            for (auto &mesh : scene->meshes) {
                const auto m_scene = rtcNewScene(device);
                {
                    const auto geometry = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
                    rtcSetSharedGeometryBuffer(geometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
                                               &mesh->vertices[0], 0, sizeof(float) * 3, mesh->vertices.size());
                    AKR_ASSERT_THROW(rtcGetDeviceError(device) == RTC_ERROR_NONE);
                    rtcSetSharedGeometryBuffer(geometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, &mesh->indices[0],
                                               0, sizeof(int) * 3, mesh->indices.size());
                    AKR_ASSERT_THROW(rtcGetDeviceError(device) == RTC_ERROR_NONE);
                    rtcCommitGeometry(geometry);
                    rtcAttachGeometry(m_scene, geometry);
                    rtcReleaseGeometry(geometry);
                }
                per_mesh_scene[mesh.get()] = m_scene;
            }
            auto recursive_build = [&](Transform parent_transform, P<scene::Node> node, auto &&self) {
                Transform node_T = parent_transform * node->transform();
                for (auto &instance : node->instances) {
                    Transform T = node_T * instance->transform();
                    const auto geometry = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_INSTANCE);
                    rtcSetGeometryTransform(geometry, 0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, &T.m);
                    rtcCommitGeometry(geometry);
                    rtcAttachGeometry(rtcScene, geometry);
                    rtcReleaseGeometry(geometry);
                }
            };
            recursive_build(Transform(), scene->root, recursive_build);
        }
        std::optional<Intersection> intersect1(const Ray &ray) const override {
            RTCRayHit rayHit;
            rayHit.ray = toRTCRay(ray);
            rayHit.ray.flags = 0;
            rayHit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
            rayHit.hit.primID = RTC_INVALID_GEOMETRY_ID;
            rayHit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
            RTCIntersectContext context;
            rtcInitIntersectContext(&context);
            rtcIntersect1(rtcScene, &context, &rayHit);
            if (rayHit.hit.geomID == RTC_INVALID_GEOMETRY_ID || rayHit.hit.primID == RTC_INVALID_GEOMETRY_ID)
                return std::nullopt;
            Intersection intersection;
            intersection.prim_id = rayHit.hit.primID;
            AKR_ASSERT(rayHit.hit.geomID == 0);
            intersection.geom_id = rayHit.hit.instID[0];
            intersection.uv = vec2(rayHit.hit.u, rayHit.hit.v);
            intersection.t = rayHit.ray.tfar;
            return intersection;
        }
        ~EmbreeAccelImpl() {
            for (auto &&[_, scene] : per_mesh_scene) {
                rtcReleaseScene(scene);
            }
            rtcReleaseScene(rtcScene);
            rtcReleaseDevice(device);
        }
    };
    std::shared_ptr<EmbreeAccel> create_embree_accel() { return std::make_shared<EmbreeAccelImpl>(); }
} // namespace akari::render