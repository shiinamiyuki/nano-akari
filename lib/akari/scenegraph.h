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

namespace akari::scene {

    template <typename T>
    using P = std::shared_ptr<T>;
    class Texture {
      public:
        std::variant<Float, Spectrum, std::string> value;
        AKR_SER(value)
        void clear() { value = Float(0.0); }
        void set_image_texture(const std::string &path) { value = path; }
        void set_color(Spectrum color) { value = color; }
        void set_float(Float v) { value = v; }
    };
    class Material {
      public:
        P<Texture> diffuse;
        P<Texture> specular;
        P<Texture> metallic;
        P<Texture> roughnes;
        AKR_SER(diffuse, specular, metallic, roughnes)
    };
    class Mesh {
      public:
        std::vector<vec3> vertices;
        std::vector<ivec3> indices;
        std::vector<vec3> normals;
        std::vector<vec2> texcoords;
        std::string path;
        AKR_SER(path)
    };
    class Instance {
      public:
        TRSTransform transform;
        P<Mesh> mesh;
        P<Material> material;
        std::vector<P<Instance>> children;
        AKR_SER(transform, mesh, material, children)
    };
    class SceneGraph {
      public:
        std::vector<P<Mesh>> meshes;
        std::vector<P<Instance>> instances;
        AKR_SER(meshes, instances)
    };
} // namespace akari::scene