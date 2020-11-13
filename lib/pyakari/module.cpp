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

#include <sstream>
#include <pyakari/module.h>
#include <pybind11/stl_bind.h>
#include <akari/scenegraph.h>
#include <cereal/archives/json.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>

PYBIND11_MAKE_OPAQUE(std::vector<akari::scene::P<akari::scene::Mesh>>);
PYBIND11_MAKE_OPAQUE(std::vector<akari::scene::P<akari::scene::Instance>>);
namespace akari::python {

    template <typename T, int N>
    void register_vec(py::module &m, const char *name1, const char *name2) {
        using V = Vector<T, N>;
        auto c = py::class_<V>(m, name1);
        m.def(name2, []() -> V { return V(); });
        if constexpr (N >= 1) {
            m.def(name2, [](T x) -> V { return V(x); });
            c.def_readwrite("x", &V::x);
        }
        if constexpr (N == 2) {
            m.def(name2, [](T x, T y) -> V { return V(x, y); });
        }
        if constexpr (N == 3) {
            m.def(name2, [](T x, T y, T z) -> V { return V(x, y, z); });
        }
        if constexpr (N == 4) {
            m.def(name2, [](T x, T y, T z, T w) -> V { return V(x, y, z, w); });
        }
        if constexpr (N >= 2) {
            c.def_readwrite("y", &V::y);
        }
        if constexpr (N >= 3) {
            c.def_readwrite("z", &V::z);
        }
        if constexpr (N >= 4) {
            c.def_readwrite("w", &V::w);
        }
    }
    void init(py::module &m) {
        using namespace akari::scene;
        m.doc() = "nano-akari python interface";
        register_vec<float, 2>(m, "_vec2", "vec2");
        register_vec<float, 3>(m, "_vec3", "vec3");
        register_vec<float, 4>(m, "_vec4", "vec4");

        register_vec<int, 2>(m, "_ivec2", "ivec2");
        register_vec<int, 3>(m, "_ivec3", "ivec3");
        register_vec<int, 4>(m, "_ivec4", "ivec4");

        register_vec<double, 2>(m, "_dvec2", "dvec2");
        register_vec<double, 3>(m, "_dvec3", "dvec3");
        register_vec<double, 4>(m, "_dvec4", "dvec4");

        register_vec<bool, 2>(m, "_bvec2", "bvec2");
        register_vec<bool, 3>(m, "_bvec3", "bvec3");
        register_vec<bool, 4>(m, "_bvec4", "bvec4");

        py::class_<TRSTransform>(m, "TRSTransform")
            .def(py::init<>())
            .def_readwrite("translation", &TRSTransform::translation)
            .def_readwrite("rotation", &TRSTransform::rotation)
            .def_readwrite("scale", &TRSTransform::scale);

        py::class_<Texture, P<Texture>>(m, "Texture")
            .def(py::init<>())
            .def("set_image_texture", &Texture::set_image_texture)
            .def("set_color", &Texture::set_color)
            .def("set_float", &Texture::set_float);
        py::class_<Material, P<Material>>(m, "Material")
            .def(py::init<>())
            .def_readwrite("diffuse", &Material::diffuse)
            .def_readwrite("specular", &Material::specular)
            .def_readwrite("metallic", &Material::metallic)
            .def_readwrite("roughnes", &Material::roughnes);
        py::class_<Instance, P<Instance>>(m, "Instance").def(py::init<>());
        py::class_<SceneGraph, P<SceneGraph>>(m, "SceneGraph")
            .def(py::init<>())
            .def_readwrite("meshes", &SceneGraph::meshes)
            .def_readwrite("instances", &SceneGraph::instances);
        m.def("save_json", [](P<SceneGraph> scene) -> std::string {
            std::ostringstream os;
            {
                cereal::JSONOutputArchive ar(os);
                ar(scene);
            }
            return os.str();
        });
        m.def("load_json", [](const std::string &s) -> P<SceneGraph> {
            std::istringstream in(s);
            cereal::JSONInputArchive ar(in);
            P<SceneGraph> scene;
            ar(scene);
            return scene;
        });
        m.def("save_xml", [](P<SceneGraph> scene) -> std::string {
            std::ostringstream os;
            {
                cereal::XMLOutputArchive ar(os);
                ar(scene);
            }
            return os.str();
        });
        m.def("load_xml", [](const std::string &s) -> P<SceneGraph> {
            std::istringstream in(s);
            cereal::XMLInputArchive ar(in);
            P<SceneGraph> scene;
            ar(scene);
            return scene;
        });
        py::bind_vector<std::vector<P<Mesh>>>(m, "MeshArray");
        py::bind_vector<std::vector<P<Instance>>>(m, "InstanceArray");
    }
} // namespace akari::python

PYBIND11_MODULE(pyakari, m) { akari::python::init(m); }