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

#pragma once
#include <akari/util.h>
#include <akari/image.h>
#include <akari/scenegraph.h>
#include <array>
namespace akari::scene {
    class SceneGraph;
}
namespace akari::render {
#pragma region sampling
    AKR_XPU inline glm::vec2 concentric_disk_sampling(const glm::vec2 &u) {
        glm::vec2 uOffset = ((float(2.0) * u) - glm::vec2(int32_t(1), int32_t(1)));
        if (((uOffset.x == float(0.0)) && (uOffset.y == float(0.0))))
            return glm::vec2(int32_t(0), int32_t(0));
        float theta = float();
        float r = float();
        if ((glm::abs(uOffset.x) > glm::abs(uOffset.y))) {
            r = uOffset.x;
            theta = (PiOver4 * (uOffset.y / uOffset.x));
        } else {
            r = uOffset.y;
            theta = (PiOver2 - (PiOver4 * (uOffset.x / uOffset.y)));
        }
        return (r * glm::vec2(glm::cos(theta), glm::sin(theta)));
    }
    AKR_XPU inline glm::vec3 cosine_hemisphere_sampling(const glm::vec2 &u) {
        glm::vec2 uv = concentric_disk_sampling(u);
        float r = glm::dot(uv, uv);
        float h = glm::sqrt(glm::max(float(float(0.0)), float((float(1.0) - r))));
        return glm::vec3(uv.x, h, uv.y);
    }
    AKR_XPU inline float cosine_hemisphere_pdf(float cosTheta) { return (cosTheta * InvPi); }
    AKR_XPU inline float uniform_sphere_pdf() { return (float(1.0) / (float(4.0) * Pi)); }
    AKR_XPU inline glm::vec3 uniform_sphere_sampling(const glm::vec2 &u) {
        float z = (float(1.0) - (float(2.0) * u[int32_t(0)]));
        float r = glm::sqrt(glm::max(float(0.0), (float(1.0) - (z * z))));
        float phi = ((float(2.0) * Pi) * u[int32_t(1)]);
        return glm::vec3((r * glm::cos(phi)), (r * glm::sin(phi)), z);
    }
    AKR_XPU inline glm::vec2 uniform_sample_triangle(const glm::vec2 &u) {
        float su0 = glm::sqrt(u[int32_t(0)]);
        float b0 = (float(1.0) - su0);
        float b1 = (u[int32_t(1)] * su0);
        return glm::vec2(b0, b1);
    }
#pragma endregion
#pragma region geometry
    AKR_XPU inline float cos_theta(const glm::vec3 &w) { return w.y; }
    AKR_XPU inline float abs_cos_theta(const glm::vec3 &w) { return glm::abs(cos_theta(w)); }
    AKR_XPU inline float cos2_theta(const glm::vec3 &w) { return (w.y * w.y); }
    AKR_XPU inline float sin2_theta(const glm::vec3 &w) { return (float(1.0) - cos2_theta(w)); }
    AKR_XPU inline float sin_theta(const glm::vec3 &w) { return glm::sqrt(glm::max(float(0.0), sin2_theta(w))); }
    AKR_XPU inline float tan2_theta(const glm::vec3 &w) { return (sin2_theta(w) / cos2_theta(w)); }
    AKR_XPU inline float tan_theta(const glm::vec3 &w) { return glm::sqrt(glm::max(float(0.0), tan2_theta(w))); }
    AKR_XPU inline float cos_phi(const glm::vec3 &w) {
        float sinTheta = sin_theta(w);
        return (sinTheta == float(0.0)) ? float(1.0) : glm::clamp((w.x / sinTheta), -float(1.0), float(1.0));
    }
    AKR_XPU inline float sin_phi(const glm::vec3 &w) {
        float sinTheta = sin_theta(w);
        return (sinTheta == float(0.0)) ? float(0.0) : glm::clamp((w.z / sinTheta), -float(1.0), float(1.0));
    }
    AKR_XPU inline float cos2_phi(const glm::vec3 &w) { return (cos_phi(w) * cos_phi(w)); }
    AKR_XPU inline float sin2_phi(const glm::vec3 &w) { return (sin_phi(w) * sin_phi(w)); }
    AKR_XPU inline bool same_hemisphere(const glm::vec3 &wo, const glm::vec3 &wi) {
        return ((wo.y * wi.y) >= float(0.0));
    }
    AKR_XPU inline std::optional<glm::vec3> refract(const glm::vec3 &wi, const glm::vec3 &n, float eta) {
        float cosThetaI = glm::dot(n, wi);
        float sin2ThetaI = glm::max(float(0.0), (float(1.0) - (cosThetaI * cosThetaI)));
        float sin2ThetaT = ((eta * eta) * sin2ThetaI);
        if ((sin2ThetaT >= float(1.0)))
            return std::nullopt;
        float cosThetaT = glm::sqrt((float(1.0) - sin2ThetaT));
        auto wt = ((eta * -wi) + (((eta * cosThetaI) - cosThetaT) * n));
        return wt;
    }
    AKR_XPU inline vec3 faceforward(const vec3 &w, const vec3 &n) { return dot(w, n) < 0.0 ? -n : n; }
    AKR_XPU inline float fr_dielectric(float cosThetaI, float etaI, float etaT) {
        bool entering = (cosThetaI > float(0.0));
        if (!entering) {
            std::swap(etaI, etaT);
            cosThetaI = glm::abs(cosThetaI);
        }
        float sinThetaI = glm::sqrt(glm::max(float(0.0), (float(1.0) - (cosThetaI * cosThetaI))));
        float sinThetaT = ((etaI / etaT) * sinThetaI);
        if ((sinThetaT >= float(1.0)))
            return float(1.0);
        float cosThetaT = glm::sqrt(glm::max(float(0.0), (float(1.0) - (sinThetaT * sinThetaT))));
        float Rpar = (((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + (etaI * cosThetaT)));
        float Rper = (((etaI * cosThetaI) - (etaT * cosThetaT)) / ((etaI * cosThetaI) + (etaT * cosThetaT)));
        return (float(0.5) * ((Rpar * Rpar) + (Rper * Rper)));
    }
    AKR_XPU inline glm::vec3 fr_conductor(float cosThetaI, const glm::vec3 &etaI, const glm::vec3 &etaT,
                                          const glm::vec3 &k) {
        float CosTheta2 = (cosThetaI * cosThetaI);
        float SinTheta2 = (float(1.0) - CosTheta2);
        glm::vec3 Eta = (etaT / etaI);
        glm::vec3 Etak = (k / etaI);
        glm::vec3 Eta2 = (Eta * Eta);
        glm::vec3 Etak2 = (Etak * Etak);
        glm::vec3 t0 = ((Eta2 - Etak2) - SinTheta2);
        glm::vec3 a2plusb2 = glm::sqrt(((t0 * t0) + ((float(4.0) * Eta2) * Etak2)));
        glm::vec3 t1 = (a2plusb2 + CosTheta2);
        glm::vec3 a = glm::sqrt((float(0.5) * (a2plusb2 + t0)));
        glm::vec3 t2 = ((float(2.0) * a) * cosThetaI);
        glm::vec3 Rs = ((t1 - t2) / (t1 + t2));
        glm::vec3 t3 = ((CosTheta2 * a2plusb2) + (SinTheta2 * SinTheta2));
        glm::vec3 t4 = (t2 * SinTheta2);
        glm::vec3 Rp = ((Rs * (t3 - t4)) / (t3 + t4));
        return (float(0.5) * (Rp + Rs));
    }

    AKR_XPU inline vec3 spherical_to_xyz(float sinTheta, float cosTheta, float phi) {
        return glm::vec3(sinTheta * glm::cos(phi), cosTheta, sinTheta * glm::sin(phi));
    }

    AKR_XPU inline float spherical_theta(const vec3 &v) { return glm::acos(glm::clamp(v.y, -1.0f, 1.0f)); }

    AKR_XPU inline float spherical_phi(const glm::vec3 v) {
        float p = glm::atan(v.z, v.x);
        return p < 0.0 ? (p + 2.0 * Pi) : p;
    }
#pragma endregion
#pragma region

    static const int32_t MicrofacetGGX = int32_t(0);
    static const int32_t MicrofacetBeckmann = int32_t(1);
    static const int32_t MicrofacetPhong = int32_t(2);
    AKR_XPU inline float BeckmannD(float alpha, const glm::vec3 &m) {
        if ((m.y <= float(0.0)))
            return float(0.0);
        float c = cos2_theta(m);
        float t = tan2_theta(m);
        float a2 = (alpha * alpha);
        return (glm::exp((-t / a2)) / (((Pi * a2) * c) * c));
    }
    AKR_XPU inline float BeckmannG1(float alpha, const glm::vec3 &v, const glm::vec3 &m) {
        if (((glm::dot(v, m) * v.y) <= float(0.0))) {
            return float(0.0);
        }
        float a = (float(1.0) / (alpha * tan_theta(v)));
        if ((a < float(1.6))) {
            return (((float(3.535) * a) + ((float(2.181) * a) * a)) /
                    ((float(1.0) + (float(2.276) * a)) + ((float(2.577) * a) * a)));
        } else {
            return float(1.0);
        }
    }
    AKR_XPU inline float PhongG1(float alpha, const glm::vec3 &v, const glm::vec3 &m) {
        if (((glm::dot(v, m) * v.y) <= float(0.0))) {
            return float(0.0);
        }
        float a = (glm::sqrt(((float(0.5) * alpha) + float(1.0))) / tan_theta(v));
        if ((a < float(1.6))) {
            return (((float(3.535) * a) + ((float(2.181) * a) * a)) /
                    ((float(1.0) + (float(2.276) * a)) + ((float(2.577) * a) * a)));
        } else {
            return float(1.0);
        }
    }
    AKR_XPU inline float PhongD(float alpha, const glm::vec3 &m) {
        if ((m.y <= float(0.0)))
            return float(0.0);
        return (((alpha + float(2.0)) / (float(2.0) * Pi)) * glm::pow(m.y, alpha));
    }
    AKR_XPU inline float GGX_D(float alpha, const glm::vec3 &m) {
        if ((m.y <= float(0.0)))
            return float(0.0);
        float a2 = (alpha * alpha);
        float c2 = cos2_theta(m);
        float t2 = tan2_theta(m);
        float at = (a2 + t2);
        return (a2 / ((((Pi * c2) * c2) * at) * at));
    }
    AKR_XPU inline float GGX_G1(float alpha, const glm::vec3 &v, const glm::vec3 &m) {
        if (((glm::dot(v, m) * v.y) <= float(0.0))) {
            return float(0.0);
        }
        return (float(2.0) / (float(1.0) + glm::sqrt((float(1.0) + ((alpha * alpha) * tan2_theta(m))))));
    }
    struct MicrofacetModel {
        int32_t type;
        float alpha;
    };
    AKR_XPU inline MicrofacetModel microfacet_new(int32_t type, float roughness) {
        float alpha = float();
        if ((type == MicrofacetPhong)) {
            alpha = ((float(2.0) / (roughness * roughness)) - float(2.0));
        } else {
            alpha = roughness;
        }
        return MicrofacetModel{type, alpha};
    }
    AKR_XPU inline float microfacet_D(const MicrofacetModel &model, const glm::vec3 &m) {
        int32_t type = model.type;
        float alpha = model.alpha;
        switch (type) {
        case MicrofacetBeckmann: {
            return BeckmannD(alpha, m);
        }
        case MicrofacetPhong: {
            return PhongD(alpha, m);
        }
        case MicrofacetGGX: {
            return GGX_D(alpha, m);
        }
        }
        return float(0.0);
    }
    AKR_XPU inline float microfacet_G1(const MicrofacetModel &model, const glm::vec3 &v, const glm::vec3 &m) {
        int32_t type = model.type;
        float alpha = model.alpha;
        switch (type) {
        case MicrofacetBeckmann: {
            return BeckmannG1(alpha, v, m);
        }
        case MicrofacetPhong: {
            return PhongG1(alpha, v, m);
        }
        case MicrofacetGGX: {
            return GGX_G1(alpha, v, m);
        }
        }
        return float(0.0);
    }
    AKR_XPU inline float microfacet_G(const MicrofacetModel &model, const glm::vec3 &i, const glm::vec3 &o,
                                      const glm::vec3 &m) {
        return (microfacet_G1(model, i, m) * microfacet_G1(model, o, m));
    }
    AKR_XPU inline glm::vec3 microfacet_sample_wh(const MicrofacetModel &model, const glm::vec3 &wo,
                                                  const glm::vec2 &u) {
        int32_t type = model.type;
        float alpha = model.alpha;
        float phi = ((float(2.0) * Pi) * u.y);
        float cosTheta = float(0.0);
        switch (type) {
        case MicrofacetBeckmann: {
            float t2 = ((-alpha * alpha) * glm::log((float(1.0) - u.x)));
            cosTheta = (float(1.0) / glm::sqrt((float(1.0) + t2)));
            break;
        }
        case MicrofacetPhong: {
            cosTheta = glm::pow(u.x, float((float(1.0) / (alpha + float(2.0)))));
            break;
        }
        case MicrofacetGGX: {
            float t2 = (((alpha * alpha) * u.x) / (float(1.0) - u.x));
            cosTheta = (float(1.0) / glm::sqrt((float(1.0) + t2)));
            break;
        }
        }
        float sinTheta = glm::sqrt(glm::max(float(0.0), (float(1.0) - (cosTheta * cosTheta))));
        glm::vec3 wh = glm::vec3((glm::cos(phi) * sinTheta), cosTheta, (glm::sin(phi) * sinTheta));
        if (!same_hemisphere(wo, wh))
            wh = -wh;
        return wh;
    }
    AKR_XPU inline float microfacet_evaluate_pdf(const MicrofacetModel &m, const glm::vec3 &wh) {
        return (microfacet_D(m, wh) * abs_cos_theta(wh));
    }
#pragma endregion

    template <class T>
    struct TFilm {
        Array2D<T> radiance;
        Array2D<Float> weight;
        explicit TFilm(const ivec2 &dimension) : radiance(dimension), weight(dimension) {}
        void add_sample(const ivec2 &p, const T &sample, Float weight_) {
            weight(p) += weight_;
            radiance(p) += sample;
        }
        [[nodiscard]] ivec2 resolution() const { return radiance.dimension(); }
        template <typename = std::enable_if_t<std::is_same_v<T, Color3f>>>
        Image to_rgb_image() const {
            Image image = rgb_image(resolution());
            thread::parallel_for(resolution().y, [&](uint32_t y, uint32_t) {
                for (int x = 0; x < resolution().x; x++) {
                    if (weight(x, y) != 0) {
                        auto color = (radiance(x, y)) / weight(x, y);
                        image(x, y, 0) = color[0];
                        image(x, y, 1) = color[1];
                        image(x, y, 2) = color[2];
                    } else {
                        auto color = radiance(x, y);
                        image(x, y, 0) = color[0];
                        image(x, y, 1) = color[1];
                        image(x, y, 2) = color[2];
                    }
                }
            });
            return image;
        }
    };
    using Film = TFilm<Spectrum>;
    struct CameraSample {
        vec2 p_lens;
        vec2 p_film;
        Float weight = 0.0f;
        Vec3 normal;
        Ray ray;
    };
    struct PerspectiveCamera {
        Transform c2w, w2c, r2c, c2r;
        ivec2 _resolution;
        Float fov;
        Float lens_radius = 0.0f;
        Float focal_distance = 0.0f;
        PerspectiveCamera(const ivec2 &_resolution, const Transform &c2w, Float fov)
            : c2w(c2w), w2c(c2w.inverse()), _resolution(_resolution), fov(fov) {
            preprocess();
        }
        ivec2 resolution() const { return _resolution; }
        CameraSample generate_ray(const vec2 &u1, const vec2 &u2, const ivec2 &raster) const {
            CameraSample sample;
            sample.p_lens = concentric_disk_sampling(u1) * lens_radius;
            sample.p_film = vec2(raster) + u2;
            sample.weight = 1;

            vec2 p = shuffle<0, 1>(r2c.apply_point(Vec3(sample.p_film.x, sample.p_film.y, 0.0f)));
            Ray ray(Vec3(0), Vec3(normalize(Vec3(p.x, p.y, 0) - Vec3(0, 0, 1))));
            if (lens_radius > 0 && focal_distance > 0) {
                Float ft = focal_distance / std::abs(ray.d.z);
                Vec3 pFocus = ray(ft);
                ray.o = Vec3(sample.p_lens.x, sample.p_lens.y, 0);
                ray.d = Vec3(normalize(pFocus - ray.o));
            }
            ray.o = c2w.apply_point(ray.o);
            ray.d = c2w.apply_vector(ray.d);
            sample.normal = c2w.apply_normal(Vec3(0, 0, -1.0f));
            sample.ray = ray;

            return sample;
        }

      private:
        void preprocess() {
            Transform m;
            m = Transform::scale(Vec3(1.0f / _resolution.x, 1.0f / _resolution.y, 1)) * m;
            m = Transform::scale(Vec3(2, 2, 1)) * m;
            m = Transform::translate(Vec3(-1, -1, 0)) * m;
            m = Transform::scale(Vec3(1, -1, 1)) * m;
            auto s = atan(fov / 2);
            if (_resolution.x > _resolution.y) {
                m = Transform::scale(Vec3(s, s * Float(_resolution.y) / _resolution.x, 1)) * m;
            } else {
                m = Transform::scale(Vec3(s * Float(_resolution.x) / _resolution.y, s, 1)) * m;
            }
            r2c = m;
            c2r = r2c.inverse();
        }
    };
    struct Camera : Variant<PerspectiveCamera> {
        using Variant::Variant;
        ivec2 resolution() const { AKR_VAR_DISPATCH(resolution); }
        CameraSample generate_ray(const vec2 &u1, const vec2 &u2, const ivec2 &raster) const {
            AKR_VAR_DISPATCH(generate_ray, u1, u2, raster);
        }
    };

    struct ConstantTexture {
        Spectrum value;
    };

    struct DeviceImageImpl;
    using DeviceImage = DeviceImageImpl *;
    struct ImageTexture {
        DeviceImage image;
    };

    struct Texture : Variant<ConstantTexture, ImageTexture> {
        using Variant::Variant;
    };
    struct ShadingPoint {
        Vec2 texcoords;
        Vec3 p;
        Vec3 dpdu, dpdv;
        Vec3 n;
        Vec3 dndu, dndv;
        Vec3 ng;
    };
    template <class R>
    class TextureEvaluator {
      public:
        AKR_XPU R evaluate(const Texture &tex, const ShadingPoint &sp);
    };
    enum class BSDFType : int {
        Unset = 0u,
        Reflection = 1u << 0,
        Transmission = 1u << 1,
        Diffuse = 1u << 2,
        Glossy = 1u << 3,
        Specular = 1u << 4,
        DiffuseReflection = Diffuse | Reflection,
        DiffuseTransmission = Diffuse | Transmission,
        GlossyReflection = Glossy | Reflection,
        GlossyTransmission = Glossy | Transmission,
        SpecularReflection = Specular | Reflection,
        SpecularTransmission = Specular | Transmission,
        All = Diffuse | Glossy | Specular | Reflection | Transmission
    };
    AKR_XPU inline BSDFType operator&(BSDFType a, BSDFType b) { return BSDFType((int)a & (int)b); }
    AKR_XPU inline BSDFType operator|(BSDFType a, BSDFType b) { return BSDFType((int)a | (int)b); }
    AKR_XPU inline BSDFType operator~(BSDFType a) { return BSDFType(~(uint32_t)a); }
    struct BSDFSample {
        Vec3 wi;
        Spectrum f;
        Float pdf = 0.0;
        BSDFType type = BSDFType::Unset;
    };
    struct Material {
        Texture color;
        Texture metallic;
        Texture roughness;
        Texture specular;
    };
    class BSDF {
      public:
        Spectrum color;
        Float metallic = 0.0;
        Float roughness = 0.0;
        Frame frame;
        BSDF() = default;
        BSDF(const Frame &frame) : frame(frame) {}
        BSDFSample sample_local(Float u0, const Vec2 &u1, const Vec3 &wo, const ShadingPoint &sp) const {
            BSDFSample sample;
            sample.wi = cosine_hemisphere_sampling(u1);
            if (same_hemisphere(wo, sample.wi)) {
                sample.f = color * InvPi;
                sample.pdf = cosine_hemisphere_pdf(abs_cos_theta(sample.wi)) * InvPi;
                sample.type = BSDFType::DiffuseReflection;
            }
            return sample;
        }
        Spectrum evaluate_local(const Vec3 &wo, const Vec3 &wi) const {
            if (same_hemisphere(wi, wo)) {
                return color * InvPi;
            }
            return Spectrum(0.0);
        }
        Float evaluate_pdf_local(const Vec3 &wo, const Vec3 &wi) const {
            if (same_hemisphere(wi, wo)) {
                return cosine_hemisphere_pdf(abs_cos_theta(wi)) * InvPi;
            }
            return 0.0;
        }
        BSDFSample sample(Float u0, const Vec2 &u1, const Vec3 &wo, const ShadingPoint &sp) const {
            auto sample = sample_local(u0, u1, frame.world_to_local(wo), sp);
            sample.wi = frame.local_to_world(sample.wi);
            return sample;
        }
        Spectrum evaluate(const Vec3 &wo, const Vec3 &wi) const {
            return evaluate_local(frame.world_to_local(wi), frame.world_to_local(wo));
        }
        Float evaluate_pdf(const Vec3 &wo, const Vec3 &wi) const {
            return evaluate_pdf_local(frame.world_to_local(wi), frame.world_to_local(wo));
        }
    };
    struct Light;
    struct Triangle {
        std::array<Vec3, 3> vertices;
        std::array<Vec3, 3> normals;
        std::array<vec2, 3> texcoords;
        const Material *material = nullptr;
        const Light *light = nullptr;
        Vec3 p(const vec2 &uv) const { return lerp3(vertices[0], vertices[1], vertices[2], uv); }
        Float area() const { return length(cross(vertices[1] - vertices[0], vertices[2] - vertices[0])) * 0.5f; }
        Vec3 ng() const { return normalize(cross(vertices[1] - vertices[0], vertices[2] - vertices[0])); }
        Vec3 ns(const vec2 &uv) const { return normalize(lerp3(normals[0], normals[1], normals[2], uv)); }
        vec2 texcoord(const vec2 &uv) const { return lerp3(texcoords[0], texcoords[1], texcoords[2], uv); }
        Vec3 dpdu(Float u) const { return dlerp3du(vertices[0], vertices[1], vertices[2], u); }
        Vec3 dpdv(Float v) const { return dlerp3du(vertices[0], vertices[1], vertices[2], v); }

        std::pair<Vec3, Vec3> dnduv(const vec2 &uv) const {
            auto n = ns(uv);
            Float il = 1.0 / length(n);
            n *= il;
            auto dn_du = (normals[1] - normals[0]) * il;
            auto dn_dv = (normals[2] - normals[0]) * il;
            dn_du = -n * dot(n, dn_du) + dn_du;
            dn_dv = -n * dot(n, dn_dv) + dn_dv;
            return std::make_pair(dn_du, dn_dv);
        }

        std::optional<std::pair<Float, Vec2>> intersect(const Ray &ray) const {
            auto &v0 = vertices[0];
            auto &v1 = vertices[1];
            auto &v2 = vertices[2];
            Vec3 e1 = (v1 - v0);
            Vec3 e2 = (v2 - v0);
            Float a, f, u, v;
            auto h = cross(ray.d, e2);
            a = dot(e1, h);
            if (a > Float(-1e-6f) && a < Float(1e-6f))
                return std::nullopt;
            f = 1.0f / a;
            auto s = ray.o - v0;
            u = f * dot(s, h);
            if (u < 0.0 || u > 1.0)
                return std::nullopt;
            auto q = cross(s, e1);
            v = f * dot(ray.d, q);
            if (v < 0.0 || u + v > 1.0)
                return std::nullopt;
            Float t = f * dot(e2, q);
            if (t > ray.tmin && t < ray.tmax) {
                return std::make_pair(t, Vec2(u, v));
            } else {
                return std::nullopt;
            }
        }
    };
    struct SurfaceInteraction {
        Triangle triangle;
        Vec3 p;
        BSDF bsdf;
        Vec3 ng, ns;
        vec2 texcoords;
        Vec3 dndu, dndv;
        Vec3 dpdu, dpdv;

        SurfaceInteraction(const vec2 &uv, const Triangle &triangle)
            : triangle(triangle), p(triangle.p(uv)), ng(triangle.ng()), ns(triangle.ns(uv)),
              texcoords(triangle.texcoord(uv)) {
            dpdu = triangle.dpdu(uv[0]);
            dpdv = triangle.dpdu(uv[1]);
            std::tie(dndu, dndv) = triangle.dnduv(uv);
        }
        ShadingPoint sp() const {
            ShadingPoint sp_;
            sp_.n = ns;
            sp_.texcoords = texcoords;
            sp_.dndu = dndu;
            sp_.dndv = dndv;
            sp_.dpdu = dpdu;
            sp_.dpdv = dpdv;
            return sp_;
        }
    };
    template <class FloatEvaluator, class SpectrumEvaluator>
    class MaterialEvaluator {
        FloatEvaluator eval_f;
        SpectrumEvaluator eval_s;

      public:
        BSDF evaluate(const Material &mat, SurfaceInteraction &si) const {
            auto sp = si.sp();
            BSDF bsdf(Frame(si.p, si.dpdu));
            bsdf.color = eval_s(mat, sp);
            return bsdf;
        }
    };
    struct MeshInstance {
        Transform transform;
        BufferView<vec3> vertices;
        BufferView<ivec3> indices;
        BufferView<vec3> normals;
        BufferView<vec2> texcoords;
        const Material *material = nullptr;

        Triangle get_triangle(int prim_id) {
            Triangle trig;
            for (int i = 0; i < 3; i++) {
                trig.vertices[i] = transform.apply_vector(vertices[indices[prim_id][i]]);
                trig.normals[i] = transform.apply_normal(normals[indices[prim_id][i]]);
                trig.texcoords[i] = texcoords[indices[prim_id][i]];
            }
            trig.material = material;
            return trig;
        }
    };

    struct Scene {
        Camera camera;
        std::vector<MeshInstance> instances;
    };

    struct Intersection {
        Float t = Inf;
        Vec2 uv;
        int geom_id = -1;
        int prim_id = -1;
        bool hit() const { return geom_id != -1; }
    };
    class EmbreeAccel {
      public:
        virtual void build(const std::shared_ptr<scene::SceneGraph> &scene) = 0;
        virtual std::optional<Intersection> intersect1(const Ray &ray) const = 0;
    };
    std::shared_ptr<EmbreeAccel> create_embree_accel();

    Film render_pt(const Scene &scene);
} // namespace akari::render