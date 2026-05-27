#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using Array3d = py::array_t<double>;
using Array1d = py::array_t<double>;
using Array1i = py::array_t<int32_t>;

// ============================================================
// PchipSpline — uniform-grid monotone cubic (Fritsch-Carlson)
// ============================================================
struct PchipSpline {
    double x_lo{0}, h{1}, h_inv{1};
    int n{0};
    std::vector<double> y, d;

    PchipSpline() = default;

    PchipSpline(double x_lo_, double x_hi_, std::vector<double> y_in)
        : x_lo(x_lo_), n(static_cast<int>(y_in.size())), y(std::move(y_in))
    {
        h     = (x_hi_ - x_lo_) / (n - 1);
        h_inv = 1.0 / h;
        d.resize(n, 0.0);
        _build();
    }

    double eval(double x) const {
        double t = (x - x_lo) * h_inv;
        int k = std::max(0, std::min(n - 2, static_cast<int>(std::floor(t))));
        double s = t - k;
        double s2 = s * s, s3 = s2 * s;
        return (2*s3 - 3*s2 + 1) * y[k]
             + (s3 - 2*s2 + s)   * h * d[k]
             + (-2*s3 + 3*s2)    * y[k+1]
             + (s3 - s2)         * h * d[k+1];
    }

private:
    // Fritsch-Carlson one-sided endpoint slope with monotonicity guards.
    // m0 is the adjacent secant, m1 is the next secant inward.
    static double _edge(double m0, double m1) {
        double dv = (3.0 * m0 - m1) / 2.0;
        if (!std::isfinite(dv) || dv * m0 <= 0.0) return 0.0;
        if (m0 * m1 < 0.0 && std::abs(dv) > 3.0 * std::abs(m0)) return 3.0 * m0;
        return dv;
    }

    void _build() {
        if (n < 2) return;
        std::vector<double> delta(n - 1);
        for (int k = 0; k < n - 1; k++)
            delta[k] = (y[k+1] - y[k]) * h_inv;

        if (n == 2) { d[0] = d[1] = delta[0]; return; }

        // Interior slopes: harmonic mean of adjacent secants (uniform-h simplification)
        for (int k = 1; k < n - 1; k++) {
            if (delta[k-1] * delta[k] <= 0.0)
                d[k] = 0.0;
            else
                d[k] = 2.0 / (1.0/delta[k-1] + 1.0/delta[k]);
        }
        // Endpoint slopes
        d[0]   = _edge(delta[0],   delta[1]);
        d[n-1] = _edge(delta[n-2], delta[n-3]);
    }
};

// ============================================================
// FreeRotorGridC — free-rotor propagator on a precomputed grid
//
// log G(x) = log Σ_{l=0}^{L_max} (2l+1)/(4π) P_l(x) exp(-λ·l(l+1))
//
// Legendre polynomials via Bonnet recursion; log G stored as a
// PCHIP spline for fast interpolation.
// ============================================================
class FreeRotorGridC {
public:
    FreeRotorGridC(double lambda_rot, int L_max = 100, int grid_size = 1000) {
        const double x_lo = -1.0, x_hi = 1.0;
        double hx = (x_hi - x_lo) / (grid_size - 1);

        std::vector<double> x(grid_size);
        for (int j = 0; j < grid_size; j++)
            x[j] = x_lo + j * hx;

        // Weights: w[l] = (2l+1)/(4π) * exp(-λ·l(l+1))
        std::vector<double> w(L_max + 1);
        for (int l = 0; l <= L_max; l++)
            w[l] = (2*l + 1) / (4.0 * M_PI) * std::exp(-lambda_rot * l * (l + 1));

        // Accumulate G via Bonnet recursion using two alternating buffers.
        // buf0 = P_{l-1}, buf1 = P_l; updated in-place per grid point.
        std::vector<double> G(grid_size, 0.0);
        std::vector<double> buf0(grid_size, 1.0); // P_0 = 1
        for (int j = 0; j < grid_size; j++) G[j] += w[0] * buf0[j];

        if (L_max >= 1) {
            std::vector<double> buf1(x);           // P_1 = x
            for (int j = 0; j < grid_size; j++) G[j] += w[1] * buf1[j];

            for (int l = 1; l < L_max; l++) {
                double a = (2.0*l + 1) / (l + 1);
                double b = static_cast<double>(l) / (l + 1);
                for (int j = 0; j < grid_size; j++) {
                    double Pnext = a * x[j] * buf1[j] - b * buf0[j];
                    G[j]    += w[l+1] * Pnext;
                    buf0[j]  = buf1[j];
                    buf1[j]  = Pnext;
                }
            }
        }

        std::vector<double> log_G(grid_size);
        for (int j = 0; j < grid_size; j++)
            log_G[j] = std::log(std::max(G[j], 1e-300));

        spline_logG_ = PchipSpline(x_lo, x_hi, std::move(log_G));
    }

    double log_eval(double x) const {
        x = std::max(-1.0, std::min(1.0, x));
        return spline_logG_.eval(x);
    }

private:
    PchipSpline spline_logG_;
};

// ============================================================
// MoveResult
// ============================================================
struct MoveResult {
    int particle{0};
    int m_lo{0}, m_hi{0};
    double log_ratio_contrib{0.0};
    bool has_rot{false};  // true if this move updates orientations

    MoveResult() = default;
    MoveResult(int particle, int m_lo, int m_hi, double lrc, bool has_rot = false)
        : particle(particle), m_lo(m_lo), m_hi(m_hi),
          log_ratio_contrib(lrc), has_rot(has_rot) {}
};

// ============================================================
// MIC function dispatch
// ============================================================
using MicFn = void(*)(double*, const double*, const double*, const double*);

static void do_mic_free(double* out, const double* a, const double* b, const double* /*Lhalf*/) {
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];
}

static inline double _mic1(double d, double Lhalf) {
    if (d > Lhalf) d -= 2.0 * Lhalf;
    else if (d < -Lhalf) d += 2.0 * Lhalf;
    return d;
}

static void do_mic_3d(double* out, const double* a, const double* b, const double* Lhalf) {
    out[0] = _mic1(a[0] - b[0], Lhalf[0]);
    out[1] = _mic1(a[1] - b[1], Lhalf[1]);
    out[2] = _mic1(a[2] - b[2], Lhalf[2]);
}

static void do_mic_quasi2d(double* out, const double* a, const double* b, const double* Lhalf) {
    out[0] = _mic1(a[0] - b[0], Lhalf[0]);
    out[1] = _mic1(a[1] - b[1], Lhalf[1]);
    out[2] = a[2] - b[2];
}

static void do_mic_quasi1d(double* out, const double* a, const double* b, const double* Lhalf) {
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = _mic1(a[2] - b[2], Lhalf[2]);
}

// ============================================================
// PathState
// ============================================================
struct PathState {
    Array3d positions;
    Array3d orientations;
    Array3d buffer_positions;
    Array3d buffer_orientations;
    int N{0}, M{0};
    double tau_prime{0.0};
    Array1d lambda_trans;
    Array1d lambda_rot;
    Array1i slice_kind;
    py::object boundary;
    // Per-particle rotational propagator grids; null when no rotational DOF.
    const std::vector<std::shared_ptr<FreeRotorGridC>>* grids{nullptr};
};

// ============================================================
// Move abstract base + Python trampoline
// ============================================================
class Move {
public:
    virtual MoveResult propose(PathState& state, py::object rng) = 0;
    virtual ~Move() = default;
};

class PyMove : public Move {
public:
    using Move::Move;
    MoveResult propose(PathState& state, py::object rng) override {
        PYBIND11_OVERRIDE_PURE(MoveResult, Move, propose, state, rng);
    }
};

// ============================================================
// Helpers: views into 3-element rows of numpy arrays
// ============================================================
static Array3d row_view(Array3d& arr, int m, int i) {
    // Returns (3,) view of arr[m, i, :]
    py::buffer_info info = arr.request();
    char* base = static_cast<char*>(info.ptr);
    char* ptr = base + m * info.strides[0] + i * info.strides[1];
    return Array3d({3}, {sizeof(double)},
                   reinterpret_cast<double*>(ptr), arr);
}

static double dot3(const double* a, const double* b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

static void normalize3(double* u) {
    double n = std::sqrt(u[0]*u[0] + u[1]*u[1] + u[2]*u[2]);
    if (n > 1e-15) { u[0] /= n; u[1] /= n; u[2] /= n; }
}

// Build two unit vectors e1, e2 perpendicular to u (u must be unit vector).
static void build_perp(const double u[3], double e1[3], double e2[3]) {
    double t[3] = {1.0, 0.0, 0.0};
    if (std::abs(u[0]) > 0.9) { t[0] = 0.0; t[1] = 1.0; t[2] = 0.0; }
    double dot = dot3(t, u);
    for (int d = 0; d < 3; d++) e1[d] = t[d] - dot * u[d];
    normalize3(e1);
    // e2 = u × e1
    e2[0] = u[1]*e1[2] - u[2]*e1[1];
    e2[1] = u[2]*e1[0] - u[0]*e1[2];
    e2[2] = u[0]*e1[1] - u[1]*e1[0];
}

// Sample a unit vector u_new uniformly within a cone of half-angle step_size
// centred on u (which must be a unit vector).
static void propose_in_cone(double u_new[3], const double u[3],
                             double step_size, py::object rng) {
    double e1[3], e2[3];
    build_perp(u, e1, e2);

    double cos_max = std::cos(step_size);
    double cos_theta = rng.attr("uniform")(py::float_(cos_max), py::float_(1.0)).cast<double>();
    double phi = rng.attr("uniform")(py::float_(0.0), py::float_(2.0 * M_PI)).cast<double>();
    double sin_theta = std::sqrt(std::max(0.0, 1.0 - cos_theta * cos_theta));

    for (int d = 0; d < 3; d++)
        u_new[d] = cos_theta * u[d]
                 + sin_theta * std::cos(phi) * e1[d]
                 + sin_theta * std::sin(phi) * e2[d];
}

// Apply Rodrigues rotation: rotate vector v by angle theta around unit axis ax.
static void rodrigues(double out[3], const double v[3],
                      const double ax[3], double theta) {
    double ct = std::cos(theta), st = std::sin(theta);
    double dot = dot3(ax, v);
    // cross = ax × v
    double cx = ax[1]*v[2] - ax[2]*v[1];
    double cy = ax[2]*v[0] - ax[0]*v[2];
    double cz = ax[0]*v[1] - ax[1]*v[0];
    out[0] = ct * v[0] + st * cx + (1.0 - ct) * dot * ax[0];
    out[1] = ct * v[1] + st * cy + (1.0 - ct) * dot * ax[1];
    out[2] = ct * v[2] + st * cz + (1.0 - ct) * dot * ax[2];
}


// ============================================================
// C++ move implementations — translational
// ============================================================
class TranslationEndMove : public Move {
public:
    explicit TranslationEndMove(double step_size) : step_size_(step_size) {}

    MoveResult propose(PathState& state, py::object rng) override {
        int i = rng.attr("integers")(0, state.N).cast<int>();
        int which = rng.attr("integers")(0, 2).cast<int>();
        int m = (which == 0) ? 0 : state.M - 1;

        auto pos = state.positions.unchecked<3>();
        auto buf = state.buffer_positions.mutable_unchecked<3>();
        auto disp = rng.attr("uniform")(
            py::float_(-step_size_), py::float_(step_size_), py::int_(3)
        ).cast<Array3d>().unchecked<1>();

        buf(m, 0, 0) = pos(m, i, 0) + disp(0);
        buf(m, 0, 1) = pos(m, i, 1) + disp(1);
        buf(m, 0, 2) = pos(m, i, 2) + disp(2);

        return MoveResult(i, m, m, 0.0);
    }

private:
    double step_size_;
};

class PyTranslationEndMove : public TranslationEndMove {
public:
    using TranslationEndMove::TranslationEndMove;
    MoveResult propose(PathState& state, py::object rng) override {
        PYBIND11_OVERRIDE(MoveResult, TranslationEndMove, propose, state, rng);
    }
};

class TranslationInteriorMove : public Move {
public:
    explicit TranslationInteriorMove(double step_size) : step_size_(step_size) {}

    MoveResult propose(PathState& state, py::object rng) override {
        int i = rng.attr("integers")(0, state.N).cast<int>();
        int m = rng.attr("integers")(1, state.M - 1).cast<int>();

        auto pos = state.positions.unchecked<3>();
        auto buf = state.buffer_positions.mutable_unchecked<3>();
        auto disp = rng.attr("uniform")(
            py::float_(-step_size_), py::float_(step_size_), py::int_(3)
        ).cast<Array3d>().unchecked<1>();

        buf(m, 0, 0) = pos(m, i, 0) + disp(0);
        buf(m, 0, 1) = pos(m, i, 1) + disp(1);
        buf(m, 0, 2) = pos(m, i, 2) + disp(2);

        return MoveResult(i, m, m, 0.0);
    }

private:
    double step_size_;
};

class PyTranslationInteriorMove : public TranslationInteriorMove {
public:
    using TranslationInteriorMove::TranslationInteriorMove;
    MoveResult propose(PathState& state, py::object rng) override {
        PYBIND11_OVERRIDE(MoveResult, TranslationInteriorMove, propose, state, rng);
    }
};

class TranslationRigidMove : public Move {
public:
    explicit TranslationRigidMove(double step_size) : step_size_(step_size) {}

    MoveResult propose(PathState& state, py::object rng) override {
        int i = rng.attr("integers")(0, state.N).cast<int>();
        auto disp = rng.attr("uniform")(
            py::float_(-step_size_), py::float_(step_size_), py::int_(3)
        ).cast<Array3d>().unchecked<1>();

        auto pos = state.positions.unchecked<3>();
        auto buf = state.buffer_positions.mutable_unchecked<3>();
        for (int m = 0; m < state.M; m++) {
            buf(m, 0, 0) = pos(m, i, 0) + disp(0);
            buf(m, 0, 1) = pos(m, i, 1) + disp(1);
            buf(m, 0, 2) = pos(m, i, 2) + disp(2);
        }
        return MoveResult(i, 0, state.M - 1, 0.0);
    }

private:
    double step_size_;
};

class PyTranslationRigidMove : public TranslationRigidMove {
public:
    using TranslationRigidMove::TranslationRigidMove;
    MoveResult propose(PathState& state, py::object rng) override {
        PYBIND11_OVERRIDE(MoveResult, TranslationRigidMove, propose, state, rng);
    }
};

class TranslationBisectionMove : public Move {
public:
    explicit TranslationBisectionMove(int level) : level_(level) {
        if (level < 1)
            throw std::invalid_argument("TranslationBisectionMove: level must be >= 1");
    }

    MoveResult propose(PathState& state, py::object rng) override {
        int n = 1 << level_;  // 2^level
        if (n > state.M - 1)
            throw std::runtime_error(
                "TranslationBisectionMove: 2^level (" + std::to_string(n) +
                ") > M-1 (" + std::to_string(state.M - 1) + "), level too large for this system");

        int i = rng.attr("integers")(0, state.N).cast<int>();
        int max_start = state.M - 1 - n;
        int m_start = rng.attr("integers")(0, max_start + 1).cast<int>();

        auto pos = state.positions.unchecked<3>();
        auto buf = state.buffer_positions.mutable_unchecked<3>();
        auto lam = state.lambda_trans.unchecked<1>();
        double lam_i = lam(i);

        // Work array: old positions for the full range [m_start .. m_start+n]
        std::vector<std::array<double, 3>> work(n + 1);
        std::vector<std::array<double, 3>> work_old(n + 1);
        for (int k = 0; k <= n; k++) {
            int m = m_start + k;
            work[k][0] = pos(m, i, 0);
            work[k][1] = pos(m, i, 1);
            work[k][2] = pos(m, i, 2);
            work_old[k] = work[k];
        }

        // Lévy bridge bisection: coarsest to finest
        int step = n;
        for (int l = level_; l >= 1; l--) {
            int half_step = step / 2;
            double t_eff = static_cast<double>(half_step) * half_step / step;
            double sigma = std::sqrt(lam_i * t_eff);

            for (int k = 0; k + step <= n; k += step) {
                int mid_k = k + half_step;
                auto z = rng.attr("standard_normal")(3).cast<Array3d>().unchecked<1>();
                for (int d = 0; d < 3; d++)
                    work[mid_k][d] = 0.5 * (work[k][d] + work[k + step][d]) + sigma * z(d);
            }
            step = half_step;
        }

        // Write interior slices to buffer
        for (int k = 1; k < n; k++) {
            int m = m_start + k;
            buf(m, 0, 0) = work[k][0];
            buf(m, 0, 1) = work[k][1];
            buf(m, 0, 2) = work[k][2];
        }

        // Compute log_ratio_contrib to cancel engine kinetic terms for free particle
        double log_ratio = 0.0;
        for (int k = 0; k < n; k++) {
            double d_old = 0.0, d_new = 0.0;
            for (int d = 0; d < 3; d++) {
                double diff_old = work_old[k][d] - work_old[k + 1][d];
                d_old += diff_old * diff_old;
                double diff_new = work[k][d] - work[k + 1][d];
                d_new += diff_new * diff_new;
            }
            log_ratio += (d_new - d_old) / (4.0 * lam_i);
        }

        return MoveResult(i, m_start + 1, m_start + n - 1, log_ratio);
    }

private:
    int level_;
};

class PyTranslationBisectionMove : public TranslationBisectionMove {
public:
    using TranslationBisectionMove::TranslationBisectionMove;
    MoveResult propose(PathState& state, py::object rng) override {
        PYBIND11_OVERRIDE(MoveResult, TranslationBisectionMove, propose, state, rng);
    }
};


// ============================================================
// C++ move implementations — rotational
// ============================================================

class RotationEndMove : public Move {
public:
    explicit RotationEndMove(double step_size) : step_size_(step_size) {}

    MoveResult propose(PathState& state, py::object rng) override {
        int i = rng.attr("integers")(0, state.N).cast<int>();
        int which = rng.attr("integers")(0, 2).cast<int>();
        int m = (which == 0) ? 0 : state.M - 1;

        auto ori = state.orientations.unchecked<3>();
        auto buf = state.buffer_orientations.mutable_unchecked<3>();

        double u[3] = {ori(m, i, 0), ori(m, i, 1), ori(m, i, 2)};
        double u_new[3];
        propose_in_cone(u_new, u, step_size_, rng);

        buf(m, 0, 0) = u_new[0];
        buf(m, 0, 1) = u_new[1];
        buf(m, 0, 2) = u_new[2];

        return MoveResult(i, m, m, 0.0, /*has_rot=*/true);
    }

private:
    double step_size_;
};

class PyRotationEndMove : public RotationEndMove {
public:
    using RotationEndMove::RotationEndMove;
    MoveResult propose(PathState& state, py::object rng) override {
        PYBIND11_OVERRIDE(MoveResult, RotationEndMove, propose, state, rng);
    }
};

class RotationInteriorMove : public Move {
public:
    explicit RotationInteriorMove(double step_size) : step_size_(step_size) {}

    MoveResult propose(PathState& state, py::object rng) override {
        int i = rng.attr("integers")(0, state.N).cast<int>();
        int m = rng.attr("integers")(1, state.M - 1).cast<int>();

        auto ori = state.orientations.unchecked<3>();
        auto buf = state.buffer_orientations.mutable_unchecked<3>();

        double u[3] = {ori(m, i, 0), ori(m, i, 1), ori(m, i, 2)};
        double u_new[3];
        propose_in_cone(u_new, u, step_size_, rng);

        buf(m, 0, 0) = u_new[0];
        buf(m, 0, 1) = u_new[1];
        buf(m, 0, 2) = u_new[2];

        return MoveResult(i, m, m, 0.0, /*has_rot=*/true);
    }

private:
    double step_size_;
};

class PyRotationInteriorMove : public RotationInteriorMove {
public:
    using RotationInteriorMove::RotationInteriorMove;
    MoveResult propose(PathState& state, py::object rng) override {
        PYBIND11_OVERRIDE(MoveResult, RotationInteriorMove, propose, state, rng);
    }
};

class RotationRigidMove : public Move {
public:
    explicit RotationRigidMove(double step_size) : step_size_(step_size) {}

    MoveResult propose(PathState& state, py::object rng) override {
        int i = rng.attr("integers")(0, state.N).cast<int>();

        // Random rotation axis (uniform on sphere) and angle
        auto ax_arr = rng.attr("standard_normal")(3).cast<Array3d>().unchecked<1>();
        double ax[3] = {ax_arr(0), ax_arr(1), ax_arr(2)};
        normalize3(ax);
        double theta = rng.attr("uniform")(
            py::float_(-step_size_), py::float_(step_size_)).cast<double>();

        auto ori = state.orientations.unchecked<3>();
        auto buf = state.buffer_orientations.mutable_unchecked<3>();

        for (int m = 0; m < state.M; m++) {
            double v[3] = {ori(m, i, 0), ori(m, i, 1), ori(m, i, 2)};
            double v_new[3];
            rodrigues(v_new, v, ax, theta);
            normalize3(v_new);
            buf(m, 0, 0) = v_new[0];
            buf(m, 0, 1) = v_new[1];
            buf(m, 0, 2) = v_new[2];
        }

        return MoveResult(i, 0, state.M - 1, 0.0, /*has_rot=*/true);
    }

private:
    double step_size_;
};

class PyRotationRigidMove : public RotationRigidMove {
public:
    using RotationRigidMove::RotationRigidMove;
    MoveResult propose(PathState& state, py::object rng) override {
        PYBIND11_OVERRIDE(MoveResult, RotationRigidMove, propose, state, rng);
    }
};

// RotationBisectionMove: hierarchical bisection on the sphere.
// Proposes new interior orientations using a cone centred on the great-circle
// midpoint of each pair of fixed endpoints. The cone half-angle is scaled to
// match the free-rotor angular variance at each level.
//
// The proposal is symmetric (forward and reverse draw from the same cone for
// fixed endpoints), so log_ratio_contrib = 0 and the engine's kinetic terms
// enter the acceptance ratio without cancellation. Acceptance < 1 for a free
// rotor is correct behaviour; the kinetic terms prefer proposals with higher
// rotational weight.
class RotationBisectionMove : public Move {
public:
    explicit RotationBisectionMove(int level) : level_(level) {
        if (level < 1)
            throw std::invalid_argument("RotationBisectionMove: level must be >= 1");
    }

    MoveResult propose(PathState& state, py::object rng) override {
        int n = 1 << level_;
        if (n > state.M - 1)
            throw std::runtime_error(
                "RotationBisectionMove: 2^level (" + std::to_string(n) +
                ") > M-1 (" + std::to_string(state.M - 1) + "), level too large");

        int i = rng.attr("integers")(0, state.N).cast<int>();
        int max_start = state.M - 1 - n;
        int m_start = rng.attr("integers")(0, max_start + 1).cast<int>();

        auto ori = state.orientations.unchecked<3>();
        auto buf = state.buffer_orientations.mutable_unchecked<3>();
        auto lam_rot = state.lambda_rot.unchecked<1>();
        double lam_rot_i = lam_rot(i);

        // Load current orientations for the full range
        std::vector<std::array<double, 3>> work(n + 1);
        for (int k = 0; k <= n; k++) {
            int m = m_start + k;
            for (int d = 0; d < 3; d++) work[k][d] = ori(m, i, d);
        }

        // Hierarchical bisection: coarsest to finest.
        // At each level, propose the midpoint of each interval from a cone
        // centred on the great-circle midpoint; cone size scales as √(λ·half_step).
        int step = n;
        for (int l = level_; l >= 1; l--) {
            int half_step = step / 2;
            double sigma_angle = std::sqrt(lam_rot_i * static_cast<double>(half_step));
            double cone = std::min(sigma_angle * 2.0, M_PI);

            for (int k = 0; k + step <= n; k += step) {
                int mid_k = k + half_step;
                // Great-circle midpoint between work[k] and work[k+step]
                double mean[3];
                for (int d = 0; d < 3; d++)
                    mean[d] = work[k][d] + work[k + step][d];
                double norm = std::sqrt(dot3(mean, mean));
                if (norm < 1e-10) {
                    // Antipodal: pick any perpendicular direction
                    build_perp(work[k].data(), mean, mean + 3);
                    norm = std::sqrt(dot3(mean, mean));
                }
                for (int d = 0; d < 3; d++) mean[d] /= norm;

                double u_new[3];
                propose_in_cone(u_new, mean, cone, rng);
                work[mid_k][0] = u_new[0];
                work[mid_k][1] = u_new[1];
                work[mid_k][2] = u_new[2];
            }
            step = half_step;
        }

        // Write interior slices to buffer
        for (int k = 1; k < n; k++) {
            int m = m_start + k;
            buf(m, 0, 0) = work[k][0];
            buf(m, 0, 1) = work[k][1];
            buf(m, 0, 2) = work[k][2];
        }

        // Symmetric proposal → log_ratio_contrib = 0; kinetic terms handled by engine.
        return MoveResult(i, m_start + 1, m_start + n - 1, 0.0, /*has_rot=*/true);
    }

private:
    int level_;
};

class PyRotationBisectionMove : public RotationBisectionMove {
public:
    using RotationBisectionMove::RotationBisectionMove;
    MoveResult propose(PathState& state, py::object rng) override {
        PYBIND11_OVERRIDE(MoveResult, RotationBisectionMove, propose, state, rng);
    }
};


// ============================================================
// Engine: owns the sweep loop
// ============================================================
struct MoveStats {
    int64_t attempts{0};
    int64_t acceptances{0};
};

class Engine {
public:
    Engine(Array3d positions, Array3d orientations,
           Array3d buf_pos, Array3d buf_ori,
           Array1d lambda_trans, Array1d lambda_rot,
           Array1i slice_kind,
           int N, int M, double tau_prime,
           py::object rng,
           int boundary_kind, Array1d box_half,
           py::object boundary_obj,
           std::vector<std::shared_ptr<FreeRotorGridC>> grids)
        : pos_(positions), ori_(orientations),
          buf_pos_(buf_pos), buf_ori_(buf_ori),
          lam_(lambda_trans), lam_rot_(lambda_rot),
          sk_(slice_kind),
          N_(N), M_(M), tau_(tau_prime), rng_(rng),
          V_ext_(py::none()), V_int_(py::none()),
          f_(py::none()), h_(py::none()),
          total_w_(0.0),
          boundary_obj_(boundary_obj),
          grids_(std::move(grids))
    {
        switch (boundary_kind) {
            case 1: mic_fn_ = do_mic_3d;     break;
            case 2: mic_fn_ = do_mic_quasi2d; break;
            case 3: mic_fn_ = do_mic_quasi1d; break;
            default: mic_fn_ = do_mic_free;  break;
        }
        auto bh = box_half.unchecked<1>();
        box_half_[0] = bh(0);
        box_half_[1] = bh(1);
        box_half_[2] = bh(2);
    }

    void add_move(std::shared_ptr<Move> move, double weight) {
        moves_.push_back(std::move(move));
        weights_.push_back(weight);
        total_w_ += weight;
        move_stats_.push_back(MoveStats{});
    }

    void set_potential(py::object V_ext, py::object V_int) {
        V_ext_ = V_ext;
        V_int_ = V_int;
    }

    void set_trial_wavefunction(py::object f, py::object h) {
        f_ = f;
        h_ = h;
    }

    // Returns (move_stats, bead_stats) where each is list of (attempts, acceptances)
    py::tuple run_sweep(int sweeps_per_block) {
        size_t n = moves_.size();
        if (n == 0) throw std::runtime_error("No moves registered");

        std::vector<MoveStats> delta_move(n);
        std::vector<MoveStats> delta_bead(M_);

        // Precompute cumulative probabilities
        std::vector<double> cum(n);
        double running = 0.0;
        for (size_t k = 0; k < n; k++) {
            running += weights_[k] / total_w_;
            cum[k] = running;
        }

        // Build PathState — wraps the same numpy arrays
        PathState ps;
        ps.positions = pos_;
        ps.orientations = ori_;
        ps.buffer_positions = buf_pos_;
        ps.buffer_orientations = buf_ori_;
        ps.N = N_;
        ps.M = M_;
        ps.tau_prime = tau_;
        ps.lambda_trans = lam_;
        ps.lambda_rot = lam_rot_;
        ps.slice_kind = sk_;
        ps.boundary = boundary_obj_;
        ps.grids = grids_.empty() ? nullptr : &grids_;

        int n_attempts = sweeps_per_block * N_ * M_;

        for (int t = 0; t < n_attempts; t++) {
            // Weighted move selection
            double u = rng_.attr("random")().cast<double>();
            int idx = static_cast<int>(n) - 1;
            for (size_t k = 0; k + 1 < n; k++) {
                if (u < cum[k]) { idx = static_cast<int>(k); break; }
            }

            MoveResult result = moves_[idx]->propose(ps, rng_);
            double log_a = compute_log_acceptance(result);

            bool accepted;
            if (log_a >= 0.0) {
                accepted = true;
            } else {
                double r = rng_.attr("random")().cast<double>();
                accepted = std::log(r) < log_a;
            }

            delta_move[idx].attempts++;
            for (int mb = result.m_lo; mb <= result.m_hi; mb++)
                delta_bead[mb].attempts++;

            if (accepted) {
                delta_move[idx].acceptances++;
                for (int mb = result.m_lo; mb <= result.m_hi; mb++)
                    delta_bead[mb].acceptances++;
                apply_move(result);
            }
        }

        py::list ms, bs;
        for (auto& s : delta_move)
            ms.append(py::make_tuple(s.attempts, s.acceptances));
        for (int m = 0; m < M_; m++)
            bs.append(py::make_tuple(delta_bead[m].attempts, delta_bead[m].acceptances));

        return py::make_tuple(ms, bs);
    }

private:
    Array3d pos_, ori_, buf_pos_, buf_ori_;
    Array1d lam_, lam_rot_;
    Array1i sk_;
    int N_, M_;
    double tau_;
    py::object rng_;
    py::object V_ext_, V_int_, f_, h_;
    std::vector<std::shared_ptr<Move>> moves_;
    std::vector<double> weights_;
    double total_w_;
    std::vector<MoveStats> move_stats_;
    MicFn mic_fn_{do_mic_free};
    std::array<double, 3> box_half_{0.0, 0.0, 0.0};
    py::object boundary_obj_;
    std::vector<std::shared_ptr<FreeRotorGridC>> grids_;

    Array3d mic_displacement(const Array3d& a, const Array3d& b) const {
        auto ra = a.unchecked<1>();
        auto rb = b.unchecked<1>();
        double ai[3] = {ra(0), ra(1), ra(2)};
        double bi[3] = {rb(0), rb(1), rb(2)};
        double res[3];
        mic_fn_(res, ai, bi, box_half_.data());
        Array3d out({3});
        auto ro = out.mutable_unchecked<1>();
        ro(0) = res[0];
        ro(1) = res[1];
        ro(2) = res[2];
        return out;
    }

    void apply_move(const MoveResult& result) {
        int i = result.particle;
        // Apply positional update
        if (!result.has_rot) {
            auto pos = pos_.mutable_unchecked<3>();
            auto buf = buf_pos_.unchecked<3>();
            for (int m = result.m_lo; m <= result.m_hi; m++) {
                pos(m, i, 0) = buf(m, 0, 0);
                pos(m, i, 1) = buf(m, 0, 1);
                pos(m, i, 2) = buf(m, 0, 2);
            }
        }
        // Apply orientational update
        if (result.has_rot) {
            auto ori = ori_.mutable_unchecked<3>();
            auto buf = buf_ori_.unchecked<3>();
            for (int m = result.m_lo; m <= result.m_hi; m++) {
                ori(m, i, 0) = buf(m, 0, 0);
                ori(m, i, 1) = buf(m, 0, 1);
                ori(m, i, 2) = buf(m, 0, 2);
            }
        }
    }

    double compute_log_acceptance(const MoveResult& result) {
        double log_a = result.log_ratio_contrib;

        int i    = result.particle;
        int m_lo = result.m_lo;
        int m_hi = result.m_hi;

        int link_lo = std::max(0,      m_lo - 1);
        int link_hi = std::min(M_ - 2, m_hi);

        // --- Translational kinetic terms ---
        if (!result.has_rot) {
            auto pos = pos_.unchecked<3>();
            auto buf = buf_pos_.unchecked<3>();
            auto lam = lam_.unchecked<1>();
            double lam_i = lam(i);

            for (int ml = link_lo; ml <= link_hi; ml++) {
                int mr = ml + 1;
                double d_old = 0.0, d_new = 0.0;
                for (int d = 0; d < 3; d++) {
                    double diff_old = pos(ml, i, d) - pos(mr, i, d);
                    d_old += diff_old * diff_old;
                    double rn_l = (ml >= m_lo) ? buf(ml, 0, d) : pos(ml, i, d);
                    double rn_r = (mr <= m_hi) ? buf(mr, 0, d) : pos(mr, i, d);
                    double diff_new = rn_l - rn_r;
                    d_new += diff_new * diff_new;
                }
                log_a += (d_old - d_new) / (4.0 * lam_i);
            }
        }

        // --- Rotational kinetic terms ---
        if (result.has_rot && !grids_.empty()) {
            auto ori = ori_.unchecked<3>();
            auto buf = buf_ori_.unchecked<3>();
            FreeRotorGridC* grid = grids_[i].get();

            for (int ml = link_lo; ml <= link_hi; ml++) {
                int mr = ml + 1;
                // Old dot product
                double old_dot = 0.0;
                for (int d = 0; d < 3; d++)
                    old_dot += ori(ml, i, d) * ori(mr, i, d);
                old_dot = std::max(-1.0, std::min(1.0, old_dot));

                // New dot product: use buffer for slices in [m_lo, m_hi]
                double uo_l[3], uo_r[3];
                for (int d = 0; d < 3; d++) {
                    uo_l[d] = (ml >= m_lo) ? buf(ml, 0, d) : ori(ml, i, d);
                    uo_r[d] = (mr <= m_hi) ? buf(mr, 0, d) : ori(mr, i, d);
                }
                double new_dot = std::max(-1.0, std::min(1.0, dot3(uo_l, uo_r)));

                log_a += grid->log_eval(new_dot) - grid->log_eval(old_dot);
            }
        }

        // --- Potential and trial-wavefunction terms (per bead) ---
        for (int m = m_lo; m <= m_hi; m++) {
            if (!V_ext_.is_none() || !V_int_.is_none()) {
                double factor = (m == 0 || m == M_ - 1) ? 0.5 : 1.0;
                auto r_old = row_view(pos_, m, i);
                auto u_i_old = row_view(ori_, m, i);

                // For rotational moves, use the old position; for translational, use buffer
                Array3d r_new = result.has_rot ? r_old : row_view(buf_pos_, m, 0);
                // For rotational moves, use buffer orientation; for translational, use old
                Array3d u_i_new = result.has_rot ? row_view(buf_ori_, m, 0) : u_i_old;

                double V_old = 0.0, V_new = 0.0;
                if (!V_ext_.is_none()) {
                    V_old += V_ext_(r_old, u_i_old).cast<double>();
                    V_new += V_ext_(r_new, u_i_new).cast<double>();
                }
                if (!V_int_.is_none()) {
                    for (int j = 0; j < N_; j++) {
                        if (j == i) continue;
                        auto r_j   = row_view(pos_, m, j);
                        auto u_j   = row_view(ori_, m, j);
                        auto rij_o = mic_displacement(r_old, r_j);
                        auto rij_n = mic_displacement(r_new, r_j);
                        V_old += V_int_(rij_o, u_i_old, u_j).cast<double>();
                        V_new += V_int_(rij_n, u_i_new, u_j).cast<double>();
                    }
                }
                log_a -= tau_ * factor * (V_new - V_old);
            }

            // Trial wavefunction (endpoint slices only)
            if ((m == 0 || m == M_ - 1) && (!f_.is_none() || !h_.is_none())) {
                auto r_old = row_view(pos_, m, i);
                auto u_i_old = row_view(ori_, m, i);
                Array3d r_new = result.has_rot ? r_old : row_view(buf_pos_, m, 0);
                Array3d u_i_new = result.has_rot ? row_view(buf_ori_, m, 0) : u_i_old;

                double psi_old = 0.0, psi_new = 0.0;
                if (!f_.is_none()) {
                    psi_old += f_(r_old, u_i_old).cast<double>();
                    psi_new += f_(r_new, u_i_new).cast<double>();
                }
                if (!h_.is_none()) {
                    for (int j = 0; j < N_; j++) {
                        if (j == i) continue;
                        auto r_j   = row_view(pos_, m, j);
                        auto u_j   = row_view(ori_, m, j);
                        auto rij_o = mic_displacement(r_old, r_j);
                        auto rij_n = mic_displacement(r_new, r_j);
                        psi_old += h_(rij_o, u_i_old, u_j).cast<double>();
                        psi_new += h_(rij_n, u_i_new, u_j).cast<double>();
                    }
                }
                log_a += psi_new - psi_old;
            }
        }

        return log_a;
    }
};

// ============================================================
// Module binding
// ============================================================
PYBIND11_MODULE(_engine, m) {
    m.doc() = "pigsmc C++ sweep engine";

    py::class_<FreeRotorGridC, std::shared_ptr<FreeRotorGridC>>(m, "FreeRotorGridC")
        .def(py::init<double, int, int>(),
             py::arg("lambda_rot"), py::arg("L_max") = 100, py::arg("grid_size") = 1000)
        .def("log_eval", &FreeRotorGridC::log_eval, py::arg("x"));

    py::class_<MoveResult>(m, "MoveResult")
        .def(py::init<int, int, int, double, bool>(),
             py::arg("particle"), py::arg("m_lo"), py::arg("m_hi"),
             py::arg("log_ratio_contrib"), py::arg("has_rot") = false)
        .def_readwrite("particle", &MoveResult::particle)
        .def_readwrite("m_lo", &MoveResult::m_lo)
        .def_readwrite("m_hi", &MoveResult::m_hi)
        .def_readwrite("log_ratio_contrib", &MoveResult::log_ratio_contrib)
        .def_readwrite("has_rot", &MoveResult::has_rot);

    py::class_<PathState>(m, "PathState")
        .def(py::init<>())
        .def_readwrite("positions", &PathState::positions)
        .def_readwrite("orientations", &PathState::orientations)
        .def_readwrite("buffer_positions", &PathState::buffer_positions)
        .def_readwrite("buffer_orientations", &PathState::buffer_orientations)
        .def_readwrite("N", &PathState::N)
        .def_readwrite("M", &PathState::M)
        .def_readwrite("tau_prime", &PathState::tau_prime)
        .def_readwrite("lambda_trans", &PathState::lambda_trans)
        .def_readwrite("lambda_rot", &PathState::lambda_rot)
        .def_readwrite("slice_kind", &PathState::slice_kind)
        .def_readwrite("boundary", &PathState::boundary);

    py::class_<Move, PyMove, std::shared_ptr<Move>>(m, "Move")
        .def(py::init<>())
        .def("propose", &Move::propose);

    py::class_<TranslationEndMove, Move, PyTranslationEndMove,
               std::shared_ptr<TranslationEndMove>>(m, "TranslationEndMove")
        .def(py::init<double>(), py::arg("step_size"))
        .def("propose", &TranslationEndMove::propose);

    py::class_<TranslationInteriorMove, Move, PyTranslationInteriorMove,
               std::shared_ptr<TranslationInteriorMove>>(m, "TranslationInteriorMove")
        .def(py::init<double>(), py::arg("step_size"))
        .def("propose", &TranslationInteriorMove::propose);

    py::class_<TranslationRigidMove, Move, PyTranslationRigidMove,
               std::shared_ptr<TranslationRigidMove>>(m, "TranslationRigidMove")
        .def(py::init<double>(), py::arg("step_size"))
        .def("propose", &TranslationRigidMove::propose);

    py::class_<TranslationBisectionMove, Move, PyTranslationBisectionMove,
               std::shared_ptr<TranslationBisectionMove>>(m, "TranslationBisectionMove")
        .def(py::init<int>(), py::arg("level"))
        .def("propose", &TranslationBisectionMove::propose);

    py::class_<RotationEndMove, Move, PyRotationEndMove,
               std::shared_ptr<RotationEndMove>>(m, "RotationEndMove")
        .def(py::init<double>(), py::arg("step_size"))
        .def("propose", &RotationEndMove::propose);

    py::class_<RotationInteriorMove, Move, PyRotationInteriorMove,
               std::shared_ptr<RotationInteriorMove>>(m, "RotationInteriorMove")
        .def(py::init<double>(), py::arg("step_size"))
        .def("propose", &RotationInteriorMove::propose);

    py::class_<RotationRigidMove, Move, PyRotationRigidMove,
               std::shared_ptr<RotationRigidMove>>(m, "RotationRigidMove")
        .def(py::init<double>(), py::arg("step_size"))
        .def("propose", &RotationRigidMove::propose);

    py::class_<RotationBisectionMove, Move, PyRotationBisectionMove,
               std::shared_ptr<RotationBisectionMove>>(m, "RotationBisectionMove")
        .def(py::init<int>(), py::arg("level"))
        .def("propose", &RotationBisectionMove::propose);

    py::class_<Engine>(m, "Engine")
        .def(py::init<Array3d, Array3d, Array3d, Array3d,
                      Array1d, Array1d, Array1i,
                      int, int, double, py::object,
                      int, Array1d, py::object,
                      std::vector<std::shared_ptr<FreeRotorGridC>>>(),
             py::arg("positions"), py::arg("orientations"),
             py::arg("buf_pos"), py::arg("buf_ori"),
             py::arg("lambda_trans"), py::arg("lambda_rot"),
             py::arg("slice_kind"),
             py::arg("N"), py::arg("M"), py::arg("tau_prime"),
             py::arg("rng"),
             py::arg("boundary_kind"), py::arg("box_half"),
             py::arg("boundary_obj"), py::arg("grids"))
        .def("add_move", &Engine::add_move, py::arg("move"), py::arg("weight"))
        .def("set_potential", &Engine::set_potential,
             py::arg("V_ext"), py::arg("V_int"))
        .def("set_trial_wavefunction", &Engine::set_trial_wavefunction,
             py::arg("f"), py::arg("h"))
        .def("run_sweep", &Engine::run_sweep, py::arg("sweeps_per_block"));
}
