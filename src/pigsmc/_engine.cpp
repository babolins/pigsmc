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
// MoveResult
// ============================================================
struct MoveResult {
    int particle{0};
    int m_lo{0}, m_hi{0};  // inclusive range of changed slices
    double log_ratio_contrib{0.0};

    MoveResult() = default;
    MoveResult(int particle, int m_lo, int m_hi, double lrc)
        : particle(particle), m_lo(m_lo), m_hi(m_hi), log_ratio_contrib(lrc) {}
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
    Array1i slice_kind;
    py::object boundary;
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


// ============================================================
// C++ move implementations
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
           Array1d lambda_trans, Array1i slice_kind,
           int N, int M, double tau_prime,
           py::object rng,
           int boundary_kind, Array1d box_half,
           py::object boundary_obj)
        : pos_(positions), ori_(orientations),
          buf_pos_(buf_pos), buf_ori_(buf_ori),
          lam_(lambda_trans), sk_(slice_kind),
          N_(N), M_(M), tau_(tau_prime), rng_(rng),
          V_ext_(py::none()), V_int_(py::none()),
          f_(py::none()), h_(py::none()),
          total_w_(0.0),
          boundary_obj_(boundary_obj)
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
        ps.slice_kind = sk_;
        ps.boundary = boundary_obj_;

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
    Array1d lam_;
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
        auto pos = pos_.mutable_unchecked<3>();
        auto buf = buf_pos_.unchecked<3>();
        int i = result.particle;
        for (int m = result.m_lo; m <= result.m_hi; m++) {
            pos(m, i, 0) = buf(m, 0, 0);
            pos(m, i, 1) = buf(m, 0, 1);
            pos(m, i, 2) = buf(m, 0, 2);
        }
    }

    double compute_log_acceptance(const MoveResult& result) {
        double log_a = result.log_ratio_contrib;

        auto pos = pos_.unchecked<3>();
        auto buf = buf_pos_.unchecked<3>();
        auto lam = lam_.unchecked<1>();

        int i    = result.particle;
        int m_lo = result.m_lo;
        int m_hi = result.m_hi;
        double lam_i = lam(i);

        // --- Kinetic terms ---
        // Each link ml--(ml+1) is visited exactly once.  Links touching the
        // changed range run from max(0, m_lo-1) to min(M_-2, m_hi).
        // Old positions always come from pos; new positions come from buf
        // for endpoints inside [m_lo, m_hi] and from pos for boundary
        // neighbors outside the changed range.
        int link_lo = std::max(0,      m_lo - 1);
        int link_hi = std::min(M_ - 2, m_hi);
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

        // --- Potential and trial-wavefunction terms (per bead) ---
        for (int m = m_lo; m <= m_hi; m++) {
            if (!V_ext_.is_none() || !V_int_.is_none()) {
                double factor = (m == 0 || m == M_ - 1) ? 0.5 : 1.0;
                auto r_old = row_view(pos_, m, i);
                auto r_new = row_view(buf_pos_, m, 0);
                auto u_i   = row_view(ori_, m, i);

                double V_old = 0.0, V_new = 0.0;
                if (!V_ext_.is_none()) {
                    V_old += V_ext_(r_old, u_i).cast<double>();
                    V_new += V_ext_(r_new, u_i).cast<double>();
                }
                if (!V_int_.is_none()) {
                    for (int j = 0; j < N_; j++) {
                        if (j == i) continue;
                        auto r_j   = row_view(pos_, m, j);
                        auto u_j   = row_view(ori_, m, j);
                        auto rij_o = mic_displacement(r_old, r_j);
                        auto rij_n = mic_displacement(r_new, r_j);
                        V_old += V_int_(rij_o, u_i, u_j).cast<double>();
                        V_new += V_int_(rij_n, u_i, u_j).cast<double>();
                    }
                }
                log_a -= tau_ * factor * (V_new - V_old);
            }

            // Trial wavefunction (endpoint slices only)
            if ((m == 0 || m == M_ - 1) && (!f_.is_none() || !h_.is_none())) {
                auto r_old = row_view(pos_, m, i);
                auto r_new = row_view(buf_pos_, m, 0);
                auto u_i   = row_view(ori_, m, i);

                double psi_old = 0.0, psi_new = 0.0;
                if (!f_.is_none()) {
                    psi_old += f_(r_old, u_i).cast<double>();
                    psi_new += f_(r_new, u_i).cast<double>();
                }
                if (!h_.is_none()) {
                    for (int j = 0; j < N_; j++) {
                        if (j == i) continue;
                        auto r_j   = row_view(pos_, m, j);
                        auto u_j   = row_view(ori_, m, j);
                        auto rij_o = mic_displacement(r_old, r_j);
                        auto rij_n = mic_displacement(r_new, r_j);
                        psi_old += h_(rij_o, u_i, u_j).cast<double>();
                        psi_new += h_(rij_n, u_i, u_j).cast<double>();
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

    py::class_<MoveResult>(m, "MoveResult")
        .def(py::init<int, int, int, double>(),
             py::arg("particle"), py::arg("m_lo"), py::arg("m_hi"),
             py::arg("log_ratio_contrib"))
        .def_readwrite("particle", &MoveResult::particle)
        .def_readwrite("m_lo", &MoveResult::m_lo)
        .def_readwrite("m_hi", &MoveResult::m_hi)
        .def_readwrite("log_ratio_contrib", &MoveResult::log_ratio_contrib);

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

    py::class_<Engine>(m, "Engine")
        .def(py::init<Array3d, Array3d, Array3d, Array3d,
                      Array1d, Array1i,
                      int, int, double, py::object,
                      int, Array1d, py::object>(),
             py::arg("positions"), py::arg("orientations"),
             py::arg("buf_pos"), py::arg("buf_ori"),
             py::arg("lambda_trans"), py::arg("slice_kind"),
             py::arg("N"), py::arg("M"), py::arg("tau_prime"),
             py::arg("rng"),
             py::arg("boundary_kind"), py::arg("box_half"),
             py::arg("boundary_obj"))
        .def("add_move", &Engine::add_move, py::arg("move"), py::arg("weight"))
        .def("set_potential", &Engine::set_potential,
             py::arg("V_ext"), py::arg("V_int"))
        .def("set_trial_wavefunction", &Engine::set_trial_wavefunction,
             py::arg("f"), py::arg("h"))
        .def("run_sweep", &Engine::run_sweep, py::arg("sweeps_per_block"));
}
