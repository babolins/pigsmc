#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cmath>
#include <memory>
#include <set>
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
    std::vector<std::pair<int, int>> changed;  // (particle, slice)
    double log_ratio_contrib;

    MoveResult() : log_ratio_contrib(0.0) {}
    MoveResult(std::vector<std::pair<int, int>> c, double lrc)
        : changed(std::move(c)), log_ratio_contrib(lrc) {}
};

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

static Array3d subtract3(Array3d& a, Array3d& b) {
    auto ra = a.unchecked<1>();
    auto rb = b.unchecked<1>();
    Array3d out({3});
    auto ro = out.mutable_unchecked<1>();
    ro(0) = ra(0) - rb(0);
    ro(1) = ra(1) - rb(1);
    ro(2) = ra(2) - rb(2);
    return out;
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

        return MoveResult({{i, m}}, 0.0);
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

        return MoveResult({{i, m}}, 0.0);
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
           py::object rng)
        : pos_(positions), ori_(orientations),
          buf_pos_(buf_pos), buf_ori_(buf_ori),
          lam_(lambda_trans), sk_(slice_kind),
          N_(N), M_(M), tau_(tau_prime), rng_(rng),
          V_ext_(py::none()), V_int_(py::none()),
          f_(py::none()), h_(py::none()),
          total_w_(0.0) {}

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

            // Collect unique beads
            std::set<int> beads;
            for (auto& [pi, pm] : result.changed) beads.insert(pm);
            for (int mb : beads) delta_bead[mb].attempts++;

            if (accepted) {
                delta_move[idx].acceptances++;
                for (int mb : beads) delta_bead[mb].acceptances++;
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

    void apply_move(const MoveResult& result) {
        auto pos = pos_.mutable_unchecked<3>();
        auto buf = buf_pos_.unchecked<3>();
        for (auto& [i, m] : result.changed) {
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

        for (auto& [i, m] : result.changed) {
            double lam_i = lam(i);

            // Free-propagator kinetic terms
            if (m > 0) {
                double d_old = 0.0, d_new = 0.0;
                for (int d = 0; d < 3; d++) {
                    double rp = pos(m - 1, i, d);
                    double ro = pos(m, i, d) - rp;
                    double rn = buf(m, 0, d) - rp;
                    d_old += ro * ro;
                    d_new += rn * rn;
                }
                log_a += (d_old - d_new) / (4.0 * lam_i);
            }
            if (m < M_ - 1) {
                double d_old = 0.0, d_new = 0.0;
                for (int d = 0; d < 3; d++) {
                    double rp = pos(m + 1, i, d);
                    double ro = pos(m, i, d) - rp;
                    double rn = buf(m, 0, d) - rp;
                    d_old += ro * ro;
                    d_new += rn * rn;
                }
                log_a += (d_old - d_new) / (4.0 * lam_i);
            }

            // Potential terms
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
                        auto rij_o = subtract3(r_old, r_j);
                        auto rij_n = subtract3(r_new, r_j);
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
                        auto rij_o = subtract3(r_old, r_j);
                        auto rij_n = subtract3(r_new, r_j);
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
        .def(py::init<std::vector<std::pair<int, int>>, double>(),
             py::arg("changed"), py::arg("log_ratio_contrib"))
        .def_readwrite("changed", &MoveResult::changed)
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
        .def_readwrite("slice_kind", &PathState::slice_kind);

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

    py::class_<Engine>(m, "Engine")
        .def(py::init<Array3d, Array3d, Array3d, Array3d,
                      Array1d, Array1i,
                      int, int, double, py::object>(),
             py::arg("positions"), py::arg("orientations"),
             py::arg("buf_pos"), py::arg("buf_ori"),
             py::arg("lambda_trans"), py::arg("slice_kind"),
             py::arg("N"), py::arg("M"), py::arg("tau_prime"),
             py::arg("rng"))
        .def("add_move", &Engine::add_move, py::arg("move"), py::arg("weight"))
        .def("set_potential", &Engine::set_potential,
             py::arg("V_ext"), py::arg("V_int"))
        .def("set_trial_wavefunction", &Engine::set_trial_wavefunction,
             py::arg("f"), py::arg("h"))
        .def("run_sweep", &Engine::run_sweep, py::arg("sweeps_per_block"));
}
