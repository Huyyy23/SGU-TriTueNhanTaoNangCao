#include <utility>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {
struct Node {
    std::vector<int> moves;
    std::vector<std::size_t> children;
    std::vector<double> priors;
    std::size_t parent = 0;
    int move = -1;
    int visits = 0;
    double value_sum = 0.0;
    bool expanded = false;
    bool terminal = false;
    double terminal_value = 0.0;
};

class Tree {
public:
    explicit Tree(double c_puct = 1.5) : c_puct_(c_puct) { reset(); }

    void reset() {
        std::vector<Node>().swap(nodes_);
        nodes_.emplace_back();
        simulations_done_ = 0;
        pending_ = false;
        pending_path_.clear();
    }

    py::tuple request(int budget, py::function choose_tie) {
        if (nodes_[0].terminal) return py::make_tuple(py::none(), py::list());
        if (pending_) throw std::runtime_error("Cannot request() while a leaf request is pending. Call receive() first.");
        if (budget < simulations_done_) throw std::invalid_argument("simulation budget is below completed count");
        while (simulations_done_ < budget) {
            std::size_t current = 0;
            std::vector<std::size_t> path;
            std::vector<int> moves;
            while (nodes_[current].expanded && !nodes_[current].terminal) {
                current = select_child(current, choose_tie);
                moves.push_back(nodes_[current].move);
                path.push_back(current);
            }
            if (nodes_[current].terminal) {
                backup_terminal(path, nodes_[current].terminal_value);
                ++simulations_done_;
                continue;
            }
            pending_path_ = std::move(path);
            pending_ = true;
            return py::make_tuple(current, moves);
        }
        return py::make_tuple(py::none(), py::list());
    }

    void set_terminal(std::size_t node, double value) {
        check_node(node);
        if (!pending_) throw std::runtime_error("no pending native leaf");
        nodes_[node].terminal = true;
        nodes_[node].terminal_value = value;
        nodes_[node].expanded = true;
        backup_terminal(pending_path_, value);
        pending_ = false;
        pending_path_.clear();
        ++simulations_done_;
    }

    void receive(std::size_t node, const std::vector<int>& moves,
                 const std::vector<double>& priors, double value) {
        check_node(node);
        if (!pending_ || moves.size() != priors.size())
            throw std::runtime_error("invalid or missing pending native leaf");
        if (!std::isfinite(value) || value < -1.0 || value > 1.0)
            throw std::invalid_argument("leaf value out of range");
        if (nodes_[node].expanded) throw std::runtime_error("native node expanded twice");
        std::vector<std::size_t> children;
        children.reserve(moves.size());
        for (std::size_t i = 0; i < moves.size(); ++i) {
            Node child;
            child.parent = node;
            child.move = moves[i];
            children.push_back(nodes_.size());
            nodes_.push_back(std::move(child));
        }
        nodes_[node].moves = moves;
        nodes_[node].priors = priors;
        nodes_[node].children = std::move(children);
        nodes_[node].expanded = true;
        ++nodes_[node].visits;
        nodes_[node].value_sum += value;
        double backed = value;
        if (pending_path_.size() > 1) {
            for (auto it = pending_path_.rbegin() + 1; it != pending_path_.rend(); ++it) {
                backed = -backed;
                ++nodes_[*it].visits;
                nodes_[*it].value_sum += backed;
            }
            backed = (pending_path_.size() % 2 == 0) ? value : -value;
            ++nodes_[0].visits;
            nodes_[0].value_sum += backed;
        } else if (!pending_path_.empty()) {
            ++nodes_[0].visits;
            nodes_[0].value_sum -= value;
        }
        ++simulations_done_;
        pending_ = false;
        pending_path_.clear();
    }

    py::array_t<int> visit_counts() const {
        py::array_t<int> counts(121);
        auto out = counts.mutable_data();
        std::fill(out, out + 121, 0);
        for (std::size_t i = 0; i < nodes_[0].children.size(); ++i)
            out[nodes_[0].moves[i]] = nodes_[nodes_[0].children[i]].visits;
        return counts;
    }

    py::tuple root_statistics() const {
        py::array_t<int> visits(121);
        py::array_t<double> priors(121);
        py::array_t<double> q_values(121);
        py::array_t<bool> legal_mask(121);
        auto* visit_data = visits.mutable_data();
        auto* prior_data = priors.mutable_data();
        auto* q_data = q_values.mutable_data();
        auto* legal_data = legal_mask.mutable_data();
        std::fill(visit_data, visit_data + 121, 0);
        std::fill(prior_data, prior_data + 121, 0.0);
        std::fill(q_data, q_data + 121, 0.0);
        std::fill(legal_data, legal_data + 121, false);
        for (std::size_t i = 0; i < nodes_[0].children.size(); ++i) {
            const int move = nodes_[0].moves[i];
            const Node& child = nodes_[nodes_[0].children[i]];
            visit_data[move] = child.visits;
            prior_data[move] = nodes_[0].priors[i];
            q_data[move] = child.visits ? -child.value_sum / child.visits : 0.0;
            legal_data[move] = true;
        }
        return py::make_tuple(std::move(visits), std::move(priors),
                              std::move(q_values), std::move(legal_mask));
    }


    int simulations_done() const { return simulations_done_; }
    std::size_t size() const { return nodes_.size(); }
    double root_value() const { return nodes_[0].visits ? nodes_[0].value_sum / nodes_[0].visits : 0.0; }
    bool root_terminal() const { return nodes_[0].terminal; }
    void mark_root_terminal(double value) {
        nodes_[0].terminal = true;
        nodes_[0].terminal_value = value;
        nodes_[0].expanded = true;
    }

private:
    void check_node(std::size_t node) const {
        if (node >= nodes_.size()) throw std::out_of_range("native node index out of range");
    }

    std::size_t select_child(std::size_t parent, py::function choose_tie) const {
        const Node& p = nodes_[parent];
        const double root_n = std::sqrt(static_cast<double>(p.visits > 0 ? p.visits : 1));
        double best = -std::numeric_limits<double>::infinity();
        std::vector<std::size_t> ties;
        for (std::size_t i = 0; i < p.children.size(); ++i) {
            const Node& child = nodes_[p.children[i]];
            const double q = child.visits ? -child.value_sum / child.visits : 0.0;
            const double u = c_puct_ * p.priors[i] * root_n / (1.0 + child.visits);
            const double score = q + u;
            if (score > best + 1e-9) {
                best = score;
                ties.assign(1, p.children[i]);
            } else if (std::abs(score - best) <= 1e-9) {
                ties.push_back(p.children[i]);
            }
        }
        if (ties.empty()) throw std::runtime_error("expanded native node has no children");
        const auto selected = choose_tie(ties.size()).cast<std::size_t>();
        if (selected >= ties.size()) throw std::out_of_range("tie-break index out of range");
        return ties[selected];
    }

    void backup_terminal(const std::vector<std::size_t>& path, double value) {
        double backed = value;
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            ++nodes_[*it].visits;
            nodes_[*it].value_sum += backed;
            backed = -backed;
        }
        ++nodes_[0].visits;
        nodes_[0].value_sum += backed;
    }

    std::vector<Node> nodes_;
    double c_puct_;
    int simulations_done_ = 0;
    bool pending_ = false;
    std::vector<std::size_t> pending_path_;
};
}  // namespace
PYBIND11_MODULE(gomoku_mcts_native, m) {
    py::class_<Tree>(m, "Tree")
        .def(py::init<double>(), py::arg("c_puct") = 1.5)
        .def("reset", &Tree::reset)
        .def("request", &Tree::request)
        .def("set_terminal", &Tree::set_terminal)
        .def("mark_root_terminal", &Tree::mark_root_terminal)
        .def("receive", &Tree::receive)
        .def("visit_counts", &Tree::visit_counts)
        .def("root_statistics", &Tree::root_statistics)
        .def_property_readonly("simulations_done", &Tree::simulations_done)
        .def_property_readonly("size", &Tree::size)
        .def_property_readonly("root_value", &Tree::root_value)
        .def_property_readonly("root_terminal", &Tree::root_terminal);
    m.attr("__version__") = "0.1.0";
}
