#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstdint>
#include <set>
#include <utility>

namespace py = pybind11;

namespace {
constexpr int size = 11;
using Cell = std::pair<int, int>;

bool inside(int r, int c) {
    return r >= 0 && r < size && c >= 0 && c < size;
}

py::list find_patterns(const py::array_t<std::int8_t, py::array::c_style>& grid,
                       int r, int c, int player) {
    if (grid.ndim() != 2 || grid.shape(0) != size || grid.shape(1) != size)
        throw py::value_error("grid must be an 11x11 contiguous int8 array");
    if (!inside(r, c) || (player != 1 && player != 2))
        throw py::value_error("invalid cell or player");
    const auto* data = grid.data();
    if (data[r * size + c] != 0)
        throw py::value_error("cell is already occupied");

    py::list patterns;
    constexpr int directions[][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};
    for (const auto& direction : directions) {
        const int dr = direction[0], dc = direction[1];
        int low = 0, high = 0;
        while (inside(r + (low - 1) * dr, c + (low - 1) * dc) &&
               data[(r + (low - 1) * dr) * size + c + (low - 1) * dc] == player)
            --low;
        while (inside(r + (high + 1) * dr, c + (high + 1) * dc) &&
               data[(r + (high + 1) * dr) * size + c + (high + 1) * dc] == player)
            ++high;
        const int length = high - low + 1;
        py::list ends;
        for (int offset : {low - 1, high + 1}) {
            const int rr = r + offset * dr, cc = c + offset * dc;
            if (inside(rr, cc) && data[rr * size + cc] == 0)
                ends.append(py::make_tuple(rr, cc));
        }
        const auto dir = py::make_tuple(dr, dc);
        if (length > 5)
            patterns.append(py::make_tuple("overline", dir, py::tuple()));
        else if (length == 4)
            patterns.append(py::make_tuple(ends.size() == 2 ? "open_four" : "blocked_four", dir, py::tuple(ends)));
        else if (length == 3)
            patterns.append(py::make_tuple(ends.size() == 2 ? "open_three" : "blocked_three", dir, py::tuple(ends)));

        if (length <= 3) {
            std::set<Cell> gaps;
            for (int start = -4; start <= 0; ++start) {
                if (!inside(r + start * dr, c + start * dc) ||
                    !inside(r + (start + 4) * dr, c + (start + 4) * dc))
                    continue;
                int values[5], stones = 0, empty = 0, first = 5, last = -1;
                for (int i = 0; i < 5; ++i) {
                    const int offset = start + i;
                    values[i] = offset == 0 ? player : data[(r + offset * dr) * size + c + offset * dc];
                    if (values[i] == player) {
                        ++stones;
                        first = std::min(first, i);
                        last = i;
                    } else if (values[i] == 0) {
                        ++empty;
                    }
                }
                if (stones != 3 || empty != 2 || last - first <= 2)
                    continue;
                for (int i = first + 1; i < last; ++i)
                    if (values[i] == 0)
                        gaps.emplace(r + (start + i) * dr, c + (start + i) * dc);
            }
            if (!gaps.empty()) {
                py::list cells;
                for (const auto& cell : gaps)
                    cells.append(py::make_tuple(cell.first, cell.second));
                patterns.append(py::make_tuple("broken_three", dir, py::tuple(cells)));
            }
        }
    }
    return patterns;
}
}  // namespace

PYBIND11_MODULE(gomoku_native, m) {
    m.def("find_patterns", &find_patterns, py::arg("grid").noconvert(),
          py::arg("r"), py::arg("c"), py::arg("player"));
}
