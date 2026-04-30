"""
Fire Truck Response Navigator
AI-powered optimal routing for fire trucks using A* Search
"""

import streamlit as st
import numpy as np
import heapq
import math
import time

# ---------------------------------------------
#  PAGE CONFIG
# ---------------------------------------------
st.set_page_config(
    page_title="Fire Truck Response Navigator",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 1.5rem; }
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .algo-badge {
        background: #c62828; color: white; padding: 3px 10px;
        border-radius: 20px; font-size: 0.8rem; font-weight: 600;
    }
    div[data-testid="stHorizontalBlock"] { gap: 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------
#  CONSTANTS
# ---------------------------------------------
EMPTY    = 0
OBSTACLE = 1
START    = 2
GOAL     = 3
PATH     = 4
VISITED  = 5

COLORS = {
    EMPTY:    "#1e2130",
    OBSTACLE: "#37474f",
    START:    "#f44336",
    GOAL:     "#4caf50",
    PATH:     "#ffc107",
    VISITED:  "#1a3a5c",
}

CELL_EMOJIS = {
    START:    "🚒",
    GOAL:     "🔥",
    OBSTACLE: "🏢",
}

# ---------------------------------------------
#  SESSION STATE
# ---------------------------------------------
def reset_grid():
    n = st.session_state.grid_size
    st.session_state.grid      = np.zeros((n, n), dtype=int)
    st.session_state.weights   = np.ones((n, n), dtype=float)
    st.session_state.result    = None
    st.session_state.start_pos = None
    st.session_state.goal_pos  = None


def init_state():
    defaults = {
        "grid_size": 15,
        "grid":      None,
        "weights":   None,
        "result":    None,
        "draw_mode": "obstacle",
        "start_pos": None,
        "goal_pos":  None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if st.session_state.grid is None:
        reset_grid()

init_state()


# ---------------------------------------------
#  A* SEARCH
# ---------------------------------------------

class SearchResult:
    def __init__(self, path, visited, cost, nodes_explored, time_ms):
        self.path           = path
        self.visited        = visited
        self.cost           = cost
        self.nodes_explored = nodes_explored
        self.time_ms        = time_ms
        self.found          = path is not None


def euclidean(a, b):
    """Euclidean distance heuristic : admissible & consistent on unit grid"""
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def get_neighbors(pos, grid):
    """Return valid 4-directional neighbors (no diagonals)"""
    r, c = pos
    rows, cols = grid.shape
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != OBSTACLE:
            yield (nr, nc)


def reconstruct_path(came_from, start, goal):
    """Trace back from goal to start using came_from map"""
    path, node = [], goal
    max_steps  = len(came_from) + 1   # safety limit to prevent infinite loop
    steps      = 0
    while node != start:
        if steps > max_steps:
            raise RuntimeError("Path reconstruction exceeded expected length : possible cycle in came_from")
        path.append(node)
        node = came_from[node]
        steps += 1
    path.append(start)
    return list(reversed(path))


def astar(grid, weights, start, goal):
    """
    A* Search: f(n) = g(n) + h(n)
    Guaranteed optimal path when heuristic is admissible
    """
    t0      = time.time()
    counter = 0  # tie-breaker

    open_heap = []
    heapq.heappush(open_heap, (euclidean(start, goal), counter, 0, start))

    came_from = {start: None}
    g_cost    = {start: 0}
    visited   = set()
    nodes_explored = 0

    while open_heap:
        f, _, g, current = heapq.heappop(open_heap)

        if current in visited:
            continue
        visited.add(current)
        nodes_explored += 1

        if current == goal:
            path    = reconstruct_path(came_from, start, goal)
            elapsed = (time.time() - t0) * 1000
            return SearchResult(path, visited, g_cost[goal], nodes_explored, elapsed)

        for neighbor in get_neighbors(current, grid):
            tentative_g = g_cost[current] + weights[neighbor]
            if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                g_cost[neighbor]    = tentative_g
                came_from[neighbor] = current
                counter += 1
                heapq.heappush(open_heap,
                    (tentative_g + euclidean(neighbor, goal), counter, tentative_g, neighbor))

    elapsed = (time.time() - t0) * 1000
    return SearchResult(None, visited, float('inf'), nodes_explored, elapsed)


def dijkstra(grid, weights, start, goal):
    """
    Dijkstra's Algorithm: f(n) = g(n) only (no heuristic)
    Guaranteed optimal but explores more nodes than A*
    """
    t0      = time.time()
    counter = 0

    open_heap = []
    heapq.heappush(open_heap, (0, counter, start))

    came_from = {start: None}
    g_cost    = {start: 0}
    visited   = set()
    nodes_explored = 0

    while open_heap:
        g, _, current = heapq.heappop(open_heap)

        if current in visited:
            continue
        visited.add(current)
        nodes_explored += 1

        if current == goal:
            path    = reconstruct_path(came_from, start, goal)
            elapsed = (time.time() - t0) * 1000
            return SearchResult(path, visited, g_cost[goal], nodes_explored, elapsed)

        for neighbor in get_neighbors(current, grid):
            tentative_g = g_cost[current] + weights[neighbor]
            if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                g_cost[neighbor]    = tentative_g
                came_from[neighbor] = current
                counter += 1
                heapq.heappush(open_heap, (tentative_g, counter, neighbor))

    elapsed = (time.time() - t0) * 1000
    return SearchResult(None, visited, float('inf'), nodes_explored, elapsed)


def greedy_bfs(grid, weights, start, goal):
    """
    Greedy Best-First Search: f(n) = h(n) only (no path cost)
    Fast but NOT guaranteed to find the optimal path
    """
    t0      = time.time()
    counter = 0

    open_heap = []
    heapq.heappush(open_heap, (euclidean(start, goal), counter, start))

    came_from = {start: None}
    visited   = set()
    nodes_explored = 0

    while open_heap:
        h, _, current = heapq.heappop(open_heap)

        if current in visited:
            continue
        visited.add(current)
        nodes_explored += 1

        if current == goal:
            path    = reconstruct_path(came_from, start, goal)
            # compute actual cost of the path found
            cost = sum(weights[path[i]] for i in range(1, len(path)))
            elapsed = (time.time() - t0) * 1000
            return SearchResult(path, visited, cost, nodes_explored, elapsed)

        for neighbor in get_neighbors(current, grid):
            if neighbor not in visited and neighbor not in came_from:
                came_from[neighbor] = current
                counter += 1
                heapq.heappush(open_heap, (euclidean(neighbor, goal), counter, neighbor))

    elapsed = (time.time() - t0) * 1000
    return SearchResult(None, visited, float('inf'), nodes_explored, elapsed)


# ---------------------------------------------
#  GRID RENDERING
# ---------------------------------------------

def render_grid(grid, result=None, show_explored=True):
    """Render the grid as an HTML table with SVG path line overlay."""
    n         = grid.shape[0]
    cell_size = max(28, min(48, 600 // n))
    display   = grid.copy()

    path_cells = set()
    if result and result.found and show_explored:
        for node in result.visited:
            if display[node] == EMPTY:
                display[node] = VISITED
        for node in (result.path or []):
            path_cells.add(node)

    table_size = cell_size * n

    # Build table HTML
    html = (
        f'<div style="overflow-x:auto;padding:4px;">'
        f'<div style="position:relative;display:inline-block;">'
        f'<table style="border-collapse:collapse;font-size:{max(10, cell_size//3)}px;">'
    )

    for r in range(n):
        html += "<tr>"
        for c in range(n):
            val      = display[r, c]
            emoji    = CELL_EMOJIS.get(val, "")
            bg_color = COLORS[EMPTY] if val == VISITED else COLORS[val]

            border = (
                "2px solid #f44336" if val == START else
                "2px solid #4caf50" if val == GOAL  else
                "1px solid #263040"
            )

            html += (
                f'<td style="width:{cell_size}px;height:{cell_size}px;'
                f'background:{bg_color};border:{border};'
                f'text-align:center;vertical-align:middle;border-radius:3px;">'
                f'{emoji}</td>'
            )
        html += "</tr>"

    html += "</table>"

    # Draw SVG line over the path
    if path_cells and result and result.path and len(result.path) > 1:
        half = cell_size // 2
        points = " ".join(
            f"{c * cell_size + half},{r * cell_size + half}"
            for r, c in result.path
        )
        html += (
            f'<svg style="position:absolute;top:0;left:0;pointer-events:none;" '
            f'width="{table_size}" height="{table_size}">'
            f'<polyline points="{points}" fill="none" stroke="#ffc107" '
            f'stroke-width="3" stroke-linecap="round" stroke-linejoin="round" '
            f'stroke-dasharray="6,3"/>'
            f'</svg>'
        )

    html += "</div></div>"
    return html


# ---------------------------------------------
#  SIDEBAR
# ---------------------------------------------

with st.sidebar:
    st.markdown("## 🚒 Fire Truck Navigator")
    st.markdown("---")

    st.markdown("### ⚙️ Grid Settings")
    new_size = st.slider("Grid Size", 8, 30, st.session_state.grid_size, 1)
    if new_size != st.session_state.grid_size:
        st.session_state.grid_size = new_size
        reset_grid()
        st.rerun()

    st.markdown("### 🤖 Algorithm")
    algorithm = st.selectbox(
        "Select algorithm:",
        ["A* Search", "Dijkstra's", "Greedy BFS"],
        index=0
    )
    st.session_state.algorithm = algorithm

    st.markdown("### 🎨 Draw Mode")
    draw_mode = st.radio(
        "Select what to place:",
        ["obstacle", "start", "goal"],
        format_func=lambda x: {
            "obstacle": "🏢 Obstacle (Road Block)",
            "start":    "🚒 Fire Station",
            "goal":     "🔥 Fire Location",
        }[x],
        index=["obstacle","start","goal"].index(st.session_state.draw_mode) if st.session_state.draw_mode in ["obstacle","start","goal"] else 0
    )
    st.session_state.draw_mode = draw_mode

    st.markdown("---")
    if st.button("🗑️ Clear All", use_container_width=True):
        reset_grid()
        st.rerun()

    st.markdown("### 🗝️ Legend")
    for color, label in [
        ("#f44336", "🚒 Fire Station"),
        ("#4caf50", "🔥 Fire Location"),
        ("#37474f", "🏢 Obstacle / Block"),
        ("#ffc107", "🛣  Optimal Route"),
    ]:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0;">'
            f'<div style="width:16px;height:16px;background:{color};border-radius:3px;'
            f'border:1px solid #444;flex-shrink:0;"></div>'
            f'<span style="font-size:0.82rem;color:#cfd8dc;">{label}</span></div>',
            unsafe_allow_html=True
        )


# ---------------------------------------------
#  MAIN PANEL
# ---------------------------------------------

st.markdown("# 🚒 AI Fire Truck Response Navigator")
st.markdown(
    f'<span class="algo-badge">{st.session_state.get("algorithm", "A* Search")}</span> &nbsp;'
    '<span style="color:#90a4ae;font-size:0.85rem;">Set fire station 🚒 & fire location 🔥 then Run</span>',
    unsafe_allow_html=True
)
st.markdown("---")

main_col, right_col = st.columns([3, 1.2])

with main_col:
    st.markdown("### 🗺️ City Map Editor")
    st.caption("Choose draw mode in the sidebar, then enter row/col and click Place")

    with st.expander("📍 Click a Cell to Edit", expanded=True):
        cc1, cc2, cc3 = st.columns([1,1,2])
        with cc1:
            click_r = st.number_input("Row", 0, st.session_state.grid_size-1, 0, key="cr")
        with cc2:
            click_c = st.number_input("Col", 0, st.session_state.grid_size-1, 0, key="cc")
        with cc3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(f"Place: {st.session_state.draw_mode.upper()}", use_container_width=True):
                r, c = int(click_r), int(click_c)
                mode = st.session_state.draw_mode
                grid = st.session_state.grid

                if mode == "start":
                    if grid[r, c] == GOAL:
                        st.error("⚠️ Cannot place Fire Station on the Fire Location cell")
                    else:
                        grid[grid == START] = EMPTY
                        grid[r, c] = START
                        st.session_state.start_pos = (r, c)
                elif mode == "goal":
                    if grid[r, c] == START:
                        st.error("⚠️ Cannot place Fire Location on the Fire Station cell")
                    else:
                        grid[grid == GOAL] = EMPTY
                        grid[r, c] = GOAL
                        st.session_state.goal_pos = (r, c)
                elif mode == "obstacle":
                    if grid[r, c] not in (START, GOAL):
                        grid[r, c] = OBSTACLE
                    else:
                        st.error("⚠️ Cannot place an obstacle on the Fire Station or Fire Location")

                st.session_state.result = None
                st.rerun()

    st.markdown(
        render_grid(st.session_state.grid, st.session_state.result, show_explored=True),
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    run_col, clear_col = st.columns([2, 1])
    with run_col:
        run_clicked = st.button("🚀 Find Optimal Route", use_container_width=True, type="primary")
    with clear_col:
        if st.button("🔄 Reset Path", use_container_width=True):
            st.session_state.result = None
            st.rerun()

    if run_clicked:
        start = st.session_state.start_pos
        goal  = st.session_state.goal_pos
        n     = st.session_state.grid_size

        # Validate positions are not stale from a previous grid size
        def in_bounds(pos):
            return pos is not None and 0 <= pos[0] < n and 0 <= pos[1] < n

        if not in_bounds(start) or not in_bounds(goal):
            st.error("⚠️ Please place both a Fire Station 🚒 and Fire Location 🔥 on the grid")
        elif start == goal:
            st.error("⚠️ Start and goal cannot be the same cell")
        elif st.session_state.grid[start] == OBSTACLE:
            st.error("⚠️ Fire Station is blocked by an obstacle. Please move it")
        elif st.session_state.grid[goal] == OBSTACLE:
            st.error("⚠️ Fire Location is blocked by an obstacle. Please move it")
        else:
            algo = st.session_state.get("algorithm", "A* Search")
            algo_funcs = {
                "A* Search":  astar,
                "Dijkstra's": dijkstra,
                "Greedy BFS": greedy_bfs,
            }
            with st.spinner(f"Running {algo}..."):
                try:
                    result = algo_funcs[algo](st.session_state.grid, st.session_state.weights, start, goal)
                    st.session_state.result = result
                    if result.found:
                        st.success(f"✅ Route found! Path length: {len(result.path)} cells : Cost: {result.cost:.2f}")
                    else:
                        st.error("❌ No route found : fire location is unreachable from fire station")
                except Exception as e:
                    st.error(f"❌ Unexpected error during search: {e}")
                    st.stop()
            st.rerun()


# ---------------------------------------------
#  RIGHT PANEL — RESULTS
# ---------------------------------------------

with right_col:
    st.markdown("### 📊 Results")
    result = st.session_state.result

    if result:
        # Status
        if result.found:
            st.success("✅ Route Found!")
        else:
            st.error("❌ No Route Found")

        st.markdown("<br>", unsafe_allow_html=True)

        # Metrics
        metrics = [
            (" Path Length",    str(len(result.path)) if result.found and result.path else "N/A"),
            (" Route Cost",     f"{result.cost:.2f}"  if result.found else "∞"),
            (" Nodes Explored", str(result.nodes_explored)),
            (" Compute Time",   f"{result.time_ms:.2f} ms"),
        ]
        for lbl, val in metrics:
            st.markdown(
                f'<div style="background:#1a2535;border:1px solid #e53935;border-radius:8px;'
                f'padding:10px 14px;margin:6px 0;">'
                f'<div style="font-size:0.75rem;color:#ffccbc;text-transform:uppercase;letter-spacing:1px;">{lbl}</div>'
                f'<div style="font-size:1.4rem;font-weight:700;color:#ff8a65;">{val}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            '<div style="background:#1a2535;border-radius:10px;padding:20px;'
            'text-align:center;color:#546e7a;margin-top:10px;">'
            '<div style="font-size:2rem;">📍</div>'
            '<div style="margin-top:8px;">Set fire station & fire location,<br>then run the algorithm.</div>'
            '</div>',
            unsafe_allow_html=True
        )