#  Fire Truck Response Navigator

AI-powered optimal routing for fire trucks using A* Search, Dijkstra's, 
and Greedy Best-First Search on a city grid.

## Project Info
- Course: AI2002 Artificial Intelligence
- University: FAST-NUCES
- Members: Adil Taimoor (I23-0104), Mohammed Anas (I23-3026)

## Algorithms Implemented
- A* Search : optimal, heuristic-guided
- Dijkstra's Algorithm : optimal, uninformed baseline
- Greedy Best-First Search : fast, non-optimal

## Installation

1. Clone the repository:
   git clone https://github.com/at51401ansari-Adil-taimoor/Fire-Truck-Response-Navigator.git

2. Navigate to the project folder:
   cd Fire-Truck-Response-Navigator

3. Install dependencies:
   pip install -r requirements.txt

4. Run the application:
   streamlit run Main.py

5. Open browser at:
   http://localhost:8501

## How to Use
1. Set grid size using the sidebar slider (8-30)
2. Select draw mode: Fire Station, Fire Location, or Obstacle
3. Enter row and column coordinates and click Place
4. Select algorithm from the dropdown (A*, Dijkstra, Greedy BFS)
5. Click Find Optimal Route to run the search
6. View the optimal path on the grid and metrics in the Results panel

## Requirements
- Python 3.8+
- streamlit
- numpy
