"""Solve a small maze with a Grover-search circuit.

The classical BFS pass finds the shortest possible path length. Grover's
algorithm then searches every sequence of that length in parallel, marking
sequences that reach the goal without crossing a wall or leaving the maze.

This is an educational example rather than a claim of quantum speedup:
building the oracle still requires the maze rules to be evaluated classically.
"""

from __future__ import annotations

from collections import deque
from math import floor, log2, pi
from typing import Iterable, Sequence

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


Maze = Sequence[str]
MOVES = ("U", "D", "L", "R")
DELTAS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}


def validate_maze(maze: Maze) -> tuple[tuple[int, int], tuple[int, int]]:
    """Validate a maze and return its start and goal coordinates."""
    if not maze or any(not row for row in maze):
        raise ValueError("The maze must contain at least one non-empty row.")
    width = len(maze[0])
    if any(len(row) != width for row in maze):
        raise ValueError("Every maze row must have the same width.")

    starts = [(r, c) for r, row in enumerate(maze) for c, cell in enumerate(row) if cell == "S"]
    goals = [(r, c) for r, row in enumerate(maze) for c, cell in enumerate(row) if cell == "G"]
    if len(starts) != 1 or len(goals) != 1:
        raise ValueError("The maze must contain exactly one S and one G.")
    if any(cell not in ".#SG" for row in maze for cell in row):
        raise ValueError("Maze cells may only be '.', '#', 'S', or 'G'.")
    return starts[0], goals[0]


def walk(maze: Maze, start: tuple[int, int], moves: str) -> tuple[int, int] | None:
    """Walk a move sequence; return the final position or None if invalid."""
    row, column = start
    for move in moves:
        delta_row, delta_column = DELTAS[move]
        row += delta_row
        column += delta_column
        if (
            row < 0
            or row >= len(maze)
            or column < 0
            or column >= len(maze[0])
            or maze[row][column] == "#"
        ):
            return None
    return row, column


def shortest_path_length(maze: Maze, start: tuple[int, int], goal: tuple[int, int]) -> int:
    """Find the shortest path length with BFS."""
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        position, distance = queue.popleft()
        if position == goal:
            return distance
        for delta_row, delta_column in DELTAS.values():
            next_position = (position[0] + delta_row, position[1] + delta_column)
            if (
                0 <= next_position[0] < len(maze)
                and 0 <= next_position[1] < len(maze[0])
                and maze[next_position[0]][next_position[1]] != "#"
                and next_position not in visited
            ):
                visited.add(next_position)
                queue.append((next_position, distance + 1))
    raise ValueError("The maze has no path from S to G.")


def encode_moves(moves: str) -> int:
    """Encode two bits per move in the order U=0, D=1, L=2, R=3."""
    return sum(MOVES.index(move) << (2 * index) for index, move in enumerate(moves))


def decode_moves(value: int, length: int) -> str:
    """Decode a computational-basis integer into a move sequence."""
    return "".join(MOVES[(value >> (2 * index)) & 3] for index in range(length))


def valid_states(
    maze: Maze, start: tuple[int, int], goal: tuple[int, int], length: int
) -> list[int]:
    """Return basis states whose move sequences solve the maze."""
    return [
        state
        for state in range(4**length)
        if (moves := decode_moves(state, length))
        and walk(maze, start, moves) == goal
    ]


def build_phase_oracle(marked_states: Iterable[int], search_qubits: int) -> QuantumCircuit:
    """Build an oracle that applies a phase flip to each marked basis state."""
    oracle = QuantumCircuit(search_qubits + 1, name="oracle")
    target = search_qubits
    for state in marked_states:
        for qubit in range(search_qubits):
            if not (state >> qubit) & 1:
                oracle.x(qubit)
        oracle.mcx(list(range(search_qubits)), target)
        for qubit in range(search_qubits):
            if not (state >> qubit) & 1:
                oracle.x(qubit)
    return oracle


def build_grover_circuit(search_qubits: int, marked_states: list[int]) -> QuantumCircuit:
    """Create a Grover circuit for the supplied marked basis states."""
    oracle = build_phase_oracle(marked_states, search_qubits)
    circuit = QuantumCircuit(search_qubits + 1, search_qubits)
    circuit.h(range(search_qubits))
    circuit.x(search_qubits)
    circuit.h(search_qubits)

    state_count = 2**search_qubits
    iterations = max(1, floor((pi / 4) * (state_count / len(marked_states)) ** 0.5))
    for _ in range(iterations):
        circuit.compose(oracle, inplace=True)
        circuit.h(range(search_qubits))
        circuit.x(range(search_qubits))
        circuit.h(search_qubits - 1)
        circuit.mcx(list(range(search_qubits - 1)), search_qubits - 1)
        circuit.h(search_qubits - 1)
        circuit.x(range(search_qubits))
        circuit.h(range(search_qubits))

    circuit.measure(range(search_qubits), range(search_qubits))
    return circuit


def solve_maze(maze: Maze, shots: int = 1024) -> str:
    """Use Grover search to return one shortest path through the maze."""
    start, goal = validate_maze(maze)
    length = shortest_path_length(maze, start, goal)
    marked = valid_states(maze, start, goal, length)
    if not marked:
        raise RuntimeError("The oracle contains no solutions at the shortest length.")

    search_qubits = 2 * length
    circuit = build_grover_circuit(search_qubits, marked)
    simulator = AerSimulator()
    result = simulator.run(transpile(circuit, simulator), shots=shots).result()
    counts = result.get_counts()

    for measured_state, _ in sorted(counts.items(), key=lambda item: item[1], reverse=True):
        candidate = decode_moves(int(measured_state, 2), length)
        if walk(maze, start, candidate) == goal:
            return candidate
    raise RuntimeError("The simulator did not measure a valid path.")


def draw_solution(maze: Maze, path: str) -> str:
    """Render the maze with the solution path marked by '*'. """
    start, _ = validate_maze(maze)
    cells = [list(row) for row in maze]
    row, column = start
    for move in path:
        delta_row, delta_column = DELTAS[move]
        row += delta_row
        column += delta_column
        if cells[row][column] == ".":
            cells[row][column] = "*"
    return "\n".join("".join(row) for row in cells)


if __name__ == "__main__":
    example_maze = (
        "S...",
        ".##.",
        "...G",
        "....",
    )
    solution = solve_maze(example_maze)
    print(f"Quantum path: {solution}")
    print(draw_solution(example_maze, solution))