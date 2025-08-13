import numpy as np
from torch.utils.data import Dataset

class SudokuGenerator:
    """
    Generates full 9x9 Sudoku solutions via backtracking,
    then creates puzzles by removing cells.
    """

    def __init__(self):
        # Pre-allocate an empty grid
        self.grid = np.zeros((9, 9), dtype=int)

    def _is_safe(self, r, c, val):
        """Check row, col, and 3×3 box for safety."""
        # Row & column
        if val in self.grid[r, :] or val in self.grid[:, c]:
            return False
        # 3×3 block
        br, bc = 3 * (r // 3), 3 * (c // 3)
        if val in self.grid[br:br + 3, bc:bc + 3]:
            return False
        return True


    def _fill_cell(self, idx=0):
        """Recursive backtracking fill from cell idx = 0..80."""
        if idx == 81:
            return True  # all filled
        r, c = divmod(idx, 9)
        # Shuffle trial order for randomness
        nums = np.random.permutation(np.arange(1, 10))
        for val in nums:
            if self._is_safe(r, c, val):
                self.grid[r, c] = val
                if self._fill_cell(idx + 1):
                    return True
                # backtrack
                self.grid[r, c] = 0
        return False  # trigger backtracking


    def generate_solution(self):
        """Public method to get a full valid solution grid."""
        self.grid.fill(0)
        success = self._fill_cell(0)
        assert success, "Backtracking failed to generate a solution"
        return self.grid.copy()


    def make_puzzle(self, holes=40):
        """
        Start from a fresh solution, then remove `holes` cells at random.
        Returns (puzzle, solution).
        """
        sol = self.generate_solution()
        puzzle = sol.copy()
        # choose unique hole positions
        idx = np.random.choice(81, size=holes, replace=False)
        puzzle.flat[idx] = 0
        return puzzle, sol



class FileDataset(Dataset):
    """
    PyTorch Dataset that yields (masked one-hot, solution one-hot) pairs.
    Shapes: both are [9, 9, 9] (channel, row, col).
    """

    def __init__(self, file):
        self.data = np.load(file, allow_pickle=True)
        print(self.data.shape)
        
        
    def __len__(self):
        return len(self.data)


    def _one_hot(self, grid):
        """
        Convert a [9×9] int grid (0 for blank, 1–9 filled) into one-hot:
        → shape [9,9,9], dtype float32, channels first.
        """
        one_hot = np.zeros((9, 9, 9), dtype=np.float32)
        # for each cell, if value d>0, set channel d-1 = 1
        nonzero = grid > 0
        rows, cols = np.where(nonzero)
        digits = grid[nonzero] - 1
        one_hot[rows, cols, digits] = 1.0
        # transpose to channels-first
        return one_hot.transpose(2, 0, 1)
    
    
    def __getitem__(self, idx):
        puzzle, solution = self.data[idx]
        x = self._one_hot(puzzle)       # masked input
        y = self._one_hot(solution)     # ground-truth solution
        return {"puzzle":x, "solution":y}
