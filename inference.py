import torch
import numpy as np
from tqdm import tqdm

import ebm  
import sudoku_dataset as dataset 

###! Configuration for Inference
CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_path": "sudoku_ebm.pth",  # Path to your saved model
    "test_data_path": "dataset_test.npy", # Path to your dataset
    "solver_steps": 2000,          # More steps for higher quality solutions
    "solver_step_size": 0.1,
}

def generate_solution(model, initial_board, puzzle_mask, steps, step_size):
    """
    Uses Langevin dynamics to find a low-energy solution starting from a puzzle.
    """

    x = initial_board.clone().detach().requires_grad_(True)
    model.eval()

    for i in tqdm(range(steps), desc="Solving Puzzle"):
        noise_scale = np.sqrt(step_size) * (1 - i / steps) # Optional: Annealing
        
        energy = model(x).sum()
        grad, = torch.autograd.grad(energy, x)
        
        # Langevin dynamics update
        x.data.add_(-0.5 * step_size * grad) # Gradient descent
        x.data.add_(torch.randn_like(x) * noise_scale) # Add noise
        
        x.data = torch.clamp(x.data, min=0) # Probabilities can't be negative
        mask_expanded = puzzle_mask.unsqueeze(1).expand_as(x)
        x.data[mask_expanded] = initial_board.data[mask_expanded]
        
        x_flat = x.view(-1, 9, 81)
        x_probs = torch.nn.functional.softmax(x_flat, dim=1)
        x.data = x_probs.view_as(x)
        
    return x.detach()

def convert_to_board(one_hot_tensor):
    """Converts a (1, 9, 81) one-hot tensor to a 9x9 numpy board."""
    if one_hot_tensor.dim() > 3: # Handles batch dimension
        one_hot_tensor = one_hot_tensor.squeeze(0)
    
    digits = torch.argmax(one_hot_tensor, dim=0) # Get the index of the '1' for each cell
    board = digits.view(9, 9).cpu().numpy()
    return board + 1 # Convert from 0-8 indices to 1-9 digits


def print_board(board_numpy, title=""):
    """Prints a 9x9 numpy board in a readable format."""
    print(title)
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("- - - - - - - - - - -")
        for j in range(9):
            if j % 3 == 0 and j != 0:
                print("| ", end="")
            digit = board_numpy[i, j]
            print(f"{digit if digit != 0 else '.'} ", end="")
        print()
    print("-" * 25)


def main():
    device = torch.device(CONFIG["device"])
    print(f"Using device: {device}")

    # 1. Load the trained model
    model = ebm.SudokuEBM()
    model.load_state_dict(torch.load(CONFIG["model_path"], map_location=device))
    model.to(device)
    model.eval() # IMPORTANT: Set model to evaluation mode


    sudoku_data = dataset.FileDataset(CONFIG["test_data_path"])
    sample = sudoku_data[0] 
    
    # Reshape and move to device, add a batch dimension of 1
    puzzle_one_hot = torch.from_numpy(sample['puzzle']).unsqueeze(0).float().to(device)
    solution_one_hot = torch.from_numpy(sample['solution']).unsqueeze(0).float().to(device)

    # 3. Prepare the initial state for the solver
    puzzle_mask = puzzle_one_hot.sum(dim=1) > 0
    random_fill = torch.nn.functional.softmax(torch.randn_like(puzzle_one_hot), dim=1)
    initial_board = torch.where(puzzle_mask.unsqueeze(1), puzzle_one_hot, random_fill)
    
    # 4. Generate the solution
    solved_one_hot = generate_solution(model, initial_board, puzzle_mask.squeeze(0),
                                    CONFIG["solver_steps"], CONFIG["solver_step_size"])

    # 5. Convert and display the results
    puzzle_board = convert_to_board(puzzle_one_hot)
    solution_board = convert_to_board(solution_one_hot)
    generated_board = convert_to_board(solved_one_hot)

    print_board(puzzle_board, "Original Puzzle:")
    print_board(generated_board, "Model's Generated Solution:")
    print_board(solution_board, "Ground Truth Solution:")

    # Check if the solution is correct
    if np.array_equal(generated_board, solution_board):
        print("\n Success! The generated solution is correct.")
    else:
        print("\n Failure. The generated solution is incorrect.")


if __name__ == "__main__":
    main()