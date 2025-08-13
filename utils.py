import torch 
import numpy as np


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
