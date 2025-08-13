import torch 
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

import ebm 
import sudoku_dataset as dataset


###! Configuration
CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "epochs": 50,
    "batch_size": 64,
    "lr": 1e-4,
    "langevin_steps": 60,
    "langevin_step_size": 0.01,
    "l2_reg_strenght": 0.1,
    "model_save_path": "sudoku_ebm.pth",
    "buffer_size": 10_000,
    "replay_probability": 0.95
}


def generate_negative_samples(model, initial_board, puzzle_mask, steps, step_size):
    
    x = initial_board.clone().detach().requires_grad_(True)
    
    for _ in range(steps):
        energy = model(x).sum()
        grad, = torch.autograd.grad(energy, x)
        
        ### langevind dynamics
        x.data.add_(-0.5 * step_size * grad)
        x.data.add_(torch.randn_like(x) * np.sqrt(step_size))
        
        ### project back to a valid state
        x.data = torch.clamp(x.data, min=0)
        
        mask_expanded = puzzle_mask.unsqueeze(1).expand_as(x)
        x.data[mask_expanded] = initial_board.data[mask_expanded]
        
        x_flat = x.view(-1, 9, 81)
        x_probs = torch.nn.functional.softmax(x_flat, dim=1)
        x.data = x_probs.view_as(x)
        
    return x.detach()



def main():
    device = torch.device(CONFIG["device"])
    print(f"Using device: {device}")
    
    sudoku_data = dataset.FileDataset("dataset_train.npy")
    dataloader = DataLoader(sudoku_data, batch_size=CONFIG['batch_size'], shuffle=True)
    
    best_loss = float('inf')
    
    ### replay buffer 
    replay_buffer = torch.rand(CONFIG['buffer_size'], 9, 9, 9, device=device) 
    
    model = ebm.SudokuEBM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    
    for epoch in range(CONFIG["epochs"]):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}"):
            optimizer.zero_grad()
            
            puzzles_one_hot = batch['puzzle'].float().to(device)
            solutions_one_hot = batch['solution'].float().to(device)
            
            x_batch_size = puzzles_one_hot.shape[0]
            num_from_buffer = int(CONFIG["replay_probability"] * x_batch_size)
            num_fresh = x_batch_size - num_from_buffer
            
            ### get samples from buffer 
            buffer_indices = torch.randint(0, CONFIG["buffer_size"], (num_from_buffer, )).to(device)
            initial_board_buffer = replay_buffer[buffer_indices].clone().detach()
            
            if num_fresh > 0:
                puzzles_fresh = puzzles_one_hot[:num_fresh]
                puzzles_mask_fresh = puzzles_fresh.sum(dim=1) > 0 
                random_fill_fresh = torch.nn.functional.softmax(torch.randn_like(puzzles_fresh), dim=1)
                initial_board_fresh = torch.where(puzzles_mask_fresh.unsqueeze(1), puzzles_fresh, random_fill_fresh)
                
                initial_board = torch.cat([initial_board_buffer, initial_board_fresh], dim=0)
            else:
                intial_board = intial_board_buffer
            
            
            puzzle_mask = puzzles_one_hot.sum(dim=1) > 0 
            negative_samples = generate_negative_samples(model, initial_board, puzzle_mask, 
                                                        CONFIG['langevin_steps'], CONFIG['langevin_step_size'])
            
            negative_energy = model(negative_samples)
            positive_energy = model(solutions_one_hot)
            
            loss_cd = positive_energy.mean() - negative_energy.mean()
            loss_reg = CONFIG['l2_reg_strenght'] * ((positive_energy**2).mean() + (negative_energy**2).mean())
            
            loss = loss_cd + loss_reg
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            ## update the replay buffer
            # Check for corrupted samples before adding them to the buffer
            is_valid = torch.isfinite(negative_samples[:num_from_buffer]).all(dim=(1,2,3)).to(device)
            valid_samples = negative_samples[:num_from_buffer][is_valid]
            valid_indices = buffer_indices[is_valid]

            if valid_indices.numel() > 0:
                replay_buffer[valid_indices] = valid_samples.detach()
        
        avg_loss = total_loss  / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), CONFIG["model_save_path"])
            print(f"Model saved to {CONFIG['model_save_path']}")
            
if __name__=="__main__":
    main()
