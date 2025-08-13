import os, argparse
import torch 
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

import ebm_improved as ebm_plus
import ebm
import sudoku_dataset as dataset



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



def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    sudoku_data = dataset.FileDataset(args.data_file)
    dataloader = DataLoader(sudoku_data, batch_size=args.batch_size, shuffle=True)
    
    best_loss = float('inf')
    
    ### replay buffer 
    replay_buffer = torch.rand(args.buffer_size, 9, 9, 9, device=device) 
    
    model = ebm_plus.SudokuEBM(hidden_dim=args.ebm_hidden_dim, num_blocks=args.ebm_num_blocks).to(device)
    # model = ebm.SudokuEBM()
    # total_params = sum(p.numel() for p in model.parameters())
    # print("params: ", total_params)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    for epoch in range(args.epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            optimizer.zero_grad()
            
            puzzles_one_hot = batch['puzzle'].float().to(device)
            solutions_one_hot = batch['solution'].float().to(device)
            
            x_batch_size = puzzles_one_hot.shape[0]
            num_from_buffer = int(args.replay_prob * x_batch_size)
            num_fresh = x_batch_size - num_from_buffer
            
            ### get samples from buffer 
            buffer_indices = torch.randint(0, args.buffer_size, (num_from_buffer, )).to(device)
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
                                                        args.langevin_steps, args.langevin_step_size)
            
            negative_energy = model(negative_samples)
            positive_energy = model(solutions_one_hot)
            
            loss_cd = positive_energy.mean() - negative_energy.mean()
            loss_reg = args.l2_reg_strenght * ((positive_energy**2).mean() + (negative_energy**2).mean())
            
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
            torch.save(model.state_dict(), args.save_file)
            print(f"Model saved to { args.save_file}")



if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--save_file",     type=str,   default="best_ebm_plus.pt")
    # p.add_argument("--save_file",     type=str,   default="/content/drive/MyDrive/ebm_runs/best_ebm_plus.pt")
    p.add_argument("--data_file",    type=str,   default="dataset_train.npy")
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--langevin_steps", type=int, default=60)
    p.add_argument("--langevin_step_size", type=float, default=0.01)
    p.add_argument("--l2_reg_strenght", type=float, default=0.1)
    p.add_argument("--buffer_size", type=int, default=10_000)
    p.add_argument("--replay_prob", type=float, default=0.95)
    p.add_argument("--ebm_hidden_dim", type=float, default=128)
    p.add_argument("--ebm_num_blocks", type=int, default=4)
    
    args = p.parse_args()
    main(args)
