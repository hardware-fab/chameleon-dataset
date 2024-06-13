# Chameleon: A Dataset of Obfuscated Power Traces for Side-Channel Analysis 

The Chameleon is a dataset designed for side-channel analysis of obfuscated power traces.
It contains real-world power traces collected from a 32-bit RISC-V System-on-Chip implementing four hiding countermeasures: Dynamic Frequency Scaling (DFS), Random Delay (RD), Morphing (MRP), and Chaffing (CHF). Each side-channel trace includes multiple cryptographic operations interleaved with general-purpose applications.

The Chameleon dataset is available on 🤗 [Hugging Face](https://huggingface.co/datasets/hardware-fab/Chameleon).

<div align="center">
   <img src="./images/chameleon_logo.png" alt="Chameleon Logo" width="150">
</div>

## Experiment reproducibility

To reproduce the side-channel attack used as dataset validation follow these steps:

1. Clone the repository with all submodules.

   ```bash
   git clone https://github.com/hardware-fab/chameleon-dataset.git --recursive
   ```

2. Download Chameleon from 🤗 [Hugging Face](https://huggingface.co/datasets/hardware-fab/Chameleon).  
   Full dataset:
   ```python
   from datasets import load_dataset
   
   dataset = load_dataset("hardware-fab/Chameleon")
   ```
   
   One sub-dataset of choice:
   ```python
   from datasets import load_dataset
   
   sub_dataset = load_dataset("hardware-fab/Chameleon", '<sub_dataset>')
   ```
   Replace `<sub_dataset>` with `DFS`, `RD`, `MRP`, or `CHF`.

4. Build the training, validation, and testing subsets using the `create_subsets.py` python script.

   ```
   python create_subsets.py -dd <chameleon_sub-datset_path> -od <output_path>  -a 100 
   ```

   Replace `<chameleon_sub-datset_path>` with the download path of Chameleon and `<output_path>` with the desired output location for the subsets files.  
   As output of the script, you will find three files for each subset, i.e.,  `_windows.npy`, `_targets.npy`, and `_meta.npy`.
   Additionally, a `config.txt` lists the subsets' sizes and shapes.

5. Update the parameters of the model of choice in `Reinforcement-Learning-for-SCA/models` according to the subsets' sizes and shapes.

6. Run the `Reinforcement-Learning-for-SCA` tool for attacking the subsets.

   ```
   python -m metaqnn.main <model>
   ```
   Replace `<model>` with the model of choice. 

## Note

This repository is protected by copyright and licensed under the [Apache-2.0 license](https://github.com/hardware-fab/chameleon-dataset/blob/main/LICENSE) file.

© 2024 hardware-fab
