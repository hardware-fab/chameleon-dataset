# Chameleon: A Dataset of Obfuscated Power Traces for Side-Channel Analysis 

The Chameleon is a dataset design for side-channel analysis of obfuscated power traces.
It containscontains real-world power traces collected from a 32-bit RISC-V System-on-Chip implementing four hiding countermeasures: Dynamic Frequency Scaling (DFS), Random Delay (RD), Morphing (MRP), and Chaffing (CHF). Each side-channel trace includes multiple cryptographic operations interleaved with general-purpose applications.

The Chameleon dataset is available [here](https://huggingface.co/datasets/hardware-fab/Chameleon) hosted by Hugging Face.

<div align="center">
   <img src="./images/chameleon_logo.png" alt="Chameleon Logo" width="150">
</div>

## Experiment reproducibility

To reproduce the side-channel attack used as dataset validation follow these steps:

1. Clone the repository with all submodules.

   ```
   git clone https://github.com/hardware-fab/chameleon-dataset.git --recursive
   ```

2. Download Chameleon from [Hugging Face](https://huggingface.co/datasets/hardware-fab/Chameleon).
3. Build the training, validation, and testing subsets using the `create_subsets.py` python script.

   ```
   python create_subsets.py -dd <chameleon_sub-datset_path> -od <output_path>  -a 100 
   ```

   Replace `<chameleon_sub-datset_path>` with the download path of Chameleon and `<output_path>` with the desired output location for the subsets files.  
   As output of the script you will find three files for each subset, i.e.,  `_windows.npy`, `_targets.npy`, and `_meta.npy`.
   Additionally, a `config.txt` lists the subsets' sizes and shapes.

4. Update the parameters of the model of choice in `Reinforcement-Learning-for-SCA/models` according to the subsets' sizes and shapes.

5. Run the `Reinforcement-Learning-for-SCA` tool for attacking the subsets.

   ```
   python -m metaqnn.main <model>
   ```
   Replace `<model>` with the model of choice. 

## Note

This repository is protected by copyright and licensed under the [Apache-2.0 license](https://github.com/hardware-fab/chameleon-dataset/blob/main/LICENSE) file.

© 2024 hardware-fab
