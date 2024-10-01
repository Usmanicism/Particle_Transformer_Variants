# Particle Transformer Variants

This repository contains modifications to the [Particle Transformer](https://github.com/jet-universe/particle_transformer) project. It introduces two new variants of the Particle Transformer model for enhanced performance in jet tagging tasks.

## New Variants
1. **MultiAxis Particle Transformer**
2. **Swin Particle Transformer (SWAPT)**

## How to Use

### 1. Initial Setup
First, clone the following repositories and place them in the same folder:
- [Particle Transformer](https://github.com/jet-universe/particle_transformer)
- [weaver-core](https://github.com/hqucms/weaver-core)

### 2. Replacing and Adding Files
Download the files from this repository and place them as follows:

- Replace `train_JetClass.sh` in the `Particle_Transformer` folder.
- Place `example_MultiAxisParticleTransformer.py` and `example_SwinParticleTransformer.py` in the `Particle_Transformer/networks` folder.
- Place `MultiAxisParticleTransformer.py` and `SwinParticleTransformer.py` in the `weaver-core/weaver/nn/model` folder.

### 3. Running the Training
To run the training, use the following command:

'''bash
./train_JetClass.sh [SWAPT|ParMat|ParT|PN|PFN|PCNN] [kin|kinpid|full] ...
'''

### Additional arguments can be passed directly to the weaver command, such as '#--batch-size', '#--start-lr', '#--gpus'. These arguments will override existing parameters in train_JetClass.sh.

## Citation
### If you use the Particle Transformer code and/or the JetClass dataset in your work, please cite the following paper:
'''
@misc{usman2024particlemultiaxistransformerjet,
      title={Particle Multi-Axis Transformer for Jet Tagging}, 
      author={Muhammad Usman and M Husnain Shahid and Maheen Ejaz and Ummay Hani and Nayab Fatima and Abdul Rehman Khan and Asifullah Khan and Nasir Majid Mirza},
      year={2024},
      eprint={2406.06638},
      archivePrefix={arXiv},
      primaryClass={hep-ph},
      url={https://arxiv.org/abs/2406.06638}, 
}
'''
