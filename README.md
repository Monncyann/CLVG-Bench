# <img src="https://github.com/Monncyann/CLVG-Bench/blob/main/figs/logo.png" width="140" align="left"> How Far Are Video Models from True Multimodal Reasoning?

> **Official repository for the paper: "How Far Are Video Models from True Multimodal Reasoning?"**.
>
> The repository is being continuously updated.

<p align="center">
  <a href="https://arxiv.org/abs/2604.19193" target="_blank"><img src="https://img.shields.io/badge/arXiv-2604.19193-red"></a>
  <a href="https://huggingface.co/datasets/Moncyan/CLVG-Bench" target="_blank"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-yellow"></a>
</p>

## 💡 About

Despite remarkable progress toward general-purpose video models, a critical question remains unanswered: how far are these models from achieving true multimodal reasoning? Existing benchmarks fail to address this question rigorously, as they remain constrained by straightforward task designs and fragmented evaluation metrics that neglect complex multimodal reasoning.

<img width="1000" height="900" alt="image" src="https://github.com/Monncyann/CLVG-Bench/blob/main/figs/CLVG-Bench_v5.png" />

## 🚀 Key Contributions

* **Context Learning in Video Generation**: We introduce CLVG-Bench, a novel evaluation framework that abstracts video generation tasks into the definition of context learning video generation, and systematically evaluates the capability of current video models in simulating and reasoning about real-world dynamics.

* **Adaptive Video Evaluator**: We propose the Adaptive Video Evaluator, a flexible evaluation framework designed for open-ended generation tasks. This evaluator dynamically adjusts based on a minimal set of human annotations, offering a versatile approach to assess tasks with varying contexts.

* **Limitation of SOTA Video Models**: Our study uncovers the limitations of current video models in multimodal reasoning. We advocate for a tighter integration of understanding and generation to enhance model performance in complex video generation scenarios.

## 🔔 CLVG-Bench Directory Structure
Here is the directory structure for CLVG-Bench along with descriptions for each folder and file type:
```python
CLVG-Bench/
├─ metadata.parquet
├─ Element_Editing/
│   ├─ Background_Modification/
│   │   ├─ 1/
│   │   ├─ 2/
│   │   └─ ...
│   ├─ Camera_Motion_Editing/
│   │   ├─ 1/
│   │   ├─ 2/
│   │   └─ ...
│   ├─ Dialogue_Editing/
│   │   ├─ 1/
│   │   └─ ...
│   ├─ Element_Addition/
│   │   └─ ...
│   ├─ Element_Removal/
│   │   └─ ...
│   ├─ Object_Replacement/
│   │   └─ ...
│   ├─ Subject_Editing/
│   │   └─ ...
│   └─ Vocal_Timbre_Editing/
│       └─ ...
├─ Partial_Reference/
│   ├─ Background_Reference/
│   │   └─ ...
│   ├─ Camera_Angle_Reference/
│   │   └─ ...
│   ├─ Camera_Motion_Reference/
│   │   └─ ...
│   ├─ Composition_Reference/
│   │   └─ ...
│   ├─ Dialogue_Reference/
│   │   └─ ...
│   ├─ Sound_Effects_Reference/
│   │   └─ ...
│   ├─ Style_Transfer/
│   │   └─ ...
│   ├─ Subject_Reference/
│   │   └─ ...
│   ├─ Transition_Style_Reference/
│   │   └─ ...
│   └─ Video_Style_Reference/
│       └─ ...
├─ Script_Continuation_Completion/
│   ├─ Backward_Continuation/
│   │   └─ ...
│   ├─ Forward_Continuation/
│   │   └─ ...
│   └─ Transition_Completion/
│       └─ ...
├─ Physical_Simulation/
│   ├─ Fluid_Dynamics&Micro-physics/
│   │   └─ ...
│   ├─ Material_Mechanics&Fracture/
│   │   └─ ...
│   ├─ Optics&Perspective/
│   │   └─ ...
│   ├─ Thermodynamics&Phase_Change/
│   │   └─ ...
│   ├─ Complex_Interaction&Environment/
│   │   └─ ...
│   ├─ Biological_Physics/
│   │   └─ ...
│   ├─ Environmental&Atmospheric_Physics/
│   │   └─ ...
│   ├─ Cooking&Chemical_Reactions/
│   │   └─ ...
│   ├─ Mechanics&Kinematics/
│   │   └─ ...
│   ├─ Destruction&High_Energy/
│   │   └─ ...
│   └─ Advanced_Soft_Body&Material/
│       └─ ...
├─ Logical_Reasoning/
│   ├─ Space&Pathfinding/
│   │   └─ ...
│   ├─ Sorting&Math/
│   │   └─ ...
│   ├─ Long-horizon&State_Changes/
│   │   └─ ...
│   └─ Games&Symbolic_Logic/
│       └─ ...
└─ Perception/
    ├─ Edge_Detection/
    │   └─ ...
    ├─ Element_Segmentation/
    │   └─ ...
    ├─ Keypoint_Localization/
    │   └─ ...
    ├─ Overall_Video_Enhancement/
    │   └─ ...
    ├─ Joint&Bundled Search/
    │   └─ ...
    └─ Local_Inpainting&Restoration/
        └─ ...
```

## ✍️ Citation
If you find this work help, please consider a citation:
```python
@misc{zhang2026farvideomodelstrue,
      title={How Far Are Video Models from True Multimodal Reasoning?}, 
      author={Xiaotian Zhang and Jianhui Wei and Yuan Wang and Jie Tan and Yichen Li and Yan Zhang and Ziyi Chen and Daoan Zhang and Dezhi YU and Wei Xu and Songtao Jiang and Zuozhu Liu},
      year={2026},
      eprint={2604.19193},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.19193}, 
}
```
