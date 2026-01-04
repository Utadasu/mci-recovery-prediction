# 项目文件清理清单

## 1. 保留文件清单

### 1.1 核心源代码文件
- `main.py` - 项目主入口文件
- `mci_recovery_prediction.py` - MCI恢复预测核心实现
- `adversarial_contrastive_learning.py` - 对抗性对比学习实现
- `optimized_models.py` - 优化的模型定义
- `data_utils.py` - 数据处理工具
- `dataset.py` - 数据集定义
- `text_encoder_module.py` - 文本编码器模块
- `losses.py` - 损失函数定义
- `advanced_trainer.py` - 高级训练器
- `optimized_contrastive_learning.py` - 优化的对比学习实现

### 1.2 训练和运行脚本
- `run_contrastive_image_encoder.py` - 图像编码器训练脚本
- `run_improved_model.py` - 改进模型训练脚本
- `run_training.py` - 核心训练脚本
- `run_with_contrastive_encoder.py` - 使用对比编码器运行
- `train_contrastive.py` - 对比学习训练脚本
- `train_image_encoder_for_contrastive.py` - 训练对比学习图像编码器

### 1.3 配置和依赖文件
- `requirements.txt` - 核心依赖清单
- `PROJECT_RULES.md` - 项目规则体系
- `README.md` - 项目说明文档

### 1.4 必要的辅助文件
- `early_fusion.py` - 早期融合实现
- `early_fusion_fixed.py` - 修复的早期融合实现
- `quick_finetune.py` - 快速微调脚本
- `deep_architecture_finetune.py` - 深度架构微调

## 2. 删除文件清单

### 2.1 旧的MCI转化预测相关文件
- `mci_conversion_prediction.py` (已删除)
- `run_optimized_mci_prediction.py` (已删除)
- `run_domain_adaptation.py` (已删除)

### 2.2 测试和验证文件
- `test.py`
- `test_contrastive_training.py`
- `test_enhanced_model.py`
- `test_improved_early_fusion.py`
- `test_memory_fix.py`
- `verify_contrastive_model.py`
- `test_mci_adaptation.py` (已删除)
- `test_mci_standard_mode.py` (已删除)

### 2.3 冗余的训练脚本
- `run_ablation_domain_adaptation.py`
- `run_ablation_studies.py`
- `run_ad_alignment_fix.py`
- `run_tta.py`
- `train_smart_downsample.py`
- `run_ablation_no_adversarial.py` (已删除)
- `run_ablation_no_cognitive.py` (已删除)

### 2.4 临时文件和缓存文件
- `__pycache__/` - Python编译缓存目录
- `.idea/` - IDE配置目录
- `.trae/` - Trae IDE配置目录

### 2.5 文档文件
- `ImprovedResNetCBAM3D_Architecture.md`
- `ImprovedResNetCBAM3D_完整架构文档.md`
- `ImprovedResNetCBAM3D_数据流更新.md`
- `ImprovedResNetCBAM3D_数据流详解.md`
- `ImprovedResNetCBAM3D_详细架构图.md`
- `PROJECT_LOGIC_AND_FRAMEWORK.md`
- `README_HierarchicalSwin.md`
- `使用增强版模型指南.md`
- `参考文献.docx`
- `对比学习代码整体思路`
- `对比学习图像编码器集成总结.md`
- `数据尺寸问题解答.md`
- `数据预处理流程图.md`
- `硬件配置与模型适配分析.md`
- `论文修改_A部分_损失函数.md`

### 2.6 前端相关文件
- `frontend/` - 前端目录

### 2.7 结果文件
- `results/` - 结果目录

### 2.8 其他冗余文件
- `ADCN数据集服务器路径及信息.txt`
- `MCI_Pre_Results.xlsx`
- `MCI数据服务器路径及信息.txt`
- `MakePic.py`
- `MakePic_Figure1.py`
- `check_files.py`
- `complete_project_architecture.html`
- `creative_architecture_diagram.html`
- `draw_image_encoder_architecture.html`
- `full_pipeline_architecture.html`
- `hierarchical_swin_results.json`
- `image_encoder_architecture.html`
- `install_dependencies.sh`
- `install_quick.sh`
- `medical_device_qa.txt`
- `multimodal_encoder_diagram.html`
- `my-cursor-rules.mdc`
- `requirements_quick.txt`
- `similarity_matrix_generator.py`

## 3. 删除说明

### 3.1 旧的MCI转化预测文件
这些文件已经被`mci_recovery_prediction.py`替代，功能已经集成到新的实现中。

### 3.2 测试和验证文件
这些文件用于开发和调试，不是项目运行的必要文件，可以在需要时重新生成。

### 3.3 冗余的训练脚本
这些脚本是特定实验或 ablation 研究的产物，不是项目的核心功能。

### 3.4 临时文件和缓存文件
这些文件是系统或IDE生成的临时文件，不影响项目的运行。

### 3.5 文档文件
保留了核心的`README.md`和`PROJECT_RULES.md`，其他文档文件可以在需要时查阅或重新生成。

### 3.6 前端相关文件
如果前端不是项目的核心功能，可以删除，或者根据项目需求保留。

### 3.7 结果文件
结果文件可以在运行时重新生成，删除后可以节省空间。

## 4. 清理后项目结构

```
项目根目录:
├── main.py                           # 项目主入口
├── mci_recovery_prediction.py        # MCI恢复预测核心实现
├── adversarial_contrastive_learning.py # 对比学习实现
├── optimized_models.py               # 优化的模型定义
├── data_utils.py                     # 数据处理工具
├── dataset.py                        # 数据集定义
├── text_encoder_module.py            # 文本编码器
├── losses.py                         # 损失函数
├── advanced_trainer.py               # 高级训练器
├── optimized_contrastive_learning.py # 优化的对比学习
├── early_fusion.py                   # 早期融合
├── early_fusion_fixed.py             # 修复的早期融合
├── quick_finetune.py                 # 快速微调
├── deep_architecture_finetune.py     # 深度架构微调
├── run_contrastive_image_encoder.py  # 图像编码器训练
├── run_improved_model.py             # 改进模型训练
├── run_training.py                   # 核心训练脚本
├── run_with_contrastive_encoder.py   # 使用对比编码器运行
├── train_contrastive.py              # 对比学习训练
├── train_image_encoder_for_contrastive.py # 训练对比学习图像编码器
├── requirements.txt                  # 核心依赖
├── PROJECT_RULES.md                  # 项目规则体系
└── README.md                         # 项目说明文档
```