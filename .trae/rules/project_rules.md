# 项目规则体系

## 1. 项目概述
本项目是基于深度学习的MCI（轻度认知障碍）恢复预测系统，用于预测MCI患者是否恢复为认知正常（CN），涉及医学图像处理、深度学习、特征提取和分类算法等技术领域。

### 1.1 技术栈
- **主要语言**: Python 3.12
- **深度学习框架**: PyTorch 2.5.1
- **医学图像处理**: nibabel, scipy.ndimage
- **机器学习算法**: scikit-learn, XGBoost
- **版本控制**: Git

### 1.2 团队规模
3-10人开发团队

### 1.3 服务器配置
- **硬件**: AMD EPYC 9654 (16核)、60GB内存、vGPU-32GB、NVIDIA驱动550.120、CUDA 12.4
- **软件**: Ubuntu 22.04、Python 3.12、PyTorch 2.5.1

## 2. 代码规范

### 2.1 Python代码规范
- 遵循PEP 8标准，4个空格缩进
- 单行代码≤100字符
- 导入顺序：标准库→第三方库→本地模块，组间空一行
- 类定义空两行，函数定义空一行
- 使用清晰注释和Google风格文档字符串

### 2.2 命名规范
- 变量/函数：snake_case（如`image_features`）
- 类名：CamelCase（如`MCIDataLoader`）
- 常量：全大写+下划线（如`RANDOM_SEED`）
- 模块名：小写+下划线（如`data_utils.py`）

### 2.3 代码质量要求
- 避免重复代码，封装为函数/类
- 单个函数≤50行，复杂功能拆分
- 使用类型提示
- 避免魔法数字，定义为常量
- 合理使用try-except，避免裸露except

### 2.4 执行与检查
- 代码风格：`flake8 --max-line-length=100 --ignore=E203,W503`
- 类型检查：`mypy --ignore-missing-imports --strict-optional`

## 3. 开发流程
1. **需求分析**：收集需求→评审→编写需求文档
2. **设计**：架构设计→详细设计→设计评审
3. **开发**：分配任务→代码开发→单元测试→集成测试
4. **测试**：功能测试→性能测试→稳定性测试→编写测试报告
5. **部署**：配置环境→编写部署脚本→设置监控
6. **维护**：bug修复→功能迭代→文档更新

## 4. 版本控制规则

### 4.1 Git分支管理
- **main**：稳定版本，仅接受develop合并
- **develop**：集成开发，接受feature合并
- **feature/***：新功能开发
- **hotfix/***：生产bug修复
- **release/***：版本发布准备

### 4.2 分支操作
- 从develop创建feature分支，从main创建hotfix分支
- 使用Pull Request（PR）合并，禁止直接push到main/develop
- 功能完成后删除feature分支

### 4.3 提交信息格式
- 标题：`<类型>: <简短描述>`（如`feat: 添加MCI恢复预测功能`）
- 类型：feat（新功能）、fix（修复）、docs（文档）、style（风格）、refactor（重构）、test（测试）、chore（构建）
- 正文：说明变更内容，可加列表
- 脚注：关联issue（如`Resolves #123`）

## 5. 文档编写规范
- 文档类型：需求、设计、用户、开发、API文档
- 格式：Markdown，清晰标题层级，包含目录，使用图表说明复杂概念
- 更新要求：功能变更同步更新，与代码一致，定期审查
- 存储：项目根目录`docs/`文件夹，重要文档在README.md链接

## 6. 测试标准
- 测试类型：单元测试、集成测试、功能测试、性能测试、稳定性测试
- 覆盖率：核心功能≥80%，使用`pytest-cov`统计
- 测试用例：覆盖正常/异常情况，可重复，清晰断言，有意义名称
- 执行：使用`pytest`，提交前运行相关测试，合并前运行所有测试，集成到CI/CD

## 7. 代码审查流程
- 审查范围：代码、文档、配置文件变更
- 人员：每个PR至少2名相关领域审查人员，避免自审
- 内容：代码规范、功能实现、潜在问题、可读性、测试充分性、文档完整性
- 流程：创建PR→审查→提出意见→修改→再次审查→通过后合并
- 时间要求：审查24小时内完成（复杂PR≤48小时），开发者24小时内处理意见
- 标准：至少2人批准，无未解决意见

## 8. 项目管理规则
- 任务管理：使用GitHub Issues/Jira，明确描述、优先级和截止日期，及时更新状态
- 会议规则：定期召开，提前准备议程，时长≤30分钟，发送会议纪要
- 进度报告：开发者每周更新，负责人定期通报，及时沟通问题
- 执行：使用项目管理工具，定期站会，建立沟通群

## 9. 违规处理
- **轻微违规**（代码风格、提交格式等）：口头提醒，立即整改
- **中度违规**（分支管理、测试要求等）：书面警告，指定时间整改
- **严重违规**（开发流程、审查流程等）：暂停开发权限，培训后恢复
- **重复违规**：通报批评，影响绩效考核
- 申诉机制：开发者可向项目负责人申诉，3个工作日内回复

## 10. 附则
- 规则定期审查更新，需团队讨论通过，更新后通知所有成员
- 自发布之日起生效，解释权归项目团队所有

## 11. MCI_DATA 数据规范

### 11.1 目录结构
- **服务器根目录**: `/root/autodl-tmp/MCI_DATA/`
- **AD**: `totalAD/ADfinal{MOD}/` (MOD: CSF, GRAY, WHITE) → 完整路径: `/root/autodl-tmp/MCI_DATA/totalAD/ADfinal{MOD}/`
- **CN**: `totalCN/CNfinal{MOD}/` (MOD: CSF, GRAY, WHITE) → 完整路径: `/root/autodl-tmp/MCI_DATA/totalCN/CNfinal{MOD}/`
- **MCIc**: `totalMCIc/total{MOD}/` (MOD: CSF, GRAY, WHITE) → 完整路径: `/root/autodl-tmp/MCI_DATA/totalMCIc/total{MOD}/`
- **MCInc**: `totalMCInc/total{MOD}/` (MOD: CSF, GRAY, WHITE) → 完整路径: `/root/autodl-tmp/MCI_DATA/totalMCInc/total{MOD}/`

### 11.2 文件分布
| 类型 | CSF | GRAY | WHITE | 总计 |
|------|-----|------|-------|------|
| AD | 414 | 414 | 414 | 1242 |
| CN | 414 | 414 | 414 | 1242 |
| MCIc | 100 | 100 | 100 | 300 |
| MCInc | 100 | 100 | 100 | 300 |
| 总计 | 1028 | 1028 | 1028 | 3084 |

### 11.3 命名模式

#### AD/CN 命名模式
- **格式**: `mwp<MOD>MRI_<site>_S_<subject>_<date>_<time>.<version>.nii`
- **模态标识**: `mwp1`=GRAY, `mwp2`=WHITE, `mwp3`=CSF
- **示例**: `mwp3MRI_002_S_0619_2006-06-01_19_48_28.0.nii`

#### MCIc/MCInc 命名模式
- **格式**: `<site>_S_<subject>_<MOD>.nii`
- **模态标识**: `mwp1T1`=GRAY, `mwp2T1`=WHITE, `mwp3T1`=CSF
- **示例**: `002_S_0729_mwp3T1.nii`

---
**创建日期**：2025年12月23日
**版本**：v1.0.0