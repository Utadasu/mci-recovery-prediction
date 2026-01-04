 
import os
import numpy as np
import torch
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import json

# 导入我们项目中的模块
from mci_conversion_prediction import MCIDataLoader, FeatureExtractor, set_seed, RECOMMENDED_CONFIG, RANDOM_SEED

# 尝试导入ONNX相关库，如果失败则提示用户安装
try:
    import skl2onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    import onnxmltools
except ImportError:
    print("❌ 错误: 'skl2onnx', 'onnxruntime', or 'onnxmltools' 未安装。")
    print("🔧 请运行: pip install skl2onnx onnxruntime onnxmltools")
    exit()

warnings.filterwarnings('ignore')

def main():
    """
    主函数: 训练并导出最终的XGBoost分类器为ONNX格式
    """
    print("🚀 开始最终模型训练与导出任务...")
    
    # 🎯 固定随机种子以确保结果可复现
    set_seed(RANDOM_SEED)
    
    # --- 1. 加载并准备全量数据 ---
    print("\n" + "="*20 + " 步骤 1: 加载全量数据 " + "="*20)
    data_dir = '/root/autodl-tmp/DATA_MCI'
    if not os.path.exists(data_dir):
        print(f"⚠️ 警告: 服务器数据路径 '{data_dir}' 在当前环境不可用。")
        print("   将尝试在本地 './' 目录寻找替代数据...")
        # 在本地开发时，可以将MCI数据放在项目根目录下的 'DATA_MCI' 文件夹中
        local_data_dir = './DATA_MCI'
        if os.path.exists(local_data_dir):
             data_dir = local_data_dir
        else:
            print(f"❌ 错误: 未能在 '{data_dir}' 或 '{local_data_dir}' 找到数据，程序终止。")
            return

    data_loader = MCIDataLoader(data_dir=data_dir)
    images, image_labels, image_patient_ids = data_loader.load_mci_images()
    texts, text_labels, text_patient_ids = data_loader.load_mci_text_data()

    if len(images) == 0:
        print("❌ 错误: 未能加载任何图像数据，程序终止。")
        return

    images, texts, labels, _ = data_loader.align_image_text_data(
        images, texts, image_labels, text_labels, image_patient_ids, text_patient_ids
    )

    if len(images) == 0:
        print("❌ 错误: 数据对齐后无可用样本，程序终止。")
        return

    print(f"✅ 数据加载完成，共 {len(images)} 个对齐样本。")

    # --- 2. 提取并融合特征 ---
    print("\n" + "="*20 + " 步骤 2: 提取并融合特征 " + "="*20)
    # 动态获取最佳模型路径
    from mci_conversion_prediction import EnhancedMCIClassifier
    best_model_path = EnhancedMCIClassifier(RECOMMENDED_CONFIG)._get_best_pretrained_model_path()
    
    feature_extractor = FeatureExtractor(
        model_path=best_model_path,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        batch_size=16
    )

    all_image_features = feature_extractor.extract_image_features(images)
    all_text_features = feature_extractor.extract_text_features(texts)

    # 使用最佳权重融合特征
    weight = RECOMMENDED_CONFIG.get('image_feature_weight', 0.8)
    fused_features = weight * all_image_features + (1 - weight) * all_text_features
    print(f"✅ 特征提取与融合完成，特征维度: {fused_features.shape}")

    # --- 3. 训练最终的分类器 ---
    print("\n" + "="*20 + " 步骤 3: 训练最终分类器 " + "="*20)
    
    # 🔥 最终解决方案：使用skl2onnx原生支持的LogisticRegression替换XGBoost，避免转换错误
    from sklearn.linear_model import LogisticRegression

    # 使用一个与我们之前调优相似的正则化强度
    # 注意: LogisticRegression的C是正则化强度的倒数
    regularization_strength = RECOMMENDED_CONFIG.get('regularization_strength', 10.0)
    final_classifier = LogisticRegression(
        C=1.0/regularization_strength,
        max_iter=RECOMMENDED_CONFIG.get('max_iter', 6000),
        solver='liblinear',
        random_state=RANDOM_SEED
    )

    # 创建一个包含标准化和分类的Pipeline
    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', final_classifier)
    ])

    # 在全部数据上训练Pipeline
    pipeline.fit(fused_features, labels)
    print("✅ 最终分类器Pipeline训练完成。")

    # --- 4. 提取组件并保存Scaler参数 ---
    print("\n" + "="*20 + " 步骤 4: 提取组件并保存Scaler " + "="*20)

    # 从训练好的Pipeline中提取scaler和classifier
    fitted_scaler = pipeline.named_steps['scaler']
    fitted_classifier = pipeline.named_steps['classifier']

    # 保存scaler的参数 (mean and scale) 到一个JSON文件
    scaler_params = {
        'mean': fitted_scaler.mean_.tolist(),
        'scale': fitted_scaler.scale_.tolist()
    }
    scaler_filename = "scaler_params.json"
    with open(scaler_filename, 'w') as f:
        json.dump(scaler_params, f, indent=4)
    print(f"✅ Scaler参数已保存到: {scaler_filename}")


    # --- 5. 转换为ONNX格式并保存 ---
    print("\n" + "="*20 + " 步骤 5: 转换分类器为ONNX格式 " + "="*20)
    
    # 定义ONNX模型的输入格式
    # [None, 512] 表示可以接受任意数量的样本，每个样本是512维的向量
    initial_type = [('float_input', FloatTensorType([None, fused_features.shape[1]]))]
    
    # 进行转换 - 这次只转换分类器
    try:
        # 目标opset是解决某些转换器问题的常用方法
        target_opset = 12
        onnx_model = convert_sklearn(
            fitted_classifier, 
            initial_types=initial_type, 
            target_opset={'': target_opset}
        )
        
        # 保存模型到文件
        onnx_filename = "mci_classifier.onnx"
        with open(onnx_filename, "wb") as f:
            f.write(onnx_model.SerializeToString())
            
        print(f"🎉 成功！模型已导出为: {onnx_filename}")
        print("   这个文件现在可以用于前端网页部署了。")

    except Exception as e:
        print(f"❌ 导出ONNX模型时发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 