import onnxruntime as ort
import os

def inspect_onnx_model(model_path):
    """
    加载一个ONNX模型并打印其输入和输出节点的名称、形状。
    """
    print(f"🕵️‍♂️ 正在检查模型: {model_path}")

    if not os.path.exists(model_path):
        print(f"❌ 错误: 模型文件未找到于 '{model_path}'")
        print("   请确保您已经成功运行 'export_model.py' 并且该文件位于项目根目录。")
        return

    try:
        # 创建一个推理会话
        session = ort.InferenceSession(model_path)

        # 获取输入信息
        print("\n--- 模型的输入 (Inputs) ---")
        inputs = session.get_inputs()
        for i, input_node in enumerate(inputs):
            print(f"  [{i}] 名称 (Name): {input_node.name}")
            print(f"      形状 (Shape): {input_node.shape}")
            print(f"      类型 (Type): {input_node.type}")

        # 获取输出信息
        print("\n--- 模型的输出 (Outputs) ---")
        outputs = session.get_outputs()
        for i, output_node in enumerate(outputs):
            print(f"  [{i}] 名称 (Name): {output_node.name}")
            print(f"      形状 (Shape): {output_node.shape}")
            print(f"      类型 (Type): {output_node.type}")

        print("\n✅ 检查完成。")
        print("👉 请将上面列出的 '输出 (Outputs)' 名称更新到 'frontend/script.js' 文件中。")

    except Exception as e:
        print(f"\n❌ 加载或检查模型时发生错误: {e}")

if __name__ == "__main__":
    # 我们要检查的模型文件，它应该在项目根目录
    onnx_file_path = "mci_classifier.onnx"
    inspect_onnx_model(onnx_file_path) 