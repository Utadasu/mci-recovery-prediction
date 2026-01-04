#!/bin/bash

# 阿尔茨海默病MRI分类系统 - 依赖安装脚本（修复版）
# 作者: AI Assistant
# 日期: 2025-12-23

# 定义颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 输出带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查网络连接
check_network() {
    print_info "检查网络连接..."
    if ping -c 1 -W 3 baidu.com > /dev/null 2>&1; then
        print_success "网络连接正常"
        return 0
    else
        print_info "Ping百度失败，尝试使用curl检查..."
        if curl -s -m 3 https://baidu.com > /dev/null; then
            print_success "网络连接正常（使用curl验证）"
            return 0
        else
            print_warning "网络连接检测失败，但仍将继续安装"
            return 0
        fi
    fi
}

# 检查Python版本
check_python() {
    print_info "检查Python版本..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | awk '{print $2}')
        print_info "Python版本: $PYTHON_VERSION"
        # 检查Python版本是否满足要求 (>=3.8)
        if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"; then
            print_success "Python版本满足要求 (>=3.8)"
            return 0
        else
            print_error "Python版本不满足要求，请升级到Python 3.8+"
            return 1
        fi
    else
        print_error "未找到Python 3，请安装Python 3.8+"
        return 1
    fi
}

# 检查pip版本
check_pip() {
    print_info "检查pip版本..."
    if command -v pip3 &> /dev/null; then
        PIP_VERSION=$(pip3 --version | awk '{print $2}')
        print_info "pip版本: $PIP_VERSION"
        # 检查pip版本是否满足要求 (>=22.0.0)
        if python3 -c "import sys; import pip; sys.exit(0 if pip.__version__ >= '22.0.0' else 1)"; then
            print_success "pip版本满足要求 (>=22.0.0)"
            return 0
        else
            print_warning "pip版本较旧，正在升级pip..."
            pip3 install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple/
            return $?
        fi
    else
        print_error "未找到pip3，请安装或升级pip"
        return 1
    fi
}

# 安装依赖
install_dependencies() {
    print_info "开始安装依赖..."
    
    # 使用pip安装requirements.txt中的依赖，使用清华镜像源
    pip3 install --no-cache-dir --timeout=120 -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
    
    if [ $? -eq 0 ]; then
        print_success "依赖安装成功"
        return 0
    else
        print_error "依赖安装失败，请检查错误信息"
        return 1
    fi
}

# 验证依赖安装
verify_dependencies() {
    print_info "验证依赖安装..."
    
    # 使用pip check验证依赖
    pip3 check
    
    if [ $? -eq 0 ]; then
        print_success "依赖验证成功，所有依赖项版本兼容"
        return 0
    else
        print_warning "依赖验证发现问题，可能存在版本不兼容"
        return 1
    fi
}

# 显示已安装的依赖版本
show_installed_deps() {
    print_info "显示已安装的核心依赖版本..."
    
    # 列出核心依赖的版本
    pip3 list | grep -E "torch|numpy|pandas|scipy|scikit-learn|nibabel|transformers"
    
    print_success "核心依赖版本信息显示完成"
}

# 主函数
main() {
    print_info "开始阿尔茨海默病MRI分类系统依赖安装..."
    
    # 检查网络连接
    check_network
    
    # 检查Python版本
    check_python
    if [ $? -ne 0 ]; then
        exit 1
    fi
    
    # 检查pip版本
    check_pip
    if [ $? -ne 0 ]; then
        exit 1
    fi
    
    # 安装依赖
    install_dependencies
    if [ $? -ne 0 ]; then
        exit 1
    fi
    
    # 验证依赖安装
    verify_dependencies
    
    # 显示已安装的依赖版本
    show_installed_deps
    
    print_success "依赖安装流程完成！"
    print_info "您可以通过运行 'python3 main.py' 启动项目"
}

# 执行主函数
main