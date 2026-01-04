import os
import re
from collections import defaultdict

root_dir = 'D:\\MCI_DATA'

# 定义要统计的类型和模态
categories = ['AD', 'CN', 'MCIc', 'MCInc']
modalities = {'CSF', 'GRAY', 'WHITE'}

# 存储统计结果
directory_structure = {}
file_counts = defaultdict(lambda: defaultdict(int))
naming_patterns = defaultdict(list)

# 遍历根目录
for category in categories:
    if category in ['AD', 'CN']:
        cat_dir = os.path.join(root_dir, f'total{category}')
        if os.path.exists(cat_dir):
            directory_structure[category] = {}
            for mod in modalities:
                mod_dir = os.path.join(cat_dir, f'{category}final{mod}')
                if os.path.exists(mod_dir):
                    files = os.listdir(mod_dir)
                    directory_structure[category][mod] = len(files)
                    file_counts[category][mod] += len(files)
                    # 收集文件名模式
                    for file in files:
                        naming_patterns[(category, mod)].append(file)
    elif category in ['MCIc', 'MCInc']:
        cat_dir = os.path.join(root_dir, f'total{category}')
        if os.path.exists(cat_dir):
            directory_structure[category] = {}
            for mod in modalities:
                mod_dir = os.path.join(cat_dir, f'total{mod}')
                if os.path.exists(mod_dir):
                    files = os.listdir(mod_dir)
                    directory_structure[category][mod] = len(files)
                    file_counts[category][mod] += len(files)
                    # 收集文件名模式
                    for file in files:
                        naming_patterns[(category, mod)].append(file)

# 分析命名模式
def analyze_patterns(files, category, modality):
    patterns = {}
    if not files:
        return patterns
    
    # 分析AD和CN的文件名模式
    if category in ['AD', 'CN']:
        # 格式: mwp3MRI_002_S_0619_2006-06-01_19_48_28.0.nii
        pattern = re.compile(r'^(mwp\d+MRI)_(\d{3})_(S)_(\d{4})_(\d{4}-\d{2}-\d{2})_(\d{2})_(\d{2})_(\d{2})\.(\d+)\.nii$')
        patterns['format'] = 'mwp<mod>MRI_<site>_<S>_<subject>_<date>_<hour>_<minute>_<second>.<version>.nii'
        patterns['modality_indicator'] = {'mwp1': 'GRAY', 'mwp2': 'WHITE', 'mwp3': 'CSF'}
    # 分析MCI的文件名模式
    elif category in ['MCIc', 'MCInc']:
        # 格式: 002_S_0729_mwp3T1.nii
        pattern = re.compile(r'^(\d{3})_(S)_(\d{4})_(mwp\d+T1)\.nii$')
        patterns['format'] = '<site>_<S>_<subject>_<modality>.nii'
        patterns['modality_indicator'] = {'mwp1T1': 'GRAY', 'mwp2T1': 'WHITE', 'mwp3T1': 'CSF'}
    
    return patterns

# 生成报告
print('=== MCI_DATA 目录结构和文件命名模式分析报告 ===\n')

print('1. 目录层次结构\n')
for category, mods in directory_structure.items():
    print(f'{category}/')
    for mod, count in mods.items():
        if category in ['AD', 'CN']:
            print(f'  └── {category}final{mod}/: {count} files')
        else:
            print(f'  └── total{mod}/: {count} files')
    print()

print('2. 文件分布统计\n')
print('| 类型 | CSF | GRAY | WHITE | 总计 |')
print('|------|-----|------|-------|------|')
total_all = 0
for category in categories:
    csf = file_counts[category]['CSF']
    gray = file_counts[category]['GRAY']
    white = file_counts[category]['WHITE']
    total = csf + gray + white
    total_all += total
    print(f'| {category} | {csf} | {gray} | {white} | {total} |')
print(f'| 总计 | {sum(file_counts[c]["CSF"] for c in categories)} | {sum(file_counts[c]["GRAY"] for c in categories)} | {sum(file_counts[c]["WHITE"] for c in categories)} | {total_all} |')
print()

print('3. 文件名命名模式\n')
for category in categories:
    print(f'{category} 文件命名模式:')
    for mod in modalities:
        files = naming_patterns[(category, mod)]
        if files:
            patterns = analyze_patterns(files, category, mod)
            print(f'  {mod} 模态:')
            print(f'    格式: {patterns.get("format", "未知")}')
            print(f'    模态标识: {patterns.get("modality_indicator", "未知")}')
            # 显示示例
            print(f'    示例: {files[0]}')
    print()

print('4. 异常和不一致性\n')
# 检查各类型是否缺少模态
for category in categories:
    for mod in modalities:
        if file_counts[category][mod] == 0:
            print(f'- {category} 缺少 {mod} 模态数据')

# 检查文件名格式一致性
def check_format_consistency(files, category, modality):
    inconsistent = []
    if not files:
        return inconsistent
    
    if category in ['AD', 'CN']:
        pattern = re.compile(r'^(mwp\d+MRI)_(\d{3})_(S)_(\d{4})_(\d{4}-\d{2}-\d{2})_(\d{2})_(\d{2})_(\d{2})\.(\d+)\.nii$')
    else:
        pattern = re.compile(r'^(\d{3})_(S)_(\d{4})_(mwp\d+T1)\.nii$')
    
    for file in files:
        if not pattern.match(file):
            inconsistent.append(file)
    return inconsistent

# 检查所有类别和模态的格式一致性
for category in categories:
    for mod in modalities:
        files = naming_patterns[(category, mod)]
        if files:
            inconsistent = check_format_consistency(files, category, mod)
            if inconsistent:
                print(f'- {category} 的 {mod} 模态中有 {len(inconsistent)} 个文件格式不一致')
                print(f'  示例: {inconsistent[:3]}')
