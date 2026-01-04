# Git 云代码库配置指南

## 1. 选择云代码平台

选择一个你喜欢的云代码平台创建远程仓库：

- **GitHub**：https://github.com/new
- **GitLab**：https://gitlab.com/projects/new
- **Gitee**（码云）：https://gitee.com/projects/new

## 2. 创建远程仓库

在所选平台创建新仓库时：
- 仓库名：建议使用 `mci-recovery-prediction`
- 描述：MCI（轻度认知障碍）恢复预测系统
- 选择 **公开** 或 **私有**（根据需要）
- **不要勾选** "Initialize this repository with a README" 或其他初始化选项
- 点击 "Create repository"

## 3. 关联本地仓库与远程仓库

创建远程仓库后，复制仓库的 HTTPS 或 SSH 地址，然后执行以下命令：

### HTTPS 方式（推荐，无需配置 SSH）
```bash
& "C:\Program Files\Git\cmd\git.exe" remote add origin https://github.com/your-username/mci-recovery-prediction.git
```

### SSH 方式（需要配置 SSH 密钥）
```bash
& "C:\Program Files\Git\cmd\git.exe" remote add origin git@github.com:your-username/mci-recovery-prediction.git
```

## 4. 推送代码到远程仓库

第一次推送需要指定分支：
```bash
& "C:\Program Files\Git\cmd\git.exe" push -u origin master
```

## 5. 常用 Git 命令

### 查看状态
```bash
& "C:\Program Files\Git\cmd\git.exe" status
```

### 提交修改
```bash
& "C:\Program Files\Git\cmd\git.exe" add .
& "C:\Program Files\Git\cmd\git.exe" commit -m "your commit message"
& "C:\Program Files\Git\cmd\git.exe" push
```

### 拉取最新代码
```bash
& "C:\Program Files\Git\cmd\git.exe" pull
```

## 6. 配置 Git 别名（可选，方便使用）

在 PowerShell 中添加 Git 别名：

```powershell
# 在 PowerShell 配置文件中添加别名
notepad $PROFILE

# 添加以下内容
function Git-Command {
    param(
        [Parameter(Mandatory=$false, Position=0, ValueFromRemainingArguments=$true)]
        [string[]]$Args
    )
    & "C:\Program Files\Git\cmd\git.exe" @Args
}
Set-Alias -Name git -Value Git-Command

# 保存后重启 PowerShell
```

配置后，你可以直接使用 `git status`、`git add .` 等命令，无需完整路径。

## 7. 在其他设备上克隆仓库

在你的笔记本电脑上：

```bash
# HTTPS 方式
git clone https://github.com/your-username/mci-recovery-prediction.git

# SSH 方式
git clone git@github.com:your-username/mci-recovery-prediction.git
```

## 8. 注意事项

1. 每次修改代码后，使用 `git add .` 和 `git commit -m "message"` 提交到本地仓库
2. 执行 `git push` 将本地提交推送到远程仓库
3. 在其他设备上工作前，先执行 `git pull` 拉取最新代码
4. 定期提交，保持代码同步
5. 合理编写提交信息，便于追踪修改历史

## 9. 分支管理（进阶）

- 主分支：`master` 或 `main`（稳定版本）
- 开发分支：`develop`（集成开发）
- 功能分支：`feature/xxx`（新功能开发）
- 修复分支：`hotfix/xxx`（bug 修复）

```bash
# 创建并切换到功能分支
git checkout -b feature/your-feature

# 开发完成后合并到主分支
git checkout master
git merge feature/your-feature

git push
```