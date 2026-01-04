# GitHub 远程仓库创建与关联指南

## 1. 在 GitHub 上创建远程仓库

请按照以下步骤在 GitHub 上创建新仓库：

1. 打开 GitHub 网站：https://github.com/new
2. 登录你的 GitHub 账号
3. 填写仓库信息：
   - **Repository name**: 建议使用 `mci-recovery-prediction`
   - **Description**: MCI（轻度认知障碍）恢复预测系统
   - **Visibility**: 根据需要选择 "Public" 或 "Private"
   - **不要勾选**任何初始化选项（如 "Add a README file"、"Add .gitignore" 等）
4. 点击 **"Create repository"** 按钮

## 2. 获取远程仓库地址

创建成功后，你会看到仓库页面，在页面上找到 **"Quick setup - if you’ve done this kind of thing before"** 部分，复制仓库的 HTTPS 地址（格式：`https://github.com/your-username/mci-recovery-prediction.git`）

## 3. 关联本地仓库与 GitHub 远程仓库

将下面命令中的 `your-username` 替换为你的 GitHub 用户名，然后执行：

```bash
& "C:\Program Files\Git\cmd\git.exe" remote add origin https://github.com/your-username/mci-recovery-prediction.git
```

## 4. 验证关联成功

执行以下命令验证远程仓库关联是否成功：

```bash
& "C:\Program Files\Git\cmd\git.exe" remote -v
```

## 5. 推送本地代码到 GitHub

执行以下命令将本地代码推送到 GitHub：

```bash
& "C:\Program Files\Git\cmd\git.exe" push -u origin master
```

## 6. 查看远程仓库地址

执行以下命令查看远程仓库地址：

```bash
& "C:\Program Files\Git\cmd\git.exe" config --get remote.origin.url
```

请按照以上步骤操作，完成后你将成功将本地仓库关联到 GitHub 远程仓库，并可以查看远程仓库地址。