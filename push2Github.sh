#!/bin/bash

# 初始化Git仓库（如果还没有）
git init

# 添加当前文件夹的所有文件到暂存区
git add .

# 提交文件到仓库，提交信息为"Initial commit"
git commit -m "Remove data"

git remote set-url origin https://github.com/monstre0731/Create_ground_truth_Lumpi.git


# 关联远程仓库（如果需要，将<远程仓库URL>替换为实际的远程仓库URL） 该过程只需要一次，所以这个可以注释掉
git remote add main https://github.com/monstre0731/Create_ground_truth_Lumpi.git

# 推送到远程仓库的主分支（将main替换为你的主分支名称）
git push -u origin main
