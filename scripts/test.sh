#!/bin/bash
# script.sh

set -x
# 输出开始的提示
echo "Running Python script..."
python --version
ls
# 运行 Python 文件
python test.py

# 输出完成的提示
echo "Python script has finished running."
