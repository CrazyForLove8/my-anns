#!/bin/bash

BUILD_DIR="build"
TEST_EXEC="scal_test"

DATASET_BASE="/root/mount/dataset/internet_search/internet_search_train.fbin"
DATASET_QUERY="/root/mount/dataset/internet_search/internet_search_test.fbin"
DATASET_GROUNDTRUTH="/root/mount/dataset/internet_search/internet_search_neighbors.fbin"
# l2, cosine
METRIC="l2"
# 日志输出目录、构建好的索引输出目录
OUTPUT_PATH="/root/mount/my-anns/output/internet_search/"
# 把数据集平分为多少个子集
SPLIT_NUM=2
# 运行线程数
THREAD_NUM=200
# 我们的方法的参数
K=20
# HNSW & Vamana 共享的参数
MAX_NEIGHBOR=32
EF_CONSTRUCTION=200
# VAMANA 的参数
ALPHA=1.2

echo "--- 开始编译项目 ---"

if [ ! -d "$BUILD_DIR" ]; then
    echo "创建构建目录: $BUILD_DIR"
    mkdir "$BUILD_DIR"
fi

cd "$BUILD_DIR" || exit

echo "运行 CMake 配置..."
cmake ..

echo "编译项目..."
cmake --build .

if [ $? -ne 0 ]; then
    echo "错误：项目编译失败！"
    exit 1
fi

echo "--- 编译完成 ---"
echo "--- 运行: $TEST_EXEC ---"

if [ -f "src/$TEST_EXEC" ]; then
    cd src || exit
    echo "正在运行 ./$TEST_EXEC $TEST_ARGS"
    ./"$TEST_EXEC" $DATASET_BASE $DATASET_QUERY $DATASET_GROUNDTRUTH $METRIC $OUTPUT_PATH $SPLIT_NUM $THREAD_NUM $K $MAX_NEIGHBOR $EF_CONSTRUCTION $ALPHA

    if [ $? -ne 0 ]; then
        echo "警告：'$TEST_EXEC' 运行失败"
    else
        echo "'$TEST_EXEC' 运行成功"
    fi
else
    echo "错误：未找到可执行文件 '$TEST_EXEC'。"
    exit 1
fi

echo "--- 所有操作完成 ---"
cd .. || exit
cd .. || exit