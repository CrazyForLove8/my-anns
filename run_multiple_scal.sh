#!/bin/bash

BUILD_DIR="build"
TEST_EXEC="scal_test"

DATASET_BASE="/root/mount/dataset/siftsmall/siftsmall_base.fvecs"
DATASET_QUERY="/root/mount/dataset/siftsmall/siftsmall_query.fvecs"
DATASET_GROUNDTRUTH="/root/mount/dataset/siftsmall/siftsmall_groundtruth.ivecs"
METRIC="l2" # l2, cosine
OUTPUT_PATH="/root/mount/my-anns/output/siftsmall/" # 日志输出目录、构建好的索引输出目录
THREAD_NUM=48 # 运行线程数
K=20 # 我们的方法的参数
MAX_NEIGHBOR=32 # HNSW & Vamana 共享的参数
EF_CONSTRUCTION=200
ALPHA=1.2 # VAMANA 的参数

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

if [ ! -f "src/$TEST_EXEC" ]; then
    echo "错误：未找到可执行文件 'src/$TEST_EXEC'。请检查编译日志。"
    exit 1
fi

cd src || exit

for current_split_num in {3..7}
do
    echo "--- 正在使用 SPLIT_NUM=$current_split_num 运行 $TEST_EXEC ---"

    ./"$TEST_EXEC" "$DATASET_BASE" "$DATASET_QUERY" "$DATASET_GROUNDTRUTH" \
                  "$METRIC" "$OUTPUT_PATH" "$current_split_num" "$THREAD_NUM" \
                  "$K" "$MAX_NEIGHBOR" "$EF_CONSTRUCTION" "$ALPHA"

    if [ $? -ne 0 ]; then
        echo "警告：'$TEST_EXEC' (SPLIT_NUM=$current_split_num) 运行失败！"
    else
        echo "'$TEST_EXEC' (SPLIT_NUM=$current_split_num) 运行成功！"
    fi

    echo "---------------------------------------------------"
done

echo "--- 所有循环运行完成 ---"
cd .. || exit
cd .. || exit