#!/bin/bash
set -e

# ANTLR jar路径
ANTLR_JAR="../lib/antlr-4.13.2-complete.jar"

# 编译选项
JAVAC_OPTS="-source 11 -target 11 -encoding UTF-8"
ANTLR_OPTS="-visitor -no-listener"

# 清理之前的编译结果
rm -rf out
mkdir -p out

# 生成ANTLR解析器
java -jar "$ANTLR_JAR" $ANTLR_OPTS Solidity.g4

# 编译Java文件
javac $JAVAC_OPTS \
    -cp "$ANTLR_JAR" \
    -d out \
    *.java

# 创建JAR
jar cvfm scana-parser.jar manifest.txt -C out .

# 显示JAR信息
jar tf scana-parser.jar

