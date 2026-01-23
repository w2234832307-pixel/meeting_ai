#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复 .env 文件编码问题
确保 .env 文件使用 UTF-8 编码
"""
import os
import sys
from pathlib import Path

# 设置标准输出编码为 UTF-8
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

def fix_env_encoding():
    """修复 .env 文件编码"""
    env_path = Path(".env")
    env_example_path = Path("env.example")
    
    print("[修复工具] .env 文件编码修复")
    print("-" * 50)
    
    # 1. 检查 .env 是否存在
    if not env_path.exists():
        print("[错误] .env 文件不存在")
        if env_example_path.exists():
            print("[成功] 发现 env.example，正在创建 .env...")
            # 读取 env.example（尝试多种编码）
            content = None
            for encoding in ['utf-8', 'gbk', 'latin-1']:
                try:
                    with open(env_example_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    print(f"   成功读取 env.example (编码: {encoding})")
                    break
                except Exception as e:
                    continue
            
            if content:
                # 写入 .env（强制UTF-8）
                with open(env_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print("[成功] 已创建 .env 文件（UTF-8编码）")
                print("[警告] 请编辑 .env 文件，填写你的实际配置（API Key等）")
            else:
                print("[错误] 无法读取 env.example")
        else:
            print("[错误] env.example 也不存在，无法自动创建")
        return
    
    # 2. 尝试读取现有 .env
    print(f"[信息] 找到 .env 文件: {env_path.absolute()}")
    
    content = None
    original_encoding = None
    
    for encoding in ['utf-8', 'gbk', 'gb2312', 'latin-1']:
        try:
            with open(env_path, 'r', encoding=encoding) as f:
                content = f.read()
            original_encoding = encoding
            print(f"[成功] 成功读取 .env (当前编码: {encoding})")
            break
        except Exception as e:
            continue
    
    if not content:
        print("[错误] 无法读取 .env 文件（所有编码都失败）")
        print("[建议] 手动删除 .env，重新从 env.example 复制")
        return
    
    # 3. 重新保存为UTF-8
    if original_encoding != 'utf-8':
        print(f"[转换] 正在转换编码: {original_encoding} -> UTF-8")
        
        # 备份原文件
        backup_path = env_path.with_suffix('.env.backup')
        with open(backup_path, 'wb') as f:
            with open(env_path, 'rb') as orig:
                f.write(orig.read())
        print(f"[备份] 已备份原文件: {backup_path}")
        
        # 重新写入UTF-8
        with open(env_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("[成功] .env 文件已转换为 UTF-8 编码")
    else:
        print("[成功] .env 文件已经是 UTF-8 编码")
    
    # 4. 验证关键配置
    print("\n[配置检查]")
    print("-" * 50)
    
    lines = content.split('\n')
    configs = {}
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            configs[key.strip()] = value.strip()
    
    # 检查关键配置
    checks = {
        'APP_PORT': '应用端口',
        'ASR_SERVICE_TYPE': 'ASR服务类型',
        'LLM_SERVICE_TYPE': 'LLM服务类型',
        'EMBEDDING_SERVICE': 'Embedding服务类型',
        'CHROMA_HOST': 'Chroma主机',
    }
    
    for key, desc in checks.items():
        value = configs.get(key, '未配置')
        status = '[OK]' if value and value != '未配置' and not value.startswith('your_') else '[!!]'
        print(f"{status} {desc:20s}: {value}")
    
    print("\n" + "=" * 50)
    print("[成功] 修复完成！现在可以运行: python main.py")
    print("=" * 50)

if __name__ == "__main__":
    try:
        fix_env_encoding()
    except Exception as e:
        print(f"[错误] 发生错误: {e}")
        import traceback
        traceback.print_exc()
