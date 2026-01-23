"""
环境检测脚本 - 检查Python环境和依赖
"""
import sys
import subprocess
import os
from pathlib import Path

# 设置Windows控制台编码
if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')
    sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

def check_python_version():
    """检查Python版本"""
    print("=" * 60)
    print("Python 环境检测")
    print("=" * 60)
    print(f"Python 路径: {sys.executable}")
    print(f"Python 版本: {sys.version}")
    print(f"版本信息: {sys.version_info}")
    
    if sys.version_info < (3, 8):
        print("⚠️ 警告: Python版本过低，建议使用Python 3.8+")
    else:
        print("✅ Python版本符合要求")
    print()

def check_dependencies():
    """检查依赖包"""
    print("=" * 60)
    print("依赖包检测")
    print("=" * 60)
    
    required_packages = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "requests",
        "python-dotenv",
        "tencentcloud-sdk-python",
        "openai",
        "pymilvus",
        "pymysql",
        "tenacity"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == "python-dotenv":
                import dotenv
                print(f"✅ {package}")
            elif package == "tencentcloud-sdk-python":
                import tencentcloud
                print(f"✅ {package}")
            else:
                __import__(package.replace("-", "_"))
                print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ 缺少以下依赖: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
    else:
        print("\n✅ 所有依赖包已安装")
    print()

def check_env_file():
    """检查.env文件"""
    print("=" * 60)
    print("环境变量文件检测")
    print("=" * 60)
    
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env 文件存在")
        
        # 读取并显示配置（隐藏敏感信息）
        with open(env_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        required_vars = {
            "TENCENT_SECRET_ID": False,
            "TENCENT_SECRET_KEY": False,
            "LLM_API_KEY": False
        }
        
        print("\n配置项检查:")
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                
                if key in required_vars:
                    required_vars[key] = True
                    # 隐藏敏感值
                    if value and len(value) > 4:
                        masked_value = value[:4] + "*" * min(len(value) - 4, 10)
                    else:
                        masked_value = "***"
                    print(f"  ✅ {key}={masked_value}")
                elif any(sensitive in key.upper() for sensitive in ["SECRET", "KEY", "PASSWORD"]):
                    print(f"  ✅ {key}=***")
                else:
                    print(f"  ✅ {key}={value}")
        
        print("\n必需配置检查:")
        missing = [k for k, v in required_vars.items() if not v]
        if missing:
            print(f"  ❌ 缺少配置: {', '.join(missing)}")
            print("  请使用 python update_env.py 配置")
        else:
            print("  ✅ 所有必需配置已设置")
    else:
        print("❌ .env 文件不存在")
        print("请创建 .env 文件并配置环境变量")
        print("可以使用 python update_env.py 进行配置")
    print()

def check_directories():
    """检查必要的目录"""
    print("=" * 60)
    print("目录结构检测")
    print("=" * 60)
    
    required_dirs = [
        "app",
        "app/api",
        "app/core",
        "app/services",
        "app/schemas",
        "logs",
        "temp_files"
    ]
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✅ {dir_path}/")
        else:
            print(f"⚠️ {dir_path}/ - 不存在（将在首次运行时自动创建）")
    print()

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("会议AI服务 - 环境检测")
    print("=" * 60 + "\n")
    
    check_python_version()
    check_dependencies()
    check_env_file()
    check_directories()
    
    print("=" * 60)
    print("检测完成")
    print("=" * 60)
    print("\n如果所有检测都通过，可以运行: python main.py")

if __name__ == "__main__":
    main()