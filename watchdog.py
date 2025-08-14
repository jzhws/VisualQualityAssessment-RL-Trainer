import subprocess
import time

# 你的shell脚本绝对路径
sh_script_path = "./finetune_RL.sh"

def is_process_running(script_path):
    # 使用 pgrep 或 ps 查找包含脚本名的进程
    try:
        # pgrep -f 会匹配完整命令行，能找到脚本对应的进程
        result = subprocess.run(["pgrep", "-f", script_path], stdout=subprocess.PIPE)
        if result.stdout:
            return True
        else:
            return False
    except Exception as e:
        print(f"检查进程时出错: {e}")
        return False

def start_process(script_path):
    try:
        # 以子进程启动脚本，且使其不阻塞当前程序
        subprocess.Popen(["bash", script_path])
        print(f"启动脚本: {script_path}")
    except Exception as e:
        print(f"启动脚本出错: {e}")

if __name__ == "__main__":
    while True:
        if is_process_running(sh_script_path):
            print("进程已运行，跳过")
        else:
            print("进程未运行，重新启动")
            start_process(sh_script_path)
        # 休眠5分钟
        time.sleep(300)
