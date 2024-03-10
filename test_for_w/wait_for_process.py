import psutil
import sys
import time
import argparse
from datetime import datetime
import os
file_dir = os.path.dirname(os.path.abspath(__file__))
# inf_mac job run python --working-dir . --exec-command "python wait_for_process.py -st" --user luofanming --use-env luofm --num-nodes 1
def wait_for_python_process(filename, wait_forever=False):
    cnt = 0
    time_total = 0
    if wait_forever:
        while wait_forever:
            time.sleep(1.0)
    while True:
        time_total += 5
        try:
            exist = False
            for proc in psutil.process_iter():
                if proc.username() == 'polixir':
                    cmdline = proc.cmdline()
                    if cmdline[0] == 'python' and cmdline[1]==filename:
                        proc.wait()
                        exist = True
                    pass
                pass
            if not exist:
                cnt += 1
                if cnt > 200:
                    os.system(f'source ~/.bash_profile && bash "{os.path.join(file_dir, "save_tmux_log.sh")}" "{os.path.join(file_dir, "logfile")}"')
                    os.system(f'source ~/.bash_profile && bash "{os.path.join(file_dir, "save_tmux_log.sh")}" "/System/Volumes/Data/mnt/公共区/luofanming_public/tmux_log"')
                    break
                print(f'{datetime.now()} cmd {filename} not exists {cnt}')
                time.sleep(5)
            else:
                cnt = 0
                time.sleep(5)
                pass
            if time_total > 600:
                time_total = 0
                os.system(
                    f'source ~/.bash_profile && bash "{os.path.join(file_dir, "save_tmux_log.sh")}" "{os.path.join(file_dir, "logfile")}"')
                os.system(
                    f'source ~/.bash_profile && bash "{os.path.join(file_dir, "save_tmux_log.sh")}" "/System/Volumes/Data/mnt/公共区/luofanming_public/tmux_log"')

        except Exception as e:
            import traceback
            print((traceback.print_exc()))
            time.sleep(5)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f'generate parallel environment')
    parser.add_argument('--file_name', '-fn', type=str, default="main.py",
                        help="file_name")
    parser.add_argument('--stuck', '-st', action='store_true', default=False,
                        help='stuck forever')
    args = parser.parse_args()
    filename = args.file_name
    wait_for_python_process(filename, wait_forever=args.stuck)
