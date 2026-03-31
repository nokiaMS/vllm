# autodl管理工具
- autodl服务器信息
  - ssh命令：ssh -p 54723 root@connect.bjb1.seetacloud.com
  - 密码：WIvmTb81tmqE

# 按照如下步骤执行任务
1. 检查是否能够登陆到autodl服务器，如果能够登陆则继续执行下述步骤，如果不能够登陆则提示并结束。
2. 检查用户的bash配置文件中是否添加了source /etc/network_turbo命令，如果没有则把此命令添加到用户的bash配置文件中，并使其生效。
3. 检查系统中是否安装了docker，如果已经安装了那么提示docker的版本信息，如果没有安装，那么请在当前系统中安装docker。
4. 检查在用户目录/root/autodl-tmp下是否已经存在了code文件夹，如果不存在则创建code文件夹。
5. 进入到/root/autodl-tmp/code文件夹检查是否已经存在vllm文件夹，如果不存在则执行命令git clone https://github.com/nokiaMS/vllm.git
6. pip的国内镜像源为https://pypi.tuna.tsinghua.edu.cn/simple，在使用pip命令时使用此镜像地址。
6. pytorch的国内镜像源为https://pypi.tuna.tsinghua.edu.cn/simple，安装pytorch时使用此国内镜像源。
7. 进入到/root/autodl-tmp/code/vllm文件夹，执行命令pip install -r requirements/build.txt
8. 继续执行命令pip install -r requirements/cuda.txt
9. 查看cuda版本，如果版本号为12.8，则不执行步骤10，11，12，13，14，15，16，17，而是直接执行步骤18.
10. 进入到/root/autodl-tmp文件夹，执行命令wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
11. 继续执行命令sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
12. 继续执行命令wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb
13. 继续执行命令sudo dpkg -i cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb
14. 继续执行命令sudo cp /var/cuda-repo-ubuntu2204-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
15. 继续执行命令sudo apt-get update 
16. 继续执行命令sudo apt-get -y install cuda-toolkit-12-8 
17. 查看cuda的版本是否更新成了12.8 
18. 执行vllm --version，如果vllm已经编译完成那么结束任务；如果没有编译完成那么继续执行。
18. 继续执行命令MAX_JOBS=20 pip install -e . --no-build-isolation -v进行vllm编译。