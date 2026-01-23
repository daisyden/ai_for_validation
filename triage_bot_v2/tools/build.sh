set -x
download=$1
build=$2
workdir="/workdir"

mkdir -p ${workdir}

export http_proxy=http://proxy.ims.intel.com:911
export https_proxy=http://proxy.ims.intel.com:911

if [ $download = "True" ]; then
    source ~/.bashrc
    
    ~/miniforge3/bin/conda remove --name pytorch_guilty_commit --all --yes 2>/dev/null
    ~/miniforge3/bin/conda create -n pytorch_guilty_commit python=3.10 -y
    source ~/miniforge3/bin/activate pytorch_guilty_commit
    if [ -d ${workdir} ]; then rm -rf ${workdir} ; fi
    mkdir -p ${workdir}
    cd ${workdir}
    git clone https://github.com/pytorch/pytorch.git  
    cd ${workdir}/pytorch
    cd third_party && git clone https://github.com/intel/torch-xpu-ops.git && cd torch-xpu-ops && git rev-parse HEAD >../xpu.txt
else
    download=False
fi

source ~/miniforge3/bin/activate pytorch_guilty_commit
cd ${workdir}/pytorch && pip install --root-user-action=ignore -r .ci/docker/requirements-ci.txt
export PYTORCH_TEST_WITH_SLOW=1
export PYTEST_ADDOPTS=' -n 1 --timeout 30 --timeout_method=thread '
pip install --root-user-action=ignore pytest-timeout
conda install -y libuv
source /tools/env.sh

if [ $build = "existing" ]; then
    echo "### Pytorch is already available in container session : $container"
else
    pip uninstall torch -y
    # If build is not source or nightly, use existing pytorch installation and code under tmux    
    if [ ${build} = "source" ]; then
        echo "### Start to build pytorch from source in container session : " + container
        
        cd ${workdir}/pytorch && git pull
        pip install --root-user-action=ignore cmake ninja pybind11
        pip install --root-user-action=ignore -r requirements.txt
        cd ${workdir}/pytorch && python3 setup.py clean
        cd ${workdir}/pytorch && python3 setup.py develop
                      
    elif [ ${build} = "nightly" ]; then
        echo "### Start to install pytorch nightly in container session : " + container
        
        cd ${workdir}/pytorch
        git pull
        pushd ${workdir}/pytorch/third_party/torch-xpu-ops
        git pull
        popd
        pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/xpu
    fi
fi

if [ -d "./triton_whl" ]; then 
    rm -rf ./triton_whl
fi
pip download --no-deps --index-url https://download.pytorch.org/whl/nightly/xpu --pre pytorch-triton-xpu --dest tritone_whl
pip install --root-user-action=ignore tritone_whl/pytorch_triton_xpu-*.whl

echo "pytorch version:\n" > ${workdir}/enviroments.txt
cd ${workdir}/pytorch && git log -1 >> ${workdir}/enviroments.txt
echo "xpu-ops version:\n" >> ${workdir}/enviroments.txt
cd ${workdir}/pytorch/third_party/torch-xpu-ops && git log -1 >> ${workdir}/enviroments.txt
echo "package version:\n" >> ${workdir}/enviroments.txt
pip list|grep torch >> ${workdir}/enviroments.txt
dpkg -l |grep intel >> ${workdir}/enviroments.txt

set +x