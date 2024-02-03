module load tools/apptainer

export VENV="$HOME/.envs/venv_cluster2former"
export JUPYTER_CONFIG_DIR="$HOME/jupyter_sing/$SLURM_JOBID/"
export JUPYTER_PATH="$VENV/share/jupyter":"$HOME/jupyter_sing/$SLURM_JOBID/jupyter_path"
export JUPYTER_DATA_DIR="$HOME/jupyter_sing/$SLURM_JOBID/jupyter_data"
export JUPYTER_RUNTIME_DIR="$HOME/jupyter_sing/$SLURM_JOBID/jupyter_runtime"
export IPYTHONDIR="$HOME/ipython_sing/$SLURM_JOBID"

mkdir -p $JUPYTER_CONFIG_DIR
mkdir -p $IPYTHONDIR

export JUPYTER_ALLOW_INSECURE_WRITES=1

#####################################
# Here you should start the apptainer instance
# you can bind your datasets into the apptainer instance e.g.:

# apptainer instance start --nv --bind /path/to/the/datasets/:/path/in/the/apptainer path/to/the/image/file cluster2former_ins
#####################################

if [ ! -d "$VENV" ];then
    #####################################
    # Here you can make the desired links to your datasets e.g.:
    
    # apptainer exec --nv instance://cluster2former_ins ln -s /path/to/ytvis_2019 datasets/ytvis_2019
    # apptainer exec --nv instance://cluster2former_ins ln -s /path/to/ytvis_2021 datasets/ytvis_2021
    #####################################

    apptainer exec --nv instance://cluster2former_ins python3 -m virtualenv -p python3.10 $VENV --system-site-packages
    apptainer run --nv instance://cluster2former_ins $VENV "python3 -m pip install --upgrade pip"

    apptainer run --nv instance://cluster2former_ins $VENV "python3 -m pip install opencv-python cython Pillow imageio"
    apptainer run --nv instance://cluster2former_ins $VENV "python3 -m pip install numpy sklearn scikit-image scikit-learn pandas matplotlib plotly"
        
    # Python3.7 needed
    apptainer run --nv instance://cluster2former_ins $VENV "python3 -m pip install --no-cache-dir torch"
    apptainer run --nv instance://cluster2former_ins $VENV "python3 -m pip install torch torchvision torchaudio"

    apptainer run --nv instance://cluster2former_ins $VENV "python3 -m pip install git+https://github.com/facebookresearch/detectron2.git"

    apptainer run --nv instance://cluster2former_ins $VENV "python3 -m pip install \"git+https://github.com/youtubevos/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI\""

    apptainer run --nv instance://cluster2former_ins $VENV "python3 -m pip install git+https://github.com/cocodataset/panopticapi.git"        

    apptainer run --nv instance://cluster2former_ins $VENV "python3 -m ipykernel install --sys-prefix --name HPC_NRSFM_PYTORCH_ENV_IPYPARALLEL_CUDA --display-name HPC_NRSFM_PYTORCH_ENV_IPYPARALLEL_CUDA"
    apptainer run --nv instance://cluster2former_ins $VENV "python3 -m pip install tensorboard"

    apptainer run --nv instance://cluster2former_ins $VENV "python3 -m pip install -r requirements.txt"
    apptainer run --nv instance://cluster2former_ins $VENV "cd mask2former/modeling/pixel_decoder/ops && sh make.sh"
fi

#create a new ipython profile appended with the job id number
profile=job_${SLURM_JOB_ID}
apptainer run --nv instance://cluster2former_ins $VENV "ipython profile create --parallel ${profile}"

# Enable IPython clusters tab in deep_nrsfm_pytorch notebook
apptainer run --nv instance://cluster2former_ins $VENV "jupyter nbextension enable --py ipyparallel"

## Start Controller and Engines
#
apptainer run --nv instance://cluster2former_ins $VENV "ipcontroller --ip="*" --profile=${profile}" &
sleep 10

##srun: runs ipengine on each available core
srun apptainer run --nv instance://cluster2former_ins  $VENV "ipengine --profile=${profile} --location=$(hostname)" &
sleep 25

export XDG_RUNTIME_DIR=""

apptainer run --nv instance://cluster2former_ins $VENV "jupyter notebook --ip $(facter ipaddress) --no-browser --port 8899" &
pid=$!
sleep 5s
apptainer run --nv instance://cluster2former_ins $VENV "jupyter notebook list"
apptainer run --nv instance://cluster2former_ins $VENV "jupyter --paths"
apptainer run --nv instance://cluster2former_ins $VENV "jupyter kernelspec list"

TENSORBOARD_DIR=output_video

apptainer run --nv instance://cluster2former_ins $VENV "tensorboard --logdir ${TENSORBOARD_DIR} --host  $(facter ipaddress) --port 8889"

wait $pid
echo "Stopping instance"
apptainer instance stop cluster2former_ins