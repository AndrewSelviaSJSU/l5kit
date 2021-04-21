# Set Up

```shell
SJSU_ID=014547273
ssh -L 10001:localhost:10001 $SJSU_ID@coe-hpc1.sjsu.edu
```

```shell
ROOT=~/cmpe258
mkdir $ROOT
mkdir $ROOT/data
```

Download [the data](https://self-driving.lyft.com/level5/download/):

```shell
cd $ROOT/data
for FILE_NAME in sample train train_full validate aerial_map semantic_map
do
    curl -O https://lyft-l5-datasets-public.s3-us-west-2.amazonaws.com/prediction/v1.1/$FILE_NAME.tar
    mkdir $FILE_NAME
    tar -xf $FILE_NAME.tar -C $FILE_NAME
    rm $FILE_NAME.tar
done
```

```shell
cd $ROOT/models
curl -O https://download.pytorch.org/models/resnet50-19c8e357.pth 
```

```shell
cd $ROOT
git clone https://github.com/AndrewSelviaSJSU/l5kit.git
cd l5kit
git checkout -t origin/cmpe258
```

```shell
cd $ROOT
module load python3/3.6.6 cuda/10.0
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

```shell
MAIL_USER=
JOB_ID=$(sbatch --mail-user=$MAIL_USER --export=ALL,ROOT=$ROOT $ROOT/l5kit/cmpe258/script.sh | awk '{print $4}')
sleep 1 # Give slurm time to allocate the resources before calling squeue (you may have to tune this based on cluster traffic)
GPU_ID=$(squeue | grep $JOB_ID | awk '{print $8}')
ssh -L 10001:localhost:10001 $(whoami)@$GPU_ID
```

Open [jupyterlab](http://localhost:10001/lab/tree/examples/visualisation/visualise_data.ipynb).

You can prove the GPU is available with the following commands:

```shell
module load python3/3.6.6 cuda/10.0
source ~/cmpe258/l5kit/venv/bin/activate
python -c 'import tensorflow as tf; print(tf.__version__); print("GPU Available: ", tf.test.is_gpu_available())'

```

If you need to cancel:

```shell
scancel $JOB_ID
```








## Development

```shell
SJSU_ID=014547273
ssh -L 10001:localhost:10001 $SJSU_ID@coe-hpc1.sjsu.edu
```

```shell
ROOT=~/cmpe258
MAIL_USER=
JOB_ID=$(sbatch --mail-user=$MAIL_USER --export=ALL,ROOT=$ROOT $ROOT/l5kit/cmpe258/script.sh | awk '{print $4}')
sleep 1 # Give slurm time to allocate the resources before calling squeue (you may have to tune this based on cluster traffic)
GPU_ID=$(squeue | grep $JOB_ID | awk '{print $8}')
ssh -L 10001:localhost:10001 $(whoami)@$GPU_ID
```

Open [jupyterlab](http://localhost:10001/lab/tree/examples/visualisation/visualise_data.ipynb).