# Self-driving using Lyft Dataset

Our semester-long Deep Learning (CMPE 258) project focuses on autonomous driving. We specifically targeted a problem that would stretch us to grow rather than one we felt comfortable achieving. The dataset we chose includes 100's of gigabytes of complex data, thereby exceeding our local systems' abilities and forcing us into the cloud. Furthermore, we had to learn to train deep neural networks using GPUs on the SJSU High Performance Computing (HPC) system in order to accelerate our research. Altogether, these factors presented us with a multitude of challenges; through combined effort, we were able to overcome them.

## Team

* Bhakti Subhash Chowkwale ([email](mailto:bhaktisubhash.chowkwale@sjsu.edu), [GitHub](https://github.com/bhaktichowkwale))
* Andrew Selvia ([email](mailto:andrew.selvia@sjsu.edu), [GitHub](https://github.com/AndrewSelviaSJSU))
* Cory Sweet ([email](mailto:cory.sweet@sjsu.edu), [GitHub](https://github.com/cx2sweet))
* Sudha Vijayakumar ([email](mailto:sudha.vijayakumar@sjsu.edu), [GitHub](https://github.com/sudha-vijayakumar))

## Data

// TODO

## Run on the SJSU HPC

Due to the size of the data and the GPU demands, you will likely need to use some cloud computing system to run the code in our project. We chose the SJSU HPC system since it is free unlike third-party clouds. The instructions below describe how to set up your environment so you can run our code.

### Set Up

#### Download Cisco AnyConnect Secure Mobility Client

In order to access the HPC system, you must first establish a VPN connection using the Cisco AnyConnect Secure Mobility Client application. Assuming you have not installed it previously, [download the latest version](https://vpn.sjsu.edu). Assuming you are a student, select **Student** from the **GROUP** drop-down list, then enter your SJSU credentials. After gaining entry, download and install the application.

#### Establish a VPN Connection

Once installed, open the Cisco AnyConnect Secure Mobility Client application, enter `vpn.sjsu.edu`, and select **Connect**. Now, input the same credentials you entered to download the application. You should now be connected to the SJSU VPN.

#### Configure Your Environment on the SJSU HPC System

With an active SJSU VPN connection, you should now be able to access the HPC system via `ssh`. The following commands will help you configure your environment in a standardized way so that the notebooks we will run later work appropriately.

Begin by creating a SSH tunnel from your `localhost` to the HPC system. If you're curious, the tunnel will be used later to allow you to view and edit Jupyter notebooks on your machine while they run on a GPU in the SJSU HPC. Remember to define your `SJSU_ID` (i.e. `SJSU_ID=123456789`) in the command below prior to executing. When prompted, enter your SJSU password.

```shell
PORT=# an arbitrary, unclaimed port (i.e. 10005); caution: the commands below assume you use the same value for each definition of PORT
SJSU_ID=# your SJSU numeric ID (i.e. 123456789)
ssh -L "$PORT":localhost:$PORT $SJSU_ID@coe-hpc1.sjsu.edu
```

With the tunnel to the head node established, you should have a shell on the HPC system. Define `ROOT` so that downstream commands and code can use it as an absolute point of reference.

```shell
ROOT=~/cmpe258
mkdir $ROOT
mkdir $ROOT/data
```

Now, download [the data](https://self-driving.lyft.com/level5/download/) the notebooks depend upon in order to train the neural networks. Be aware, this process will likely take quite a while since it needs to download 100's of gigabytes of data. Feel free to go eat a meal while it executes.

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

Once all the data is downloaded, you will also need to download a pre-trained resnet50 model used within a few of the notebooks. You must pre-download it since the code in the notebooks will attempt to download it at runtime otherwise. Since the code will be running on a GPU which cannot reach the Internet, this will cause the code to crash. 

```shell
cd $ROOT/models
curl -O https://download.pytorch.org/models/resnet50-19c8e357.pth 
```

With the data and models downloaded, it is finally time to clone the repository. In the commands below you will be cloning our fork of Lyft's `l5kit` repo and checking out the branch which contains our modifications.

```shell
cd $ROOT
git clone https://github.com/AndrewSelviaSJSU/l5kit.git
cd l5kit
git checkout -t origin/cmpe258
```

Next, load Python and CUDA system libraries into your session. Then, you can create a virtual environment and install the required packages. 

```shell
cd $ROOT
module load python3/3.6.6 cuda/10.0
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Run JupyterLab

By now, your environment should be properly configured on the SJSU HPC system. It is now time to run JupyterLab on the GPU and verify that things are working properly. In the commands below, you can define `MAIL_USER` (i.e. `MAIL_USER=first-name.last-name@sjsu.edu`) so that the batch job you submit to run on SLURM will email you upon completion. These commands rely on [script.sh](script.sh) which sets up the environment and runs JupyterLab on a GPU node. The surrounding shell commands simply automate processes so you don't have to think about how to obtain the `GPU_ID` which you will use at the end to create a second SSH tunnel into the GPU running your job.

```shell
MAIL_USER=# an email address to which an email should be sent upon completion of the batch job
PORT=# an arbitrary, unclaimed port (i.e. 10005); caution: the commands below assume you use the same value for each definition of PORT
OUTPUT_PATH=/home/$(whoami)/cmpe258/l5kit.log
JOB_ID=$(sbatch --mail-user=$MAIL_USER --output=$OUTPUT_PATH --export=ALL,ROOT=$ROOT,PORT=$PORT,OUTPUT_PATH=$OUTPUT_PATH $ROOT/l5kit/cmpe258/script.sh | awk '{print $4}')
sleep 1 # Give slurm time to allocate the resources before calling squeue (you may have to tune this based on cluster traffic)
GPU_ID=$(squeue | grep $JOB_ID | awk '{print $8}')
ssh -L "$PORT":localhost:$PORT $(whoami)@$GPU_ID
```

Within the GPU shell, you now must compute the URL you will use to log in to JupyterLab from your system:

```shell
cat $OUTPUT_PATH | grep http://localhost | tail -1 | xargs
```

Copy the URL produced by that command and paste it in a web browser on your system to open JupyterLab.

#### Troubleshooting

If for any reason things aren't working properly, you can execute the commands below on the GPU to troubleshoot that the GPU is available:

```shell
module load python3/3.6.6 cuda/10.0
source ~/cmpe258/l5kit/venv/bin/activate
python -c 'import tensorflow as tf; print(tf.__version__); print("GPU Available: ", tf.test.is_gpu_available())'
```

If you need to prematurely cancel your batch job, exit the SSH tunnel into the GPU then run the following command:

```shell
scancel $JOB_ID
```

### Development

Once your environment is properly configured on the SJSU HPC system, you need not do many of the steps above again. Instead, the commands below streamline the steps you must take to establish a chain of SSH tunnels to a GPU node on the SJSU HPC system.

Again, create the first SSH tunnel to the SJSU HPC system head node:

```shell
PORT=# an arbitrary, unclaimed port (i.e. 10005); caution: the commands below assume you use the same value for each definition of PORT
SJSU_ID=# your SJSU numeric ID (i.e. 123456789)
ssh -L "$PORT":localhost:$PORT $SJSU_ID@coe-hpc1.sjsu.edu
```

Now, create the second SSH tunnel to the GPU running your batch job:

```shell
MAIL_USER=# an email address to which an email should be sent upon completion of the batch job
ROOT=~/cmpe258
PORT=# an arbitrary, unclaimed port (i.e. 10005); caution: the commands below assume you use the same value for each definition of PORT
OUTPUT_PATH=/home/$(whoami)/cmpe258/l5kit.log
JOB_ID=$(sbatch --mail-user=$MAIL_USER --output=$OUTPUT_PATH --export=ALL,ROOT=$ROOT,PORT=$PORT,OUTPUT_PATH=$OUTPUT_PATH $ROOT/l5kit/cmpe258/script.sh | awk '{print $4}')
sleep 1 # Give slurm time to allocate the resources before calling squeue (you may have to tune this based on cluster traffic)
GPU_ID=$(squeue | grep $JOB_ID | awk '{print $8}')
ssh -L "$PORT":localhost:$PORT $(whoami)@$GPU_ID
```

Within the GPU shell, you now must compute the URL you will use to log in to JupyterLab from your system:

```shell
cat $OUTPUT_PATH | grep http://localhost | tail -1 | xargs
```

Copy the URL produced by that command and paste it in a web browser on your system to open JupyterLab.