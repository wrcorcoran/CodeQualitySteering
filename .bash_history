exit
nvidia-smi
bash
exit()
nvidia-smi
ls
exit
ssh pod-gpu
ls
cd /scratch/nour
ssh pod-gpu
conda activate turbenv
bash
source anaconda3/bin/activate
conda init
ssh pod-gpu
wget https://repo.anaconda.com/archive/Anaconda3-2023.07-2-Linux-x86_64.sh
sh Anaconda3.*.sh
ls
sh Anaconda3-2023.07-2-Linux-x86_64.sh
ls
source anaconda3/bin/activate
nvidia-smi
conda create --name turbenv python=3.9
conda activate turbenv
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
conda install conda-forge::pytorch_scatter
conda install pytorch_scatter
conda install conda-forge::pytorch_scatter
myquota
conda install matplotlib tqdm numba
mamba
conda
bash
ssh pod-gpu
conda activate turbenv
wget -c -r https://dataset-dl.liris.cnrs.fr/eagle_dataset/eagle_clusters.tar.gz
ls
mkdir eagle
cd eagle
pwd
cd ..
mv dataset-dl.liris.cnrs.fr /home/nour/eagle/
ls
cd eagle
ls
cd dataset-dl.liris.cnrs.fr/
ls
cd eagle_dataset/
ls
cd ..
mkdir dataset
pwd
mv dataset-dl.liris.cnrs.fr/eagle_dataset .
ls
cd dataset-dl.liris.cnrs.fr/
ls
cd ..
rm -r dataset-dl.liris.cnrs.fr/
ls
rm dataset/
rm -r dataset/
ls
cd eagle_dataset/
ls
tar -xzvf eagle_clusters.tar.gz 
conda activate turbenv
ls
git clone https://github.com/eagle-dataset/EagleMeshTransformer.git
ls
mv EagleMeshTransformer/ eagle
ls
cd eagle
ls
exit
ssh pod-gpu
conda activate trubenv
conda activate turbenv
ls
cd eagle/
ls
wget -c -r https://dataset-dl.liris.cnrs.fr/eagle_dataset/step.tar.gz
ls
cd dataset-dl.liris.cnrs.fr/
ls
cd eagle_dataset/
ls
pwd
 cd ../eagle_dataset/
mv /home/nour/eagle/dataset-dl.liris.cnrs.fr/eagle_dataset/step.tar.gz
mv /home/nour/eagle/dataset-dl.liris.cnrs.fr/eagle_dataset/step.tar.gz .
ls
pwd
cd ../eagle_dataset/
ls
pwd
cd ../../eagle_dataset/
mv /home/nour/eagle/dataset-dl.liris.cnrs.fr/eagle_dataset/step.tar.gz .
ls
cd Eagle_dataset/
ls
cd ..
ls
tar -xzvf step.tar.gz
cd eagle/eagle_dataset/
ls
cd Eagle_dataset/
cd Eagle_dataset/Cre/126/2/
cd ..
cd Eagle_dataset/Cre/126/2/
ls
cd 
ls
cd eagle/
ls
cd eagle_dataset/
ls
cd Eagle_dataset/
ls
cd ../..
ls
cd dataset-dl.liris.cnrs.fr/
ls
cd eagle_dataset/
ls
cd ..
rm -r dataset-dl.liris.cnrs.fr/
wget -c -r https://dataset-dl.liris.cnrs.fr/eagle_dataset/triangular.tar.gz
ls
rm -r dataset-dl.liris.cnrs.fr/
tmux new -s eagle
tmux attach -t eagle
ls
conda activate turbenv
ls
cd eagle/
ls
cd EagleMeshTransformer/
ls
cd ..
ls
cd dataset-dl.liris.cnrs.fr/
ls
cd eagle_dataset/
ls
cd ..
ls
cd eagle_dataset/
ls
cd Eagle_dataset/
ls
cd ..
ls
tar -xzvf step.tar.gz
ls
cd Eagle_dataset/Cre/163/1/
ls
cd
ls
cd eagle/
ls
cd eagle_dataset/
ls
tmux new -s turb
tmux attach -s turb
tmux attach -t turb
tmux attach -t eagle
tmux attach -t turb
tmux attach -t eagle
ls
cd eagle/
ls
cd eagle_dataset/
ls
cd Eagle_dataset/
ls
cd Cre
ls
cd 1
ls
cd 1
ls
conda activate turbenv
tmux attach -t turb
tmux attach -t eagle
sacct
scontrol show job 66503
scontrol show job job66503
sacct -L
sacct -e
sacct --CPUTime
sacct
sacct --format="JobID,JobName,Partition,Account,AllocCPUS,State,ExitCode"
sacct -e
sacct --format="JobID,JobName,Partition,Account,AllocCPUS,State,ExitCode,NodeList,ReqNodes"
sacct --helpformat
sacct --format="JobID,JobName,Partition,Account,AllocCPUS,State,ExitCode,NodeList,AllocTRES"
    sinfo -o "%50N %10c %20m %30G"
nvidia-smi
exit
sacct
exit
    sinfo -o "%50N %10c %20m %30G"
sacct
nvidia-smi
ssh podGPU
ssh pod-gpu
tmux attach -t eagle
tmux new -s eagle
    sinfo -o "%50N %10c %20m %30G"
tmux attach -t eagle
conda activate base
tmux attach -t eagle
ssh pod-gpu
tmux attach -t eagle
tmux new -s download
tmux attach -t download
tmux attach -t eagle
ls
mkdir eagle_repo
cd eagle_repo/
git clone https://github.com/cstam5916/292f_turbulence.git
rm -r 292f_turbulence/
ls
rm -r 292f_turbulence/
ls
cd ..
git clone git@github.com:NourAbdelmoneim/292f_turbulence.git
ssh-keygen -t ed25519 -C na2247@nyu.edu
eval "$(ssh-agent -s)"
open ~/.ssh/config
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
git clone git@github.com:NourAbdelmoneim/292f_turbulence.git
cd 292f_turbulence/
git remote add eagle_292f https://github.com/cstam5916/292f_turbulence.git
git remote -v
ls
cd Models/
cp /home/nour/eagle/EagleMeshTransformer/Models/graphViTPINN_clean.py .
ls
pwd
mv /home/nour/292f_turbulence/Models/graphViTPINN_clean.py ./graphViTPINN.py
ls
cd ..
cp /home/nour/eagle/EagleMeshTransformer/train_graphvit_pinn.py .
ls
cp /home/nour/eagle/EagleMeshTransformer/run_job_pinn.sh .
cp /home/nour/eagle/EagleMeshTransformer/eval_graphvit.py ./eval_graphvit_pinn.py
ls
git add .
git commit -m "add pinn code"
git push
ls
git clone https://github.com/cvenhoff/steering-thinking-llms.git
cd steering-thinking-llms
conda env create -f environment.yaml
conda activate stllms_env
pip install ruckig
pip install sae-lens==5.3.0 safetensors==0.4.5 sapien==2.2.2 scikit-learn==1.6.0 scipy==1.14.1 sentencepiece==0.2.0 sentry-sdk==2.19.2 setproctitle==1.3.4 shellingham==1.5.4 simple-parsing==0.1.6 smmap==5.0.1 sniffio==1.3.1 soupsieve==2.6 statsmodels==0.14.4 sympy==1.13.1 tabulate==0.9.0 tenacity==9.0.0 threadpoolctl==3.5.0 tiktoken==0.6.0 tokenizers==0.21.0 tomli==2.2.1 torch==2.5.1 tqdm==4.67.0 transformer-lens==2.11.0 transformers==4.47.1 transforms3d==0.4.2 trimesh==4.5.2 triton==3.1.0 typeguard==4.4.1 typer==0.12.5 tzdata==2024.2 urllib3==2.2.3 uvloop==0.21.0 wandb==0.19.1 xxhash==3.5.0 yarl==1.18.3 zstandard==0.22.0 openai==1.70.0 anthropic==0.49.0 nnsight==0.4.5 bitsandbytes==0.45.4 seaborn==0.13.2 mdmm==0.1.3
conda env list
conda activate stllms_env
ls
cd steering-thinking-llms/
cd train-steering-vectors/
ls
cd results/
ls
cd ..
python explore_files.py 
l
ls
cd steering-thinking-llms/
ls
cd train-steering-vectors/results/
python explore_files.py 
cd ..
python explore_files.py 
python project_steering.py 
python explore_files.py 
python project_steering.py 
python explore_files.py 
python evaluate_steering.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --n_examples 50
cd ../steering/
python evaluate_steering.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --n_examples 50
pip install python-dotenv
conda activate stllms_env
pip install python-dotenv
python evaluate_steering.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --n_examples 50
pip install messages
python evaluate_steering.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --n_examples 50
pip uninstall messages
cd ..
python -m steering.evaluate_steering --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --n_examples 50
cd
conda deactivate
conda create -n sae_reasoning python=3.11
conda activate sae_reasoning
git clone https://github.com/AIRI-Institute/SAE-Reasoning.git
cd SAE-Reasoning/
ls
pip install -r requirements.txt
cd TransformerLens
pip install -e .
conda activate sae-reasoning
exit
conda env list
nvidia-smi
conda env list
conda activate sae_reasoning
pip show sae_lens
pip install sae_lens==5.5.2 sae-dashboard
nvidia-smi
ls
cd SAE-Reasoning/evaluation/
./evaluation.sh 
cd ..
cd evaluation/lm-evaluation-harness
pip install -e '.[vllm]'
nvidia-smi
cd ..
./evaluation.sh 
tmux new -s sae
conda activate sae_reasoning
cd SAE-Reasoning/evaluation/
ls
sbatch eval_job.sh
squeue --me
scancel 169559
sbatch eval_job.sh
squeue --me
scancel 169561
tmux attach -t sae
squeue --me
tmux attach -t sae
squeue --me
nvidia-smi
conda env list
conda activate sae_reasoning
ls
cd SAE-Reasoning/
ls
cd evaluation/
ls
python3 sae.py
python project_feature.py 
python steer_model.py 
python
python steer_model.py 
hf --help
hf auth login
python steer_model.py 
python new_steer_model.py 
conda activate sae_reasoning
cd SAE-Reasoning/
cd evaluation/
ls
python new_steer_model.py 
python your_script.py --target_layer 8 --epsilon 1.5 --apply_to_generated_only true --max_new_tokens 256 --num_examples 100 --dataset_split test
python steer_model_final.py --target_layer 8 --epsilon 1.5 --apply_to_generated_only true --max_new_tokens 256 --num_examples 100 --dataset_split test --run_baseline false
python steer_model_final.py --target_layer 8 --epsilon 1.5 --apply_to_generated_only true --max_new_tokens 1024 --num_examples 1000 --dataset_split train --run_baseline true
python steer_model_final.py --target_layer 8 --epsilon 1.5 --apply_to_generated_only true --max_new_tokens 1024 --num_examples 20 --dataset_split test --run_baseline true
bash eval_steering.sh 
sbatch eval_steering.sh
squeue --me
scancel 172258
sbatch eval_steering.sh
squeue --me
sbatch eval_steering.sh
squeue --me
python project
python project_feature.py 
python steer_model_final.py --target_layer 8 --epsilon 1.5 --apply_to_generated_only true --max_new_tokens 1024 --num_examples 300 --dataset_split test --run_baseline true --projected_feature_path projected_feature_48026.pt
python steer_model_final.py --target_layer 8 --epsilon 1.5 --apply_to_generated_only true --max_new_tokens 1024 --num_examples 10 --dataset_split test --run_baseline false --projected_feature_path projected_feature_48026.pt
sbatch eval_steering.sh
squeue --me
sbatch eval_steering.sh
squeue --me
scancel 172279
scancel 172278
scancel 172276
scancel 172274
squeue --me
sbatch eval_steering.sh
squeue --me
sbatch eval_steering.sh
squeue --me
sbatch eval_steering.sh
squeue --me
sbatch eval_steering.sh
squeue --me
conda activate sae_reasoning
cd SAE-Reasoning/evaluation/
python steer_model_final.py --target_layer 8 --epsilon 1.5 --apply_to_generated_only true --max_new_tokens 1024 --num_examples 1309 --dataset_split test --run_baseline false --projected_feature_path projected_feature_48026.pt --testing true
python steer_model_final.py --target_layer 8 --epsilon 1.5 --apply_to_generated_only true --max_new_tokens 1024 --num_examples 1309 --dataset_split test --run_baseline true --projected_feature_path projected_feature_48026.pt --testing true
sbatch eval_steering.sh
squeue --me
sbatch eval_steering.sh
squeue --me
python project_feature.py 
sbatch eval_steering.sh
squeue --me
sbatch eval_steering.sh
squeue --me
sbatch eval_steering.sh
squeue --me
sbatch eval_steering.sh
squeue --me
sbatch eval_steering.sh
scancel 172362
squeue --me
sbatch eval_steering.sh
squeue --me
sbatch eval_steering.sh
squeue --me
sbatch eval_steering.sh
squeue --me
sbatch eval_steering.sh
squeue --me
sbatch eval_steering.sh
squeue --me
sbatch eval_steering.sh
squeue --me
sbatch eval_steering.sh
squeue --me
python3 project_feature.py 
sbatch eval_steering.sh
scancel 172392
python steer_model_final.py --target_layer 7 --epsilon 0.5 --apply_to_generated_only true --max_new_tokens 1024 --num_examples 300 --dataset_split test --run_baseline false --projected_feature_path projected_feature_25953.pt --testing false
sbatch eval_steering.sh
squeue --me
python3 project_feature.py 
sbatch eval_steering.sh
scancel 172400
sbatch eval_steering.sh
python3 project_feature.py 
sbatch eval_steering.sh
scancel 172418
sbatch eval_steering.sh
scancel 172419
sbatch eval_steering.sh
scancel 172421
sbatch eval_steering.sh
squeue --me
conda activate sae_reasoning
cd SAE-Reasoning/evaluation/
sbatch eval_steering.sh
squeue -me
squeue --me
sbatch eval_steering.sh
squeue --me
sbatch eval_steering.sh
squeue --me
scancel 172571
sbatch eval_steering.sh
squeue --me
sbatch eval_steering.sh
squeue --me
conda activate sae_reasoning
cd SAE-Reasoning/evaluation/
sbatch eval_steering.sh
squeue --me
conda activate sae_reasoning
cd SAE-Reasoning/
ls
cd evaluation/
ls
sbatch eval_steering.sh
squeue --me
sbatch eval_steering.sh
squeue --me
sbatch eval_steering.sh
squeue --me
sbatch eval_steering.sh
squeue --me
sbatch eval_steering.sh
squeue --me
sbatch eval_steering.sh
squeue --me
sbatch eval_steering.sh
squeue --me
sbatch eval_steering.sh
squeue --me
sbatch eval_steering.sh
squeue --me
sbatch eval_steering.sh
squeue --me
sbatch eval_steering.sh
squeue --me
sbatch eval_steering.sh
squeue --me
sbatch eval_steering.sh
squeue --me
sbatch eval_steering.sh
conda activate sae_reasoning
cd SAE-Reasoning/evaluation/
sbatch eval_steering.sh
squeue
squeue --me
sbatch eval_steering.sh
squeue
squeue --me
conda activate sae_reasoning
cd SAE-Reasoning/evaluation/
python3 extract_results.py 
python3 analyze_results.py 
conda activate sae_reasoning
cd SAE-Reasoning/evaluation/
python analyze_results.py 
sbatch eval_steering.sh
squeue --me
conda activate sae_reasoning
cd SAE-Reasoning/evaluation/
python3 extract_results.py 
python3 analyze_results.py 
sbatch eval_steering.sh
squeue
squeue --me
squeue
sbatch eval_steering.sh
squeue --me
conda activate sae_reasoning
cd SAE-Reasoning/evaluation/
python3 extract_results.py 
squeue --me
python3 analyze_results.py 
sbatch eval_steering.sh
squeue --me
conda activate sae_reasoning
cd SAE-Reasoning/evaluation/
squeue --me
python3 extract_results.py 
python3 analyze_results.py 
conda activate turbenv
wget -c -r https://dataset-dl.liris.cnrs.fr/eagle_dataset/triangular.tar.gz
d
ls
cd eagle_dataset/
ls
cd ..
cd eagle_dataset/
pwd
cd ../dataset-dl.liris.cnrs.fr/
ls
cd eagle_dataset/
ls
mv triangular.tar.gz /home/nour/eagle/eagle_dataset
ls
cd ../..
rm -r dataset-dl.liris.cnrs.fr/
ls
cd eagle_dataset/
ls
tar -xzvf triangular.tar.gz 
ls
cd Eagle_dataset/
pwd
ls
cd
ls
cd eagle/
ls
cd eagle_dataset/
ls
cd Eagle_dataset/
ls
cd Cre
ls
cd 1
ls
cd 1
ls
cd ../../..
cd ..
ls
cd ..
ls
cd EagleMeshTransformer/
ls
python train_graphvit.py --dataset-path /home/nour/eagle/eagle_dataset/Eagle_dataset --cluster-path /home/nour/eagle/eagle_dataset/Eagle_dataset --run-id "graphvit_10" --output-path "trained_models/graphvit" --n-cluster=10
conda install conda-forge::torch-scatter
pip install torch-scatter
python
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu124.html
python -c "import torch-scatter"
python train_graphvit.py --dataset-path /home/nour/eagle/eagle_dataset/Eagle_dataset --cluster-path /home/nour/eagle/eagle_dataset/Eagle_dataset --run-id "graphvit_10" --output-path "trained_models/graphvit" --n-cluster=10
nvidia-smi
python train_graphvit.py --dataset-path /home/nour/eagle/eagle_dataset/Eagle_dataset --cluster-path /home/nour/eagle/eagle_dataset/Eagle_dataset --run-id "graphvit_10" --output-path "trained_models/graphvit" --n-cluster=10
CUDA_VISIBLE_DEVICES
echo $CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0
echo $CUDA_VISIBLE_DEVICES
python train_graphvit.py --dataset-path /home/nour/eagle/eagle_dataset/Eagle_dataset --cluster-path /home/nour/eagle/eagle_dataset/Eagle_dataset --run-id "graphvit_10" --output-path "trained_models/graphvit" --n-cluster=10
export CUDA_VISIBLE_DEVICES=0,1
echo $CUDA_VISIBLE_DEVICES
python train_graphvit.py --dataset-path /home/nour/eagle/eagle_dataset/Eagle_dataset --cluster-path /home/nour/eagle/eagle_dataset/Eagle_dataset --run-id "graphvit_10" --output-path "trained_models/graphvit" --n-cluster=10
python train_graphvit.py --dataset-path /home/nour/eagle/eagle_dataset/Eagle_dataset --cluster-path /home/nour/eagle/eagle_dataset/Eagle_dataset --run-id "graphvit_10" --output-path "trained_models/graphvit" --n-cluster=10 --batch_size 1
python train_graphvit.py --dataset-path /home/nour/eagle/eagle_dataset/Eagle_dataset --cluster-path /home/nour/eagle/eagle_dataset/Eagle_dataset --run-id "graphvit_10" --output-path "trained_models/graphvit" --n-cluster=10 --batch-size 1
python train_graphvit.py --dataset-path /home/nour/eagle/eagle_dataset/Eagle_dataset --cluster-path /home/nour/eagle/eagle_dataset/Eagle_dataset --run-id "graphvit_10" --output-path "trained_models/graphvit" --n-cluster=10 --batch-size 1 --w-size 128
python train_graphvit.py --dataset-path /home/nour/eagle/eagle_dataset/Eagle_dataset --cluster-path /home/nour/eagle/eagle_dataset/Eagle_dataset --run-id "graphvit_10" --output-path "trained_models/graphvit" --n-cluster=10 --batch-size 1 --w-size 64
python train_graphvit.py --dataset-path /home/nour/eagle/eagle_dataset/Eagle_dataset --cluster-path /home/nour/eagle/eagle_dataset/Eagle_dataset --run-id "graphvit_10" --output-path "trained_models/graphvit" --n-cluster=10 --batch-size 1 --w-size 16
python train_graphvit.py --dataset-path /home/nour/eagle/eagle_dataset/Eagle_dataset --cluster-path /home/nour/eagle/eagle_dataset/Eagle_dataset --run-id "graphvit_10" --output-path "trained_models/graphvit" --n-cluster=10 --batch-size 1 --w-size 256
sbatch run_job.sh
squeue
sbatch run_job.sh
squeue
sbatch run_job.sh
cd ../eagle_dataset/
ls
cd Eagle_dataset/
ls
cd Tri/
ls
cd 65/1
ls
python
pwd
python
cd ../../..
cd ..
ls
cd ../EagleMeshTransformer/
sbatch run_job.sh
cd /home/nour/eagle/eagle_dataset/Eagle_dataset/Spl/35/2
ls
cd /home/nour/eagle/eagle_dataset/Eagle_dataset/Spl/150/2
ls
cd 
ls
cd eagle/EagleMeshTransformer/
ls
sbatch run_job.sh
cd /home/nour/eagle/eagle_dataset/Eagle_dataset/Spl/67/2
ls
python
cd 
cd eagle/EagleMeshTransformer/
sbatch run_job.sh
sbatch eval_job.sh
sbatch run_job.sh
scancel 66498
sbatch run_job.sh
sbatch run_job_pinn.sh
scancel 66501
scancel 66499
sbatch run_job.sh
sbatch run_job_pinn.sh
squeue
scancel 67622
sbatch run_job_pinn.sh
scancel 67623
sbatch run_job_pinn.sh
conda activate turbenv
ls
wget -c -r https://dataset-dl.liris.cnrs.fr/eagle_dataset/spline.tar.gz
ls
pwd
cd dataset-dl.liris.cnrs.fr/
ls
cd eagle_dataset/
ls
mv spline.tar.gz /home/nour/eagle/eagle_dataset
ls
cd ..
ls
cd ..
rm -r dataset-dl.liris.cnrs.fr/
ls
tar -xzvf spline.tar.gz 
conda activate sae_reasoning
./evaluation.sh 
sbatch evaluation.sh
sbatch eval_job.sh
squeue --me
ls
sbatch eval_job.sh
squeue
squeue --me
export VLLM_USE_V1=0
sbatch eval_job.sh
squeue --me
  
squeue --me
python3 sae.py
python -c "import vllm; print(vllm.__version__)"
python -c "import inspect; from vllm import LLM; print(inspect.signature(LLM.generate))"
grep -n "prompt_token_ids" /home/nour/SAE-Reasoning/evaluation/lm-evaluation-harness/lm_eval/models/vllm_causallms.py
grep -n "model_executor" /home/nour/SAE-Reasoning/evaluation/lm-evaluation-harness/lm_eval/models/vllm_causallms.py
cd evaluation/lm-evaluation-harness
pip install -e '.[vllm]'
cd lm-evaluation-harness
pip install -e '.[vllm]'
python -c "import vllm; print(vllm.__version__)"
python -c "import inspect; from vllm import LLM; print(inspect.signature(LLM.generate))"
grep -n "prompt_token_ids" /home/nour/SAE-Reasoning/evaluation/lm-evaluation-harness/lm_eval/models/vllm_causallms.py
grep -n "model_executor" /home/nour/SAE-Reasoning/evaluation/lm-evaluation-harness/lm_eval/models/vllm_causallms.py
pip uninstall vllm
pip install -e '.[vllm]'
python -c "import vllm; print(vllm.__version__)"
pip uninstall vllm -y
pip install vllm==0.5.4
python -c "import vllm; print(vllm.__version__)"
pip install pyairports
python -c "import vllm; print(vllm.__version__)"
pip install pyairports
pip uninstall -y pyairports
pip install "https://files.pythonhosted.org/packages/6e/75/b424aebc9f2fc5db319d5df5fff62fa19254c8ef974c254588d48c480df2/pyairports-2.1.1-py3-none-any.whl"
python -c "import pyairports; print(pyairports.__file__)"
python -c "import vllm; print(vllm.__version__)"
cd ..
ls
sbatch eval_job.sh
squeue --me
scancel 170526
sbatch eval_job.sh
squeue --me
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_WORKER_MULTIPROC_METHOD=spawn
sbatch eval_job.sh
squeue --me
sbatch eval_job.sh
squeue --me
sbatch eval_job.sh
squeue --me
sbatch eval_job.sh
squeue --me
sbatch eval_job.sh
squeue --me
squeue --start
squeue --me --start
squeue
squeue --me --start
squeue --me
sbatch eval_job.sh
squeue --me
scancel 171688
sbatch eval_job.sh
squeue --me
squeue --me --start
sbatch eval_job.sh
squeue --me --start
squeue --me
squeue --me --start
sbatch eval_job.sh
squeue --me
squeue --me --start
squeue --me
squeue --me --start
nvidia-smi
sbatch eval_job.sh
squeue --me --start
python3 sae.py
ls
pwd
wget -c -r https://zenodo.org/records/7870707/files/NsCircle.zip?download=1
conda activate turbenv
ls
cd eagle/EagleMeshTransformer/
ls
sbatch run_job_pinn.sh 
scancel 66562
sbatch run_job_pinn.sh 
sacct
sacct -U
scancel 66565
sbatch run_job_pinn.sh 
squeue -U
squeue
sbatch run_job_pinn.sh 
sbatch eval_job.sh 
sbatch run_job_pinn.sh 
scancel 68237
sbatch run_job_pinn.sh 
scancel 68240
sbatch run_job_pinn.sh 
scancel 68241
scancel 68239
squeue

sbatch run_job_pinn.sh 
sbatch run_job.sh 
sbatch run_job_pinn.sh 
python
ssh pod-gpu
python
sbatch run_job_pinn.sh 
python make_vis.py --log_file /home/nour/eagle/EagleMeshTransformer/jobLogs/outLogPINN_2 --output_dir /home/nour/eagle/EagleMeshTransformer/loss_plots
python make_vis.py --log-file /home/nour/eagle/EagleMeshTransformer/jobLogs/outLogPINN_2 --output-dir /home/nour/eagle/EagleMeshTransformer/loss_plots
python3 make_vis.py --log_file /home/nour/eagle/EagleMeshTransformer/jobLogs/outLogPINN_2 --output_dir /home/nour/eagle/EagleMeshTransformer/loss_plots
make_vis.py [-h]
python make_vis.py [-h]
python3 make_vis.py --log_file /home/nour/eagle/EagleMeshTransformer/jobLogs/outLogPINN_2 --output_dir /home/nour/eagle/EagleMeshTransformer/loss_plots
python3 make_vis.py --log_file /home/nour/eagle/EagleMeshTransformer/jobLogs/outLogPINN_divab2 --output_dir /home/nour/eagle/EagleMeshTransformer/loss_plots 
python3 make_vis.py --log_file /home/nour/eagle/EagleMeshTransformer/jobLogs/outLogPINN_2 --output_dir /home/nour/eagle/EagleMeshTransformer/loss_plots
sbatch run_job_pinn.sh 
scancel 70179
sbatch run_job_pinn.sh 
python3 make_vis.py --log_file /home/nour/eagle/EagleMeshTransformer/jobLogs/outLogPINN_divmomab2 --output_dir /home/nour/eagle/EagleMeshTransformer/loss_plots 
python3 make_vis.py --log_file /home/nour/eagle/EagleMeshTransformer/jobLogs/outLog_OG --output_dir /home/nour/eagle/EagleMeshTransformer/loss_plots 
python3 make_vis_pinn.py --log_file1 /home/nour/eagle/EagleMeshTransformer/jobLogs/outLogPINN_divab2 --log_file2 /home/nour/eagle/EagleMeshTransformer/outLogPINN_divabres1 output_dir /home/nour/eagle/EagleMeshTransformer/loss_plots 
python3 make_vis_pinn.py --log_file1 /home/nour/eagle/EagleMeshTransformer/jobLogs/outLogPINN_divab2 --log_file2 /home/nour/eagle/EagleMeshTransformer/outLogPINN_divabres1 —output_dir /home/nour/eagle/EagleMeshTransformer/loss_plots 
python3 make_vis_pinn.py --log_file1 /home/nour/eagle/EagleMeshTransformer/jobLogs/outLogPINN_divab2 --log_file2 /home/nour/eagle/EagleMeshTransformer/outLogPINN_divabres1 --output_dir /home/nour/eagle/EagleMeshTransformer/loss_plots 
python3 make_vis_pinn.py --log_file1 /home/nour/eagle/EagleMeshTransformer/jobLogs/outLogPINN_divmomab2 --log_file2 /home/nour/eagle/EagleMeshTransformer/outLogPINN_divmomabres1 --output_dir /home/nour/eagle/EagleMeshTransformer/loss_plots 
python3 make_vis_pinn.py --log_file1 /home/nour/eagle/EagleMeshTransformer/jobLogs/outLogPINN_2 --log_file2 /home/nour/eagle/EagleMeshTransformer/outLogPINN_2_res1 --output_dir /home/nour/eagle/EagleMeshTransformer/loss_plots 
sbatch run_job_pinn.sh 
squeue
sbatch run_job_pinn.sh 
squeue
python3 make_vis_pinn.py --log_file1 /home/nour/eagle/EagleMeshTransformer/jobLogs/outLogPINN_2 --log_file2 /home/nour/eagle/EagleMeshTransformer/outLogPINN_2_res1 --output_dir /home/nour/eagle/EagleMeshTransformer/loss_plots 
python3 make_vis_pinn.py --log_file1 /home/nour/eagle/EagleMeshTransformer/jobLogs/outLogPINN_divmomab2 --log_file2 /home/nour/eagle/EagleMeshTransformer/outLogPINN_divmomabres1 --output_dir /home/nour/eagle/EagleMeshTransformer/loss_plots 
python3 make_vis_pinn.py --log_file1 /home/nour/eagle/EagleMeshTransformer/jobLogs/outLogPINN_divab2 --log_file2 /home/nour/eagle/EagleMeshTransformer/outLogPINN_divabres1 --output_dir /home/nour/eagle/EagleMeshTransformer/loss_plots 
python3 make_vis_pinn.py --log_file1 /home/nour/eagle/EagleMeshTransformer/jobLogs/outLogPINN_2 --log_file2 /home/nour/eagle/EagleMeshTransformer/outLogPINN_2_res1 --output_dir /home/nour/eagle/EagleMeshTransformer/loss_plots 
sbatch eval_job.sh 
vim eval_job.sh 
sbatch eval_job.sh 
ls outLogEval*
vim eval_job.sh 
sbatch eval_job.sh 
ls outLogEval*
vim eval_job.sh 
sbatch eval_job.sh 
vim eval_job.sh 
sbatch eval_job.sh 
ls outLogEval*
1;95;0c
ls outLogEval*
sbatch eval_job.sh 
ls outLogEval*
vim eval_job.sh 
sbatch eval_job.sh 
ls outLogEval*
vim eval_job.sh 
sbatch eval_job.sh 
ls outLogEval*
squeue
ls outLogEval*
vim eval_job.sh 
sbatch eval_job.sh 
ls outLogEval*
vim eval_job.sh 
sbatch eval_job.sh 
ls outLogEval*
