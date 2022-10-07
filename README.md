# Prompt on Speech Self-Supervised Learning Model
This repo is for the research of adding prefix tuning (prompt) on speech self-supervised learning model. The codes are modified from https://github.com/s3prl/s3prl and https://github.com/facebookresearch/fairseq. I experimented with HuBERT, DeCoAR2, and Wav2Vec2 on 6 SUPERB downstream tasks (ASR, IC, PR, SF, KS, SD, SID), and explore its benefits in stability and low data regime. My teamates and my work is together submitted SLT 2022. Publication: Zih-Ching Chen, Allen Fu, Chih-Ying Liu, Hung-yi Lee, and Shang-Wen Li, “Exploring Efficient-tuning Methods in Self-supervised Speech Models”, IEEE Spoken Language Technology Workshop, 2022.

## Usage
1. Get initialization
```
cd s3prl/s3prl/
python run_downstream.py -m train -n {ANY NAME} -u {UPSTREAM} -d {DOWNSTREAM} --get_init True --task {task, ex: ASR}
```
The embeddings will be saved to ```hubert_{task}_emb_weight/```
3. Training
    ```
    cd s3prl/s3prl/
    python run_downstream.py -m train -n {EXP NAME} -u {UPSTREAM} -d {DOWNSTREAM} -c {config path}
    --prompt {"prefix" or "preinput"} --prompt_len {prompt len}
    --prompt_init {True or False} --task {task, ex: ASR}
    --pretrain_downstream_path {checkpoint for loading pretrained downstream model}
    ```
- prompt
    - prefix: for deep prefix prompt
    - preinput: for shallo prefix prompt
- prompt_len
    - integer, prompt length
- prompt_init
    - True or False, whether initialize the prompt with the embedding saved in the previous step
- task
    - Initialize prompt from ```hubert_{task}_emb_weight```. Need to specify when ```prompt_init``` is True
- pretrain_downstream_path
    - checkpoint path of pretrained downstream model (In default, don't need to specify)
5. Testing
    ```
    cd s3prl/s3prl
    python run_downstream.py -m evaluate -e {CHECKPOINT} --prompt --prompt_len
    ```
    Need to specify ```prompt``` and ```prompt_len``` if the checkpoint contains prompt.
## Modified files
- ```fairseq/fairseq/models/wav2vec/wav2vec2.py```
- ```s3prl/s3prl/optimizers.py```
    Add prompt parameters to the optimizer.
    Modify functions: get_optimizer(), get_TorchOptim()
    In downstream config,
    ```
    optimizer:
      name: TorchOptim 
      torch_optim_name: Adam
      lr: 1.0e-4
      prompt_lr: 1.0e-3 
    ```
    - If the name isn't TorchOptim, need to modify corresponding get_{name} function in ```optimizers.py```
    - Can modify prompt_lr (If it isn't speicfied, the the default value is lr)
- ```s3prl/s3prl/run_downstream.py```
    Add arguments
- ```s3prl/s3prl/downstream/runner.py```
- ```s3prl/s3prl/upstream/interfaces.py```
Modify function to_list(), to deal with the assertion error of violating the input length when adding prompt


