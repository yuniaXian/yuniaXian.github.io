After the emergence of ChatGPT, there have been many open-source projects attempting to replicate its effects, including LLaMa, DeepSpeed-Chat, ColossalChat, ChatGLM, etc. Among them, DeepSpeed-Chat is an open-source project from the Microsoft Deep Speed team, which fully provides the code for three phases: Supervised Fine-tuning, Reward Model Training, and RLHF PPO Training. The logic is straightforward, and the module division is clear. Additionally, because Deep Speed is commonly used in large model training, I have recently been studying the code of DeepSpeed-Chat. This article will introduce the practical situation of running all three phases: SFT, RW, and RLHF on a 13b model.

# Step 1: SFT
This step we train a referenced model which will be used in PPO. The fine-tuned model will be used to initialize actor model in Step 3. Choose llms such as llama2, GPT3, OPT etc to used as referenced model.

## Tokenizer

Use original code when online, or modify it as following in offline environment

DeepSpeedExamples/applications/DeepSpeed-Chat/training/utils/utils.py：
```
def load_hf_tokenizer(model_name_or_path, fast_tokenizer=True):
   #if os.path.exists(model_name_or_path):
   #    # Locally tokenizer loading has some issue, so we need to force download
   #    model_json = os.path.join(model_name_or_path, "config.json")
   #    if os.path.exists(model_json):
   #        model_json_file = json.load(open(model_json))
   #        model_name = model_json_file["_name_or_path"]
   #        tokenizer = AutoTokenizer.from_pretrained(model_name,
   #                                                  fast_tokenizer=True)
   #else:
   tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, fast_tokenizer=True)
   return tokenizer
```

DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/main.py：

```
#tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
tokenizer = load_hf_tokenizer('your_dir/facebook_opt_13b', fast_tokenizer=True)

```
#### Data loader
Use cache for training data，modify file: DeepSpeedExamples/applications/DeepSpeed-Chat/training/utils/data/raw_datasets.py：

class PromptRawDataset(object):
    def __init__(self, output_path, seed, local_rank, dataset_name):
        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank
        if not dataset_name == 'local/jsonfile':
            #self.raw_datasets = load_dataset(dataset_name) # 即使dataset_name是本地目录，也会先联网，可以设置export HF_DATASETS_OFFLINE=1或换用load_from_disk
            self.raw_datasets = datasets.load_from_disk(dataset_name)

#### Script
modify original file: DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/training_scripts/single_node/run_13b.sh：
```
deepspeed main.py \
   --data_path your_dir/dahoas_rm_static \
   --data_split 2,4,4 \
   --model_name_or_path your_dir/facebook_opt_13b \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 512 \
   --learning_rate 1e-4 \
   --weight_decay 0. \
   --num_train_epochs 16  \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --lora_dim 128 \
   --lora_module_name decoder.layers. \
   --deepspeed \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log
```   

Start training

```
cd DeepSpeedExamples/applications/DeepSpeed-Chat
python train.py --step 1 --actor-model 13b --deployment-type single_node
```

You can check training status in `DeepSpeed-Chat/output/actor-models/13b/training.log`. Due to the nature of PPO, the performance can fluctuate.

##### Time estimation
It could take 1.5 hours per epoch, GPU takes $47$ G, $24$ G of it is occupied by model parameters. The following is an analysis of cache usage:
$$
n_{\text {transformer multi block memory }}=n_{\text {batch }} n_{\text {sequence }}\left(8 n_{\text {heads }} d_{\text {head }} n_{\text {layers }}+2 n_{\text {heads }} n_{\text {sequence }} n_{\text {layers }}+10 d_{\text {model }} n_{\text {layers }}+6 d_{f f n} n_{\text {layers }}\right)
$$

Let's say $n_{\text {batch }}=4, n_{\text {sequence }}=512, n_{\text {heads }}=40, n_{\text {layers }}=40, d_{\text {head }}=\frac{d_{\text {meade }}}{n_{\text {heads }}}, d_{\text {model }}=5120, d_{f f n}=20480$, it takes $19 G$. Full fine tune requires another $19$ G for store gradients and intermediate results. When we use Lora, we reduce the size of parameters. The ZeRO-Offload put intermediate results from optimizer to $\mathrm{CPU}$ .Model parameters: $24 \mathrm{G}$ with intermediate result $19 \mathrm{G}$ totally requires $43 \mathrm{G}$. 

# Step 2: Training the Reward Model
In this step, a reward model is trained to served to predict immediate reward (score) for each word predicted in step 3.
### tokenizer
DeepSpeedExamples/applications/DeepSpeed-Chat/training/step2_reward_model_finetuning/main.py：

```
tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
tokenizer = load_hf_tokenizer('your_dir/facebook_opt_350m', fast_tokenizer=True)
```

### Data loader
Same with SFT step

#### run script
Modify `DeepSpeedExamples/applications/DeepSpeed-Chat/training/step2_reward_model_finetuning/training_scripts/single_node/run_350m.sh：`

```
deepspeed main.py \
   --data_path your_dir/dahoas_rm_static \
   --data_split 2,4,4 \
   --model_name_or_path your_dir/facebook_opt_350m \
   --num_padding_at_beginning 1 \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 512 \
   --learning_rate 5e-5 \
   --weight_decay 0.1 \
   --num_train_epochs 1 \
   --disable_dropout \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log
```
```
cd DeepSpeedExamples/applications/DeepSpeed-Chat
python train.py --step 2 --reward-model 350m --deployment-type single_node
```

Training log is stored in DeepSpeed-Chat/output/reward-models/350m/training.log

# Step 3: RLHF
+ There are four models joining the Reinforcement Learning training process:
  + Reference model (from step 1)
  + Reward model(from step 2)
  + Actor model (initiated by parameters from referenced model)
  + Critic model (initiated by Reward model)

+ During the training, parameters in reference model and reward model are freezed. The rest of the models are fine-tuned. The final model we want is the actor model.
+ Referenced model offer loss which serves as part in KL divergence. Reward model provide immediate score for each prediction.

### Tokenizer
In default, DeepSpeed-Chat assumes that actor shares tokenizer with critic model.

### Run script
Warning: `--enable_hybrid_engine`: System could report error when we turn on DeepSpeed Hybrid Engine
.Disable it. In case of OOM: CUDA out of memory，we add --offload_reference_model.

```
deepspeed --master_port 12346 main.py \
   --data_path your_dir/dahoas_rm_static \
   --data_split 2,4,4 \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --num_padding_at_beginning 1 \
   --per_device_train_batch_size 16 \
   --per_device_mini_train_batch_size 16 \
   --generation_batch_numbers 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 256 \
   --actor_learning_rate ${Actor_Lr} \
   --critic_learning_rate ${Critic_Lr} \
   --num_train_epochs 1 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 1 \
   --num_warmup_steps 100 \
   --deepspeed --seed 1234 \
   --offload_reference_model \
   --inference_tp_size 2 \
   --actor_zero_stage $ACTOR_ZERO_STAGE \
   --critic_zero_stage $CRITIC_ZERO_STAGE \
   --actor_gradient_checkpointing \
   --disable_actor_dropout \
   --actor_lora_dim 128 \
   --actor_lora_module_name decoder.layers. \
   --output_dir $OUTPUT \
    &> $OUTPUT/training.log
```

Run command
```
cd DeepSpeedExamples/applications/DeepSpeed-Chat
python train.py --step 3 --actor-model 13b --reward-model 350m --deployment-type single_node
```
we can check log in `DeepSpeed-Chat/step3-models/13b/training.log`

This concludes the training.

# Reference:
https://github.com/microsoft/DeepSpeed
