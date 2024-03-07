Usage
Use run.py for all functions (zero-shot/ICL/fine-tuning/MeZO):

python run.py {ARGUMENTS}
Please read run.py for a complete list of arguments. We introduce some of the most important ones below.

--num_train: Number of training examples. For ICL, this is the number of demonstrations.
--num_dev: Number of validation examples.
--num_test: Number of testing examples.
--model_name: HuggingFace model name or path.
--task_name: Task name.
--trainer: can be none (zero-shot/ICL), regular (fine-tuning), or zo (MeZO).
--train_as_classification: turn this on for classification tasks (Cross Entropy over likelihood of each class' label words). Otherwise it is LM-style teacher forcing.
--zo_eps: MeZO hyperparameter epsilon.
--prefix_tuning: use prefix-tuning.
--lora: use LoRA.
We also support all HuggingFace trainer arguments for easily setting fine-tuning hyperparameters.

We provide example scripts below for reproducing our experiments. All our examples sample 1,000 training examples, 500 validation examples, and 1,000 testing examples.


# HiZOO (full-parameter, prefix-tuning, and LoRA)
MODEL=facebook/opt-13b TASK=CB MODE=ft LR=1e-6 EPS=1e-3 HESSIAN_SMOOTH_TYPE=constant1e-10 bash HiZOO.sh
MODEL=facebook/opt-13b TASK=CB MODE=prefix LR=1e-2 EPS=1e-1 HESSIAN_SMOOTH_TYPE=constant1e-10 bash HiZOO.sh
MODEL=facebook/opt-13b TASK=CB MODE=lora LR=1e-5 EPS=1e-2 HESSIAN_SMOOTH_TYPE=constant1e-6 bash HiZOO.sh

# HiZOO with non-differentiable objective (SQuAD (F1) + HiZOO prefix as an example)
MODEL=facebook/opt-13b TASK=SQuAD MODE=prefix LR=1e-2 EPS=1e-1 HESSIAN_SMOOTH_TYPE=constant1e-4 bash HiZOO.sh --non_diff --evaluation_strategy no --save_strategy no --save_model
Our recommended hyperparameter search range for OPT-family can be found in appendix of our paper.
