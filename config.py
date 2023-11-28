import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default='train')
parser.add_argument("--dataset", type=str, default='nyt10')
parser.add_argument("--model_name", type=str, default='exp')
parser.add_argument("--pretrain_path", type=str, default='pretrain/bert-base-uncased')
parser.add_argument("--seed", type=int, default=36)
parser.add_argument("--train_batch_size", type=int, default=16)
parser.add_argument("--eval_batch_size", type=int, default=8)
parser.add_argument("--max_epoch", type=int, default=3)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--eps", type=float, default=1e-8)
parser.add_argument("--warmup_steps", type=int, default=500)
parser.add_argument("--max_steps", type=int, default=100000)
parser.add_argument("--max_grad_norm", type=float, default=5.0)
parser.add_argument("--grad_acc_steps", type=int, default=1)
parser.add_argument("--save_steps", type=int, default=1000)
parser.add_argument("--dropout_prob", type=float, default=0.1)
parser.add_argument("--temperature", type=float, default=0.05)
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument("--bag_size", type=int, default=3)
parser.add_argument("--max_bag_size", type=int, default=10)
parser.add_argument("--loss_weight", action='store_true')
parser.add_argument("--writer", action='store_true')
parser.add_argument("--mil", type=str, default='att')
parser.add_argument("--dont_save_logger", action='store_true')
parser.add_argument("--tau_c_max", type=float, default=0.97)
parser.add_argument("--tau_clean", type=float, default=0.85)
parser.add_argument("--filter_stop_epoch", type=int, default=2)
parser.add_argument("--T", type=float, default=0.5)
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--lambda_u", type=float, default=16.0)
parser.add_argument("--mixmatch_begin_epoch", type=int, default=1)

args, unknown = parser.parse_known_args()