import argparse
import os

args = None


def get_args():
    global args
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", type=str, default="data/longalign64")
        parser.add_argument("--minibatch_size", type=int, default=4)
        parser.add_argument("--micro_batch_size", type=int, default=2)
        parser.add_argument("--max_token_len", type=int, default=65536)
        parser.add_argument("--limit_dataset_token_len", type=int, default=None)
        parser.add_argument(
            "--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        )
        parser.add_argument("--packing_method", type=str, default="None")
        parser.add_argument("--run_name", type=str, default=None)
        parser.add_argument("--project_name", type=str, default="odc-training")
        parser.add_argument("--forward_only", action="store_true")
        parser.add_argument("--cost_model", type=str, default="linear")
        args = parser.parse_args()
        if os.environ.get("FORWARD_ONLY", "0") == "1":
            args.forward_only = True
        if "RUN_NAME" in os.environ:
            args.run_name = os.environ["RUN_NAME"]
    return args
