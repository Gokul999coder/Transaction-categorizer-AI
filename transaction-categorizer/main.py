
import argparse, subprocess, sys, os
ROOT = os.path.dirname(__file__)

def run(cmd):
    print(">>",cmd)
    return subprocess.call(cmd, shell=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate', action='store_true', help='Generate synthetic data')
    parser.add_argument('--train', action='store_true', help='Train baseline')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate')
    parser.add_argument('--predict', type=str, help='Predict text')
    parser.add_argument('--retrain', action='store_true', help='Retrain from feedback')
    parser.add_argument('--data', default='data/synthetic_transactions.csv')
    parser.add_argument('--models', default='models/')
    args = parser.parse_args()

    if args.generate:
        run(f'python -m src.data_ingestion.generate_synthetic --out {args.data} --n 3000')
    if args.train:
        run(f'python -m src.model.train_baseline --data {args.data} --models {args.models}')
    if args.evaluate:
        run(f'python -m src.evaluation.evaluate --data {args.data} --models {args.models}')
    if args.predict:
        run(f'python -m src.model.predict --text \"{args.predict}\" --models {args.models}')
    if args.retrain:
        run(f'python -m src.model.retrain_from_feedback --data {args.data} --feedback data/feedback.csv --models {args.models}')
