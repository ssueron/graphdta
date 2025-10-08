import os
import json
import csv
from datetime import datetime
import torch
import subprocess

class ExperimentManager:
    def __init__(self, model_name, dataset_name, hyperparams, exp_name=None):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.hyperparams = hyperparams

        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        if exp_name:
            exp_id = f"{timestamp}_{exp_name}"
        else:
            exp_id = f"{timestamp}_{model_name}_{dataset_name}"

        self.exp_dir = os.path.join('experiments', 'runs', exp_id)
        self.checkpoint_dir = os.path.join(self.exp_dir, 'checkpoints')
        self.log_dir = os.path.join(self.exp_dir, 'logs')
        self.result_dir = os.path.join(self.exp_dir, 'results')

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)

        self.best_checkpoint_path = os.path.join(self.checkpoint_dir, 'best.pt')
        self.latest_checkpoint_path = os.path.join(self.checkpoint_dir, 'latest.pt')
        self.metrics_log_path = os.path.join(self.log_dir, 'metrics.csv')
        self.metadata_path = os.path.join(self.exp_dir, 'metadata.json')
        self.summary_csv = os.path.join('experiments', 'summary.csv')

        self._save_metadata()
        self._init_metrics_log()

    def _get_git_commit(self):
        try:
            return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
        except:
            return 'unknown'

    def _save_metadata(self):
        metadata = {
            'model': self.model_name,
            'dataset': self.dataset_name,
            'timestamp': datetime.now().isoformat(),
            'git_commit': self._get_git_commit(),
            'hyperparameters': self.hyperparams,
            'experiment_dir': self.exp_dir
        }
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _init_metrics_log(self):
        with open(self.metrics_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'test_rmse', 'test_mse', 'test_pearson', 'test_spearman', 'test_ci'])

    def log_epoch(self, epoch, train_loss, metrics):
        with open(self.metrics_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss] + metrics)

    def save_checkpoint(self, model, optimizer, epoch, metrics, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }

        torch.save(checkpoint, self.latest_checkpoint_path)

        if is_best:
            torch.save(checkpoint, self.best_checkpoint_path)

    def load_checkpoint(self, checkpoint_type='latest'):
        path = self.best_checkpoint_path if checkpoint_type == 'best' else self.latest_checkpoint_path

        if os.path.exists(path):
            checkpoint = torch.load(path, weights_only=False)
            return checkpoint
        return None

    def save_final_results(self, predictions, labels, metrics):
        import pandas as pd

        df = pd.DataFrame({
            'true': labels,
            'predicted': predictions
        })
        df.to_csv(os.path.join(self.result_dir, 'predictions.csv'), index=False)

        metrics_dict = {
            'rmse': metrics[0],
            'mse': metrics[1],
            'pearson': metrics[2],
            'spearman': metrics[3],
            'ci': metrics[4]
        }
        with open(os.path.join(self.result_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics_dict, f, indent=2)

    def update_summary(self, best_metrics, best_epoch, total_epochs, duration_hours):
        os.makedirs('experiments', exist_ok=True)

        exp_id = os.path.basename(self.exp_dir)
        summary_row = {
            'experiment_id': exp_id,
            'model': self.model_name,
            'dataset': self.dataset_name,
            'best_mse': best_metrics[1],
            'best_ci': best_metrics[4],
            'best_epoch': best_epoch,
            'total_epochs': total_epochs,
            'duration_hours': f'{duration_hours:.2f}',
            'timestamp': datetime.now().isoformat()
        }

        file_exists = os.path.isfile(self.summary_csv)

        with open(self.summary_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summary_row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(summary_row)

def find_latest_experiment(model_name, dataset_name):
    runs_dir = os.path.join('experiments', 'runs')
    if not os.path.exists(runs_dir):
        return None

    pattern = f"{model_name}_{dataset_name}"
    matching_dirs = [d for d in os.listdir(runs_dir) if pattern in d]

    if not matching_dirs:
        return None

    matching_dirs.sort(reverse=True)
    latest_dir = os.path.join(runs_dir, matching_dirs[0])

    checkpoint_path = os.path.join(latest_dir, 'checkpoints', 'latest.pt')
    if os.path.exists(checkpoint_path):
        return checkpoint_path

    return None
