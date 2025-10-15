import os
import json
import csv
from datetime import datetime
import torch
import subprocess
import numpy as np

def convert_to_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    return obj

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
            'hyperparameters': convert_to_serializable(self.hyperparams),
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
            'rmse': float(metrics[0]),
            'mse': float(metrics[1]),
            'pearson': float(metrics[2]),
            'spearman': float(metrics[3]),
            'ci': float(metrics[4])
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
            'protein_model': self.hyperparams.get('protein_model', 'N/A'),
            'protein_model_index': self.hyperparams.get('protein_model_index', 'N/A'),
            'best_mse': float(best_metrics[1]),
            'best_ci': float(best_metrics[4]),
            'best_epoch': best_epoch,
            'total_epochs': total_epochs,
            'duration_hours': f'{duration_hours:.2f}',
            'timestamp': datetime.now().isoformat()
        }

        fieldnames = list(summary_row.keys())
        existing_rows = []

        if os.path.isfile(self.summary_csv):
            with open(self.summary_csv, 'r', newline='') as f:
                reader = csv.DictReader(f)
                existing_fieldnames = reader.fieldnames or []
                for row in reader:
                    existing_rows.append(row)

            for key in existing_fieldnames:
                if key not in fieldnames:
                    fieldnames.append(key)

            for row in existing_rows:
                for key in fieldnames:
                    row.setdefault(key, 'N/A')
        else:
            existing_fieldnames = []

        for key in fieldnames:
            summary_row.setdefault(key, 'N/A')

        existing_rows.append(summary_row)

        with open(self.summary_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in existing_rows:
                writer.writerow(row)

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
