#!/usr/bin/env python3
"""Upload evaluation logs to Hugging Face Hub."""

import argparse
from pathlib import Path

from huggingface_hub import HfApi, login


def upload_logs(
    logs_dir: str,
    repo_id: str,
    repo_type: str = "dataset",
    token: str | None = None,
) -> None:
    """Upload logs directory to Hugging Face Hub.

    Args:
        logs_dir: Path to logs directory
        repo_id: Repository ID (e.g., 'username/repo-name')
        repo_type: Type of repository ('dataset' or 'model')
        token: HF token (if None, will use HF_TOKEN env var or prompt login)
    """
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        raise ValueError(f"Logs directory not found: {logs_dir}")

    # Login if token provided
    if token:
        login(token=token)

    api = HfApi()

    # Upload the entire logs folder
    print(f"Uploading {logs_dir} to {repo_id}...")
    api.upload_folder(
        folder_path=str(logs_path),
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo="logs",
    )
    print(f"✓ Upload complete: https://huggingface.co/{repo_type}s/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload logs to Hugging Face Hub")
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="logs",
        help="Path to logs directory (default: logs)",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Repository ID (e.g., 'username/repo-name')",
    )
    parser.add_argument(
        "--repo_type",
        type=str,
        default="dataset",
        choices=["dataset", "model"],
        help="Repository type (default: dataset)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HF token (if not provided, uses HF_TOKEN env var)",
    )

    args = parser.parse_args()
    upload_logs(args.logs_dir, args.repo_id, args.repo_type, args.token)


if __name__ == "__main__":
    main()