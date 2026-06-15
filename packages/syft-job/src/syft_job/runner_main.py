#!/usr/bin/env python3
"""
SyftJob Runner - Monitors inbox folder for new jobs.
"""

import argparse
import sys
from pathlib import Path

from .job_runner import create_runner


def main():
    """Main entry point for syft-job-runner."""
    parser = argparse.ArgumentParser(
        description="SyftJob Runner - Monitors inbox folder for new jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start monitoring jobs
  syft-job-runner --syftbox-folder /path/to/SyftBox --email user@example.com

  # Reset all jobs and recreate folder structure
  syft-job-runner --syftbox-folder /path/to/SyftBox --email user@example.com --reset

  # Reset and then start monitoring
  syft-job-runner --syftbox-folder /path/to/SyftBox --email user@example.com --reset --run
        """,
    )
    parser.add_argument(
        "--syftbox-folder", required=True, help="Path to the SyftBox folder"
    )
    parser.add_argument("--email", required=True, help="User email address")
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=5,
        help="How often to check for new jobs (in seconds, default: 5)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete all jobs and recreate folder structure",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Start monitoring after reset (only used with --reset)",
    )

    args = parser.parse_args()

    syftbox_folder_path = Path(args.syftbox_folder).expanduser().resolve()
    if not syftbox_folder_path.exists():
        print(f"❌ Error: SyftBox folder not found: {syftbox_folder_path}")
        sys.exit(1)

    if not syftbox_folder_path.is_dir():
        print(f"❌ Error: Path is not a directory: {syftbox_folder_path}")
        sys.exit(1)

    try:
        runner = create_runner(str(syftbox_folder_path), args.email, args.poll_interval)

        # Handle reset flag
        if args.reset:
            print("🔄 Reset requested - deleting all jobs and recreating structure")
            runner.reset_all_jobs()
            print()  # Add spacing

            # Determine if we should run after reset
            should_run = args.run
            if not args.run:
                # Ask user if they want to start monitoring
                print(
                    "🤔 Jobs have been reset. Would you like to start monitoring? (y/N): ",
                    end="",
                )
                response = input().strip().lower()
                should_run = response in ["y", "yes"]

            if should_run:
                print("🚀 Starting job monitoring...")
                runner.run()
            else:
                print("✅ Reset completed. Job monitoring not started.")
        else:
            # Normal monitoring mode
            runner.run()

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
