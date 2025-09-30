#!/usr/bin/env python3
"""
Detect if TeamFight Tactics (League of Legends) is running on macOS
"""

import subprocess
import psutil
import time

def is_tft_running_psutil():
    """Check if TFT/League is running using psutil (cross-platform)"""
    tft_processes = [
        "League of Legends",
        "LeagueClient",
        "LeagueClientUx",
        "RiotClientServices",
        "Riot Client",
        "League of Legends Helper"
    ]

    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for tft_name in tft_processes:
                if tft_name.lower() in proc.info['name'].lower():
                    return True, proc.info['name'], proc.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    return False, None, None

def is_tft_running_ps():
    """Check if TFT/League is running using ps command (macOS/Linux)"""
    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True,
            check=True
        )

        processes_to_check = [
            "League of Legends",
            "LeagueClient",
            "RiotClient",
            "League of Legends Helper"
        ]

        for line in result.stdout.splitlines():
            for proc_name in processes_to_check:
                if proc_name in line and "grep" not in line:
                    return True, line.split()[10:], line.split()[1]  # process name and PID

    except subprocess.CalledProcessError:
        pass

    return False, None, None

def monitor_tft_status(check_interval=5):
    """Monitor TFT status continuously"""
    print("Monitoring TeamFight Tactics status...")
    print("-" * 50)

    was_running = False

    while True:
        is_running, process_name, pid = is_tft_running_psutil()

        if is_running and not was_running:
            print(f"✅ TFT/League STARTED - Process: {process_name} (PID: {pid})")
            was_running = True
        elif not is_running and was_running:
            print(f"❌ TFT/League STOPPED")
            was_running = False
        elif is_running:
            print(f"🎮 TFT/League is running - {process_name} (PID: {pid})")
        else:
            print(f"⏸️  TFT/League is not running")

        time.sleep(check_interval)

if __name__ == "__main__":
    # Single check
    print("Checking if TeamFight Tactics is running...")

    # Method 1: Using psutil
    is_running, process_name, pid = is_tft_running_psutil()
    if is_running:
        print(f"✅ TFT/League is running! Process: {process_name} (PID: {pid})")
    else:
        print("❌ TFT/League is not currently running")

    # Uncomment to start monitoring
    # monitor_tft_status(check_interval=3)