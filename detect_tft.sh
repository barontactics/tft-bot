#!/bin/bash

# Function to check if TFT/League is running
check_tft() {
    if pgrep -f "League of Legends" > /dev/null || \
       pgrep -f "LeagueClient" > /dev/null || \
       pgrep -f "RiotClient" > /dev/null; then
        echo "✅ TeamFight Tactics/League is running"
        return 0
    else
        echo "❌ TeamFight Tactics/League is not running"
        return 1
    fi
}

# Single check
check_tft

# Optional: Monitor continuously
# while true; do
#     check_tft
#     sleep 5
# done