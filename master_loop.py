import os
import sys
import time
import subprocess
from datetime import datetime

def main_live_loop():
    print("==================================================================")
    print("🤖 MASTER LOOP EXECUTIVE ACTIVATED — ALL SYSTEM CORES ARMED")
    print("==================================================================")
    print("This supervisor script initializes and protects your live_desk engine.")
    print("Intraday minute-by-minute scanning engine is completely live.\n")
    
    while True:
        try:
            # Transfer core control seamlessly to your live desk loop
            print("🚀 Starting real-time micro-desk controller...")
            subprocess.run([sys.executable, "live_desk.py"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Warning: Live desk exited with code {e.returncode}. Restarting core process in 10 seconds...")
            time.sleep(10)
        except KeyboardInterrupt:
            print("\n👋 System safely disarmed by supervisor request. Exiting terminal.")
            sys.exit(0)
        except Exception as e:
            print(f"❌ Unexpected interruption: {e}. Re-booting monitor stack...")
            time.sleep(10)

if __name__ == "__main__":
    main_live_loop()