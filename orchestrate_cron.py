import subprocess
import sys
import datetime
import time

def execute_daily_routine():
    pipeline_scripts = [
        "get_universe.py",
        "live_portfolio.py",
        "execute_trades.py"
    ]
    
    for script in pipeline_scripts:
        now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n[{now_str}] Executing: {script}...")
        
        try:
            result = subprocess.run([sys.executable, script], check=True)
            if result.returncode == 0:
                print(f"✅ Success: {script} finished.")
        except subprocess.CalledProcessError as e:
            print(f"❌ [CRITICAL ERROR] Pipeline broken at {script}: {e}")
            return False
            
    print("\n🎉 Daily portfolio rotation completed successfully.")
    return True

def main_orchestrator_loop():
    print("==================================================================")
    print("🤖 QUANT ORCHESTRATOR IS NOW LIVE ON HIGH-PRECISION CLOCK MONITOR")
    print("==================================================================")
    
    startup_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"✅ [SUCCESS] Clock monitor initialized successfully at {startup_time}!")
    print("📡 Monitoring active. Waiting for target windows (06:50 AM / 06:30 PM)...")
    print("------------------------------------------------------------------")
    
    last_morning_execution_date = ""
    last_evening_execution_date = ""
    
    while True:
        try:
            now = datetime.datetime.now()
            current_date_str = now.strftime("%Y-%m-%d")
            
            if now.weekday() < 5:
                if now.hour == 6 and now.minute == 50 and last_morning_execution_date != current_date_str:
                    print(f"\n⚡ [CLOCK ALERT] Target 06:50:00 hit! Firing morning portfolio rotation...")
                    execute_daily_routine()
                    last_morning_execution_date = current_date_str
                    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Morning cycle complete.")

                elif now.hour == 18 and now.minute == 30 and last_evening_execution_date != current_date_str:
                    print(f"\n⚡ [CLOCK ALERT] Target 18:30:00 hit! Firing evening portfolio rotation...")
                    execute_daily_routine()
                    last_evening_execution_date = current_date_str
                    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Evening cycle complete.")

                if now.second % 10 == 0:
                    sys.stdout.write(f"\r🔍 [Scanning Clock] Current Time: {now.strftime('%H:%M:%S')} | Standby Mode active... ")
                    sys.stdout.flush()
                time.sleep(1)
            else:
                time.sleep(1)
                
        # --- FIXED: Global safeguard so infrastructure does not completely brick on network/OS hitches ---
        except Exception as system_fault:
            print(f"\n⚠️ [ORCHESTRATOR WARNING] Loop hit an internal snag: {system_fault}")
            print("Cooldown activated... Resuming scan in 10 seconds.")
            time.sleep(10)

if __name__ == "__main__":
    try:
        main_orchestrator_loop()
    except KeyboardInterrupt:
        print("\n\n👋 Background orchestrator safely shut down.")
        sys.exit(0)