import subprocess
import sys

print("📦 Installing requirements …")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "-r", "requirements.txt"])

print("🔑 Logging in to HuggingFace …")
try:
    from google.colab import userdata
    from huggingface_hub import login
    token = userdata.get("HF_TOKEN")
    login(token=token, add_to_git_credential=False)
    print("✅ HuggingFace login successful.")
except Exception as e:
    print(f"⚠️  Could not auto-login ({e}).")
    print("   Run:  from huggingface_hub import login; login()")

print("\n✅ Setup complete. Run:  !python run_benchmark.py --limits")
