import subprocess
import sys

def main():
    print("JANITOR: Delegating to compiler-pipeline...")
    # The new janitor is inside compiler-pipeline.py
    subprocess.run([sys.executable, "compiler-pipeline.py", "--cleanup"])

if __name__ == "__main__":
    main()
