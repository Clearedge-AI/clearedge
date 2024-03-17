import subprocess
import sys
import platform

def install_system_packages_linux():
  try:
    subprocess.check_call(['sudo', 'apt', 'update'])
    subprocess.check_call(['sudo', 'apt', 'install', '-y', 'tesseract-ocr', 'poppler-utils'])
  except subprocess.CalledProcessError as e:
    print(f"Failed to install system packages on Linux: {e}", file=sys.stderr)
    sys.exit(1)

def install_system_packages_macos():
  try:
    subprocess.check_call(['brew', 'update'])
    subprocess.check_call(['brew', 'upgrade'])
    subprocess.check_call(['brew', 'install', 'tesseract'])
    subprocess.check_call(['brew', 'install', 'poppler'])
  except subprocess.CalledProcessError as e:
    print(f"Failed to install system packages on macOS: {e}", file=sys.stderr)
    sys.exit(1)

def install_wheel_file():
  try:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-U', 'layoutparser-0.0.0-py3-none-any.whl'])
  except subprocess.CalledProcessError as e:
    print(f"Failed to install the wheel file: {e}", file=sys.stderr)
    sys.exit(1)

def main():
  os_type = platform.system()
  if os_type == "Linux":
    install_system_packages_linux()
  elif os_type == "Darwin":
    install_system_packages_macos()
  else:
    print(f"Unsupported operating system: {os_type}", file=sys.stderr)
    sys.exit(1)

  install_wheel_file()

if __name__ == "__main__":
  main()
