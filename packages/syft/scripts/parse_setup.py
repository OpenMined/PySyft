# stdlib
import platform

p = platform.system().lower().replace("darwin", "macos")
lines = open("setup.cfg").readlines()
mark = "#install-custom-dependency"

for line in lines:
    if line.startswith(";") and line.endswith("#install-custom-dependency\n"):
        if p == "windows" and "petlib" in line:
            print("Skipping petlib on Windows")
        else:
            print(line[1:].rsplit(mark)[0])
