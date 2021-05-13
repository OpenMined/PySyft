lines = open("setup.cfg").readlines()
mark = "#install-custom-dependency"
trigger = False

for line in lines:
    if "[lib]" in line:
        trigger = True
        continue

    if trigger:
        print(line[1:].rsplit(mark)[0])
