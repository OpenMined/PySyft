lines = open("setup.cfg").readlines()
mark = "#install-custom-dependency"

for line in lines:
    if line.startswith(";") and line.endswith("#install-custom-dependency\n"):
        print(line[1:].rsplit(mark)[0])
