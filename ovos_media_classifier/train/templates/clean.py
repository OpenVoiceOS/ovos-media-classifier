import os


def validate_line(line):
    if "{" not in line:
        return False, "line without slots"
    line = line.replace("{{", "{").replace("}}", "}")
    expected_slots=line.split(",", 1)[0].split("+")
    if not all("{" + s + "}" in line for s in expected_slots):
        return False, "invalid slots label"
    return True, ""

for csv in os.listdir(os.path.dirname(os.path.realpath(__file__))):
    if not csv.endswith(".csv"):
        continue
    print("csv:", csv)
    errors = {}
    with open(csv) as csvfile:
        lines = csvfile.read().split("\n")
        header = lines[0]
        lines = set(lines[1:])
        total = len(lines)
        validated = []
        for l in lines:
            valid, err = validate_line(l)
            if not valid:
                errors.setdefault(err, [])
                errors[err].append(l)
                print(err, l)
            else:
                validated.append(l)

    print("total:", total, "errors:", total - len(validated))
    

    with open(csv, "w") as csvfile:
        CSV = [header] + sorted(validated)
        csvfile.write("\n".join(CSV))
