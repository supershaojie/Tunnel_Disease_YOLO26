def check_version(current, required):
    return int(current.split(".")[0]) >= int(required.strip(">=").split(".")[0])
