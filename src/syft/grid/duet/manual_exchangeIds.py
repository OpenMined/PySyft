# stdlib


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def server_send_then_get(id_to_send: str) -> str:
    # send Server ID
    print()
    print(
        "♫♫♫ > "
        + bcolors.HEADER
        + "STEP 1:"
        + bcolors.ENDC
        + " Send the following Duet Server ID to your Duet Partner!"
    )
    #         print(f"♫♫♫ > Duet Node ID:{domain.id.value}")
    print("♫♫♫ > Duet Server ID: " + bcolors.BOLD + id_to_send + bcolors.ENDC)
    # get Client ID
    print(
        "\n♫♫♫ > "
        + bcolors.HEADER
        + "STEP 2:"
        + bcolors.ENDC
        + " Have your duet partner send Client ID to you and enter it below!"
    )
    while True:
        target_id = input("♫♫♫ > Duet Partner's Client ID: ")  # nosec
        if len(target_id) == 32:
            break
        else:
            print("    > Error: Invalid Client ID. Please try again.")
    print()
    return target_id


def client_get_then_send(id_to_send: str) -> str:
    # get server ID
    print(
        "\n♫♫♫ > "
        + bcolors.HEADER
        + "STEP 1:"
        + bcolors.ENDC
        + " Have your duet partner send an Server ID to you and enter it below!"
    )
    while True:
        target_id = input("♫♫♫ > Duet Partner's Server ID: ")  # nosec
        if len(target_id) == 32:
            break
        else:
            print("    > Error: Invalid Server ID. Please try again.")

    # send client ID

    print()
    print(
        "♫♫♫ > "
        + bcolors.HEADER
        + "STEP 2:"
        + bcolors.ENDC
        + " Send the following Duet Client ID to your duet partner!"
    )
    print("♫♫♫ > Duet Client ID: " + bcolors.BOLD + id_to_send + bcolors.ENDC)
    print()
    print("♫♫♫ > ...waiting for partner to connect...")

    return target_id
