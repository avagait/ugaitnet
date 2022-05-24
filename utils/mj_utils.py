import sys


def mj_isDebugging():
    """
    Check if program is running in a debugger
    :return: True if debugger is detected, False otherwise
    """
    gettrace = getattr(sys, 'gettrace', None)

    if gettrace is None:
        isD = False
    elif gettrace():
        isD = True
    else:
        isD = False

    return False#isD
