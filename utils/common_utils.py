def say_tp_message(desc: str, strdata: str) -> None:
    """
    Print a simple (two part) message consisting of two strings
    :param desc: string describing what data will be output next:
    :param strdata: meaningful data represented as a string
    :return: returns nothing, but does print to STDOUT
    """
    message = "".join([desc, strdata])
    print(message)
