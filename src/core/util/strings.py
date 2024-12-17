from functools import wraps

def truncate_string(string : str, max_length : int, truncate_suffix : str = ""):
    """
    Truncate a string to a maximum length.

    :param string: the string to truncate
    :param max_length: the maximum length of the return value
    :param truncate_suffix: An optional suffix to add when the string is truncated.
        May contain a {chars} format string which will be formatted
         with the number of characters truncated.

    :return: The truncated string
    """
    trunc_chars = len(string) - max_length
    if trunc_chars <= 0:
        return string
    total_trunc_chars = 0
    while True:
        string = string[:-trunc_chars]
        total_trunc_chars += trunc_chars
        formatted_suffix = truncate_suffix.format(chars=total_trunc_chars)
        trunc_chars = len(string) + len(formatted_suffix) - max_length
        if trunc_chars <= 0:
            return string + formatted_suffix
        elif len(string) == 0:
            # We can't truncate anymore, truncate the suffix instead
            return formatted_suffix[:max_length]

def multiline_repr(obj, recursive : bool = False, **override_fields : str | None):
    """
    Create a multiline repr for an object.
    """
    body = []
    value_repr = multiline_repr if recursive else repr
    def prepend_lines(string):
        lines = string.splitlines()
        prepended = [f"\t{line}" for line in lines[1:]]
        return lines[0] + ''.join(prepended)
    for key, value in obj.__dict__.items():
        if key in override_fields:
            value = override_fields[key]
            if value is None:
                continue
        body.append(f"\t{key}={prepend_lines(value_repr(value))}")
    return f"{obj.__class__.__name__}(\n{',\n'.join(body)}\n)"

