from typing import Iterable

def multiline_str(obj, level = 0) -> str:
    if isinstance(obj, str):
        return obj
    elif isinstance(obj, dict):
        body = ",\n".join([f"{'\t'*(level + 1)}{k}: {multiline_str(v, level=level+2)}" for k, v in obj.items()])
        return ("{\n" + 
                body +
                "\n" + "\t"*level + "}"
                )
    elif isinstance(obj, Iterable):
        enclosing = ('(', ')')
        if isinstance(obj, list):
            enclosing = ('[', ']')
        elif isinstance(obj, set):
            enclosing = ('{', '}')
        body = ",\n".join(f"{'\t'*(level + 1)}{multiline_str(x, level=level+2)}" for x in obj)
        return (f"{enclosing[0]}\n" + body + f"\n{'\t'*level}{enclosing[1]}")
    else:
        return str(obj)
        