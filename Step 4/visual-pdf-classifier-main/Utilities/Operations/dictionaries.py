def merge_dictionaries(a, b, path=None):
    "merges b into a"

    a = dict(a)
    b= dict(b)

    if path is None: path = []
    for key in b.keys():
        if key in a.keys():
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dictionaries(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass # same leaf value
            else:
                raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a