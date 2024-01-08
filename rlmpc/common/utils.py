import yaml

ACADOS_MULTIPLIER_ORDER = [
    "lbu",
    "lbx",
    "lg",
    "lh",
    "lphi",
    "ubu",
    "ubx",
    "ug",
    "uh",
    "uphi",
    "lsbu",
    "lsbx",
    "lsg",
    "lsh",
    "lsphi",
    "usbu",
    "usbx",
    "usg",
    "ush",
    "usphi",
]


def rename_key_in_dict(d: dict, old_key: str, new_key: str):
    d[new_key] = d.pop(old_key)
    return d


def rename_item_in_list(lst: list, old_item: str, new_item: str):
    if old_item in lst:
        index_old = lst.index(old_item)
        lst[index_old] = new_item

    return lst


def read_config(config_file):
    with open(config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config
