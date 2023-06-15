
def pick_max_answer(f):
    f_copy = deepcopy(f)
    fnal = []
    for i, new in enumerate(f):
        item = f_copy[i]
        new_id = new['node_bindings']['input_chemical'][0]['id']
        if new_id in [binding['id'] for binding in new['node_bindings']['input_chemical']]:
            f.remove(item)
            item.update(new)
            fnal.append(item)
    return fnal
