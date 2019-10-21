def batch_sizes_to_tuple(object_literal):
    if 'batch_sizes' in object_literal:
        object_literal['batch_sizes'] = tuple(object_literal['batch_sizes'])
    return object_literal
