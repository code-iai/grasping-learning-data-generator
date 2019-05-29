_cram_to_word_net_object_ = {'BOWL': 'bowl.n.01',
                             'CUP': 'cup.n.01',
                             'SPOON': 'spoon.n.01',
                             'BREAKFAST-CEREAL': 'cereal.n.03',
                             'DRAWER': 'drawer.n.01',
                             'MILK': 'box.n.01'
                             }


def get_word_net_object(cram_object, default_object=None):
    if default_object:
        return _cram_to_word_net_object_.get(cram_object, default_object)

    return _cram_to_word_net_object_.get(cram_object, cram_object)