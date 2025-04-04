from typing import Union, List


class InputExample:
    """
    Structure for one input example with texts, the label and a unique id
    """
    def __init__(self, guid: str = '', texts: List[str] = None, label: Union[int, float] = 0, edge_index=None,
                 edge_type=None, pos_ids=None,edge_index_mask=None,edge_type_mask=None,mask_idx=None,masked_labels=None):
        """
        Creates one InputExample with the given texts, guid and label


        :param guid
            id for the example
        :param texts
            the texts for the example.
        :param label
            the label for the example
        """
        self.guid = guid
        self.texts = texts
        self.label = label
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.pos_ids = pos_ids
        self.edge_index_mask = edge_index_mask
        self.edge_type_mask = edge_type_mask
        self.mask_idx = mask_idx
        self.masked_labels = masked_labels

    def __str__(self):
        return "<InputExample> label: {}, texts: {}".format(str(self.label), "; ".join(self.texts))
    def set_label(self, label):
        self.label = label
