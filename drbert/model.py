import logging
import warnings

from torch import nn
from transformers import BertModel
from transformers.modeling_bert import BertPreTrainedModel

from .constants import SEQUENCE_CLASSIFICATION_TASKS
from .constants import TASKS
from .modules.heads import SequenceClassificationHead
from .modules.heads import SequenceLabellingHead

logger = logging.getLogger(__name__)


class DrBERT(BertPreTrainedModel):
    def __init__(self, config):
        super(DrBERT, self).__init__(config)

        # Core BERT model
        self.bert = BertModel(config)
        self.init_weights()

        self.classification_heads = nn.ModuleDict()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, labels=None, name=None):
        if name is not None and name not in set(self.classification_heads.keys()):
            err_msg = (f"name must belong to one of the registered classification heads. Expected"
                       f" one of {list(self.classification_heads.keys())}. Got '{name}'.")
            logger.error('ValueError: %s', err_msg)
            raise ValueError(err_msg)

        inputs = {
            "bert": self.bert,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
            "head_mask": head_mask,
            "labels": labels,
        }

        # If name is provided, we are feeding the data through BERT + ONE classification head
        if name is not None:
            outputs = self.classification_heads[name](**inputs)
        # Otherwise, assume we are performing inference using ALL classification heads
        else:
            if labels is not None:
                warnings.warn('"task" is None but "labels" were given, they will be ignored.')
            outputs = {}
            for name, head in self.classification_heads.items():
                outputs[name] = head(**inputs)

        return outputs  # (loss), scores, (hidden_states), (attentions)

    def register_classification_head(self, name, task, num_labels):
        """Register a new classification head.

        Args:
            name (str): A unique name which can be used to access the classification head.
            task (str): Which type of task this classification head will perform. Must be in
                `drbert.constants.TASKS`.
            num_labels (int or tuple): The number of target classes for this head. Will be a tuple
                in the case of multi-label datasets.

        Raises:
            ValueError: If `name` is already registered (i.e. it is in `self.classification_heads`).
            ValueError: If `task` is not in `drbert.constants.TASKS`.

        Returns:
            torch.nn.Module: The classification head created.
        """
        if name in self.classification_heads:
            err_msg = (f'{name} is already a registered classification head. Please use'
                       ' unique names for each head.')
            logger.error('ValueError: %s', err_msg)
            raise ValueError(err_msg)
        if task not in TASKS:
            err_msg = f'task must be one of {TASKS}. Got {task}'
            logger.error('ValueError: %s', err_msg)
            raise ValueError(err_msg)

        if task == 'sequence_labelling':
            self.classification_heads[name] = SequenceLabellingHead(self.config, num_labels)
        elif task == 'document_classification':
            err_msg = 'Document classification is not yet implemented.'
            logger.error('ValueError: %s', err_msg)
            raise NotImplementedError('Document classification is not yet implemented.')
        elif task in SEQUENCE_CLASSIFICATION_TASKS:
            self.classification_heads[name] = SequenceClassificationHead(self.config, num_labels)

        # HACK (John): This assumes all params on same device. Is there a better way to do this?
        # Place the head on the same device as its parent model
        self.classification_heads[name].to(next(self.parameters()).device)

        return self.classification_heads[name]
