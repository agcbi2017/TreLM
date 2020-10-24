import warnings

import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertConfig
from transformers.modeling_bert import BertLayer, BertPooler, BertOnlyMLMHead, BertPreTrainingHeads
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput
from torch.nn import CrossEntropyLoss
from transformers.file_utils import ModelOutput
from dataclasses import dataclass
from typing import Optional, List, Union, Dict, Tuple

class TrelmBertConfig(BertConfig):

    model_type = "trelm_bert"

    def __init__(self, n_langs=2, langs_to_id=None, **kwargs):
        super().__init__(n_langs=n_langs, langs_to_id=langs_to_id, **kwargs)


class TrelmBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

        assert config.num_hidden_layers % 2 == 0, "num_hidden_layers must be even in Trelm!"
        self.tlayer = BertLayer(config) #
        self.tlayer_position = int(config.num_hidden_layers / 2)

    def forward(
        self,
        hidden_states,
        tlayer_lang_embs, # B x T x D
        tlayer_pos_embs, # B x T x D
        ordering_alignment=None, # B x T
        ordering_attention_mask=None, # B x T
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            # tlayer encoding
            if i == self.tlayer_position:
                if ordering_alignment is not None:
                    bs, ts, hd = hidden_states.size()
                    assert ordering_alignment.size() == (bs, ts)
                    ordered_hidden_states = []
                    for bidx in range(bs):
                        ordered_hidden_states.append(hidden_states[bidx].index_select(0, ordering_alignment[bidx]).unsqueeze(0))
                    hidden_states = torch.cat(ordered_hidden_states, dim=0)

                hidden_states = hidden_states + tlayer_lang_embs + tlayer_pos_embs

                layer_outputs = self.tlayer(
                    hidden_states,
                    attention_mask if ordering_attention_mask is None else ordering_attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )
                hidden_states = layer_outputs[0]

            
            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


class TrelmBertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.lang_embeddings = nn.Embedding(config.n_langs, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, lang_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if lang_ids is None:
            lang_ids = torch.ones(input_shape, dtype=torch.long, device=self.position_ids.device)
        elif lang_ids.size(1) == 1:
            lang_ids = lang_ids.expand(input_shape)
        else:
            assert lang_ids.size() == input_shape

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        lang_embeddings = self.lang_embeddings(lang_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings + lang_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings, position_embeddings


class TrelmBertModel(BertPreTrainedModel):

    config_class = TrelmBertConfig
    base_model_prefix = "trelm_bert"

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = TrelmBertEmbeddings(config)
        self.encoder = TrelmBertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)
            if layer == self.encoder.tlayer_position:
                self.encoder.tlayer.attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        lang_ids=None, # B x 1
        tlayer_lang_ids=None, # B x 1
        ordering_alignment=None, # B x T
        ordering_attention_mask=None, # B x T
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        #--Trelm--
        if lang_ids is None:
            lang_ids = torch.ones(input_shape, dtype=torch.long, device=device) # default is the new language
        elif lang_ids.size(1) == 1:
            lang_ids = lang_ids.expand(input_shape)
        else:
            assert lang_ids.size() == input_shape
        
        if tlayer_lang_ids is None:
            tlayer_lang_ids = lang_ids
        elif tlayer_lang_ids.size(1) == 1:
            tlayer_lang_ids = tlayer_lang_ids.expand(input_shape)
        else:
            assert tlayer_lang_ids.size() == input_shape
        
        tlayer_lang_embs = self.embeddings.lang_embeddings(tlayer_lang_ids)
        extended_ordering_attention_mask = None
        if ordering_alignment is not None:
            assert ordering_attention_mask is not None, "ordering_attention_mask can not be none when ordering_alignment is not None!"
            extended_ordering_attention_mask: torch.Tensor = self.get_extended_attention_mask(ordering_attention_mask, input_shape, device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output, position_embeddings = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        encoder_outputs = self.encoder(
            embedding_output,
            tlayer_lang_embs=tlayer_lang_embs, # B x T x D
            tlayer_pos_embs=position_embeddings, # B x T x D
            ordering_alignment=ordering_alignment, # B x T
            ordering_attention_mask=extended_ordering_attention_mask, # B x T
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class TrelmBertForMaskedLM(BertPreTrainedModel):

    config_class = TrelmBertConfig
    base_model_prefix = "trelm_bert"

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `TrelmBertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.trelm_bert = TrelmBertModel(config)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(
        self,
        input_ids=None,
        lang_ids=None, # B x 1
        tlayer_lang_ids=None, # B x 1
        ordering_alignment=None, # B x T
        ordering_attention_mask=None, # B x T
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        if "masked_lm_labels" in kwargs:
            warnings.warn(
                "The `masked_lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("masked_lm_labels")
        assert "lm_labels" not in kwargs, "Use `BertWithLMHead` for autoregressive language modeling task."
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.trelm_bert(
            input_ids,
            lang_ids=lang_ids, # B x 1
            tlayer_lang_ids=tlayer_lang_ids, # B x 1
            ordering_alignment=ordering_alignment, # B x T
            ordering_attention_mask=ordering_attention_mask, # B x T
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class TrelmBertForSequenceClassification(BertPreTrainedModel):

    config_class = TrelmBertConfig
    base_model_prefix = "trelm_bert"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.trelm_bert = TrelmBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        lang_ids=None, # B x 1
        tlayer_lang_ids=None, # B x 1
        ordering_alignment=None, # B x T
        ordering_attention_mask=None, # B x T
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.trelm_bert(
            input_ids,
            lang_ids=lang_ids, # B x 1
            tlayer_lang_ids=tlayer_lang_ids, # B x 1
            ordering_alignment=ordering_alignment, # B x T
            ordering_attention_mask=ordering_attention_mask, # B x T
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@dataclass
class TrelmBertForPreTrainingOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class TrelmBertForPreTraining(BertPreTrainedModel):

    config_class = TrelmBertConfig
    base_model_prefix = "trelm_bert"

    def __init__(self, config):
        super().__init__(config)

        self.trelm_bert = TrelmBertModel(config)
        self.cls = BertPreTrainingHeads(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        next_sentence_label=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        if "masked_lm_labels" in kwargs:
            warnings.warn(
                "The `masked_lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("masked_lm_labels")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.trelm_bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        total_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return TrelmBertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )