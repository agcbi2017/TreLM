import warnings

import torch
import torch.nn as nn
from transformers import ElectraPreTrainedModel, ElectraConfig
from transformers.modeling_electra import ElectraLayer, ElectraGeneratorPredictions
from torch.nn import CrossEntropyLoss


class TrelmElectraConfig(ElectraConfig):

    model_type = "trelm_electra"

    def __init__(self, n_langs=2, langs_to_id=None, **kwargs):
        super().__init__(n_langs=n_langs, langs_to_id=langs_to_id, **kwargs)


class TrelmElectraEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([ElectraLayer(config) for _ in range(config.num_hidden_layers)])

        assert config.num_hidden_layers % 2 == 0, "num_hidden_layers must be even in Trelm!"
        self.tlayer = ElectraLayer(config) #
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


class TrelmElectraEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)

        self.lang_embeddings = nn.Embedding(config.n_langs, config.embedding_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
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
            lang_ids = torch.ones(input_shape, dtype=torch.long, device=self.position_ids.device) # default is the new language
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


class TrelmElectraModel(ElectraPreTrainedModel):

    config_class = TrelmElectraConfig
    base_model_prefix = "trelm_electra"

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = TrelmElectraEmbeddings(config)

        if config.embedding_size != config.hidden_size:
            self.embeddings_project = nn.Linear(config.embedding_size, config.hidden_size)

        self.encoder = TrelmElectraEncoder(config)
        self.config = config
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
            lang_ids = torch.ones(input_shape, dtype=torch.long, device=device)
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
        if hasattr(self, "embeddings_project"):
            tlayer_lang_embs = self.embeddings_project(tlayer_lang_embs)
        extended_ordering_attention_mask = None
        if ordering_alignment is not None:
            assert ordering_attention_mask is not None, "ordering_attention_mask can not be none when ordering_alignment is not None!"
            extended_ordering_attention_mask: torch.Tensor = self.get_extended_attention_mask(ordering_attention_mask, input_shape, device)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        hidden_states, position_embeddings = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        if hasattr(self, "embeddings_project"):
            hidden_states = self.embeddings_project(hidden_states)
            position_embeddings = self.embeddings_project(position_embeddings)

        hidden_states = self.encoder(
            hidden_states,
            tlayer_lang_embs=tlayer_lang_embs, # B x T x D
            tlayer_pos_embs=position_embeddings, # B x T x D
            ordering_alignment=ordering_alignment, # B x T
            ordering_attention_mask=extended_ordering_attention_mask, # B x T
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return hidden_states


class TrelmElectraForMaskedLM(ElectraPreTrainedModel):
    
    config_class = TrelmElectraConfig
    base_model_prefix = "trelm_electra"

    def __init__(self, config):
        super().__init__(config)

        self.trelm_electra = TrelmElectraModel(config)
        self.generator_predictions = ElectraGeneratorPredictions(config)
        self.generator_lm_head = nn.Linear(config.embedding_size, config.vocab_size)
        self.init_weights()

    def get_output_embeddings(self):
        return self.generator_lm_head

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

        generator_hidden_states = self.trelm_electra(
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
        generator_sequence_output = generator_hidden_states[0]

        prediction_scores = self.generator_predictions(generator_sequence_output)
        prediction_scores = self.generator_lm_head(prediction_scores)

        loss = None
        # Masked language modeling softmax layer
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + generator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=generator_hidden_states.hidden_states,
            attentions=generator_hidden_states.attentions,
        )