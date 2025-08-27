from typing import Optional, Tuple, Union
from transformers import GPT2LMHeadModel
import torch
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, CausalLMOutputWithPast
from torch.nn import CrossEntropyLoss
from transformers import BioGptTokenizer, BioGptForCausalLM, GPT2LMHeadModel,GPT2Tokenizer, LlamaForCausalLM 


class QAModelLlama(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
       

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values:Optional[Tuple[Tuple[torch.Tensor]]] = None,# Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()
        

        logits = logits[..., :-1, :] #shifitng
        out = logits[...,-1, -2:] #TRUE and FALSE probs only - after shifitng

        loss = None
        if labels is not None:
            labels = labels[..., -1].clone()
            labels[labels == TRUE_TOKEN_ID] = 0 # TRUE token is 50262/263 , 2nd to the last column of lm_logits, 0th column of out
            labels[labels != 0] = 1
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            out = out.contiguous()
            labels = labels.contiguous()

            loss_fct = CrossEntropyLoss()

            loss = loss_fct(out.view(-1, out.size(-1)), labels.view(-1))
            # print("loss:\t", loss)
            # print("labels:\t", labels)
            # print("input:\t", input_ids)
            # sys.stdout.flush()
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class QAModelGPT2(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
       
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = True, #None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)


        lm_logits = lm_logits[..., :-1, :] #shifitng
        out = lm_logits[...,-1, -2:] #TRUE and FALSE probs only - after shifitng
        loss = None
        if labels is not None:
            labels = labels[..., -1].clone()
            labels[labels == TRUE_TOKEN_ID] = 0 # TRUE token is 50262/263 , 2nd to the last column of lm_logits, 0th column of out
            labels[labels != 0] = 1
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            out = out.contiguous()
            labels = labels.contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(out.view(-1, out.size(-1)), labels.view(-1))
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )




class QAModelBioGPT(BioGptForCausalLM):
    def __init__(self, config):
        super().__init__(config)
       

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # print(return_dict)
        outputs = self.biogpt(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # print(outputs)
        sequence_output = outputs[0]
        prediction_scores = self.output_projection(sequence_output)

        prediction_scores = prediction_scores[..., :-1, :] #shifitng
        out = prediction_scores[...,-1, -2:] #TRUE and FALSE probs only - after shifitng

        lm_loss = None
        if labels is not None:

            labels = labels[..., -1].clone()
            labels[labels == TRUE_TOKEN_ID] = 0 # TRUE token is  2nd to the last column of lm_logits, 0th column of out
            labels[labels != 0] = 1
            # move labels to correct device to enable model parallelism
            labels = labels.to(prediction_scores.device)
            out = out.contiguous()
            labels = labels.contiguous()
            loss_fct = CrossEntropyLoss()#weight=torch.Tensor([POS_WEIGHT, NEG_WEIGHT]).to(prediction_scores.device))
            lm_loss = loss_fct(out.view(-1, out.size(-1)), labels.view(-1))


        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )