#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2025/04/21 11:17:07

import torch
import torch.nn as nn

from transformers import LlamaForCausalLM, LlamaConfig
from copy import deepcopy
import pdb

torch.manual_seed(42)

class PolicyLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.eps = 0.2

    def forward(self, new_actor_log_probs, old_actor_log_probs, advantage):
        '''
        new_actor_log_probs 新actor的概率分布，就是off-policy那部分，用采样推理出的老数据再训练actor 
        old_actor_log_probs 老actor的概率分布，off-policy之前那部分
        
        Advantage = Penalized_Reward(Reward+Critic模型，但是没看到gamma项？) - Values(Critic模型)
        
        最后loss返回加上负号的作用
        如果advantage > 0 要提高ratio 那这个损失要变小（从大的负值->小的负值）
        如果advantage < 0 要降低ratio 那这个损失要变小（从大的正值->小的正值） 
        '''

        kl_divergence = new_actor_log_probs - old_actor_log_probs   # log(a/b)
        ratio = torch.exp(kl_divergence)
        surrogate_objectives = torch.min(ratio * advantage, ratio.clamp(1-self.eps, 1+self.eps) * advantage)   
     
        return torch.mean(-surrogate_objectives) 

class ValueLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.eps = 0.2

    def forward(self, values, reward, old_values):
        '''
        Reward 是实际预期收益(做了某个行为，有一些repo里命名成returns)，
        它应该由两部分组成：即时收益R_t（reward_model）和未来收益折现 gamma * V_t+1 (critic_model)

        Values 是预估预期收益（不做某个行为）
        values 实时critic_model跑出来的预估预期收益 
        old_values 老critic_model跑出来的预估预期收益
    
        Advantage = Reward - Values
        求损失的时候是
        loss = (Reward - Values) ** 2   
        P.S 有的写成：(Values - Reward) ** 2  也对，surrogate_values越小越好 对应loss数值越小越好
        '''
        values_clipped = old_values + (values - old_values).clamp(-self.eps, self.eps) # 让values的值不要超过old_values太多,有个上下限
        surrogate_values = torch.max(torch.square(values - reward), torch.square(values_clipped - reward))
        return torch.mean(surrogate_values)

class RewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sft_model = LlamaForCausalLM(config=LlamaConfig(vocab_size=1000, num_hidden_layers=1, hidden_size=128))
        self.reward_head = nn.Linear(128, 1) # 加一个线性层

    def forward(self, input_ids):
        hidden_states = self.sft_model(input_ids, output_hidden_states=True)["hidden_states"]
        last_hidden = hidden_states[-1] 
        rewards = self.reward_head(last_hidden).mean(dim=1) #输出是(seq_length, 1), 然后求平均值变成一个标量    
        return rewards

import trl


def run_ppo(): # one epoch
    # T = length
    
    def generate(model, input_ids, max_new_tokens=5, temperature=0.8, top_k=20, top_p=0.95): # 推理
        # top_p 和 top_k 不应该混着用，这里只做个示例
        for _ in range(max_new_tokens):
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]

            v, _ = torch.topk(logits, top_k, dim=-1)  # top_k 
            logits[logits < v[:, [-1]]] = -float('Inf')
    
            probs = torch.softmax(logits / temperature, dim=-1)
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1) # 从最大的开始累加
            mask = (probs_sum - probs_sort) > top_p # 不包含当前位置的概率
            probs_sort[mask] = 0.0
        
            next_token_idx = torch.multinomial(probs_sort, num_samples=1)# 注意是索引位置
           
            next_token_id = torch.gather(probs_idx, -1, next_token_idx)
            input_ids = torch.cat((input_ids, next_token_id), dim=1)
            
        return input_ids
    
    def forward(model, input_ids, indices):  # 训练
        outputs = model(input_ids)
        log_prob_all_vocab =  torch.log_softmax(outputs.logits[:, :-1, :], dim=-1) 
        indices = indices.unsqueeze(-1)[:, 1:, :]
        log_prob_output = torch.gather(log_prob_all_vocab, dim=-1, index=indices) 
        # loss = -log_prob_output.sum()/log_prob_output.numel()  # BTW loss 是这么计算出来的
        return log_prob_output

    actor_criterion = PolicyLoss()
    critic_criterion = ValueLoss()

    sft_model = LlamaForCausalLM(config=LlamaConfig(vocab_size=1000, num_hidden_layers=1, hidden_size=128))
    actor_model = deepcopy(sft_model)

    reward_model = RewardModel()
    critic_model = deepcopy(reward_model) 

    prompt_ids = [1, 2, 3, 4, 5, 6] # Question
    prompt_ids = torch.LongTensor([prompt_ids,]) 

    target_ids = [11, 10, 9, 8, 7]  # Answer(Ground Truth)
    target_ids = torch.LongTensor([target_ids,]) 
    
    completion = generate(sft_model, prompt_ids) # completion = Question + Response(非GT)
    indices = torch.cat((prompt_ids, target_ids), dim=-1) # indices = Question + Answer(GT)
    
    sft_log_probs = forward(sft_model, completion, indices)
    actor_log_probs = forward(actor_model, completion, indices)
    
    reward = reward_model(completion)
    values = critic_model(completion)

    # deepspeed-chat RLHF 里的缩放Reward 
    def kl_penalized_reward(reward, log_prob_rl, log_prob_sft):
        kl_beta = 0.1
        kl_divergence = log_prob_rl - log_prob_sft   # log(a/b)
        ratio = torch.exp(kl_divergence)
        
        # 这里不太明白 MinChatGPT 选择了另一种方式k3  http://joschu.net/blog/kl-approx.html  
        k1 = -kl_divergence
        k2 = 1/2* torch.square(kl_divergence)
        k3 = (ratio - 1) - kl_divergence
        estimated_kl = torch.mean(k3)
        return reward - kl_beta * estimated_kl

    penalized_reward = kl_penalized_reward(reward, actor_log_probs, sft_log_probs) 

    advantage = penalized_reward - values
    
    actor_loss = actor_criterion(actor_log_probs, sft_log_probs, advantage)
    actor_loss.backward()

    critic_loss = critic_criterion(values, penalized_reward, values)
    critic_loss.backward() # 算梯度
    # optimizer.step() # 更新参数....
    pdb.set_trace()
    




def main():
    pass

    run_ppo()

if __name__ == '__main__':
    pass
    main()


