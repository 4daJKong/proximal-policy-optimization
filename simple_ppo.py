#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2025/04/21 11:17:07
import trl

import torch
import torch.nn as nn

from transformers import LlamaForCausalLM, LlamaConfig
from copy import deepcopy
import pdb

torch.manual_seed(42)

'''
（解决）问题1 reward现在输入直接是reward_model计算的， 但是定义上是V_t = R_t + gamma * V_t+1 ? 但是后面这个折现怎么做呢
（解决）问题2 现在没有GAE 
        问题3 也没有off-policy
（解决）问题4 make_experience / generate_experience 是否在生成completion的时候就是解决了
        问题5 reward_forward 的输出应该是[seq_len,1] 但是actor_model的输出log_prob应该是[seq_len-1,1],这个没有解决，主要是概率分布差一位的问题
    问题6 log_prob 的index 用什么做计算？是label的index还是completion的index？

'''
class PolicyLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.eps = 0.2

    def forward(self, new_actor_log_probs, old_actor_log_probs, advantage):
        '''
        这个不太确定：
        new_actor_log_probs 新actor的概率分布，就是off-policy那部分，用采样推理出的老数据再训练actor 
        old_actor_log_probs 老actor的概率分布，off-policy之前那部分
        
        Advantage = Penalized_Reward(主要由Reward模型，但具体V_t+1和GAE还要考虑) - Values(Critic模型)
        
        最后loss返回加上负号的作用
        如果advantage > 0 要提高ratio 那这个损失要变小（从大的负值->小的负值）
        如果advantage < 0 要降低ratio 那这个损失要变小（从大的正值->小的正值） 
        '''

        kl_divergence = new_actor_log_probs - old_actor_log_probs   # log(a/b)
        ratio = torch.exp(kl_divergence)
        surrogate_objectives = torch.min(advantage * ratio, advantage * ratio.clamp(1-self.eps, 1+self.eps))   
     
        return torch.mean(-surrogate_objectives)  # 也有的用的是torch.sum(surrogate_objectives)

class ValueLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.eps = 0.2

    def forward(self, values, old_values, returns):
        '''
        从概念上：
        Reward 是实际预期收益（做了某个行为后）应由两部分组成：即时收益R_t（reward_model产生）和未来收益折现 gamma * V_t+1 (critic_model产生) P.S 还有GAE的部分
        Values 是预估预期收益（不做某个行为）
        Advantage = Reward - Values （returns实际上是Reward）
        求损失的时候是
        loss = (Reward - Values) ** 2   
        P.S 有的写成：(Values - Reward) ** 2  也对，surrogate_values越小越好 对应loss数值越小越好
        
        实现：
        values      实时critic_model跑出来的预估预期收益 （是变动的，随着ppo epoch迭代而改变）
        old_values  老critic_model跑出来的预估预期收益（固定值？）
    
        '''
        values_clipped = old_values + (values - old_values).clamp(-self.eps, self.eps)  # 让values的值不要超过old_values太多,有个上下限
        # 上面这个式子等价于：
        # values_clipped = torch.clamp(values, old_values-self.eps, old_values+self.eps)

        surrogate_values = 0.5 * torch.max(torch.square(values - returns), torch.square(values_clipped - returns))
    
        return torch.mean(surrogate_values) # 也有的用的是torch.sum(surrogate_values)

class RewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sft_model = LlamaForCausalLM(config=LlamaConfig(vocab_size=1000, num_hidden_layers=1, hidden_size=128))
        self.reward_head = nn.Linear(128, 1) # 加一个线性层

    def forward(self, input_ids):
        hidden_states = self.sft_model(input_ids, output_hidden_states=True)["hidden_states"]
        last_hidden = hidden_states[-1]  
        rewards = self.reward_head(last_hidden).squeeze(-1) # (bsz, seq_len)
        
        # rewards.mean(dim=-1) # 求平均值变成一个标量，有的repo(MinChatGPT)是这样做的
      
        return rewards[:, :-1] # 是[:, :-1] 还是 [:, 1:]  还是直接返回呢？这里不太明白，因为reward和log_prob shape一致，但是log_prob 应该是少一位的


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
        log_prob_all_vocab =  torch.log_softmax(outputs.logits[:, :-1, :], dim=-1) # 这里不太明白，因为相当于log_prob[:-1],但是怎么和critic和reward形状上对齐呢？
        indices = indices.unsqueeze(-1)[:, 1:, :]
        log_prob_output = torch.gather(log_prob_all_vocab, dim=-1, index=indices) 
        # loss = -log_prob_output.sum()/log_prob_output.numel()  # BTW, loss 是这么计算出来的
        return log_prob_output

    actor_criterion = PolicyLoss()
    critic_criterion = ValueLoss()

    sft_model = LlamaForCausalLM(config=LlamaConfig(vocab_size=1000, num_hidden_layers=1, hidden_size=128))
    actor_model = LlamaForCausalLM(config=LlamaConfig(vocab_size=1000, num_hidden_layers=1, hidden_size=128))

    reward_model = RewardModel()
    critic_model = RewardModel()

    prompt_ids = [1, 2, 3, 4, 5, 6] # Question
    prompt_ids = torch.LongTensor([prompt_ids,]) 

    target_ids = [11, 10, 9, 8, 7]  # Ground Truth (GT)
    target_ids = torch.LongTensor([target_ids,])  # (bsz=1,seq_len) 注意不要：torch.LongTensor(target_ids)  
    
    indices = torch.cat((prompt_ids, target_ids), dim=-1) # indices = Question + GT
    ''' for _ in mini_epochs:(off-policy的事，)  这个应该是和重要性采样相关吧，但是该没有加'''
    
    # generate_experience 部分，
    completion = generate(actor_model, prompt_ids) # completion = Question + Response(非GT)

    sft_log_probs = forward(sft_model, input_ids=completion, indices=completion)
    actor_log_probs = forward(actor_model, input_ids=completion, indices=completion)
 
    rewards = reward_model(completion) # shape(bsz,seq_len) -> reward.mean(dim=-1) scalar
    values = critic_model(completion) # shape(bsz,seq_len) -> values.mean(dim=-1) scalar

    def kl_penalized_reward(rewards, log_prob_rl, log_prob_ref):
        kl_ctl = 0.1
        kl_divergence = log_prob_rl - log_prob_ref   # log(a/b)
        
        # http://joschu.net/blog/kl-approx.html
        # 这里不太明白 MinChatGPT 选择了另一种方式 k3  
        # 具体地，有如下几种不同计算  kl_divergence 的实现方法
        # kl_divergence = -(log_prob_rl - log_prob_ref)  # k1
        # kl_divergence = 1/2 * torch.square(log_prob_rl - log_prob_ref) # k2
        # kl_divergence = (torch.exp(log_prob_rl - log_prob_ref) - 1) - (log_prob_rl - log_prob_ref) # k3
        
        '''deepspeed 的实现是这样的：'''
        kl_divergence_estimate =  - kl_ctl * kl_divergence
        kl_divergence_estimate[-1, -1, 0] =  rewards[:,-1] + kl_divergence_estimate[-1, -1, 0]
        return kl_divergence_estimate.squeeze(-1)
    penalized_rewards = kl_penalized_reward(rewards, actor_log_probs, sft_log_probs) 
    


    '''
    GAE 简单的例子
    idx         0           1           2           3           4
    rewards     0.7         0.8         0.9         1.0         1.1
    values      0.3         0.4         0.5         0.6         0.7
    1:          0.4         0.4         0.4         0.4         0.4
    2:          0.4+γ0.4    0.4+γ0.5    0.4+γ0.6    0.4+γ0.7    0.4+γ0
    3:
    其中
    1: rewards-values          
    2: delta = rewards-values + γ*nextvalues (其中nextvalues时values的下一个值)
    3：lastgaelam = rewards-values + γ*nextvalues + γ*λ*lastgaelam  是从后往前计算的 
        idx = 4 时  0.4 + γ0 + 0
        idx = 3 时  0.4 + γ0.7 + γλ(0.4)
        idx = 2 时  0.4 + γ0.6 + γλ(0.4 + γ0.7 + γλ(0.4))
        idx = 1 时  0.4 + γ0.5 + γλ(0.4 + γ0.6 + γλ(0.4 + γ0.7 + γλ(0.4)))
    returns ~ 移除了advantage 中 -value 的部分 (做个某个action的纯奖励了)
    '''
    def get_advantages_and_returns(rewards, values):
        gamma = 1.0
        lambd = 0.95

        lastgaelam = 0
        advantages_reversed = []
        length = rewards.shape[-1]
       
        for t in reversed(range(length)):
            nextvalues = values[:, t + 1] if t < length - 1 else 0.0
            delta = rewards[:, t] - values[:, t] + gamma * nextvalues 
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=-1)
        returns = advantages + values
        return  advantages, returns
    advantages, returns =  get_advantages_and_returns(penalized_rewards, values)

    actor_loss = actor_criterion(actor_log_probs, sft_log_probs, advantages)
    actor_loss.backward()  # 算梯度
    
    old_values = values # 这个old_values 直接用values代替不太合适，至少应该写在前面
    critic_loss = critic_criterion(values, old_values, returns)
    critic_loss.backward()
    
    
    
    # optimizer.step() # 更新参数....
    pdb.set_trace()
    

def main():
    pass
    # test_sth()
    run_ppo()

if __name__ == '__main__':
    pass
    main()


